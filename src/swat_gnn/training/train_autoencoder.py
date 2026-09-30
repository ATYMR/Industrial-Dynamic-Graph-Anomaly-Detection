from __future__ import annotations

from pathlib import Path
import sys
import copy
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


# ============================================================
# PROJECT IMPORTS
# ============================================================

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
    load_test_data,
)

from swat_gnn.models.autoencoder import (
    AutoencoderConfig,
    FeedForwardAutoencoder,
    get_device,
    set_random_seed,
)

from swat_gnn.evaluation.metrics import (
    apply_threshold,
    compute_binary_metrics,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

INPUT_DIM = 66

HIDDEN_DIM_1 = 32
HIDDEN_DIM_2 = 16
LATENT_DIM = 8

LEARNING_RATE = 1e-3

BATCH_SIZE = 1024

MAX_EPOCHS = 50

PATIENCE = 7

WEIGHT_DECAY = 0.0

DEVICE = "auto"

# ------------------------------------------------------------
# Threshold protocol
# ------------------------------------------------------------
#
# Validation is normal-only.
#
# Therefore we cannot choose a threshold using validation F1.
#
# The primary operating point is the 99th percentile of
# normal validation reconstruction errors.
#
# This is frozen before evaluating the labeled test set.
# ------------------------------------------------------------

PRIMARY_PERCENTILE = 99.0

THRESHOLD_PERCENTILES = [
    95.0,
    97.0,
    98.0,
    99.0,
    99.5,
]


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_variable_features(
    training_segments: list[pd.DataFrame],
) -> list[str]:

    if not training_segments:
        raise ValueError(
            "No training segments were provided."
        )

    feature_names = [
        column
        for column in training_segments[0].columns
        if column != "timestamp"
    ]

    variable_features: list[str] = []

    for feature in feature_names:

        is_constant = all(
            df[feature].nunique(
                dropna=False
            ) <= 1
            for df in training_segments
        )

        if not is_constant:
            variable_features.append(
                feature
            )

    return variable_features


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    segments: list[pd.DataFrame],
    feature_names: list[str],
) -> list[np.ndarray]:

    if not segments:
        raise ValueError(
            "No dataframe segments were provided."
        )

    arrays: list[np.ndarray] = []

    for segment_index, df in enumerate(segments):

        missing = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Segment {segment_index} is missing "
                f"features: {missing}"
            )

        values = df[
            feature_names
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Segment {segment_index} contains "
                "NaN or infinite values."
            )

        arrays.append(values)

    return arrays


# ============================================================
# CONCATENATION
# ============================================================

def concatenate_segments(
    segments: list[np.ndarray],
) -> np.ndarray:

    if not segments:
        raise ValueError(
            "No arrays were provided."
        )

    return np.concatenate(
        segments,
        axis=0,
    )


# ============================================================
# DATASET
# ============================================================

class SensorDataset(TensorDataset):
    """
    TensorDataset wrapper used to make the training intent
    explicit.
    """

    def __init__(
        self,
        data: np.ndarray,
    ) -> None:

        if data.ndim != 2:
            raise ValueError(
                "data must have shape "
                "(samples, features)."
            )

        if not np.isfinite(data).all():
            raise ValueError(
                "data contains NaN or infinite values."
            )

        tensor = torch.from_numpy(
            data.astype(
                np.float32,
                copy=False,
            )
        )

        super().__init__(
            tensor
        )


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_validation_threshold(
    validation_scores: np.ndarray,
    percentile: float,
) -> tuple[float, float]:

    scores = np.asarray(
        validation_scores,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError(
            "validation_scores must be 1D."
        )

    if scores.size == 0:
        raise ValueError(
            "validation_scores cannot be empty."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "validation_scores contain NaN or infinite values."
        )

    if not 0.0 < percentile < 100.0:
        raise ValueError(
            "percentile must be between 0 and 100."
        )

    threshold = float(
        np.percentile(
            scores,
            percentile,
        )
    )

    predictions = apply_threshold(
        scores,
        threshold,
    )

    flag_rate = float(
        predictions.mean()
    )

    return (
        threshold,
        flag_rate,
    )


# ============================================================
# TRAINING
# ============================================================

def train_autoencoder(
    model: FeedForwardAutoencoder,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> dict[str, list[float] | int | float]:

    criterion = nn.MSELoss(
        reduction="mean"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_loss = float(
        "inf"
    )

    best_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    train_history: list[float] = []

    validation_history: list[float] = []

    model.to(device)

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        training_loss_sum = 0.0
        training_samples = 0

        for batch in train_loader:

            x = batch[0].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            reconstruction = model(
                x
            )

            loss = criterion(
                reconstruction,
                x,
            )

            loss.backward()

            optimizer.step()

            batch_size = x.shape[0]

            training_loss_sum += (
                loss.item()
                * batch_size
            )

            training_samples += (
                batch_size
            )

        train_loss = (
            training_loss_sum
            / training_samples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss_sum = 0.0
        validation_samples = 0

        with torch.no_grad():

            for batch in validation_loader:

                x = batch[0].to(
                    device,
                    non_blocking=True,
                )

                reconstruction = model(
                    x
                )

                loss = criterion(
                    reconstruction,
                    x,
                )

                batch_size = x.shape[0]

                validation_loss_sum += (
                    loss.item()
                    * batch_size
                )

                validation_samples += (
                    batch_size
                )

        validation_loss = (
            validation_loss_sum
            / validation_samples
        )

        train_history.append(
            float(train_loss)
        )

        validation_history.append(
            float(validation_loss)
        )

        print(
            f"Epoch {epoch:03d}"
            f" | train_loss={train_loss:.8f}"
            f" | val_loss={validation_loss:.8f}"
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if validation_loss < (
            best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                f"Early stopping after "
                f"{epoch} epochs."
            )

            break

    # --------------------------------------------------------
    # Restore best validation model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    best_epoch = int(
        np.argmin(
            validation_history
        )
        + 1
    )

    return {
        "train_loss": train_history,
        "validation_loss": validation_history,
        "best_validation_loss": (
            float(best_validation_loss)
        ),
        "best_epoch": best_epoch,
        "epochs_trained": len(
            train_history
        ),
    }


# ============================================================
# SCORE DATA
# ============================================================

def compute_reconstruction_scores(
    model: FeedForwardAutoencoder,
    data: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:

    if data.ndim != 2:
        raise ValueError(
            "data must have shape "
            "(samples, features)."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "data contains NaN or infinite values."
        )

    dataset = SensorDataset(
        data
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model.eval()

    scores: list[np.ndarray] = []

    with torch.no_grad():

        for batch in loader:

            x = batch[0].to(
                device,
                non_blocking=True,
            )

            batch_scores = (
                model.reconstruction_error(
                    x
                )
            )

            scores.append(
                batch_scores
                .detach()
                .cpu()
                .numpy()
            )

    result = np.concatenate(
        scores,
        axis=0,
    )

    if not np.isfinite(result).all():
        raise ValueError(
            "Reconstruction scores contain "
            "NaN or infinite values."
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 1. Reproducibility
    # --------------------------------------------------------

    set_random_seed(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # 2. Configuration validation
    # --------------------------------------------------------

    model_config = AutoencoderConfig(
        input_dim=INPUT_DIM,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        latent_dim=LATENT_DIM,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        random_state=RANDOM_STATE,
        device=DEVICE,
    )

    model_config.validate()

    # --------------------------------------------------------
    # 3. Device
    # --------------------------------------------------------

    device = get_device(
        DEVICE
    )

    print()
    print("=" * 70)
    print("HAI 23.05 — Autoencoder Baseline")
    print("=" * 70)

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # 4. Load configuration
    # --------------------------------------------------------

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    config = load_config(
        str(config_path)
    )

    # --------------------------------------------------------
    # 5. Load datasets
    # --------------------------------------------------------

    print()
    print("Loading data...")

    training_segments = load_training_data(
        config
    )

    validation_segments = load_validation_data(
        config
    )

    test_segments, test_labels = load_test_data(
        config
    )

    # --------------------------------------------------------
    # 6. Select variable features
    # --------------------------------------------------------

    feature_names = get_variable_features(
        training_segments
    )

    print(
        f"Total model features: "
        f"{len(feature_names)}"
    )

    if len(feature_names) != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} variable features, "
            f"but found {len(feature_names)}."
        )

    # --------------------------------------------------------
    # 7. Extract features
    # --------------------------------------------------------

    X_train_segments = extract_features(
        training_segments,
        feature_names,
    )

    X_validation_segments = extract_features(
        validation_segments,
        feature_names,
    )

    X_test_segments = extract_features(
        test_segments,
        feature_names,
    )

    # --------------------------------------------------------
    # 8. Concatenate
    # --------------------------------------------------------

    X_train = concatenate_segments(
        X_train_segments
    )

    X_validation = concatenate_segments(
        X_validation_segments
    )

    X_test = concatenate_segments(
        X_test_segments
    )

    y_test = np.concatenate(
        [
            labels.to_numpy(
                dtype=np.int64
            )
            for labels in test_labels
        ]
    )

    # --------------------------------------------------------
    # 9. Dataset summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Training observations   : "
        f"{X_train.shape}"
    )

    print(
        f"Validation observations : "
        f"{X_validation.shape}"
    )

    print(
        f"Test observations       : "
        f"{X_test.shape}"
    )

    print(
        f"Test attack observations: "
        f"{int(y_test.sum())}"
    )

    print(
        f"Test normal observations: "
        f"{int((y_test == 0).sum())}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 10. Standardization
    #
    # FIT ONLY ON TRAINING DATA.
    # --------------------------------------------------------

    print()
    print(
        "Fitting StandardScaler on training data..."
    )

    scaler = StandardScaler()

    scaler.fit(
        X_train
    )

    X_train_scaled = scaler.transform(
        X_train
    ).astype(
        np.float32,
        copy=False,
    )

    X_validation_scaled = scaler.transform(
        X_validation
    ).astype(
        np.float32,
        copy=False,
    )

    X_test_scaled = scaler.transform(
        X_test
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "Standardization complete."
    )

    # --------------------------------------------------------
    # 11. Create PyTorch datasets
    # --------------------------------------------------------

    train_dataset = SensorDataset(
        X_train_scaled
    )

    validation_dataset = SensorDataset(
        X_validation_scaled
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # 12. Create model
    # --------------------------------------------------------

    model = FeedForwardAutoencoder(
        input_dim=INPUT_DIM,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        latent_dim=LATENT_DIM,
    )

    model.to(
        device
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("=" * 70)
    print("AUTOENCODER")
    print("=" * 70)

    print(
        f"Input dimension      : {INPUT_DIM}"
    )

    print(
        f"Hidden dimensions    : "
        f"{HIDDEN_DIM_1} → "
        f"{HIDDEN_DIM_2}"
    )

    print(
        f"Latent dimension     : {LATENT_DIM}"
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 13. Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    training_history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )

    print()
    print(
        f"Best epoch: "
        f"{training_history['best_epoch']}"
    )

    print(
        f"Best validation loss: "
        f"{training_history['best_validation_loss']:.8f}"
    )

    # --------------------------------------------------------
    # 14. Compute validation anomaly scores
    # --------------------------------------------------------

    print()
    print(
        "Computing validation reconstruction errors..."
    )

    validation_scores = compute_reconstruction_scores(
        model=model,
        data=X_validation_scaled,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------------
    # 15. Threshold sensitivity
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold_rows: list[dict[str, float]] = []

    for percentile in THRESHOLD_PERCENTILES:

        threshold, flag_rate = (
            select_validation_threshold(
                validation_scores,
                percentile,
            )
        )

        threshold_rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "validation_flag_rate": flag_rate,
            }
        )

        print(
            f"{percentile:6.1f}th percentile"
            f" | threshold={threshold:.8f}"
            f" | validation flag rate="
            f"{flag_rate:.6%}"
        )

    # --------------------------------------------------------
    # 16. Freeze primary threshold
    # --------------------------------------------------------

    threshold, validation_flag_rate = (
        select_validation_threshold(
            validation_scores,
            PRIMARY_PERCENTILE,
        )
    )

    print()
    print(
        f"Primary percentile: "
        f"{PRIMARY_PERCENTILE:.1f}"
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.8f}"
    )

    print(
        f"Validation flag rate: "
        f"{validation_flag_rate:.6%}"
    )

    # --------------------------------------------------------
    # 17. Compute test anomaly scores
    # --------------------------------------------------------

    print()
    print(
        "Computing test reconstruction errors..."
    )

    test_scores = compute_reconstruction_scores(
        model=model,
        data=X_test_scaled,
        device=device,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------------
    # 18. Apply frozen threshold
    # --------------------------------------------------------

    test_predictions = apply_threshold(
        test_scores,
        threshold,
    )

    # --------------------------------------------------------
    # 19. Compute metrics
    # --------------------------------------------------------

    metrics = compute_binary_metrics(
        labels=y_test,
        predictions=test_predictions,
        scores=test_scores,
    )

    # --------------------------------------------------------
    # 20. Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Precision : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"FPR       : "
        f"{metrics['fpr']:.6f}"
    )

    print(
        f"AUROC     : "
        f"{metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC     : "
        f"{metrics['auprc']:.6f}"
    )

    print()

    print(
        f"Predicted anomalies : "
        f"{int(test_predictions.sum())}"
    )

    print(
        f"Actual anomalies    : "
        f"{int(y_test.sum())}"
    )

    print(
        f"Total test samples  : "
        f"{len(y_test)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 21. Save results
    # --------------------------------------------------------

    results_dir = (
        PROJECT_ROOT
        / "results"
        / "tables"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_row = {
        "dataset": "HAI_23.05",
        "model": "Feed-Forward Autoencoder",
        "input_dim": INPUT_DIM,
        "hidden_dim_1": HIDDEN_DIM_1,
        "hidden_dim_2": HIDDEN_DIM_2,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": training_history[
            "best_epoch"
        ],
        "best_validation_loss": (
            training_history[
                "best_validation_loss"
            ]
        ),
        "random_state": RANDOM_STATE,
        "device": str(device),
        "threshold_percentile": (
            PRIMARY_PERCENTILE
        ),
        "threshold": threshold,
        "validation_flag_rate": (
            validation_flag_rate
        ),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "test_samples": len(y_test),
        "test_attack_samples": int(
            y_test.sum()
        ),
        "predicted_anomalies": int(
            test_predictions.sum()
        ),
    }

    results_df = pd.DataFrame(
        [result_row]
    )

    results_path = (
        results_dir
        / "autoencoder_baseline.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # 22. Save threshold analysis
    # --------------------------------------------------------

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_path = (
        results_dir
        / "autoencoder_thresholds.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    # --------------------------------------------------------
    # 23. Save training history
    # --------------------------------------------------------

    history_df = pd.DataFrame(
        {
            "epoch": np.arange(
                1,
                training_history[
                    "epochs_trained"
                ]
                + 1,
            ),
            "train_loss": training_history[
                "train_loss"
            ],
            "validation_loss": training_history[
                "validation_loss"
            ],
        }
    )

    history_path = (
        results_dir
        / "autoencoder_training_history.csv"
    )

    history_df.to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # 24. Save experiment configuration
    # --------------------------------------------------------

    config_output = {
        "dataset": "HAI_23.05",
        "model": "Feed-Forward Autoencoder",
        "input_dim": INPUT_DIM,
        "hidden_dim_1": HIDDEN_DIM_1,
        "hidden_dim_2": HIDDEN_DIM_2,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "random_state": RANDOM_STATE,
        "device": str(device),
        "threshold_percentile": (
            PRIMARY_PERCENTILE
        ),
        "feature_count": len(
            feature_names
        ),
        "feature_names": feature_names,
    }

    config_output_path = (
        results_dir
        / "autoencoder_experiment_config.json"
    )

    with open(
        config_output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config_output,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # 25. Completion
    # --------------------------------------------------------

    print()
    print(
        "Results saved:"
    )

    print(
        f"  {results_path}"
    )

    print(
        f"  {threshold_path}"
    )

    print(
        f"  {history_path}"
    )

    print(
        f"  {config_output_path}"
    )

    print()
    print(
        "Autoencoder experiment completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()