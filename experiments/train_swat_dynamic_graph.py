from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swat_gnn.data.swat_temporal_dataset import (
    build_swat_temporal_datasets,
)
from swat_gnn.models.dynamic_graph_autoencoder import (
    DynamicGraphAutoencoder,
)


# =====================================================================
# Configuration
# =====================================================================
import os
SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))

SEQUENCE_LENGTH = 60

TRAIN_STRIDE = 5
VALIDATION_STRIDE = 1
TEST_STRIDE = 1

BATCH_SIZE = 256

TEMPORAL_HIDDEN_DIM = 32
EMBEDDING_DIM = 32
GAT_HIDDEN_DIM = 64
GAT_HEADS = 4
GRAPH_K = int(os.environ.get("GRAPH_K", "5"))

LEARNING_RATE = 1e-3

MAX_EPOCHS = 30
PATIENCE = 5

# ---------------------------------------------------------------------
# Smoke-test configuration
#
# Set this to True for the first run.
# After the smoke test succeeds, change it to False for full training.
# ---------------------------------------------------------------------

SMOKE_TEST = False
EVALUATE_ONLY = False

SMOKE_TRAIN_WINDOWS = 2000
SMOKE_VALIDATION_WINDOWS = 2000

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
)

TABLE_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / f"swat_dynamic_graph_autoencoder_k{GRAPH_K}_seed{SEED}_best.pt"
)

RESULTS_PATH = (
    TABLE_DIR
    / f"swat_dynamic_graph_autoencoder_k{GRAPH_K}_seed{SEED}_results.json"
)

HISTORY_PATH = (
    TABLE_DIR
    / f"swat_dynamic_graph_autoencoder_k{GRAPH_K}_seed{SEED}_history.json"
)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility is useful for the research experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Utility functions
# =====================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def count_parameters(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def limit_dataset(
    dataset,
    max_windows: int | None,
) -> torch.utils.data.Dataset:
    """
    Return a subset containing only the first max_windows windows.

    The full dataset remains untouched.
    """

    if max_windows is None:
        return dataset

    max_windows = min(
        max_windows,
        len(dataset),
    )

    return torch.utils.data.Subset(
        dataset,
        range(max_windows),
    )


# =====================================================================
# Dataset preparation
# =====================================================================

def prepare_data():
    print("\nLoading SWaT data...")

    datasets = build_swat_temporal_datasets(
        sequence_length=SEQUENCE_LENGTH,
        train_stride=TRAIN_STRIDE,
        evaluation_stride=VALIDATION_STRIDE,
    )

    feature_names = datasets["feature_names"]
    train_dataset = datasets["train_dataset"]
    validation_dataset = datasets["validation_dataset"]
    test_datasets = datasets["test_datasets"]
    test_labels = datasets["test_labels"]

    print(
        f"Training windows:   {len(train_dataset):,}"
    )

    print(
        f"Validation windows: {len(validation_dataset):,}"
    )

    for episode_idx, test_dataset in enumerate(test_datasets):
        print(
            f"Test episode {episode_idx + 1} windows: "
            f"{len(test_dataset):,}"
        )

    print(
        f"Features:            {len(feature_names)}"
    )

    return (
        train_dataset,
        validation_dataset,
        test_datasets,
        test_labels,
        feature_names,
    )


# =====================================================================
# DataLoader
# =====================================================================

def create_loader(
    dataset,
    shuffle: bool,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# =====================================================================
# Training
# =====================================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(
        loader
    ):

        batch = batch.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        reconstruction = model(batch)

        target = batch[:, -1, :]

        loss = torch.mean(
            (reconstruction - target) ** 2
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Training loss became NaN or infinite."
            )

        loss.backward()

        optimizer.step()

        batch_size = batch.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"  batch "
                f"{batch_idx + 1:,}/"
                f"{len(loader):,} "
                f"loss={loss.item():.6f}"
            )

    return total_loss / total_samples


# =====================================================================
# Validation loss
# =====================================================================

@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        batch = batch.to(
            device,
            non_blocking=True,
        )

        reconstruction = model(batch)

        target = batch[:, -1, :]

        loss = torch.mean(
            (reconstruction - target) ** 2
        )

        batch_size = batch.shape[0]

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


# =====================================================================
# Anomaly scoring
# =====================================================================

@torch.no_grad()
def score_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:

    model.eval()

    scores = []

    for batch in loader:

        batch = batch.to(
            device,
            non_blocking=True,
        )

        reconstruction = model(batch)

        target = batch[:, -1, :]

        batch_scores = torch.mean(
            (reconstruction - target) ** 2,
            dim=1,
        )

        scores.append(
            batch_scores.detach()
            .cpu()
            .numpy()
        )

    if not scores:
        return np.empty(
            0,
            dtype=np.float64,
        )

    return np.concatenate(
        scores
    )


# =====================================================================
# Threshold
# =====================================================================

def calculate_threshold(
    validation_scores: np.ndarray,
    percentile: float = 99.0,
) -> float:

    if validation_scores.ndim != 1:
        raise ValueError(
            "validation_scores must be 1D."
        )

    if len(validation_scores) == 0:
        raise ValueError(
            "No validation scores."
        )

    if not np.isfinite(
        validation_scores
    ).all():
        raise ValueError(
            "Validation scores contain "
            "NaN or infinite values."
        )

    return float(
        np.percentile(
            validation_scores,
            percentile,
        )
    )


# =====================================================================
# Metrics
# =====================================================================

def compute_test_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:

    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    labels = labels.astype(
        np.int64,
        copy=False,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    negatives = labels == 0

    if negatives.sum() > 0:
        false_positive_rate = float(
            (
                predictions[negatives] == 1
            ).mean()
        )
    else:
        false_positive_rate = float("nan")

    if len(np.unique(labels)) == 2:

        auroc = roc_auc_score(
            labels,
            scores,
        )

        auprc = average_precision_score(
            labels,
            scores,
        )

    else:

        auroc = float("nan")
        auprc = float("nan")

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(false_positive_rate),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "predicted_anomalies": int(
            predictions.sum()
        ),
        "actual_anomalies": int(
            labels.sum()
        ),
        "total_samples": int(
            len(labels)
        ),
    }


# =====================================================================
# Save JSON
# =====================================================================

def save_json(
    path: Path,
    data,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
        )


# =====================================================================
# Main
# =====================================================================

def main() -> None:

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Smoke test: {SMOKE_TEST}"
    )

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        test_datasets,
        test_labels,
        feature_names,
    ) = prepare_data()

    # ---------------------------------------------------------------
    # Smoke-test dataset sizes
    # ---------------------------------------------------------------

    if SMOKE_TEST:

        train_dataset = limit_dataset(
            train_dataset,
            SMOKE_TRAIN_WINDOWS,
        )

        validation_dataset = (
            limit_dataset(
                validation_dataset,
                SMOKE_VALIDATION_WINDOWS,
            )
        )

        # We do not use the full test set
        # during smoke testing.

        print(
            "\nSMOKE TEST DATA:"
        )

        print(
            f"Training windows: "
            f"{len(train_dataset):,}"
        )

        print(
            f"Validation windows: "
            f"{len(validation_dataset):,}"
        )

    # ---------------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------------

    train_loader = create_loader(
        train_dataset,
        shuffle=True,
    )

    validation_loader = create_loader(
        validation_dataset,
        shuffle=False,
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    model = DynamicGraphAutoencoder(
        sequence_length=SEQUENCE_LENGTH,
        num_nodes=len(feature_names),
        temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        gat_hidden_dim=GAT_HIDDEN_DIM,
        gat_heads=GAT_HEADS,
        k=GRAPH_K,
        dropout=0.0,
    ).to(device)

    print(
        "\nModel:"
    )

    print(model)

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    history = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    if not EVALUATE_ONLY:
        for epoch in range(
            1,
            MAX_EPOCHS + 1,
        ):

            print(
                f"\nEpoch "
                f"{epoch:02d}/{MAX_EPOCHS}"
            )

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

            validation_loss = evaluate_loss(
                model=model,
                loader=validation_loader,
                device=device,
            )

            print(
                f"Train loss: "
                f"{train_loss:.8f}"
            )

            print(
                f"Validation loss: "
                f"{validation_loss:.8f}"
            )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(
                        train_loss
                    ),
                    "validation_loss": float(
                        validation_loss
                    ),
                }
            )

            # -----------------------------------------------------------
            # Early stopping
            # -----------------------------------------------------------

            if validation_loss < (
                best_validation_loss
            ):

                best_validation_loss = (
                    validation_loss
                )

                best_epoch = epoch

                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "optimizer_state_dict":
                            optimizer.state_dict(),
                        "validation_loss":
                            validation_loss,
                        "config": {
                            "sequence_length":
                                SEQUENCE_LENGTH,
                            "num_nodes":
                                len(feature_names),
                            "temporal_hidden_dim":
                                TEMPORAL_HIDDEN_DIM,
                            "embedding_dim":
                                EMBEDDING_DIM,
                            "gat_hidden_dim":
                                GAT_HIDDEN_DIM,
                            "gat_heads":
                                GAT_HEADS,
                            "graph_k":
                                GRAPH_K,
                            "learning_rate":
                                LEARNING_RATE,
                            "seed":
                                SEED,
                        },
                        "feature_names":
                            feature_names,
                    },
                    CHECKPOINT_PATH,
                )

                print(
                    "  New best model saved."
                )

            else:

                epochs_without_improvement += 1

                print(
                    "  No validation improvement."
                )

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping."
                )

                break

    # ---------------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------------

    save_json(
        HISTORY_PATH,
        history,
    )

    print(
        f"\nBest epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.8f}"
    )

    # ---------------------------------------------------------------
    # Smoke-test only
    # ---------------------------------------------------------------

    if SMOKE_TEST:

        print(
            "\n" + "=" * 60
        )

        print(
            "SMOKE TEST COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            "Training pipeline executed "
            "successfully."
        )

        print(
            "No test-set evaluation was "
            "performed."
        )

        print(
            "\nIf everything above looks "
            "correct, change:"
        )

        print(
            "    SMOKE_TEST = False"
        )

        print(
            "and run the full experiment."
        )

        return

    # ---------------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    if EVALUATE_ONLY:
        best_epoch = int(
            checkpoint["epoch"]
        )
        best_validation_loss = float(
            checkpoint["validation_loss"]
        )

    print(
        f"\nBest checkpoint restored."
    )

    print(
        f"Checkpoint epoch: {best_epoch}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{best_validation_loss:.8f}"
    )

    # ---------------------------------------------------------------
    # Validation scoring
    # ---------------------------------------------------------------

    print(
        "\nScoring validation set..."
    )

    validation_scores = score_dataset(
        model=model,
        loader=validation_loader,
        device=device,
    )

    threshold = calculate_threshold(
        validation_scores,
        percentile=99.0,
    )

    print(
        f"Validation p99 threshold: "
        f"{threshold:.8f}"
    )

    # ---------------------------------------------------------------
    # Full test episodes
    # ---------------------------------------------------------------

    print(
        "\nScoring SWaT test episodes..."
    )

    test_score_parts = []

    for episode_idx, test_dataset in enumerate(
        test_datasets
    ):
        print(
            f"Scoring test episode "
            f"{episode_idx + 1}/{len(test_datasets)}..."
        )

        test_loader = create_loader(
            test_dataset,
            shuffle=False,
        )

        episode_scores = score_dataset(
            model=model,
            loader=test_loader,
            device=device,
        )

        test_score_parts.append(
            episode_scores
        )

    test_scores = np.concatenate(
        test_score_parts
    )

    # SWaT labels are already endpoint-aligned
    # with the scored windows.

    scored_labels = np.concatenate(
        test_labels
    )

    if len(test_scores) != len(
        scored_labels
    ):
        raise RuntimeError(
            "Test score/label length mismatch: "
            f"{len(test_scores)} vs "
            f"{len(scored_labels)}"
        )

    # ---------------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------------

    metrics = compute_test_metrics(
        scores=test_scores,
        labels=scored_labels,
        threshold=threshold,
    )

    metrics.update(
        {
            "model":
                "DynamicGraphAutoencoder",
            "dataset":
                "SWaT",
            "sequence_length":
                SEQUENCE_LENGTH,
            "train_stride":
                TRAIN_STRIDE,
            "validation_stride":
                VALIDATION_STRIDE,
            "test_stride":
                TEST_STRIDE,
            "graph_k":
                GRAPH_K,
            "best_epoch":
                best_epoch,
            "best_validation_loss":
                best_validation_loss,
            "seed":
                SEED,
            "num_features":
                len(feature_names),
            "test_score_coverage":
                len(test_scores)
                / sum(
                    len(labels)
                    for labels in test_labels
                ),
        }
    )

    save_json(
        RESULTS_PATH,
        metrics,
    )

    # ---------------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 60
    )

    for key in [
        "precision",
        "recall",
        "f1",
        "fpr",
        "auroc",
        "auprc",
    ]:

        print(
            f"{key:>12}: "
            f"{metrics[key]:.6f}"
        )

    print(
        f"{'threshold':>12}: "
        f"{metrics['threshold']:.8f}"
    )

    print(
        f"{'predicted':>12}: "
        f"{metrics['predicted_anomalies']:,}"
    )

    print(
        f"{'actual':>12}: "
        f"{metrics['actual_anomalies']:,}"
    )

    print(
        f"{'coverage':>12}: "
        f"{metrics['test_score_coverage']:.6f}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_PATH
    )

    print(
        "\nCheckpoint saved to:"
    )

    print(
        CHECKPOINT_PATH
    )


if __name__ == "__main__":
    main()



