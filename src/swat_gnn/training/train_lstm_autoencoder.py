from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from swat_gnn.data.feature_analysis import analyze_features
from swat_gnn.data.loader import (
    get_dataset_root,
    load_config,
    load_test_data,
    load_training_data,
    load_validation_data,
)
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.evaluation.metrics import compute_binary_metrics
from swat_gnn.models.lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderConfig,
)


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------

def make_loader(
    dataset: TemporalWindowDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        reconstruction = model(batch)

        loss = nn.functional.mse_loss(
            reconstruction,
            batch,
        )

        loss.backward()
        optimizer.step()

        batch_size = batch.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_reconstruction(
    model: LSTMAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_scores: list[np.ndarray] = []

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        reconstruction = model(batch)

        loss = nn.functional.mse_loss(
            reconstruction,
            batch,
        )

        batch_size = batch.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Primary anomaly score:
        # reconstruction MSE at the final timestep.
        final_error = (
            reconstruction[:, -1, :] - batch[:, -1, :]
        ).pow(2).mean(dim=1)

        all_scores.append(
            final_error.detach().cpu().numpy()
        )

    mean_loss = total_loss / total_samples
    scores = np.concatenate(all_scores)

    return mean_loss, scores


# ---------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------

def calculate_thresholds(
    validation_scores: np.ndarray,
) -> dict[str, float]:
    percentiles = [95, 97, 98, 99, 99.5]

    return {
        f"p{str(percentile).replace('.', '_')}": float(
            np.percentile(validation_scores, percentile)
        )
        for percentile in percentiles
    }


# ---------------------------------------------------------------------
# Test scoring
# ---------------------------------------------------------------------

@torch.no_grad()
def score_test_segments(
    model: LSTMAutoencoder,
    test_segments: list[np.ndarray],
    test_labels: list[np.ndarray],
    sequence_length: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model.eval()

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    total_timesteps = 0
    scored_timesteps = 0

    segment_information = []

    for segment_id, (segment, labels) in enumerate(
        zip(test_segments, test_labels)
    ):
        total_timesteps += len(segment)

        dataset = TemporalWindowDataset(
            segments=[segment],
            sequence_length=sequence_length,
            stride=1,
        )

        loader = make_loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            device=device,
        )

        segment_scores: list[np.ndarray] = []

        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            reconstruction = model(batch)

            final_error = (
                reconstruction[:, -1, :] - batch[:, -1, :]
            ).pow(2).mean(dim=1)

            segment_scores.append(
                final_error.detach().cpu().numpy()
            )

        scores = np.concatenate(segment_scores)

        # A 60-second window ending at timestep t produces
        # a score for timestep t. Therefore the first 59
        # timesteps have no score.
        aligned_labels = labels[sequence_length - 1 :]

        if len(scores) != len(aligned_labels):
            raise RuntimeError(
                f"Score/label alignment mismatch for test segment "
                f"{segment_id}: scores={len(scores)}, "
                f"labels={len(aligned_labels)}"
            )

        all_scores.append(scores)
        all_labels.append(aligned_labels)

        scored_timesteps += len(scores)

        segment_information.append(
            {
                "segment_id": segment_id,
                "raw_timesteps": len(segment),
                "scored_timesteps": len(scores),
                "excluded_initial_timesteps": sequence_length - 1,
                "attack_timesteps_scored": int(aligned_labels.sum()),
            }
        )

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    coverage = scored_timesteps / total_timesteps

    coverage_information = {
        "total_test_timesteps": total_timesteps,
        "scored_test_timesteps": scored_timesteps,
        "coverage": coverage,
        "excluded_timesteps": total_timesteps - scored_timesteps,
        "segments": segment_information,
    }

    return scores, labels, coverage_information


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    config_path = project_root / "configs" / "data.yaml"

    config = load_config(str(config_path))

    # -------------------------------------------------------------
    # Experiment settings
    # -------------------------------------------------------------

    sequence_length = 60

    train_stride = 5
    validation_stride = 1
    test_stride = 1  # currently fixed to 1 in test scoring

    batch_size = 256

    model_config = LSTMAutoencoderConfig(
        input_dim=66,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1,
        learning_rate=1e-3,
        batch_size=batch_size,
        max_epochs=30,
        patience=5,
        random_state=42,
        device="auto",
    )

    seed = model_config.random_state

    set_seed(seed)

    device = get_device()

    print("=" * 70)
    print("LSTM AUTOENCODER — HAI 23.05")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Sequence length: {sequence_length}")
    print(f"Training stride: {train_stride}")
    print(f"Validation stride: {validation_stride}")
    print(f"Batch size: {batch_size}")
    print()

    # -------------------------------------------------------------
    # Load raw data
    # -------------------------------------------------------------

    print("Loading HAI data...")

    training_segments = load_training_data(config)
    validation_segments = load_validation_data(config)
    test_sensor_segments, test_labels = load_test_data(config)

    print(
        f"Training segments: {len(training_segments)}"
    )
    print(
        f"Validation segments: {len(validation_segments)}"
    )
    print(
        f"Test segments: {len(test_sensor_segments)}"
    )

    # -------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------

    print()
    print("Analyzing training features...")

    feature_analysis = analyze_features(training_segments)

    feature_names = feature_analysis["variable_features"]
    constant_features = feature_analysis["constant_features"]

    print(f"Total features: {len(feature_analysis['feature_names'])}")
    print(f"Constant features: {len(constant_features)}")
    print(f"Variable features: {len(feature_names)}")

    print()
    print("Excluded constant features:")
    for feature in constant_features:
        print(f"  - {feature}")

    # -------------------------------------------------------------
    # Select only variable features
    # -------------------------------------------------------------

    def select_features(segments):
        selected = []

        for df in segments:
            selected.append(
                df[feature_names].copy()
            )

        return selected

    training_selected = select_features(training_segments)
    validation_selected = select_features(validation_segments)
    test_selected = select_features(test_sensor_segments)

    # -------------------------------------------------------------
    # Train-only normalization
    # -------------------------------------------------------------

    print()
    print("Fitting StandardScaler on training data only...")

    preprocessor = Preprocessor()

    # Preprocessor expects the same feature set as the dataframes.
    scaled_training = preprocessor.fit_transform(
        training_selected
    )

    scaled_validation = preprocessor.transform(
        validation_selected
    )

    scaled_test = preprocessor.transform(
        test_selected
    )

    print("Normalization complete.")

    # -------------------------------------------------------------
    # Temporal datasets
    # -------------------------------------------------------------

    print()
    print("Creating temporal datasets...")

    train_dataset = TemporalWindowDataset(
        segments=scaled_training,
        sequence_length=sequence_length,
        stride=train_stride,
    )

    validation_dataset = TemporalWindowDataset(
        segments=scaled_validation,
        sequence_length=sequence_length,
        stride=validation_stride,
    )

    print(f"Training windows: {len(train_dataset):,}")
    print(f"Validation windows: {len(validation_dataset):,}")

    train_loader = make_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        device=device,
    )

    validation_loader = make_loader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        device=device,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    model = LSTMAutoencoder(
        input_dim=len(feature_names),
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        num_layers=model_config.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config.learning_rate,
    )

    print()
    print("Model:")
    print(model)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(f"Trainable parameters: {parameter_count:,}")

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_history = []

    checkpoint_dir = (
        project_root
        / "results"
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_checkpoint_path = (
        checkpoint_dir
        / "lstm_autoencoder_best.pt"
    )

    for epoch in range(1, model_config.max_epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, _ = evaluate_reconstruction(
            model=model,
            loader=validation_loader,
            device=device,
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:02d}/{model_config.max_epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "input_dim": len(feature_names),
                    "hidden_dim": model_config.hidden_dim,
                    "latent_dim": model_config.latent_dim,
                    "num_layers": model_config.num_layers,
                    "sequence_length": sequence_length,
                    "feature_names": feature_names,
                    "seed": seed,
                },
                best_checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= model_config.patience
        ):
            print()
            print(
                f"Early stopping after epoch {epoch}."
            )
            break

    # -------------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------------

    print()
    print(
        f"Restoring best model from epoch {best_epoch}..."
    )

    checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -------------------------------------------------------------
    # Validation scores
    # -------------------------------------------------------------

    print()
    print("Calculating validation reconstruction scores...")

    validation_loss, validation_scores = (
        evaluate_reconstruction(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    thresholds = calculate_thresholds(
        validation_scores
    )

    print()
    print("Validation thresholds:")

    for name, threshold in thresholds.items():
        print(
            f"  {name}: {threshold:.8f}"
        )

    primary_threshold = thresholds["p99"]

    print()
    print(
        f"Primary threshold (99th percentile): "
        f"{primary_threshold:.8f}"
    )

    # -------------------------------------------------------------
    # Test scoring
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST EVALUATION")
    print("=" * 70)

    # test_stride is conceptually part of the protocol.
    # score_test_segments currently uses stride=1 deliberately.
    if test_stride != 1:
        raise ValueError(
            "Test scoring is currently implemented with stride=1."
        )

    test_scores, test_ground_truth, coverage = (
        score_test_segments(
            model=model,
            test_segments=scaled_test,
            test_labels=[
                labels.to_numpy()
                for labels in test_labels
            ],
            sequence_length=sequence_length,
            device=device,
            batch_size=batch_size,
        )
    )

    test_predictions = (
        test_scores >= primary_threshold
    ).astype(np.int64)

    metrics = compute_binary_metrics(
        labels=test_ground_truth,
        predictions=test_predictions,
        scores=test_scores,
    )

    print()
    print("Test results:")
    print("-" * 70)

    for metric_name, value in metrics.items():
        print(
            f"{metric_name:>12}: {value:.6f}"
        )

    print()
    print(
        f"Predicted anomaly timesteps: "
        f"{test_predictions.sum():,}"
    )

    print(
        f"Actual anomaly timesteps: "
        f"{test_ground_truth.sum():,}"
    )

    print(
        f"Test timesteps scored: "
        f"{coverage['scored_test_timesteps']:,} / "
        f"{coverage['total_test_timesteps']:,}"
    )

    print(
        f"Test coverage: "
        f"{coverage['coverage'] * 100:.4f}%"
    )

    print(
        f"Excluded initial timesteps: "
        f"{coverage['excluded_timesteps']}"
    )

    # -------------------------------------------------------------
    # Save experiment results
    # -------------------------------------------------------------

    results_dir = (
        project_root
        / "results"
        / "tables"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiment_results = {
        "experiment": "lstm_autoencoder_hai_23_05",
        "dataset": "HAI_23.05",
        "seed": seed,
        "device": str(device),
        "num_total_features": len(
            feature_analysis["feature_names"]
        ),
        "num_constant_features": len(
            constant_features
        ),
        "num_variable_features": len(
            feature_names
        ),
        "sequence_length": sequence_length,
        "train_stride": train_stride,
        "validation_stride": validation_stride,
        "test_stride": test_stride,
        "batch_size": batch_size,
        "hidden_dim": model_config.hidden_dim,
        "latent_dim": model_config.latent_dim,
        "num_layers": model_config.num_layers,
        "learning_rate": model_config.learning_rate,
        "max_epochs": model_config.max_epochs,
        "patience": model_config.patience,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "validation_thresholds": thresholds,
        "primary_threshold": primary_threshold,
        "test_metrics": {
            key: (
                None
                if not np.isfinite(value)
                else float(value)
            )
            for key, value in metrics.items()
        },
        "predicted_anomaly_timesteps": int(
            test_predictions.sum()
        ),
        "actual_anomaly_timesteps": int(
            test_ground_truth.sum()
        ),
        "test_coverage": coverage,
        "feature_names": feature_names,
        "constant_features": constant_features,
        "training_history": training_history,
    }

    results_path = (
        results_dir
        / "lstm_autoencoder_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            experiment_results,
            f,
            indent=2,
        )

    history_path = (
        results_dir
        / "lstm_autoencoder_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            training_history,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to: {results_path}"
    )

    print(
        f"Training history saved to: {history_path}"
    )

    print(
        f"Best checkpoint saved to: "
        f"{best_checkpoint_path}"
    )


if __name__ == "__main__":
    main()