from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from swat_gnn.data.swat_temporal_dataset import build_swat_temporal_datasets
from swat_gnn.evaluation.metrics import compute_binary_metrics
from swat_gnn.models.autoencoder import FeedForwardAutoencoder


# ============================================================
# Configuration
# ============================================================

SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))

SEQUENCE_LENGTH = 60
TRAIN_STRIDE = 5
EVALUATION_STRIDE = 1

HIDDEN_DIM_1 = 32
HIDDEN_DIM_2 = 16
LATENT_DIM = 8

LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 50
PATIENCE = 7
WEIGHT_DECAY = 0.0

THRESHOLD_PERCENTILE = 99.0

CHECKPOINT_PATH = Path(
    "results/checkpoints/swat_autoencoder_best.pt"
)

RESULTS_PATH = Path(
    "results/tables/swat_autoencoder_baseline.csv"
)

CONFIG_PATH = Path(
    "results/tables/swat_autoencoder_experiment_config.json"
)

HISTORY_PATH = Path(
    "results/tables/swat_autoencoder_training_history.csv"
)

THRESHOLD_PATH = Path(
    "results/tables/swat_autoencoder_thresholds.csv"
)

SCORES_PATH = Path(
    "results/tables/swat_autoencoder_scores.npz"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Utilities
# ============================================================

def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def score_dataset(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    scores = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(
                device,
                non_blocking=True,
            )

            # Feature-only baseline:
            # use only the endpoint observation.
            x = batch[:, -1, :]

            reconstruction = model(x)

            batch_scores = torch.mean(
                (reconstruction - x) ** 2,
                dim=1,
            )

            scores.append(
                batch_scores.cpu().numpy()
            )

    return np.concatenate(scores)


def evaluate_episode(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    return compute_binary_metrics(
        labels=labels,
        predictions=predictions,
        scores=scores,
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load validated SWaT pipeline
    # --------------------------------------------------------

    print("\nLoading SWaT datasets...")

    datasets = build_swat_temporal_datasets(
        sequence_length=SEQUENCE_LENGTH,
        train_stride=TRAIN_STRIDE,
        evaluation_stride=EVALUATION_STRIDE,
    )

    feature_names = datasets["feature_names"]
    train_dataset = datasets["train_dataset"]
    validation_dataset = datasets["validation_dataset"]
    test_datasets = datasets["test_datasets"]
    test_labels = datasets["test_labels"]

    input_dim = len(feature_names)

    print(f"Features: {input_dim}")
    print(f"Training windows: {len(train_dataset):,}")
    print(
        f"Validation windows: "
        f"{len(validation_dataset):,}"
    )

    print(
        "Test windows: "
        + ", ".join(
            f"{len(dataset):,}"
            for dataset in test_datasets
        )
    )

    print(
        "Test attack windows: "
        + ", ".join(
            f"{int(labels.sum()):,}"
            for labels in test_labels
        )
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FeedForwardAutoencoder(
        input_dim=input_dim,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        latent_dim=LATENT_DIM,
    ).to(device)

    print("\nModel:")
    print(model)

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        sample_count = 0

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch = batch.to(
                device,
                non_blocking=True,
            )

            x = batch[:, -1, :]

            optimizer.zero_grad(
                set_to_none=True
            )

            reconstruction = model(x)

            loss = criterion(
                reconstruction,
                x,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss "
                    f"at epoch {epoch}, "
                    f"batch {batch_index}: "
                    f"{loss.item()}"
                )

            loss.backward()
            optimizer.step()

            actual_batch_size = batch.shape[0]

            running_loss += (
                loss.item()
                * actual_batch_size
            )

            sample_count += actual_batch_size

        train_loss = (
            running_loss / sample_count
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        validation_loss_total = 0.0
        validation_samples = 0

        with torch.no_grad():

            for batch in validation_loader:

                batch = batch.to(
                    device,
                    non_blocking=True,
                )

                x = batch[:, -1, :]

                reconstruction = model(x)

                loss = criterion(
                    reconstruction,
                    x,
                )

                validation_loss_total += (
                    loss.item()
                    * batch.shape[0]
                )

                validation_samples += (
                    batch.shape[0]
                )

        validation_loss = (
            validation_loss_total
            / validation_samples
        )

        print(
            f"Train loss: {train_loss:.8f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.8f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        # ----------------------------------------------------
        # Checkpoint / early stopping
        # ----------------------------------------------------

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
                    "feature_names": feature_names,
                    "input_dim": input_dim,
                    "sequence_length": SEQUENCE_LENGTH,
                    "train_stride": TRAIN_STRIDE,
                    "evaluation_stride": EVALUATION_STRIDE,
                    "hidden_dim_1": HIDDEN_DIM_1,
                    "hidden_dim_2": HIDDEN_DIM_2,
                    "latent_dim": LATENT_DIM,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                },
                CHECKPOINT_PATH,
            )

            print("  New best model saved.")

        else:

            epochs_without_improvement += 1

            print("  No validation improvement.")

            if epochs_without_improvement >= PATIENCE:

                print("\nEarly stopping.")
                break

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------

    print(f"\nBest epoch: {best_epoch}")

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.8f}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    print("\nScoring validation set...")

    validation_scores = score_dataset(
        model,
        validation_dataset,
        device,
    )

    threshold = float(
        np.percentile(
            validation_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    validation_flag_rate = float(
        np.mean(validation_scores >= threshold)
    )

    print(
        f"Validation p99 threshold: "
        f"{threshold:.8f}"
    )

    print(
        f"Validation flag rate: "
        f"{validation_flag_rate:.6f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("\nScoring SWaT test episodes...")

    episode_scores = []
    episode_metrics = []

    for episode_index, (
        test_dataset,
        labels,
    ) in enumerate(
        zip(test_datasets, test_labels),
        start=1,
    ):

        scores = score_dataset(
            model,
            test_dataset,
            device,
        )

        if len(scores) != len(labels):
            raise RuntimeError(
                "Test score/label mismatch "
                f"for episode {episode_index}: "
                f"scores={len(scores)}, "
                f"labels={len(labels)}"
            )

        metrics = evaluate_episode(
            scores,
            labels,
            threshold,
        )

        episode_scores.append(scores)
        episode_metrics.append(metrics)

        print(
            f"\nEpisode {episode_index}:"
        )

        for name, value in metrics.items():
            print(
                f"  {name}: {value:.8f}"
            )

    # --------------------------------------------------------
    # Combined test evaluation
    # --------------------------------------------------------

    combined_scores = np.concatenate(
        episode_scores
    )

    combined_labels = np.concatenate(
        test_labels
    )

    combined_predictions = (
        combined_scores >= threshold
    ).astype(np.int64)

    combined_metrics = compute_binary_metrics(
        labels=combined_labels,
        predictions=combined_predictions,
        scores=combined_scores,
    )

    print(
        "\n============================================================"
    )
    print("SWaT FF-AE FINAL TEST RESULTS")
    print(
        "============================================================"
    )

    for name, value in combined_metrics.items():
        print(
            f"{name}: {value:.8f}"
        )

    print(
        f"test_windows: {len(combined_scores):,}"
    )

    print(
        f"test_attack_windows: "
        f"{int(combined_labels.sum()):,}"
    )

    print(
        f"predicted_anomalies: "
        f"{int(combined_predictions.sum()):,}"
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "dataset": "SWaT",
        "model": "Feed-Forward Autoencoder",
        "input_dim": input_dim,
        "hidden_dim_1": HIDDEN_DIM_1,
        "hidden_dim_2": HIDDEN_DIM_2,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
        "random_state": SEED,
        "device": str(device),
        "sequence_length": SEQUENCE_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "evaluation_stride": EVALUATION_STRIDE,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "feature_count": input_dim,
        "feature_names": feature_names,
        "skipped_train_segments": datasets[
            "skipped_train_segments"
        ],
    }

    CONFIG_PATH.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "epoch,train_loss,validation_loss\n"
        )

        for row in history:

            handle.write(
                f"{row['epoch']},"
                f"{row['train_loss']},"
                f"{row['validation_loss']}\n"
            )

    # --------------------------------------------------------
    # Save threshold information
    # --------------------------------------------------------

    with THRESHOLD_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "threshold_percentile,"
            "threshold,"
            "validation_flag_rate\n"
        )

        handle.write(
            f"{THRESHOLD_PERCENTILE},"
            f"{threshold},"
            f"{validation_flag_rate}\n"
        )

    # --------------------------------------------------------
    # Save scores
    # --------------------------------------------------------

    np.savez_compressed(
        SCORES_PATH,
        validation_scores=validation_scores,
        test_scores=combined_scores,
        test_labels=combined_labels,
        threshold=np.asarray(threshold),
    )

    # --------------------------------------------------------
    # Save final baseline summary
    # --------------------------------------------------------

    row = {
        "dataset": "SWaT",
        "model": "Feed-Forward Autoencoder",
        "input_dim": input_dim,
        "hidden_dim_1": HIDDEN_DIM_1,
        "hidden_dim_2": HIDDEN_DIM_2,
        "latent_dim": LATENT_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "random_state": SEED,
        "device": str(device),
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": threshold,
        "validation_flag_rate": validation_flag_rate,
        **combined_metrics,
        "test_windows": len(combined_scores),
        "test_attack_windows": int(combined_labels.sum()),
        "predicted_anomalies": int(
            combined_predictions.sum()
        ),
    }

    columns = list(row.keys())

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            ",".join(columns) + "\n"
        )

        handle.write(
            ",".join(
                str(row[column])
                for column in columns
            )
            + "\n"
        )

    print(
        f"\nSaved checkpoint: {CHECKPOINT_PATH}"
    )
    print(
        f"Saved results:    {RESULTS_PATH}"
    )
    print(
        f"Saved config:     {CONFIG_PATH}"
    )
    print(
        f"Saved history:    {HISTORY_PATH}"
    )
    print(
        f"Saved threshold:  {THRESHOLD_PATH}"
    )
    print(
        f"Saved scores:     {SCORES_PATH}"
    )


if __name__ == "__main__":
    main()
