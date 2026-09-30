from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import to_undirected

from swat_gnn.data.feature_analysis import analyze_features
from swat_gnn.data.loader import (
    load_config,
    load_test_data,
    load_training_data,
    load_validation_data,
)
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.evaluation.metrics import compute_binary_metrics
from swat_gnn.graph.static_graph import build_top_k_correlation_graph
from swat_gnn.models.gat_autoencoder import (
    GATAutoencoder,
    GATAutoencoderConfig,
    get_device,
    set_random_seed,
)


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42

TOP_K = 5

# The GPU sees only this many graphs at once.
MICRO_BATCH_SIZE = 32

# Optimization sees an effective batch of 256 samples.
EFFECTIVE_BATCH_SIZE = 256

GRADIENT_ACCUMULATION_STEPS = (
    EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
)

MAX_EPOCHS = 30
PATIENCE = 5

LEARNING_RATE = 1e-3

HIDDEN_DIM = 64
LATENT_DIM = 32
HEADS = 4
DROPOUT = 0.0

THRESHOLD_PERCENTILE = 99.0

DEVICE_NAME = "auto"

# Number of validation samples used during training
# for early stopping only.
EARLY_STOPPING_VALIDATION_SIZE = 20_000


# ============================================================
# Validation
# ============================================================


def validate_batch_configuration() -> None:
    if MICRO_BATCH_SIZE <= 0:
        raise ValueError(
            "MICRO_BATCH_SIZE must be positive."
        )

    if EFFECTIVE_BATCH_SIZE <= 0:
        raise ValueError(
            "EFFECTIVE_BATCH_SIZE must be positive."
        )

    if (
        EFFECTIVE_BATCH_SIZE
        % MICRO_BATCH_SIZE
        != 0
    ):
        raise ValueError(
            "EFFECTIVE_BATCH_SIZE must be divisible "
            "by MICRO_BATCH_SIZE."
        )

    expected_steps = (
        EFFECTIVE_BATCH_SIZE
        // MICRO_BATCH_SIZE
    )

    if (
        expected_steps
        != GRADIENT_ACCUMULATION_STEPS
    ):
        raise ValueError(
            "Gradient accumulation configuration "
            "is inconsistent."
        )


# ============================================================
# Graph batching
# ============================================================


def build_batched_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Replicate one graph topology across a batch.

    Input:
        edge_index: [2, E]

    Output:
        [2, batch_size * E]
    """

    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index must be 2-dimensional."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape (2, E)."
        )

    if edge_index.dtype != torch.long:
        raise TypeError(
            "edge_index must have dtype torch.long."
        )

    if num_nodes <= 0:
        raise ValueError(
            "num_nodes must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    offsets = (
        torch.arange(
            batch_size,
            device=edge_index.device,
            dtype=edge_index.dtype,
        )
        * num_nodes
    )

    offsets = offsets.view(
        1,
        batch_size,
        1,
    )

    batched = (
        edge_index.unsqueeze(1)
        + offsets
    )

    return batched.permute(
        0,
        2,
        1,
    ).reshape(2, -1)


def make_graph_batch(
    data: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert [B, N] sensor data into a disconnected
    batch of B graphs.
    """

    batch = torch.as_tensor(
        data,
        dtype=torch.float32,
        device=device,
    )

    if batch.ndim != 2:
        raise ValueError(
            "data must have shape (batch, nodes)."
        )

    batch_size, num_nodes = batch.shape

    # One scalar feature per sensor node.
    x = batch.unsqueeze(-1)

    flat_x = x.reshape(
        batch_size * num_nodes,
        1,
    )

    batch_edges = build_batched_edge_index(
        edge_index=edge_index,
        num_nodes=num_nodes,
        batch_size=batch_size,
    )

    return flat_x, batch_edges


# ============================================================
# Forward / scoring
# ============================================================


def batch_reconstruction_loss(
    model: GATAutoencoder,
    batch_data: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute mean reconstruction MSE for a batch.
    """

    batch_size, num_nodes = batch_data.shape

    x, batch_edges = make_graph_batch(
        data=batch_data,
        edge_index=edge_index,
        device=device,
    )

    reconstruction = model(
        x,
        batch_edges,
    )

    reconstruction = reconstruction.reshape(
        batch_size,
        num_nodes,
    )

    original = torch.as_tensor(
        batch_data,
        dtype=torch.float32,
        device=device,
    )

    return (
        reconstruction - original
    ).pow(2).mean()


def score_data(
    model: GATAutoencoder,
    data: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Compute one graph-level anomaly score per timestep.
    """

    model.eval()

    scores = []

    with torch.no_grad():

        for start in range(
            0,
            len(data),
            batch_size,
        ):
            batch_data = data[
                start : start + batch_size
            ]

            batch_size_actual = len(
                batch_data
            )

            x, batch_edges = make_graph_batch(
                data=batch_data,
                edge_index=edge_index,
                device=device,
            )

            reconstruction = model(
                x,
                batch_edges,
            )

            reconstruction = reconstruction.reshape(
                batch_size_actual,
                -1,
            )

            original = torch.as_tensor(
                batch_data,
                dtype=torch.float32,
                device=device,
            )

            batch_scores = (
                reconstruction - original
            ).pow(2).mean(dim=1)

            scores.append(
                batch_scores.cpu().numpy()
            )

    return np.concatenate(scores)


def mean_validation_loss(
    model: GATAutoencoder,
    data: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    """
    Compute mean reconstruction loss without
    storing every score.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for start in range(
            0,
            len(data),
            batch_size,
        ):
            batch_data = data[
                start : start + batch_size
            ]

            loss = batch_reconstruction_loss(
                model=model,
                batch_data=batch_data,
                edge_index=edge_index,
                device=device,
            )

            count = len(batch_data)

            total_loss += (
                loss.item() * count
            )

            total_samples += count

    return total_loss / total_samples


# ============================================================
# Training
# ============================================================


def train_one_epoch(
    model: GATAutoencoder,
    optimizer: torch.optim.Optimizer,
    data: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Train for one epoch using micro-batches and
    gradient accumulation.
    """

    model.train()

    indices = np.random.permutation(
        len(data)
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_samples = 0

    accumulation_count = 0

    for start in range(
        0,
        len(indices),
        MICRO_BATCH_SIZE,
    ):
        batch_indices = indices[
            start : start + MICRO_BATCH_SIZE
        ]

        batch_data = data[
            batch_indices
        ]

        loss = batch_reconstruction_loss(
            model=model,
            batch_data=batch_data,
            edge_index=edge_index,
            device=device,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        # Scale the loss because we accumulate
        # several micro-batches.
        scaled_loss = (
            loss
            / GRADIENT_ACCUMULATION_STEPS
        )

        scaled_loss.backward()

        accumulation_count += 1

        batch_count = len(
            batch_indices
        )

        total_loss += (
            loss.item() * batch_count
        )

        total_samples += batch_count

        should_step = (
            accumulation_count
            >= GRADIENT_ACCUMULATION_STEPS
        )

        is_last_batch = (
            start + MICRO_BATCH_SIZE
            >= len(indices)
        )

        if should_step or is_last_batch:

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            accumulation_count = 0

    return total_loss / total_samples


# ============================================================
# Utility
# ============================================================


def create_early_stopping_subset(
    data: np.ndarray,
    size: int,
    seed: int,
) -> np.ndarray:
    """
    Create a deterministic subset for early stopping.

    This subset is used ONLY during training.
    The complete validation set is used later for
    threshold selection.
    """

    if size <= 0:
        raise ValueError(
            "Validation subset size must be positive."
        )

    if size >= len(data):
        return data

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(data),
        size=size,
        replace=False,
    )

    indices.sort()

    return data[indices]


def save_json(
    path: Path,
    payload: dict,
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
            payload,
            file,
            indent=2,
        )


# ============================================================
# Main
# ============================================================


def main() -> None:

    validate_batch_configuration()

    set_random_seed(SEED)

    config = load_config()

    device = get_device(
        DEVICE_NAME
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Micro-batch size: "
        f"{MICRO_BATCH_SIZE}"
    )

    print(
        f"Effective batch size: "
        f"{EFFECTIVE_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation steps: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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
    # Feature analysis
    # --------------------------------------------------------

    feature_info = analyze_features(
        training_segments
    )

    feature_names = feature_info[
        "variable_features"
    ]

    constant_features = feature_info[
        "constant_features"
    ]

    print(
        f"Total features: "
        f"{len(feature_info['feature_names'])}"
    )

    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )

    print(
        f"Variable features: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Training-only normalization
    # --------------------------------------------------------

    preprocessor = Preprocessor()

    preprocessor.fit(
        [
            df[
                ["timestamp"] + feature_names
            ]
            for df in training_segments
        ]
    )

    scaled_train_segments = (
        preprocessor.transform(
            [
                df[
                    ["timestamp"] + feature_names
                ]
                for df in training_segments
            ]
        )
    )

    scaled_validation_segments = (
        preprocessor.transform(
            [
                df[
                    ["timestamp"] + feature_names
                ]
                for df in validation_segments
            ]
        )
    )

    scaled_test_segments = (
        preprocessor.transform(
            [
                df[
                    ["timestamp"] + feature_names
                ]
                for df in test_segments
            ]
        )
    )

    train_data = np.concatenate(
        scaled_train_segments,
        axis=0,
    )

    validation_data = np.concatenate(
        scaled_validation_segments,
        axis=0,
    )

    test_data = np.concatenate(
        scaled_test_segments,
        axis=0,
    )

    labels = np.concatenate(
        [
            label.to_numpy(
                dtype=np.int64
            )
            for label in test_labels
        ],
        axis=0,
    )

    print(
        f"Training timesteps: "
        f"{len(train_data):,}"
    )

    print(
        f"Validation timesteps: "
        f"{len(validation_data):,}"
    )

    print(
        f"Test timesteps: "
        f"{len(test_data):,}"
    )

    print(
        f"Test attack timesteps: "
        f"{labels.sum():,}"
    )

    # --------------------------------------------------------
    # Static graph
    # --------------------------------------------------------

    combined_training = np.concatenate(
        scaled_train_segments,
        axis=0,
    )

    correlation_matrix = np.corrcoef(
        combined_training,
        rowvar=False,
    )

    correlation_df = pd.DataFrame(
        correlation_matrix,
        index=feature_names,
        columns=feature_names,
    )

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation_df,
        k=TOP_K,
    )

    edge_index = torch.as_tensor(
        graph.edge_index,
        dtype=torch.long,
        device=device,
    )

    edge_index = to_undirected(
        edge_index,
        num_nodes=graph.num_nodes,
    )

    print(
        f"Graph type: "
        f"top-{TOP_K} training correlation graph"
    )

    print(
        f"Graph nodes: "
        f"{graph.num_nodes}"
    )

    print(
        f"Undirected graph edges: "
        f"{graph.num_edges}"
    )

    print(
        f"Mean degree: "
        f"{2 * graph.num_edges / graph.num_nodes:.2f}"
    )

    print(
        f"Message-passing edges: "
        f"{edge_index.shape[1]}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_config = GATAutoencoderConfig(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        heads=HEADS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        batch_size=EFFECTIVE_BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        random_state=SEED,
        device=DEVICE_NAME,
    )

    model = GATAutoencoder(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        heads=model_config.heads,
        dropout=model_config.dropout,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(model)

    print(
        f"Trainable params: "
        f"{parameter_count:,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Early stopping validation subset
    # --------------------------------------------------------

    early_stop_validation = (
        create_early_stopping_subset(
            validation_data,
            EARLY_STOPPING_VALIDATION_SIZE,
            SEED,
        )
    )

    print(
        f"Early-stopping validation samples: "
        f"{len(early_stop_validation):,}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_state = None
    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data=train_data,
            edge_index=edge_index,
            device=device,
        )

        validation_loss = (
            mean_validation_loss(
                model=model,
                data=early_stop_validation,
                edge_index=edge_index,
                device=device,
                batch_size=MICRO_BATCH_SIZE,
            )
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "early_stopping_validation_loss": (
                    validation_loss
                ),
            }
        )

        print(
            f"Epoch {epoch:02d} "
            f"train {train_loss:.6f} "
            f"val {validation_loss:.6f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_state = deepcopy(
                model.state_dict()
            )

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"Early stopping after "
                f"{epoch} epochs; "
                f"best epoch {best_epoch}."
            )

            break

    if best_state is None:
        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # FULL validation scoring
    # --------------------------------------------------------

    print(
        "\nScoring full validation set..."
    )

    validation_scores = score_data(
        model=model,
        data=validation_data,
        edge_index=edge_index,
        device=device,
        batch_size=MICRO_BATCH_SIZE,
    )

    threshold_percentiles = {
        "95": float(
            np.percentile(
                validation_scores,
                95,
            )
        ),
        "97": float(
            np.percentile(
                validation_scores,
                97,
            )
        ),
        "98": float(
            np.percentile(
                validation_scores,
                98,
            )
        ),
        "99": float(
            np.percentile(
                validation_scores,
                99,
            )
        ),
        "99.5": float(
            np.percentile(
                validation_scores,
                99.5,
            )
        ),
    }

    threshold = threshold_percentiles[
        "99"
    ]

    print(
        "\nValidation thresholds:"
    )

    for percentile, value in (
        threshold_percentiles.items()
    ):
        print(
            f"  p{percentile}: "
            f"{value:.8f}"
        )

    print(
        f"\nPrimary threshold "
        f"(p99): {threshold:.8f}"
    )

    # --------------------------------------------------------
    # FULL test scoring
    # --------------------------------------------------------

    print(
        "\nScoring full test set..."
    )

    test_scores = score_data(
        model=model,
        data=test_data,
        edge_index=edge_index,
        device=device,
        batch_size=MICRO_BATCH_SIZE,
    )

    predictions = (
        test_scores >= threshold
    ).astype(np.int64)

    metrics = compute_binary_metrics(
        labels=labels,
        predictions=predictions,
        scores=test_scores,
    )

    print(
        "\nTest metrics:"
    )

    for key, value in metrics.items():
        print(
            f"  {key}: {value:.6f}"
        )

    print(
        f"\nPredicted anomalies: "
        f"{predictions.sum():,}"
    )

    print(
        f"Actual anomalies: "
        f"{labels.sum():,}"
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(
        "results/checkpoints/"
        "gat_autoencoder_best.pt"
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": model_config.input_dim,
                "hidden_dim": model_config.hidden_dim,
                "latent_dim": model_config.latent_dim,
                "heads": model_config.heads,
                "dropout": model_config.dropout,
            },
            "feature_names": feature_names,
            "graph": {
                "type": (
                    f"top-{TOP_K}_"
                    "training_correlation"
                ),
                "num_nodes": graph.num_nodes,
                "num_edges": graph.num_edges,
                "message_passing_edges": int(
                    edge_index.shape[1]
                ),
                "k": TOP_K,
            },
            "training": {
                "micro_batch_size": (
                    MICRO_BATCH_SIZE
                ),
                "effective_batch_size": (
                    EFFECTIVE_BATCH_SIZE
                ),
                "gradient_accumulation_steps": (
                    GRADIENT_ACCUMULATION_STEPS
                ),
            },
            "threshold": threshold,
            "best_epoch": best_epoch,
            "best_validation_loss": (
                best_validation_loss
            ),
            "seed": SEED,
        },
        checkpoint_path,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "model": "GAT_Autoencoder",
        "dataset": "HAI_23.05",
        "seed": SEED,
        "device": str(device),
        "features": {
            "total": len(
                feature_info["feature_names"]
            ),
            "constant": len(
                constant_features
            ),
            "variable": len(
                feature_names
            ),
        },
        "graph": {
            "type": (
                f"top-{TOP_K}_"
                "training_correlation"
            ),
            "nodes": graph.num_nodes,
            "undirected_edges": graph.num_edges,
            "message_passing_edges": int(
                edge_index.shape[1]
            ),
            "k": TOP_K,
            "mean_degree": (
                2
                * graph.num_edges
                / graph.num_nodes
            ),
        },
        "training": {
            "micro_batch_size": (
                MICRO_BATCH_SIZE
            ),
            "effective_batch_size": (
                EFFECTIVE_BATCH_SIZE
            ),
            "gradient_accumulation_steps": (
                GRADIENT_ACCUMULATION_STEPS
            ),
            "learning_rate": LEARNING_RATE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "best_epoch": best_epoch,
            "early_stopping_validation_size": (
                len(early_stop_validation)
            ),
            "best_early_stopping_validation_loss": (
                best_validation_loss
            ),
        },
        "model_config": {
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "heads": HEADS,
            "dropout": DROPOUT,
            "trainable_parameters": (
                parameter_count
            ),
        },
        "threshold": {
            "method": (
                "full_validation_percentile"
            ),
            "primary_percentile": (
                THRESHOLD_PERCENTILE
            ),
            "primary_threshold": threshold,
            "validation_percentiles": (
                threshold_percentiles
            ),
        },
        "test": {
            "timesteps": len(test_data),
            "attack_timesteps": int(
                labels.sum()
            ),
            "predicted_anomalies": int(
                predictions.sum()
            ),
        },
        "metrics": metrics,
    }

    save_json(
        Path(
            "results/tables/"
            "gat_autoencoder_results.json"
        ),
        results,
    )

    save_json(
        Path(
            "results/tables/"
            "gat_autoencoder_history.json"
        ),
        {
            "history": history
        },
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {checkpoint_path}"
    )

    print(
        "  results/tables/"
        "gat_autoencoder_results.json"
    )

    print(
        "  results/tables/"
        "gat_autoencoder_history.json"
    )


if __name__ == "__main__":
    main()