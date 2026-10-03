from __future__ import annotations
import os

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
    load_test_data,
)
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.evaluation.metrics import compute_binary_metrics


# ============================================================
# Configuration
# ============================================================

SEED = int(os.environ.get("EXPERIMENT_SEED", "42"))

SEQUENCE_LENGTH = 60

TRAIN_STRIDE = 5
VALIDATION_STRIDE = 1
TEST_STRIDE = 1

BATCH_SIZE = 256

TEMPORAL_HIDDEN = 32
EMBEDDING_DIM = 32

GAT_HIDDEN = 64
GAT_HEADS = 4

GRAPH_K = 5

LEARNING_RATE = 1e-3

MAX_EPOCHS = 30
PATIENCE = 5

THRESHOLD_PERCENTILE = 99

GRAPH_PATH = Path(
    "results/tables/fixed_cosine_graph.pt"
)

CHECKPOINT_PATH = Path(
    "results/checkpoints/"
    "static_temporal_graph_autoencoder_best.pt"
)

RESULTS_PATH = Path(
    "results/tables/"
    "static_temporal_graph_autoencoder_results.json"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Fixed graph loading
# ============================================================

def load_fixed_cosine_graph(
    path: Path,
    num_nodes: int,
    k: int = GRAPH_K,
) -> torch.Tensor:
    """
    Load the fixed cosine-similarity graph generated from
    the training windows.

    The graph is expected to be a PyTorch edge_index tensor
    with shape [2, E].

    For 66 nodes and k=5 directed top-k neighbors:

        E = 66 * 5 = 330
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Fixed graph not found: {path.resolve()}"
        )

    edge_index = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(edge_index, torch.Tensor):
        raise TypeError(
            "Fixed graph file must contain a torch.Tensor."
        )

    if edge_index.ndim != 2:
        raise ValueError(
            "Fixed graph must have shape [2, E]. "
            f"Got {tuple(edge_index.shape)}."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "Fixed graph must have shape [2, E]. "
            f"Got {tuple(edge_index.shape)}."
        )

    edge_index = edge_index.long()

    if edge_index.numel() == 0:
        raise ValueError(
            "Fixed graph contains no edges."
        )

    if edge_index.min().item() < 0:
        raise ValueError(
            "Fixed graph contains negative node indices."
        )

    if edge_index.max().item() >= num_nodes:
        raise ValueError(
            "Fixed graph contains a node index outside "
            f"the valid range [0, {num_nodes - 1}]."
        )

    self_loops = (
        edge_index[0] == edge_index[1]
    ).sum().item()

    if self_loops != 0:
        raise ValueError(
            f"Fixed graph contains {self_loops} self-loops."
        )

    expected_edges = num_nodes * k

    if edge_index.shape[1] != expected_edges:
        raise ValueError(
            "Unexpected number of graph edges. "
            f"Expected {expected_edges} for "
            f"{num_nodes} nodes and k={k}, "
            f"got {edge_index.shape[1]}."
        )

    print(
        "\nLoaded fixed cosine graph:"
    )
    print(
        f"  Path: {path.resolve()}"
    )
    print(
        f"  Shape: {tuple(edge_index.shape)}"
    )
    print(
        f"  Nodes: {num_nodes}"
    )
    print(
        f"  Directed edges: {edge_index.shape[1]}"
    )
    print(
        f"  Self-loops: {self_loops}"
    )
    print(
        f"  Top-k: {k}"
    )

    return edge_index


# ============================================================
# Graph batching
# ============================================================

def batch_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    Create a disconnected batched graph.

    Every sample receives the exact same fixed graph.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    device = edge_index.device

    offsets = (
        torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        )
        * num_nodes
    )

    offsets = offsets.view(
        -1,
        1,
        1,
    )

    batched_edges = (
        edge_index.view(
            1,
            2,
            -1,
        )
        + offsets
    )

    batched_edges = batched_edges.permute(
        1,
        0,
        2,
    )

    return batched_edges.reshape(
        2,
        -1,
    )


# ============================================================
# Model
# ============================================================

class SensorTemporalEncoder(nn.Module):
    """
    Shared temporal encoder.

    Input:
        [B, T, N]

    Output:
        [B, N, embedding_dim]
    """

    def __init__(
        self,
        hidden_dim: int = TEMPORAL_HIDDEN,
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.projection = nn.Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "[batch, sequence_length, num_nodes]."
            )

        batch_size, sequence_length, num_nodes = x.shape

        sensor_sequences = (
            x.transpose(1, 2)
            .contiguous()
            .view(
                batch_size * num_nodes,
                sequence_length,
                1,
            )
        )

        _, (hidden, _) = self.lstm(
            sensor_sequences
        )

        sensor_embeddings = hidden[-1]

        sensor_embeddings = self.projection(
            sensor_embeddings
        )

        return sensor_embeddings.view(
            batch_size,
            num_nodes,
            -1,
        )


class StaticTemporalGraphAutoencoder(
    nn.Module
):
    """
    Static graph + temporal encoder autoencoder.

    Architecture:

        [B, 60, 66]
              |
        shared sensor LSTM
              |
        [B, 66, 32]
              |
        fixed cosine graph
              |
        GAT 32 -> 64 x 4
              |
        GAT 256 -> 32
              |
        GAT 32 -> 64 x 4
              |
        GAT 256 -> 1
              |
        [B, 66]
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        temporal_hidden: int = TEMPORAL_HIDDEN,
        embedding_dim: int = EMBEDDING_DIM,
        gat_hidden: int = GAT_HIDDEN,
        gat_heads: int = GAT_HEADS,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes

        self.temporal_encoder = (
            SensorTemporalEncoder(
                hidden_dim=temporal_hidden,
                embedding_dim=embedding_dim,
            )
        )

        self.register_buffer(
            "edge_index",
            edge_index,
        )

        self.gat_encoder_1 = GATConv(
            embedding_dim,
            gat_hidden,
            heads=gat_heads,
        )

        self.gat_encoder_2 = GATConv(
            gat_hidden * gat_heads,
            embedding_dim,
            heads=1,
        )

        self.gat_decoder_1 = GATConv(
            embedding_dim,
            gat_hidden,
            heads=gat_heads,
        )

        self.gat_decoder_2 = GATConv(
            gat_hidden * gat_heads,
            1,
            heads=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = x.shape[0]

        # ----------------------------------------------------
        # Temporal encoding
        # ----------------------------------------------------

        sensor_embeddings = (
            self.temporal_encoder(x)
        )

        # ----------------------------------------------------
        # Fixed disconnected batch graph
        # ----------------------------------------------------

        batched_edges = batch_edge_index(
            self.edge_index,
            batch_size=batch_size,
            num_nodes=self.num_nodes,
        )

        h = sensor_embeddings.reshape(
            batch_size * self.num_nodes,
            -1,
        )

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        h = self.gat_encoder_1(
            h,
            batched_edges,
        )

        h = torch.relu(h)

        h = self.gat_encoder_2(
            h,
            batched_edges,
        )

        h = torch.relu(h)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        h = self.gat_decoder_1(
            h,
            batched_edges,
        )

        h = torch.relu(h)

        h = self.gat_decoder_2(
            h,
            batched_edges,
        )

        reconstruction = h.view(
            batch_size,
            self.num_nodes,
        )

        return reconstruction


# ============================================================
# Utility functions
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_anomaly_scores(
    model: nn.Module,
    dataset: TemporalWindowDataset,
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

            reconstruction = model(
                batch
            )

            target = batch[:, -1, :]

            batch_scores = torch.mean(
                (
                    reconstruction - target
                ) ** 2,
                dim=1,
            )

            scores.append(
                batch_scores.detach()
                .cpu()
                .numpy()
            )

    return np.concatenate(scores)


def get_endpoint_labels(
    labels: list,
    sequence_length: int,
    stride: int,
) -> np.ndarray:
    """
    Convert timestep labels into labels
    corresponding to the endpoint of every
    temporal window.
    """

    endpoint_labels = []

    for segment_labels in labels:

        values = segment_labels.to_numpy(
            dtype=np.int64
        )

        if len(values) < sequence_length:
            raise ValueError(
                "Label segment shorter than "
                "sequence length."
            )

        endpoints = values[
            sequence_length - 1 :: stride
        ]

        endpoint_labels.append(
            endpoints
        )

    return np.concatenate(
        endpoint_labels
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading HAI data..."
    )

    config = load_config()

    training_segments_df = (
        load_training_data(config)
    )

    validation_segments_df = (
        load_validation_data(config)
    )

    test_segments_df, test_labels = (
        load_test_data(config)
    )

    print(
        f"Training segments:   "
        f"{len(training_segments_df)}"
    )

    print(
        f"Validation segments: "
        f"{len(validation_segments_df)}"
    )

    print(
        f"Test segments:       "
        f"{len(test_segments_df)}"
    )

    # --------------------------------------------------------
    # Select variable features using training data only
    # --------------------------------------------------------

    feature_names = [
        column
        for column in training_segments_df[0].columns
        if column != "timestamp"
    ]

    training_values = np.concatenate(
        [
            df[feature_names].to_numpy(
                dtype=np.float64
            )
            for df in training_segments_df
        ],
        axis=0,
    )

    feature_std = training_values.std(
        axis=0
    )

    variable_mask = (
        feature_std != 0
    )

    variable_features = [
        feature_names[i]
        for i, keep in enumerate(
            variable_mask
        )
        if keep
    ]

    print(
        f"\nTotal features:      "
        f"{len(feature_names)}"
    )

    print(
        f"Constant features:   "
        f"{len(feature_names) - len(variable_features)}"
    )

    print(
        f"Variable features:   "
        f"{len(variable_features)}"
    )

    if len(variable_features) != 66:
        raise RuntimeError(
            "Expected 66 variable features "
            f"for HAI 23.05, got "
            f"{len(variable_features)}."
        )

    # Keep only variable features.

    training_segments_df = [
        df[
            ["timestamp"]
            + variable_features
        ].copy()
        for df in training_segments_df
    ]

    validation_segments_df = [
        df[
            ["timestamp"]
            + variable_features
        ].copy()
        for df in validation_segments_df
    ]

    test_segments_df = [
        df[
            ["timestamp"]
            + variable_features
        ].copy()
        for df in test_segments_df
    ]

    # --------------------------------------------------------
    # Train-only normalization
    # --------------------------------------------------------

    print(
        "\nFitting StandardScaler on "
        "training data only..."
    )

    preprocessor = Preprocessor()

    training_scaled = (
        preprocessor.fit_transform(
            training_segments_df
        )
    )

    validation_scaled = (
        preprocessor.transform(
            validation_segments_df
        )
    )

    test_scaled = (
        preprocessor.transform(
            test_segments_df
        )
    )

    # --------------------------------------------------------
    # Load precomputed fixed cosine graph
    # --------------------------------------------------------

    edge_index = load_fixed_cosine_graph(
        path=GRAPH_PATH,
        num_nodes=len(variable_features),
        k=GRAPH_K,
    )

    # --------------------------------------------------------
    # Temporal datasets
    # --------------------------------------------------------

    print(
        "\nCreating temporal datasets..."
    )

    train_dataset = (
        TemporalWindowDataset(
            training_scaled,
            sequence_length=SEQUENCE_LENGTH,
            stride=TRAIN_STRIDE,
        )
    )

    validation_dataset = (
        TemporalWindowDataset(
            validation_scaled,
            sequence_length=SEQUENCE_LENGTH,
            stride=VALIDATION_STRIDE,
        )
    )

    test_dataset = (
        TemporalWindowDataset(
            test_scaled,
            sequence_length=SEQUENCE_LENGTH,
            stride=TEST_STRIDE,
        )
    )

    print(
        f"Training windows:   "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation windows: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Test windows:       "
        f"{len(test_dataset):,}"
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

    model = (
        StaticTemporalGraphAutoencoder(
            edge_index=edge_index,
            num_nodes=len(
                variable_features
            ),
            temporal_hidden=TEMPORAL_HIDDEN,
            embedding_dim=EMBEDDING_DIM,
            gat_hidden=GAT_HIDDEN,
            gat_heads=GAT_HEADS,
        )
        .to(device)
    )

    print(
        "\nModel:"
    )

    print(model)

    print(
        f"\nTrainable parameters: "
        f"{count_parameters(model):,}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

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
            f"\nEpoch {epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        for (
            batch_index,
            batch,
        ) in enumerate(
            train_loader,
            start=1,
        ):

            batch = batch.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            reconstruction = model(
                batch
            )

            target = batch[:, -1, :]

            loss = criterion(
                reconstruction,
                target,
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

            batch_size_actual = (
                batch.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size_actual
            )

            sample_count += (
                batch_size_actual
            )

            if batch_index % 100 == 0:

                print(
                    f"  batch "
                    f"{batch_index}/"
                    f"{len(train_loader)} "
                    f"loss="
                    f"{loss.item():.6f}"
                )

        train_loss = (
            running_loss
            / sample_count
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

                reconstruction = (
                    model(batch)
                )

                target = batch[:, -1, :]

                loss = criterion(
                    reconstruction,
                    target,
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
                "train_loss": train_loss,
                "validation_loss": (
                    validation_loss
                ),
            }
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "epoch": epoch,
                    "validation_loss": (
                        validation_loss
                    ),
                    "edge_index": (
                        edge_index.cpu()
                    ),
                    "feature_names": (
                        variable_features
                    ),
                    "sequence_length": (
                        SEQUENCE_LENGTH
                    ),
                    "graph_k": GRAPH_K,
                    "graph_type": (
                        "fixed_cosine"
                    ),
                    "graph_path": str(
                        GRAPH_PATH
                    ),
                    "seed": SEED,
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

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------

    print(
        f"\nBest epoch: "
        f"{best_epoch}"
    )

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
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "\nBest checkpoint restored."
    )

    # --------------------------------------------------------
    # Validation anomaly scores
    # --------------------------------------------------------

    print(
        "\nScoring validation set..."
    )

    validation_scores = (
        calculate_anomaly_scores(
            model,
            validation_dataset,
            device,
        )
    )

    threshold = float(
        np.percentile(
            validation_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    print(
        f"Validation p99 threshold: "
        f"{threshold:.8f}"
    )

    # --------------------------------------------------------
    # Test anomaly scores
    # --------------------------------------------------------

    print(
        "\nScoring test set..."
    )

    test_scores = (
        calculate_anomaly_scores(
            model,
            test_dataset,
            device,
        )
    )

    # --------------------------------------------------------
    # Convert labels to window-endpoint labels
    # --------------------------------------------------------

    test_window_labels = (
        get_endpoint_labels(
            test_labels,
            sequence_length=(
                SEQUENCE_LENGTH
            ),
            stride=TEST_STRIDE,
        )
    )

    if len(test_scores) != len(
        test_window_labels
    ):

        raise RuntimeError(
            "Test score/label length "
            "mismatch: "
            f"scores={len(test_scores)}, "
            f"labels="
            f"{len(test_window_labels)}"
        )

    # --------------------------------------------------------
    # Apply validation-derived threshold
    # --------------------------------------------------------

    test_predictions = (
        test_scores >= threshold
    ).astype(np.int64)

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    metrics = compute_binary_metrics(
        labels=test_window_labels,
        predictions=test_predictions,
        scores=test_scores,
    )

    print(
        "\n============================================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "============================================================"
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
            f"{key:>15}: "
            f"{metrics[key]:.6f}"
        )

    print(
        f"{'threshold':>15}: "
        f"{threshold:.8f}"
    )

    print(
        f"{'predicted':>15}: "
        f"{int(test_predictions.sum()):,}"
    )

    print(
        f"{'actual':>15}: "
        f"{int(test_window_labels.sum()):,}"
    )

    print(
        f"{'coverage':>15}: "
        f"{len(test_scores) / sum(len(x) for x in test_scaled):.6f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "experiment": (
            "static_temporal_graph_autoencoder"
        ),
        "dataset": "HAI_23.05",
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "train_stride": TRAIN_STRIDE,
        "validation_stride": (
            VALIDATION_STRIDE
        ),
        "test_stride": TEST_STRIDE,
        "batch_size": BATCH_SIZE,
        "temporal_hidden": (
            TEMPORAL_HIDDEN
        ),
        "embedding_dim": EMBEDDING_DIM,
        "gat_hidden": GAT_HIDDEN,
        "gat_heads": GAT_HEADS,
        "graph_k": GRAPH_K,
        "graph_type": "fixed_cosine",
        "graph_path": str(
            GRAPH_PATH
        ),
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "threshold_percentile": (
            THRESHOLD_PERCENTILE
        ),
        "num_features": len(
            variable_features
        ),
        "num_graph_edges": int(
            edge_index.shape[1]
        ),
        "train_windows": len(
            train_dataset
        ),
        "validation_windows": len(
            validation_dataset
        ),
        "test_windows": len(
            test_dataset
        ),
        "trainable_parameters": (
            count_parameters(model)
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "threshold": threshold,
        "metrics": {
            key: float(value)
            for key, value in metrics.items()
        },
        "history": history,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_PATH.resolve()
    )

    print(
        "\nCheckpoint saved to:"
    )

    print(
        CHECKPOINT_PATH.resolve()
    )


if __name__ == "__main__":
    main()
