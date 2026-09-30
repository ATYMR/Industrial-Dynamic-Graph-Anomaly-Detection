from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
from swat_gnn.graph.static_graph import (
    build_top_k_correlation_graph,
)
from swat_gnn.models.gcn_autoencoder import (
    GCNAutoencoder,
    GCNAutoencoderConfig,
    get_device,
)


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# DataLoader
# =====================================================================

def make_loader(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """
    Convert a [timesteps, features] NumPy array into a DataLoader.
    """

    tensor = torch.from_numpy(
        data.astype(np.float32, copy=False)
    )

    dataset = TensorDataset(tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


# =====================================================================
# Batched graph construction
# =====================================================================

def create_batched_edge_index(
    edge_index: torch.Tensor,
    batch_size: int,
    num_nodes: int,
) -> torch.Tensor:
    """
    Replicate one graph across a batch.

    Original graph:
        num_nodes nodes
        num_edges edges

    Batch:
        batch_size independent copies

    The resulting graphs are disconnected from one another.

    Example:
        256 graphs × 66 nodes = 16,896 nodes.

    This allows PyTorch Geometric to process the entire batch
    with one GCN forward pass instead of calling the model
    once per timestep.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if num_nodes <= 0:
        raise ValueError(
            "num_nodes must be positive."
        )

    if edge_index.ndim != 2:
        raise ValueError(
            "edge_index must be a 2D tensor."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, num_edges]."
        )

    num_edges = edge_index.shape[1]

    if num_edges == 0:
        return edge_index.new_empty(
            (2, 0)
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
        1,
        batch_size,
    )

    batched_edge_index = (
        edge_index.unsqueeze(2)
        + offsets
    )

    batched_edge_index = (
        batched_edge_index.reshape(
            2,
            num_edges * batch_size,
        )
    )

    return batched_edge_index


# =====================================================================
# Training
# =====================================================================

def train_one_epoch(
    model: GCNAutoencoder,
    loader: DataLoader,
    edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_nodes: int,
) -> float:
    """
    Train the GCN for one epoch.

    Each timestep is treated as one independent graph.

    A batch of B timesteps becomes B disconnected copies
    of the same 66-node graph.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for (batch,) in loader:

        batch = batch.to(
            device,
            non_blocking=True,
        )

        batch_size_actual = batch.shape[0]

        # -------------------------------------------------------------
        # Original:
        #
        # [B, 66]
        #
        # Convert to:
        #
        # [B * 66, 1]
        #
        # Each timestep becomes one graph.
        # -------------------------------------------------------------

        x = batch.reshape(
            batch_size_actual * num_nodes,
            1,
        )

        # -------------------------------------------------------------
        # Replicate the graph for every sample in the batch.
        # -------------------------------------------------------------

        batched_edge_index = (
            create_batched_edge_index(
                edge_index=edge_index,
                batch_size=batch_size_actual,
                num_nodes=num_nodes,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # -------------------------------------------------------------
        # ONE GCN forward pass for the entire batch.
        # -------------------------------------------------------------

        reconstruction = model(
            x,
            batched_edge_index,
        )

        # -------------------------------------------------------------
        # Reconstruction loss.
        # -------------------------------------------------------------

        loss = nn.functional.mse_loss(
            reconstruction,
            x,
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * batch_size_actual
        )

        total_samples += batch_size_actual

    if total_samples == 0:
        raise RuntimeError(
            "Training loader produced zero samples."
        )

    return total_loss / total_samples


# =====================================================================
# Dataset scoring
# =====================================================================

@torch.no_grad()
def score_dataset(
    model: GCNAutoencoder,
    loader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    num_nodes: int,
) -> tuple[float, np.ndarray]:
    """
    Calculate reconstruction loss and one anomaly score per timestep.

    Returns:
        mean_loss
        scores with shape [num_timesteps]
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_scores: list[np.ndarray] = []

    for (batch,) in loader:

        batch = batch.to(
            device,
            non_blocking=True,
        )

        batch_size_actual = batch.shape[0]

        # -------------------------------------------------------------
        # [B, 66] → [B*66, 1]
        # -------------------------------------------------------------

        x = batch.reshape(
            batch_size_actual * num_nodes,
            1,
        )

        # -------------------------------------------------------------
        # Create B independent copies of the graph.
        # -------------------------------------------------------------

        batched_edge_index = (
            create_batched_edge_index(
                edge_index=edge_index,
                batch_size=batch_size_actual,
                num_nodes=num_nodes,
            )
        )

        # -------------------------------------------------------------
        # One batched GCN forward pass.
        # -------------------------------------------------------------

        reconstruction = model(
            x,
            batched_edge_index,
        )

        # -------------------------------------------------------------
        # Reshape back to:
        #
        # [B, 66]
        # -------------------------------------------------------------

        reconstruction = reconstruction.reshape(
            batch_size_actual,
            num_nodes,
        )

        original = batch

        # -------------------------------------------------------------
        # Per-node reconstruction error.
        # -------------------------------------------------------------

        node_errors = (
            reconstruction - original
        ).pow(2)

        # -------------------------------------------------------------
        # One anomaly score per timestep.
        #
        # Mean reconstruction error across the 66 sensors.
        # -------------------------------------------------------------

        graph_scores = node_errors.mean(
            dim=1
        )

        batch_loss = node_errors.mean()

        total_loss += (
            batch_loss.item()
            * batch_size_actual
        )

        total_samples += batch_size_actual

        all_scores.append(
            graph_scores
            .detach()
            .cpu()
            .numpy()
        )

    if total_samples == 0:
        raise RuntimeError(
            "Scoring loader produced zero samples."
        )

    scores = np.concatenate(
        all_scores
    )

    mean_loss = (
        total_loss / total_samples
    )

    return mean_loss, scores


# =====================================================================
# Threshold calculation
# =====================================================================

def calculate_thresholds(
    validation_scores: np.ndarray,
) -> dict[str, float]:
    """
    Calculate candidate anomaly thresholds from normal validation data.
    """

    percentiles = [
        95,
        97,
        98,
        99,
        99.5,
    ]

    return {
        f"p{str(percentile).replace('.', '_')}":
        float(
            np.percentile(
                validation_scores,
                percentile,
            )
        )
        for percentile in percentiles
    }


# =====================================================================
# Segment concatenation
# =====================================================================

def concatenate_segments(
    segments: list[np.ndarray],
) -> np.ndarray:
    """
    Concatenate independent sensor segments.

    This is safe here because the GCN operates on individual
    timesteps rather than temporal windows.
    """

    if not segments:
        raise ValueError(
            "No segments were provided."
        )

    return np.concatenate(
        segments,
        axis=0,
    )


# =====================================================================
# Main experiment
# =====================================================================

def main() -> None:

    # -----------------------------------------------------------------
    # Project paths
    # -----------------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    config_path = (
        project_root
        / "configs"
        / "data.yaml"
    )

    config = load_config(
        str(config_path)
    )

    # -----------------------------------------------------------------
    # Experiment configuration
    # -----------------------------------------------------------------

    graph_k = 5

    model_config = GCNAutoencoderConfig(
        input_dim=1,
        hidden_dim=64,
        latent_dim=32,
        output_dim=1,
        learning_rate=1e-3,
        batch_size=256,
        max_epochs=30,
        patience=5,
        random_state=42,
        device="auto",
    )

    seed = model_config.random_state

    set_seed(seed)

    device = get_device(
        model_config.device
    )

    # -----------------------------------------------------------------
    # Experiment header
    # -----------------------------------------------------------------

    print("=" * 70)
    print(
        "STATIC GCN AUTOENCODER — HAI 23.05"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Graph type: "
        f"top-{graph_k} training correlation graph"
    )

    print(
        f"Batch size: "
        f"{model_config.batch_size}"
    )

    print()

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    print("Loading HAI data...")

    training_segments = (
        load_training_data(config)
    )

    validation_segments = (
        load_validation_data(config)
    )

    test_sensor_segments, test_labels = (
        load_test_data(config)
    )

    print(
        f"Training segments: "
        f"{len(training_segments)}"
    )

    print(
        f"Validation segments: "
        f"{len(validation_segments)}"
    )

    print(
        f"Test segments: "
        f"{len(test_sensor_segments)}"
    )

    # -----------------------------------------------------------------
    # Feature analysis
    # -----------------------------------------------------------------

    print()
    print(
        "Analyzing training features..."
    )

    feature_analysis = analyze_features(
        training_segments
    )

    feature_names = (
        feature_analysis[
            "variable_features"
        ]
    )

    constant_features = (
        feature_analysis[
            "constant_features"
        ]
    )

    print(
        f"Total features: "
        f"{len(feature_analysis['feature_names'])}"
    )

    print(
        f"Constant features: "
        f"{len(constant_features)}"
    )

    print(
        f"Variable features: "
        f"{len(feature_names)}"
    )

    # -----------------------------------------------------------------
    # Select variable features
    # -----------------------------------------------------------------

    def select_features(segments):
        return [
            df[feature_names].copy()
            for df in segments
        ]

    training_selected = select_features(
        training_segments
    )

    validation_selected = select_features(
        validation_segments
    )

    test_selected = select_features(
        test_sensor_segments
    )

    # -----------------------------------------------------------------
    # Train-only normalization
    # -----------------------------------------------------------------

    print()
    print(
        "Fitting StandardScaler on "
        "training data only..."
    )

    preprocessor = Preprocessor()

    scaled_training = (
        preprocessor.fit_transform(
            training_selected
        )
    )

    scaled_validation = (
        preprocessor.transform(
            validation_selected
        )
    )

    scaled_test = (
        preprocessor.transform(
            test_selected
        )
    )

    print(
        "Normalization complete."
    )

    # -----------------------------------------------------------------
    # Build training-only correlation graph
    # -----------------------------------------------------------------

    print()
    print(
        "Building static correlation graph..."
    )

    training_values = concatenate_segments(
        scaled_training
    )

    correlation_matrix = np.corrcoef(
        training_values,
        rowvar=False,
    )

    correlation_dataframe = pd.DataFrame(
        correlation_matrix,
        index=feature_names,
        columns=feature_names,
    )

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation_dataframe,
        k=graph_k,
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
        f"{(2 * graph.num_edges) / graph.num_nodes:.2f}"
    )

    # -----------------------------------------------------------------
    # Convert to bidirectional message-passing graph
    # -----------------------------------------------------------------

    edge_index = torch.from_numpy(
        graph.edge_index.astype(
            np.int64,
            copy=False,
        )
    )

    edge_index = to_undirected(
        edge_index,
        num_nodes=graph.num_nodes,
    )

    edge_index = edge_index.to(
        device
    )

    print(
        f"Message-passing edges: "
        f"{edge_index.shape[1]}"
    )

    # -----------------------------------------------------------------
    # Flatten training and validation data
    # -----------------------------------------------------------------

    train_data = concatenate_segments(
        scaled_training
    )

    validation_data = concatenate_segments(
        scaled_validation
    )

    print()
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
        f"{sum(len(x) for x in scaled_test):,}"
    )

    # -----------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------

    train_loader = make_loader(
        data=train_data,
        batch_size=model_config.batch_size,
        shuffle=True,
        device=device,
    )

    validation_loader = make_loader(
        data=validation_data,
        batch_size=model_config.batch_size,
        shuffle=False,
        device=device,
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    model = GCNAutoencoder(
        input_dim=1,
        hidden_dim=model_config.hidden_dim,
        latent_dim=model_config.latent_dim,
        output_dim=1,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config.learning_rate,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Model:")
    print(model)

    print()
    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_validation_loss = float(
        "inf"
    )

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

    checkpoint_path = (
        checkpoint_dir
        / "gcn_autoencoder_best.pt"
    )

    for epoch in range(
        1,
        model_config.max_epochs + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            edge_index=edge_index,
            optimizer=optimizer,
            device=device,
            num_nodes=graph.num_nodes,
        )

        validation_loss, _ = (
            score_dataset(
                model=model,
                loader=validation_loader,
                edge_index=edge_index,
                device=device,
                num_nodes=graph.num_nodes,
            )
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss":
                    validation_loss,
            }
        )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{model_config.max_epochs} | "
            f"train_loss="
            f"{train_loss:.6f} | "
            f"val_loss="
            f"{validation_loss:.6f}"
        )

        # -------------------------------------------------------------
        # Save best checkpoint
        # -------------------------------------------------------------

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
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        validation_loss,

                    "input_dim":
                        1,

                    "hidden_dim":
                        model_config.hidden_dim,

                    "latent_dim":
                        model_config.latent_dim,

                    "output_dim":
                        1,

                    "graph_k":
                        graph_k,

                    "edge_index":
                        graph.edge_index,

                    "edge_weight":
                        graph.edge_weight,

                    "feature_names":
                        feature_names,

                    "seed":
                        seed,
                },
                checkpoint_path,
            )

        else:

            epochs_without_improvement += 1

        # -------------------------------------------------------------
        # Early stopping
        # -------------------------------------------------------------

        if (
            epochs_without_improvement
            >= model_config.patience
        ):

            print()

            print(
                f"Early stopping after "
                f"epoch {epoch}."
            )

            break

    # -----------------------------------------------------------------
    # Restore best checkpoint
    # -----------------------------------------------------------------

    print()

    print(
        f"Restoring best model from "
        f"epoch {best_epoch}..."
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # -----------------------------------------------------------------
    # Validation scoring
    # -----------------------------------------------------------------

    print()

    print(
        "Calculating validation "
        "reconstruction scores..."
    )

    validation_loss, validation_scores = (
        score_dataset(
            model=model,
            loader=validation_loader,
            edge_index=edge_index,
            device=device,
            num_nodes=graph.num_nodes,
        )
    )

    thresholds = calculate_thresholds(
        validation_scores
    )

    print()

    print(
        "Validation thresholds:"
    )

    for name, threshold in (
        thresholds.items()
    ):

        print(
            f"  {name}: "
            f"{threshold:.8f}"
        )

    # -----------------------------------------------------------------
    # Primary threshold
    # -----------------------------------------------------------------

    primary_threshold = thresholds[
        "p99"
    ]

    print()

    print(
        f"Primary threshold "
        f"(99th percentile): "
        f"{primary_threshold:.8f}"
    )

    # -----------------------------------------------------------------
    # Test evaluation
    # -----------------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "TEST EVALUATION"
    )

    print("=" * 70)

    all_test_scores = []

    all_test_labels = []

    for segment, labels in zip(
        scaled_test,
        test_labels,
    ):

        test_loader = make_loader(
            data=segment,
            batch_size=model_config.batch_size,
            shuffle=False,
            device=device,
        )

        _, scores = score_dataset(
            model=model,
            loader=test_loader,
            edge_index=edge_index,
            device=device,
            num_nodes=graph.num_nodes,
        )

        labels_array = (
            labels.to_numpy(
                dtype=np.int64
            )
        )

        if len(scores) != len(
            labels_array
        ):

            raise RuntimeError(
                "Test score/label "
                "length mismatch."
            )

        all_test_scores.append(
            scores
        )

        all_test_labels.append(
            labels_array
        )

    test_scores = np.concatenate(
        all_test_scores
    )

    test_ground_truth = np.concatenate(
        all_test_labels
    )

    # -----------------------------------------------------------------
    # Frozen threshold
    # -----------------------------------------------------------------

    test_predictions = (
        test_scores
        >= primary_threshold
    ).astype(
        np.int64
    )

    # -----------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------

    metrics = compute_binary_metrics(
        labels=test_ground_truth,
        predictions=test_predictions,
        scores=test_scores,
    )

    print()

    print(
        "Test results:"
    )

    print(
        "-" * 70
    )

    for metric_name, value in (
        metrics.items()
    ):

        print(
            f"{metric_name:>12}: "
            f"{value:.6f}"
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

    # -----------------------------------------------------------------
    # Save experiment results
    # -----------------------------------------------------------------

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
        "experiment":
            "gcn_autoencoder_hai_23_05",

        "dataset":
            "HAI_23.05",

        "model":
            "Static GCN Autoencoder",

        "seed":
            seed,

        "device":
            str(device),

        "num_total_features":
            len(
                feature_analysis[
                    "feature_names"
                ]
            ),

        "num_constant_features":
            len(
                constant_features
            ),

        "num_variable_features":
            len(
                feature_names
            ),

        "graph": {
            "type":
                "top_k_absolute_correlation",

            "k":
                graph_k,

            "num_nodes":
                graph.num_nodes,

            "undirected_edges":
                graph.num_edges,

            "message_passing_edges":
                int(
                    edge_index.shape[1]
                ),

            "mean_degree":
                (
                    2
                    * graph.num_edges
                )
                / graph.num_nodes,
        },

        "hidden_dim":
            model_config.hidden_dim,

        "latent_dim":
            model_config.latent_dim,

        "learning_rate":
            model_config.learning_rate,

        "batch_size":
            model_config.batch_size,

        "max_epochs":
            model_config.max_epochs,

        "patience":
            model_config.patience,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_validation_loss,

        "validation_loss":
            validation_loss,

        "validation_thresholds":
            thresholds,

        "primary_threshold":
            primary_threshold,

        "test_metrics": {
            key: (
                None
                if not np.isfinite(value)
                else float(value)
            )
            for key, value in metrics.items()
        },

        "predicted_anomaly_timesteps":
            int(
                test_predictions.sum()
            ),

        "actual_anomaly_timesteps":
            int(
                test_ground_truth.sum()
            ),

        "feature_names":
            feature_names,

        "constant_features":
            constant_features,

        "training_history":
            training_history,
    }

    results_path = (
        results_dir
        / "gcn_autoencoder_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_results,
            file,
            indent=2,
        )

    history_path = (
        results_dir
        / "gcn_autoencoder_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            training_history,
            file,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    print(
        f"Results saved to: "
        f"{results_path}"
    )

    print(
        f"Training history saved to: "
        f"{history_path}"
    )

    print(
        f"Best checkpoint saved to: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()