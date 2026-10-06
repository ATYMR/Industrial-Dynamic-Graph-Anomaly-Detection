from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
    load_test_data,
)
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset

from train_static_temporal_graph import (
    StaticTemporalGraphAutoencoder,
    load_fixed_cosine_graph,
    calculate_anomaly_scores,
)

from train_dynamic_graph import (
    DynamicGraphAutoencoder,
)


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 60
BATCH_SIZE = 256
GRAPH_K = 5

TEMPORAL_HIDDEN = 32
EMBEDDING_DIM = 32
GAT_HIDDEN = 64
GAT_HEADS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULTS_DIR = Path("results/tables")
CHECKPOINT_DIR = Path("results/checkpoints")

GRAPH_PATH = Path(
    "results/tables/fixed_cosine_graph.pt"
)

CHECKPOINTS = {
    "static_temporal_graph":
        CHECKPOINT_DIR
        / "static_temporal_graph_autoencoder_best.pt",

    "dynamic_graph":
        CHECKPOINT_DIR
        / "dynamic_graph_autoencoder_best.pt",
}


# ============================================================
# Feature selection
# ============================================================

def get_variable_features(training_segments):

    feature_names = [
        column
        for column in training_segments[0].columns
        if column != "timestamp"
    ]

    training_values = np.concatenate(
        [
            df[feature_names].to_numpy(dtype=np.float64)
            for df in training_segments
        ],
        axis=0,
    )

    feature_std = training_values.std(axis=0)

    variable_features = [
        feature_names[i]
        for i, keep in enumerate(feature_std != 0)
        if keep
    ]

    return variable_features


def select_features(
    segments,
    feature_names,
):

    return [
        df[["timestamp"] + feature_names].copy()
        for df in segments
    ]


# ============================================================
# Save scores
# ============================================================

def save_scores(
    model_name,
    scores,
    threshold,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"{model_name}_scores.npz"
    )

    np.savez_compressed(
        output_path,
        scores=scores.astype(np.float32),
        threshold=np.float64(threshold),
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("SAVE EXISTING MODEL ANOMALY SCORES")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Load HAI data
    # ========================================================

    print("\nLoading HAI data...")

    config = load_config()

    training_segments_df = (
        load_training_data(config)
    )

    validation_segments_df = (
        load_validation_data(config)
    )

    test_segments_df, _ = (
        load_test_data(config)
    )

    # ========================================================
    # Select 66 variable features
    # ========================================================

    variable_features = (
        get_variable_features(
            training_segments_df
        )
    )

    print(
        f"Variable features: "
        f"{len(variable_features)}"
    )

    training_segments_df = (
        select_features(
            training_segments_df,
            variable_features,
        )
    )

    validation_segments_df = (
        select_features(
            validation_segments_df,
            variable_features,
        )
    )

    test_segments_df = (
        select_features(
            test_segments_df,
            variable_features,
        )
    )

    # ========================================================
    # Train-only normalization
    # ========================================================

    print(
        "\nFitting StandardScaler on training data only..."
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

    # ========================================================
    # Create validation/test datasets
    # ========================================================

    validation_dataset = (
        TemporalWindowDataset(
            validation_scaled,
            sequence_length=SEQUENCE_LENGTH,
            stride=1,
        )
    )

    test_dataset = (
        TemporalWindowDataset(
            test_scaled,
            sequence_length=SEQUENCE_LENGTH,
            stride=1,
        )
    )

    print(
        f"Validation windows: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Test windows: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # STATIC + TEMPORAL GRAPH
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STATIC + TEMPORAL GRAPH"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load exact fixed cosine graph
    # --------------------------------------------------------

    static_edge_index = load_fixed_cosine_graph(
        path=GRAPH_PATH,
        num_nodes=len(variable_features),
        k=GRAPH_K,
    )

    print(
        f"Static graph edges: "
        f"{static_edge_index.shape[1]}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    static_model = (
        StaticTemporalGraphAutoencoder(
            edge_index=static_edge_index,
            num_nodes=len(variable_features),
            temporal_hidden=TEMPORAL_HIDDEN,
            embedding_dim=EMBEDDING_DIM,
            gat_hidden=GAT_HIDDEN,
            gat_heads=GAT_HEADS,
        ).to(DEVICE)
    )

    static_checkpoint_path = (
        CHECKPOINTS[
            "static_temporal_graph"
        ]
    )

    print(
        "\nLoading checkpoint:"
    )

    print(
        static_checkpoint_path
    )

    static_checkpoint = torch.load(
        static_checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    static_model.load_state_dict(
        static_checkpoint[
            "model_state_dict"
        ]
    )

    static_model.eval()

    print(
        f"Checkpoint epoch: "
        f"{static_checkpoint['epoch']}"
    )

    # --------------------------------------------------------
    # Validation scores
    # --------------------------------------------------------

    print(
        "\nScoring validation set..."
    )

    static_validation_scores = (
        calculate_anomaly_scores(
            static_model,
            validation_dataset,
            DEVICE,
            batch_size=BATCH_SIZE,
        )
    )

    static_threshold = float(
        np.percentile(
            static_validation_scores,
            99,
        )
    )

    print(
        f"Validation p99 threshold: "
        f"{static_threshold:.8f}"
    )

    # --------------------------------------------------------
    # Test scores
    # --------------------------------------------------------

    print(
        "\nScoring test set..."
    )

    static_test_scores = (
        calculate_anomaly_scores(
            static_model,
            test_dataset,
            DEVICE,
            batch_size=BATCH_SIZE,
        )
    )

    save_scores(
        "static_temporal_graph",
        static_test_scores,
        static_threshold,
    )

    del static_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # DYNAMIC GRAPH
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DYNAMIC GRAPH"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    dynamic_model = (
        DynamicGraphAutoencoder(
            sequence_length=SEQUENCE_LENGTH,
            num_nodes=len(variable_features),
            temporal_hidden_dim=TEMPORAL_HIDDEN,
            embedding_dim=EMBEDDING_DIM,
            gat_hidden_dim=GAT_HIDDEN,
            gat_heads=GAT_HEADS,
            k=GRAPH_K,
            dropout=0.0,
        ).to(DEVICE)
    )

    dynamic_checkpoint_path = (
        CHECKPOINTS[
            "dynamic_graph"
        ]
    )

    print(
        "\nLoading checkpoint:"
    )

    print(
        dynamic_checkpoint_path
    )

    dynamic_checkpoint = torch.load(
        dynamic_checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    dynamic_model.load_state_dict(
        dynamic_checkpoint[
            "model_state_dict"
        ]
    )

    dynamic_model.eval()

    print(
        f"Checkpoint epoch: "
        f"{dynamic_checkpoint['epoch']}"
    )

    print(
        f"Checkpoint config: "
        f"{dynamic_checkpoint['config']}"
    )

    # --------------------------------------------------------
    # Validation scores
    # --------------------------------------------------------

    print(
        "\nScoring validation set..."
    )

    dynamic_validation_scores = (
        calculate_anomaly_scores(
            dynamic_model,
            validation_dataset,
            DEVICE,
            batch_size=BATCH_SIZE,
        )
    )

    dynamic_threshold = float(
        np.percentile(
            dynamic_validation_scores,
            99,
        )
    )

    print(
        f"Validation p99 threshold: "
        f"{dynamic_threshold:.8f}"
    )

    # --------------------------------------------------------
    # Test scores
    # --------------------------------------------------------

    print(
        "\nScoring test set..."
    )

    dynamic_test_scores = (
        calculate_anomaly_scores(
            dynamic_model,
            test_dataset,
            DEVICE,
            batch_size=BATCH_SIZE,
        )
    )

    save_scores(
        "dynamic_graph",
        dynamic_test_scores,
        dynamic_threshold,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()