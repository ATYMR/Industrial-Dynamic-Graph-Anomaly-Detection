from __future__ import annotations

import numpy as np
import torch
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

from train_static_temporal_graph import (
    StaticTemporalGraphAutoencoder,
    build_fixed_correlation_graph,
    calculate_anomaly_scores,
    get_endpoint_labels,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    TRAIN_STRIDE,
    VALIDATION_STRIDE,
    TEST_STRIDE,
    GRAPH_K,
    TEMPORAL_HIDDEN,
    EMBEDDING_DIM,
    GAT_HIDDEN,
    GAT_HEADS,
    CHECKPOINT_PATH,
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading HAI data...")

    config = load_config()

    training_segments_df = load_training_data(config)
    validation_segments_df = load_validation_data(config)
    test_segments_df, test_labels = load_test_data(config)

    # --------------------------------------------------------
    # Select the same 66 variable features
    # --------------------------------------------------------

    feature_names = [
        column
        for column in training_segments_df[0].columns
        if column != "timestamp"
    ]

    training_values = np.concatenate(
        [
            df[feature_names].to_numpy(dtype=np.float64)
            for df in training_segments_df
        ],
        axis=0,
    )

    feature_std = training_values.std(axis=0)

    variable_features = [
        feature_names[i]
        for i, keep in enumerate(feature_std != 0)
        if keep
    ]

    print(
        f"Using variable features: "
        f"{len(variable_features)}"
    )

    # --------------------------------------------------------
    # Keep only variable features
    # --------------------------------------------------------

    training_segments_df = [
        df[["timestamp"] + variable_features].copy()
        for df in training_segments_df
    ]

    validation_segments_df = [
        df[["timestamp"] + variable_features].copy()
        for df in validation_segments_df
    ]

    test_segments_df = [
        df[["timestamp"] + variable_features].copy()
        for df in test_segments_df
    ]

    # --------------------------------------------------------
    # Train-only normalization
    # --------------------------------------------------------

    print(
        "\nFitting StandardScaler on training data only..."
    )

    preprocessor = Preprocessor()

    training_scaled = preprocessor.fit_transform(
        training_segments_df
    )

    validation_scaled = preprocessor.transform(
        validation_segments_df
    )

    test_scaled = preprocessor.transform(
        test_segments_df
    )

    # --------------------------------------------------------
    # Reconstruct the exact fixed graph
    # --------------------------------------------------------

    edge_index = build_fixed_correlation_graph(
        training_scaled,
        k=GRAPH_K,
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\nCreating temporal datasets...")

    validation_dataset = TemporalWindowDataset(
        validation_scaled,
        sequence_length=SEQUENCE_LENGTH,
        stride=VALIDATION_STRIDE,
    )

    test_dataset = TemporalWindowDataset(
        test_scaled,
        sequence_length=SEQUENCE_LENGTH,
        stride=TEST_STRIDE,
    )

    print(
        f"Validation windows: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Test windows: "
        f"{len(test_dataset):,}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = StaticTemporalGraphAutoencoder(
        edge_index=edge_index,
        num_nodes=len(variable_features),
        temporal_hidden=TEMPORAL_HIDDEN,
        embedding_dim=EMBEDDING_DIM,
        gat_hidden=GAT_HIDDEN,
        gat_heads=GAT_HEADS,
    ).to(device)

    # --------------------------------------------------------
    # Load existing checkpoint
    # --------------------------------------------------------

    print(
        f"\nLoading checkpoint:"
    )

    print(
        CHECKPOINT_PATH.resolve()
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.8f}"
    )

    model.eval()

    # --------------------------------------------------------
    # Score validation set
    # --------------------------------------------------------

    print(
        "\nScoring validation set..."
    )

    validation_scores = calculate_anomaly_scores(
        model,
        validation_dataset,
        device,
        batch_size=BATCH_SIZE,
    )

    threshold = float(
        np.percentile(
            validation_scores,
            99,
        )
    )

    print(
        f"Validation p99 threshold: "
        f"{threshold:.8f}"
    )

    # --------------------------------------------------------
    # Score test set
    # --------------------------------------------------------

    print(
        "\nScoring test set..."
    )

    test_scores = calculate_anomaly_scores(
        model,
        test_dataset,
        device,
        batch_size=BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Convert labels to window-endpoint labels
    # --------------------------------------------------------

    test_window_labels = get_endpoint_labels(
        test_labels,
        sequence_length=SEQUENCE_LENGTH,
        stride=TEST_STRIDE,
    )

    if len(test_scores) != len(test_window_labels):
        raise RuntimeError(
            "Test score/label length mismatch: "
            f"scores={len(test_scores)}, "
            f"labels={len(test_window_labels)}"
        )

    # --------------------------------------------------------
    # Apply validation-derived threshold
    # --------------------------------------------------------

    test_predictions = (
        test_scores >= threshold
    ).astype(np.int64)

    # --------------------------------------------------------
    # Compute final metrics
    # --------------------------------------------------------

    metrics = compute_binary_metrics(
        labels=test_window_labels,
        predictions=test_predictions,
        scores=test_scores,
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print(
        "\n============================================================"
    )

    print(
        "STATIC + TEMPORAL GRAPH FINAL TEST RESULTS"
    )

    print(
        "============================================================"
    )

    print(
        f"{'precision':>15}: "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"{'recall':>15}: "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"{'f1':>15}: "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"{'fpr':>15}: "
        f"{metrics['fpr']:.6f}"
    )

    print(
        f"{'auroc':>15}: "
        f"{metrics['auroc']:.6f}"
    )

    print(
        f"{'auprc':>15}: "
        f"{metrics['auprc']:.6f}"
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

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()