from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_test_data,
)
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.models.dynamic_graph_autoencoder import (
    DynamicGraphAutoencoder,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 60
STRIDE = 1

GRAPH_K = 5

TEMPORAL_HIDDEN = 32
EMBEDDING_DIM = 32
GAT_HIDDEN = 64
GAT_HEADS = 4

BATCH_SIZE = 256

SAMPLE_WINDOWS = 5000

CHECKPOINT_PATH = Path(
    "results/checkpoints/dynamic_graph_autoencoder_best.pt"
)

OUTPUT_PATH = Path(
    "results/tables/graph_change_score_results.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Helpers
# ============================================================

def get_variable_features(training_segments):

    feature_names = [
        c
        for c in training_segments[0].columns
        if c != "timestamp"
    ]

    values = np.concatenate(
        [
            df[feature_names].to_numpy(dtype=np.float64)
            for df in training_segments
        ],
        axis=0,
    )

    std = values.std(axis=0)

    return [
        feature_names[i]
        for i in range(len(feature_names))
        if std[i] != 0
    ]


def select_features(segments, feature_names):

    return [
        df[["timestamp"] + feature_names].copy()
        for df in segments
    ]


def graph_to_edge_set(edge_index):

    edge_index = (
        edge_index.detach()
        .cpu()
        .numpy()
    )

    return {
        (
            int(edge_index[0, i]),
            int(edge_index[1, i]),
        )
        for i in range(edge_index.shape[1])
    }


def jaccard_similarity(graph_a, graph_b):

    union = graph_a | graph_b

    if not union:
        return 1.0

    return len(graph_a & graph_b) / len(union)


# ============================================================
# Window-label construction
# ============================================================

def create_window_labels(
    test_labels,
    sequence_length,
):

    window_labels = []

    for labels in test_labels:

        labels_np = labels.to_numpy(
            dtype=np.int64
        )

        if len(labels_np) < sequence_length:
            raise ValueError(
                "Test segment is shorter than "
                "the sequence length."
            )

        # Endpoint label.
        window_labels.append(
            labels_np[
                sequence_length - 1:
            ]
        )

    return np.concatenate(
        window_labels
    )


# ============================================================
# Metric calculation
# ============================================================

def calculate_threshold_metrics(
    labels,
    scores,
    threshold,
):

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    tp = np.sum(
        (predictions == 1)
        & (labels == 1)
    )

    fp = np.sum(
        (predictions == 1)
        & (labels == 0)
    )

    fn = np.sum(
        (predictions == 0)
        & (labels == 1)
    )

    tn = np.sum(
        (predictions == 0)
        & (labels == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "predicted_anomalies": int(
            predictions.sum()
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("GRAPH-CHANGE ANOMALY SCORE")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ========================================================
    # Load HAI
    # ========================================================

    print("\nLoading HAI data...")

    config = load_config()

    training_df = load_training_data(
        config
    )

    test_df, test_labels = load_test_data(
        config
    )

    # ========================================================
    # Variable features
    # ========================================================

    variable_features = (
        get_variable_features(
            training_df
        )
    )

    print(
        f"Variable features: "
        f"{len(variable_features)}"
    )

    training_df = select_features(
        training_df,
        variable_features,
    )

    test_df = select_features(
        test_df,
        variable_features,
    )

    # ========================================================
    # Train-only normalization
    # ========================================================

    print(
        "\nFitting StandardScaler "
        "on training data only..."
    )

    preprocessor = Preprocessor()

    preprocessor.fit(
        training_df
    )

    test_scaled = preprocessor.transform(
        test_df
    )

    # ========================================================
    # Window dataset
    # ========================================================

    dataset = TemporalWindowDataset(
        test_scaled,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
    )

    print(
        f"Test windows: "
        f"{len(dataset):,}"
    )

    window_labels = create_window_labels(
        test_labels,
        SEQUENCE_LENGTH,
    )

    if len(window_labels) != len(dataset):

        raise RuntimeError(
            f"Window-label mismatch: "
            f"{len(window_labels)} labels "
            f"vs {len(dataset)} windows."
        )

    print(
        f"Window labels: "
        f"{len(window_labels):,}"
    )

    print(
        f"Attack windows: "
        f"{int(window_labels.sum()):,}"
    )

    # ========================================================
    # Load trained dynamic model
    # ========================================================

    print(
        "\nLoading dynamic graph model..."
    )

    model = DynamicGraphAutoencoder(
        sequence_length=SEQUENCE_LENGTH,
        num_nodes=len(variable_features),
        temporal_hidden_dim=TEMPORAL_HIDDEN,
        embedding_dim=EMBEDDING_DIM,
        gat_hidden_dim=GAT_HIDDEN,
        gat_heads=GAT_HEADS,
        k=GRAPH_K,
        dropout=0.0,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    # ========================================================
    # Sample test windows
    # ========================================================

    rng = np.random.default_rng(
        SEED
    )

    sample_size = min(
        SAMPLE_WINDOWS,
        len(dataset),
    )

    sampled_indices = np.sort(
        rng.choice(
            len(dataset),
            size=sample_size,
            replace=False,
        )
    )

    sampled_labels = (
        window_labels[
            sampled_indices
        ]
    )

    print(
        f"Sampled windows: "
        f"{sample_size:,}"
    )

    print(
        f"Sampled attack windows: "
        f"{int(sampled_labels.sum()):,}"
    )

    # ========================================================
    # Extract graphs
    # ========================================================

    print(
        "\nExtracting dynamic graphs..."
    )

    graphs = []

    for start in range(
        0,
        sample_size,
        BATCH_SIZE,
    ):

        batch_indices = (
            sampled_indices[
                start:
                start + BATCH_SIZE
            ]
        )

        windows = torch.stack(
            [
                dataset[int(i)]
                for i in batch_indices
            ]
        ).to(DEVICE)

        with torch.no_grad():

            batch_graphs = (
                model.get_dynamic_graphs(
                    windows
                )
            )

        if isinstance(
            batch_graphs,
            (list, tuple),
        ):

            current_graphs = batch_graphs

        elif isinstance(
            batch_graphs,
            torch.Tensor,
        ):

            if batch_graphs.ndim != 3:

                raise RuntimeError(
                    "Unexpected graph tensor "
                    f"shape: {batch_graphs.shape}"
                )

            current_graphs = [
                batch_graphs[i]
                for i in range(
                    batch_graphs.shape[0]
                )
            ]

        else:

            raise RuntimeError(
                "Unexpected graph return type: "
                f"{type(batch_graphs)}"
            )

        for graph in current_graphs:

            graphs.append(
                graph_to_edge_set(
                    graph
                )
            )

        processed = len(graphs)

        if (
            processed == sample_size
            or processed % 1024 < BATCH_SIZE
        ):

            print(
                f"Processed "
                f"{processed:,} / "
                f"{sample_size:,}"
            )

    # ========================================================
    # IMPORTANT:
    #
    # Graph-change score requires consecutive windows.
    #
    # Because the sampled windows are not necessarily adjacent,
    # we calculate the score between consecutive SAMPLED
    # windows. The corresponding original window indices are
    # retained so the gap can be reported.
    # ========================================================

    graph_change_scores = []
    graph_change_labels = []
    window_gaps = []

    for i in range(
        1,
        len(graphs),
    ):

        similarity = (
            jaccard_similarity(
                graphs[i - 1],
                graphs[i],
            )
        )

        score = 1.0 - similarity

        graph_change_scores.append(
            score
        )

        # Use the label of the current
        # window, matching the endpoint-label
        # convention.
        graph_change_labels.append(
            sampled_labels[i]
        )

        gap = (
            sampled_indices[i]
            - sampled_indices[i - 1]
        )

        window_gaps.append(
            int(gap)
        )

    graph_change_scores = np.asarray(
        graph_change_scores,
        dtype=np.float64,
    )

    graph_change_labels = np.asarray(
        graph_change_labels,
        dtype=np.int64,
    )

    window_gaps = np.asarray(
        window_gaps,
        dtype=np.int64,
    )

    # ========================================================
    # Report sampling gaps
    # ========================================================

    print(
        "\nGraph-change score pairs: "
        f"{len(graph_change_scores):,}"
    )

    print(
        "Median sampled-window gap: "
        f"{np.median(window_gaps):.0f}"
    )

    print(
        "Maximum sampled-window gap: "
        f"{np.max(window_gaps):,}"
    )

    # ========================================================
    # Ranking metrics
    # ========================================================

    auroc = roc_auc_score(
        graph_change_labels,
        graph_change_scores,
    )

    auprc = average_precision_score(
        graph_change_labels,
        graph_change_scores,
    )

    print(
        "\nRanking performance:"
    )

    print(
        f"AUROC: "
        f"{auroc:.6f}"
    )

    print(
        f"AUPRC: "
        f"{auprc:.6f}"
    )

    # ========================================================
    # Thresholds
    #
    # We use percentiles only for descriptive analysis here.
    # There is no training/validation-derived threshold because
    # this experiment is explicitly exploratory.
    # ========================================================

    percentile_results = {}

    for percentile in [
        95.0,
        97.0,
        98.0,
        99.0,
        99.5,
    ]:

        threshold = np.percentile(
            graph_change_scores,
            percentile,
        )

        percentile_results[
            str(percentile)
        ] = calculate_threshold_metrics(
            graph_change_labels,
            graph_change_scores,
            threshold,
        )

    # ========================================================
    # Normal vs attack score distributions
    # ========================================================

    normal_scores = (
        graph_change_scores[
            graph_change_labels == 0
        ]
    )

    attack_scores = (
        graph_change_scores[
            graph_change_labels == 1
        ]
    )

    distribution_results = {

        "normal_count":
            int(len(normal_scores)),

        "attack_count":
            int(len(attack_scores)),

        "normal_mean":
            float(np.mean(normal_scores)),

        "normal_median":
            float(np.median(normal_scores)),

        "normal_p95":
            float(np.percentile(
                normal_scores,
                95,
            )),

        "attack_mean":
            float(np.mean(attack_scores)),

        "attack_median":
            float(np.median(attack_scores)),

        "attack_p95":
            float(np.percentile(
                attack_scores,
                95,
            )),
    }

    # ========================================================
    # Save
    # ========================================================

    results = {

        "configuration": {

            "seed": SEED,

            "sequence_length":
                SEQUENCE_LENGTH,

            "graph_k":
                GRAPH_K,

            "sample_windows":
                sample_size,

            "checkpoint_epoch":
                int(checkpoint["epoch"]),

            "score_definition":
                "1 - Jaccard(previous_graph, current_graph)",

            "label_definition":
                "current_window_endpoint_label",
        },

        "sampling": {

            "median_window_gap":
                int(np.median(window_gaps)),

            "mean_window_gap":
                float(np.mean(window_gaps)),

            "maximum_window_gap":
                int(np.max(window_gaps)),
        },

        "ranking_metrics": {

            "auroc":
                float(auroc),

            "auprc":
                float(auprc),
        },

        "percentile_thresholds":
            percentile_results,

        "score_distributions":
            distribution_results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # Print summary
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GRAPH-CHANGE SCORE RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"AUROC: "
        f"{auroc:.6f}"
    )

    print(
        f"AUPRC: "
        f"{auprc:.6f}"
    )

    print(
        "\nScore distributions:"
    )

    print(
        f"Normal mean: "
        f"{distribution_results['normal_mean']:.6f}"
    )

    print(
        f"Attack mean: "
        f"{distribution_results['attack_mean']:.6f}"
    )

    print(
        f"Normal median: "
        f"{distribution_results['normal_median']:.6f}"
    )

    print(
        f"Attack median: "
        f"{distribution_results['attack_median']:.6f}"
    )

    print(
        "\nPercentile threshold results:"
    )

    for percentile, metrics in (
        percentile_results.items()
    ):

        print(
            f"P{percentile}: "
            f"F1={metrics['f1']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"FPR={metrics['fpr']:.4f}"
        )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()