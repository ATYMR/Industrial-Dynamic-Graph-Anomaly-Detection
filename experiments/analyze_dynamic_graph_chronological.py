from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

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

SEED = 456

SEQUENCE_LENGTH = 60
STRIDE = 1
GRAPH_K = 5

TEMPORAL_HIDDEN = 32
EMBEDDING_DIM = 32
GAT_HIDDEN = 64
GAT_HEADS = 4

BATCH_SIZE = 256

CHECKPOINT_PATH = Path(
    "results/checkpoints/dynamic_graph_autoencoder_seed456_best.pt"
)

OUTPUT_PATH = Path(
    "results/tables/dynamic_graph_chronological_analysis_seed456.json"
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
            df[feature_names].to_numpy(
                dtype=np.float64
            )
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
        edge_index
        .detach()
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


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("CHRONOLOGICAL DYNAMIC GRAPH ANALYSIS")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load HAI data
    # --------------------------------------------------------

    print("\nLoading HAI data...")

    config = load_config()

    training_df = load_training_data(config)

    test_df, test_labels = load_test_data(config)

    variable_features = get_variable_features(
        training_df
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

    # --------------------------------------------------------
    # Train-only normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Create test dataset
    # --------------------------------------------------------

    dataset = TemporalWindowDataset(
        test_scaled,
        sequence_length=SEQUENCE_LENGTH,
        stride=STRIDE,
    )

    print(
        f"Test windows: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Create endpoint labels
    # --------------------------------------------------------

    window_labels_by_segment = []

    for labels in test_labels:

        labels_np = labels.to_numpy(
            dtype=np.int64
        )

        if len(labels_np) < SEQUENCE_LENGTH:
            raise ValueError(
                "Test segment is shorter "
                "than sequence length."
            )

        segment_window_labels = (
            labels_np[
                SEQUENCE_LENGTH - 1:
            ]
        )

        window_labels_by_segment.append(
            segment_window_labels
        )

    window_labels = np.concatenate(
        window_labels_by_segment
    )

    if len(window_labels) != len(dataset):
        raise RuntimeError(
            "Window-label mismatch: "
            f"{len(window_labels)} labels "
            f"vs {len(dataset)} windows."
        )

    print(
        f"Attack windows: "
        f"{int(window_labels.sum()):,}"
    )

    print(
        f"Normal windows: "
        f"{int((window_labels == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # Calculate segment boundaries
    # --------------------------------------------------------

    segment_window_lengths = [
        len(labels)
        for labels in window_labels_by_segment
    ]

    segment_boundaries = set(
        np.cumsum(
            segment_window_lengths
        )[:-1]
    )

    print(
        "\nTest segment window lengths:",
        segment_window_lengths,
    )

    print(
        "Segment boundaries:",
        sorted(segment_boundaries),
    )

    # --------------------------------------------------------
    # Load dynamic graph model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Extract graphs chronologically
    # --------------------------------------------------------

    print(
        "\nExtracting graphs in chronological order..."
    )

    graphs = []

    with torch.no_grad():

        for start in range(
            0,
            len(dataset),
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                len(dataset),
            )

            windows = torch.stack(
                [
                    dataset[i]
                    for i in range(
                        start,
                        end,
                    )
                ]
            ).to(DEVICE)

            batch_graphs = (
                model.get_dynamic_graphs(
                    windows
                )
            )

            for graph in batch_graphs:

                graphs.append(
                    graph_to_edge_set(
                        graph
                    )
                )

            processed = len(graphs)

            if (
                processed == len(dataset)
                or processed % 1024 < BATCH_SIZE
            ):
                print(
                    f"Processed "
                    f"{processed:,} / "
                    f"{len(dataset):,}"
                )

    # --------------------------------------------------------
    # Chronological graph changes
    # --------------------------------------------------------

    graph_changes = []
    transition_records = []

    # Maps the GLOBAL transition index
    # to its graph-change value.
    graph_change_by_index = {}

    transition_values = {
        "normal_to_normal": [],
        "normal_to_attack": [],
        "attack_to_attack": [],
        "attack_to_normal": [],
    }

    transition_counts = {
        "normal_to_normal": 0,
        "normal_to_attack": 0,
        "attack_to_attack": 0,
        "attack_to_normal": 0,
    }

    for i in range(
        1,
        len(graphs),
    ):

        # IMPORTANT:
        # Do not compare the last window of
        # test1 with the first window of test2.
        if i in segment_boundaries:
            continue

        previous_label = int(
            window_labels[i - 1]
        )

        current_label = int(
            window_labels[i]
        )

        similarity = jaccard_similarity(
            graphs[i - 1],
            graphs[i],
        )

        change = 1.0 - similarity

        graph_changes.append(change)

        transition_records.append(
    {
        "transition_index": int(i),
        "previous_label": previous_label,
        "current_label": current_label,
        "graph_change": float(change),
    }
)

        graph_change_by_index[i] = (
            change
        )

        if (
            previous_label == 0
            and current_label == 0
        ):
            transition_type = (
                "normal_to_normal"
            )

        elif (
            previous_label == 0
            and current_label == 1
        ):
            transition_type = (
                "normal_to_attack"
            )

        elif (
            previous_label == 1
            and current_label == 1
        ):
            transition_type = (
                "attack_to_attack"
            )

        else:
            transition_type = (
                "attack_to_normal"
            )

        transition_values[
            transition_type
        ].append(change)

        transition_counts[
            transition_type
        ] += 1

    # --------------------------------------------------------
    # Attack onsets and recoveries
    # --------------------------------------------------------

    attack_onset_indices = [
        i
        for i in range(
            1,
            len(window_labels),
        )
        if (
            window_labels[i - 1] == 0
            and window_labels[i] == 1
            and i in graph_change_by_index
        )
    ]

    recovery_indices = [
        i
        for i in range(
            1,
            len(window_labels),
        )
        if (
            window_labels[i - 1] == 1
            and window_labels[i] == 0
            and i in graph_change_by_index
        )
    ]

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "configuration": {
            "seed": SEED,
            "sequence_length": SEQUENCE_LENGTH,
            "stride": STRIDE,
            "graph_k": GRAPH_K,
            "checkpoint_epoch": int(
                checkpoint["epoch"]
            ),
            "num_features": len(
                variable_features
            ),
            "test_windows": len(
                dataset
            ),
            "label_definition":
                "endpoint_timestep_label",
            "graph_change_definition":
                "1 - Jaccard(previous_graph, current_graph)",
            "cross_test_segment_transitions_excluded":
                True,
        },

        "test_segment_window_lengths":
            segment_window_lengths,

        "transition_counts":
            transition_counts,

        "transition_distributions": {
            key: summarize(values)
            for key, values
            in transition_values.items()
        },

        "overall_graph_change":
            summarize(
                graph_changes
            ),

        "attack_onsets": {
            "count": int(
                len(attack_onset_indices)
            ),

            "mean_change": (
                float(
                    np.mean(
                        [
                            graph_change_by_index[
                                i
                            ]
                            for i in attack_onset_indices
                        ]
                    )
                )
                if attack_onset_indices
                else None
            ),

            "median_change": (
                float(
                    np.median(
                        [
                            graph_change_by_index[
                                i
                            ]
                            for i in attack_onset_indices
                        ]
                    )
                )
                if attack_onset_indices
                else None
            ),
        },

        "recoveries": {
            "count": int(
                len(recovery_indices)
            ),
        },
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

    TRANSITION_OUTPUT_PATH = Path(
        "results/tables/"
        "dynamic_graph_transition_records_seed456.json"
    )

    with open(
        TRANSITION_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            transition_records,
            f,
            indent=2,
        )

    print(
        f"  count = "
        f"{results['recoveries']['count']}"
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