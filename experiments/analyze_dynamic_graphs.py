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
    "results/tables/dynamic_graph_attack_normal_analysis.json"
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


def jaccard(a, b):

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def turnover(a, b):

    union = a | b

    if not union:
        return 0.0

    return len(a.symmetric_difference(b)) / len(union)


def summarize_graphs(graphs):

    if len(graphs) == 0:

        return {
            "count": 0,
            "unique_graphs": 0,
            "unique_graph_fraction": 0.0,
            "mean_edges": None,
            "mean_consecutive_jaccard": None,
            "median_consecutive_jaccard": None,
            "mean_consecutive_turnover": None,
            "median_consecutive_turnover": None,
        }

    edge_counts = np.array(
        [len(g) for g in graphs],
        dtype=np.int64,
    )

    signatures = [
        tuple(sorted(g))
        for g in graphs
    ]

    unique_count = len(set(signatures))

    if len(graphs) > 1:

        jaccards = np.array(
            [
                jaccard(
                    graphs[i - 1],
                    graphs[i],
                )
                for i in range(1, len(graphs))
            ]
        )

        turnovers = np.array(
            [
                turnover(
                    graphs[i - 1],
                    graphs[i],
                )
                for i in range(1, len(graphs))
            ]
        )

    else:

        jaccards = np.array([])
        turnovers = np.array([])

    return {
        "count": int(len(graphs)),
        "unique_graphs": int(unique_count),
        "unique_graph_fraction": float(
            unique_count / len(graphs)
        ),
        "mean_edges": float(
            np.mean(edge_counts)
        ),
        "mean_consecutive_jaccard": (
            float(np.mean(jaccards))
            if len(jaccards)
            else None
        ),
        "median_consecutive_jaccard": (
            float(np.median(jaccards))
            if len(jaccards)
            else None
        ),
        "mean_consecutive_turnover": (
            float(np.mean(turnovers))
            if len(turnovers)
            else None
        ),
        "median_consecutive_turnover": (
            float(np.median(turnovers))
            if len(turnovers)
            else None
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("ATTACK vs NORMAL DYNAMIC GRAPH ANALYSIS")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading HAI data...")

    config = load_config()

    train_df = load_training_data(config)

    test_df, test_labels = load_test_data(config)

    variable_features = get_variable_features(
        train_df
    )

    print(
        f"Variable features: "
        f"{len(variable_features)}"
    )

    train_df = select_features(
        train_df,
        variable_features,
    )

    test_df = select_features(
        test_df,
        variable_features,
    )

    # --------------------------------------------------------
    # Normalize using training data only
    # --------------------------------------------------------

    print(
        "\nFitting StandardScaler "
        "on training data only..."
    )

    preprocessor = Preprocessor()

    preprocessor.fit(train_df)

    test_scaled = preprocessor.transform(
        test_df
    )

    # --------------------------------------------------------
    # Build test window dataset
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
    # Create window-level labels
    #
    # IMPORTANT:
    # Each window gets the label of its endpoint timestep.
    # This matches the existing evaluation protocol.
    # --------------------------------------------------------

    window_labels = []

    for labels in test_labels:

        labels_np = labels.to_numpy(
            dtype=np.int64
        )

        if len(labels_np) < SEQUENCE_LENGTH:

            raise ValueError(
                "Test segment shorter than "
                "sequence length."
            )

        window_labels.append(
            labels_np[
                SEQUENCE_LENGTH - 1:
            ]
        )

    window_labels = np.concatenate(
        window_labels
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

    print(
        f"Normal windows: "
        f"{int((window_labels == 0).sum()):,}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading dynamic graph model...")

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
    # Sample windows
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

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

    sampled_labels = window_labels[
        sampled_indices
    ]

    print(
        f"Sampled windows: "
        f"{sample_size:,}"
    )

    print(
        f"Sampled normal windows: "
        f"{int((sampled_labels == 0).sum()):,}"
    )

    print(
        f"Sampled attack windows: "
        f"{int((sampled_labels == 1).sum()):,}"
    )

    # --------------------------------------------------------
    # Extract graphs
    # --------------------------------------------------------

    graphs = []

    print("\nExtracting dynamic graphs...")

    for start in range(
        0,
        sample_size,
        BATCH_SIZE,
    ):

        batch_indices = sampled_indices[
            start:start + BATCH_SIZE
        ]

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
                    "Unexpected graph tensor shape: "
                    f"{batch_graphs.shape}"
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
                graph_to_edge_set(graph)
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

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    normal_graphs = [
        graphs[i]
        for i in range(sample_size)
        if sampled_labels[i] == 0
    ]

    attack_graphs = [
        graphs[i]
        for i in range(sample_size)
        if sampled_labels[i] == 1
    ]

    overall = summarize_graphs(
        graphs
    )

    normal = summarize_graphs(
        normal_graphs
    )

    attack = summarize_graphs(
        attack_graphs
    )

    # --------------------------------------------------------
    # Attack-transition analysis
    #
    # Find sampled windows where the endpoint label changes:
    # 0 -> 1 or 1 -> 0.
    #
    # We compare the graph immediately before and after the
    # transition when both are present in the sampled sequence.
    # --------------------------------------------------------

    transition_jaccards = []
    transition_turnovers = []

    attack_onset_jaccards = []
    attack_onset_turnovers = []

    for i in range(1, sample_size):

        previous_label = sampled_labels[i - 1]
        current_label = sampled_labels[i]

        if previous_label != current_label:

            sim = jaccard(
                graphs[i - 1],
                graphs[i],
            )

            change = turnover(
                graphs[i - 1],
                graphs[i],
            )

            transition_jaccards.append(
                sim
            )

            transition_turnovers.append(
                change
            )

            if (
                previous_label == 0
                and current_label == 1
            ):

                attack_onset_jaccards.append(
                    sim
                )

                attack_onset_turnovers.append(
                    change
                )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "configuration": {

            "seed": SEED,

            "sequence_length":
                SEQUENCE_LENGTH,

            "stride": STRIDE,

            "graph_k": GRAPH_K,

            "sample_windows":
                sample_size,

            "num_nodes":
                len(variable_features),

            "checkpoint_epoch":
                int(checkpoint["epoch"]),

            "window_label_definition":
                "endpoint_timestep_label",
        },

        "sample_counts": {

            "total":
                int(sample_size),

            "normal":
                int((sampled_labels == 0).sum()),

            "attack":
                int((sampled_labels == 1).sum()),
        },

        "overall":
            overall,

        "normal":
            normal,

        "attack":
            attack,

        "label_transitions": {

            "total_transitions":
                len(transition_jaccards),

            "mean_jaccard":
                (
                    float(
                        np.mean(
                            transition_jaccards
                        )
                    )
                    if transition_jaccards
                    else None
                ),

            "mean_turnover":
                (
                    float(
                        np.mean(
                            transition_turnovers
                        )
                    )
                    if transition_turnovers
                    else None
                ),
        },

        "attack_onsets": {

            "count":
                len(attack_onset_jaccards),

            "mean_jaccard":
                (
                    float(
                        np.mean(
                            attack_onset_jaccards
                        )
                    )
                    if attack_onset_jaccards
                    else None
                ),

            "mean_turnover":
                (
                    float(
                        np.mean(
                            attack_onset_turnovers
                        )
                    )
                    if attack_onset_turnovers
                    else None
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

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ATTACK vs NORMAL RESULTS")
    print("=" * 70)

    print("\nOVERALL")
    print(
        f"Graphs: {overall['count']:,}"
    )
    print(
        f"Unique graph fraction: "
        f"{overall['unique_graph_fraction']:.4f}"
    )
    print(
        f"Mean Jaccard: "
        f"{overall['mean_consecutive_jaccard']:.4f}"
    )
    print(
        f"Mean turnover: "
        f"{overall['mean_consecutive_turnover']:.4f}"
    )

    print("\nNORMAL")
    print(
        f"Graphs: {normal['count']:,}"
    )
    print(
        f"Unique graph fraction: "
        f"{normal['unique_graph_fraction']:.4f}"
    )
    print(
        f"Mean Jaccard: "
        f"{normal['mean_consecutive_jaccard']}"
    )
    print(
        f"Mean turnover: "
        f"{normal['mean_consecutive_turnover']}"
    )

    print("\nATTACK")
    print(
        f"Graphs: {attack['count']:,}"
    )
    print(
        f"Unique graph fraction: "
        f"{attack['unique_graph_fraction']:.4f}"
    )
    print(
        f"Mean Jaccard: "
        f"{attack['mean_consecutive_jaccard']}"
    )
    print(
        f"Mean turnover: "
        f"{attack['mean_consecutive_turnover']}"
    )

    print("\nATTACK ONSETS")

    print(
        f"Transitions detected in sample: "
        f"{len(attack_onset_jaccards)}"
    )

    if attack_onset_jaccards:

        print(
            f"Mean onset Jaccard: "
            f"{np.mean(attack_onset_jaccards):.4f}"
        )

        print(
            f"Mean onset turnover: "
            f"{np.mean(attack_onset_turnovers):.4f}"
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