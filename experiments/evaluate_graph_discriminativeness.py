from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


RESULTS_DIR = Path("results/tables")

TRANSITION_RECORDS_PATH = (
    RESULTS_DIR
    / "dynamic_graph_transition_records_seed456.json"
)

OVERLAP_PATH = (
    RESULTS_DIR
    / "dynamic_static_edge_overlap.npy"
)

OVERLAP_LABELS_PATH = (
    RESULTS_DIR
    / "dynamic_static_edge_overlap_labels.npy"
)


def evaluate_feature(name, feature, labels):
    feature = np.asarray(feature, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    valid = np.isfinite(feature)

    feature = feature[valid]
    labels = labels[valid]

    if len(feature) != len(labels):
        raise RuntimeError(
            f"{name}: feature/label length mismatch."
        )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            f"{name}: both classes are required."
        )

    normal = feature[labels == 0]
    attack = feature[labels == 1]

    auroc = roc_auc_score(labels, feature)
    auprc = average_precision_score(labels, feature)

    u_stat, p_value = mannwhitneyu(
        normal,
        attack,
        alternative="two-sided",
    )

    return {
        "n_windows": int(len(feature)),
        "normal_windows": int(len(normal)),
        "attack_windows": int(len(attack)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "normal_mean": float(np.mean(normal)),
        "attack_mean": float(np.mean(attack)),
        "normal_median": float(np.median(normal)),
        "attack_median": float(np.median(attack)),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_value),
    }


def zscore(x):
    x = np.asarray(x, dtype=np.float64)

    mean = np.mean(x)
    std = np.std(x)

    if std == 0:
        return np.zeros_like(x)

    return (x - mean) / std


def load_transition_records():
    with TRANSITION_RECORDS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise TypeError(
            "Transition records must be a list."
        )

    return records


def main():
    print("=" * 70)
    print("GRAPH DISCRIMINATIVENESS EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load window-level overlap and labels
    # --------------------------------------------------------

    print("\nLoading window-level graph overlap...")

    overlap = np.load(OVERLAP_PATH)
    overlap_labels = np.load(
        OVERLAP_LABELS_PATH
    )

    if len(overlap) != len(overlap_labels):
        raise RuntimeError(
            "Overlap and overlap-label arrays differ "
            f"in length: {len(overlap)} vs "
            f"{len(overlap_labels)}"
        )

    overlap_labels = overlap_labels.astype(
        np.int64
    )

    print(
        f"Windows: {len(overlap):,}"
    )

    print(
        f"Normal windows: "
        f"{int(np.sum(overlap_labels == 0)):,}"
    )

    print(
        f"Attack windows: "
        f"{int(np.sum(overlap_labels == 1)):,}"
    )

    # --------------------------------------------------------
    # Feature 1: dynamic/static divergence
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "FEATURE 1: DYNAMIC/STATIC EDGE DIVERGENCE"
    )
    print("=" * 70)

    # Higher divergence = greater difference from static graph.
    divergence = 1.0 - overlap

    divergence_result = evaluate_feature(
        "dynamic_static_divergence",
        divergence,
        overlap_labels,
    )

    print(
        f"AUROC: "
        f"{divergence_result['auroc']:.6f}"
    )

    print(
        f"AUPRC:  "
        f"{divergence_result['auprc']:.6f}"
    )

    print(
        f"Normal mean: "
        f"{divergence_result['normal_mean']:.6f}"
    )

    print(
        f"Attack mean: "
        f"{divergence_result['attack_mean']:.6f}"
    )

    print(
        f"Mann-Whitney p: "
        f"{divergence_result['mann_whitney_p']:.6g}"
    )

    # --------------------------------------------------------
    # Load graph-change transition records
    # --------------------------------------------------------

    print("\nLoading transition-level graph changes...")

    records = load_transition_records()

    print(
        f"Transition records: "
        f"{len(records):,}"
    )

    # The transition records contain:
    # transition_index = endpoint window index.
    #
    # Therefore we can directly create a window-level graph-change
    # array. The first window has no previous graph, and the
    # test1 -> test2 boundary is intentionally absent.

    graph_change = np.full(
        len(overlap_labels),
        np.nan,
        dtype=np.float64,
    )

    for record in records:
        index = int(
            record["transition_index"]
        )

        if index < 0 or index >= len(
            graph_change
        ):
            raise ValueError(
                f"Invalid transition index: {index}"
            )

        graph_change[index] = float(
            record["graph_change"]
        )

    valid_graph = np.isfinite(
        graph_change
    )

    print(
        f"Graph-change windows: "
        f"{int(np.sum(valid_graph)):,}"
    )

    print(
        f"Graph-change unavailable: "
        f"{int(np.sum(~valid_graph)):,}"
    )

    # Verify expected count.
    if int(np.sum(valid_graph)) != 284280:
        raise RuntimeError(
            "Unexpected number of graph-change values."
        )

    # Verify that the missing positions are exactly:
    # first test window + test1/test2 boundary.
    missing_indices = np.where(
        ~valid_graph
    )[0]

    expected_missing = np.array(
        [0, 53941],
        dtype=np.int64,
    )

    if not np.array_equal(
        missing_indices,
        expected_missing,
    ):
        raise RuntimeError(
            "Unexpected missing graph-change indices: "
            f"{missing_indices}"
        )

    # --------------------------------------------------------
    # Feature 2: graph change
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE 2: GRAPH CHANGE")
    print("=" * 70)

    graph_change_result = evaluate_feature(
        "graph_change",
        graph_change[valid_graph],
        overlap_labels[valid_graph],
    )

    print(
        f"AUROC: "
        f"{graph_change_result['auroc']:.6f}"
    )

    print(
        f"AUPRC:  "
        f"{graph_change_result['auprc']:.6f}"
    )

    print(
        f"Normal mean: "
        f"{graph_change_result['normal_mean']:.6f}"
    )

    print(
        f"Attack mean: "
        f"{graph_change_result['attack_mean']:.6f}"
    )

    print(
        f"Mann-Whitney p: "
        f"{graph_change_result['mann_whitney_p']:.6g}"
    )

    # --------------------------------------------------------
    # Feature 3: combined exploratory signal
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE 3: COMBINED GRAPH SIGNAL")
    print("=" * 70)

    combined_valid = (
        valid_graph
        & np.isfinite(divergence)
    )

    graph_change_valid = (
        graph_change[combined_valid]
    )

    divergence_valid = (
        divergence[combined_valid]
    )

    labels_valid = (
        overlap_labels[combined_valid]
    )

    combined = (
        zscore(graph_change_valid)
        + zscore(divergence_valid)
    ) / 2.0

    combined_result = evaluate_feature(
        "combined_graph_signal",
        combined,
        labels_valid,
    )

    print(
        f"AUROC: "
        f"{combined_result['auroc']:.6f}"
    )

    print(
        f"AUPRC:  "
        f"{combined_result['auprc']:.6f}"
    )

    print(
        f"Normal mean: "
        f"{combined_result['normal_mean']:.6f}"
    )

    print(
        f"Attack mean: "
        f"{combined_result['attack_mean']:.6f}"
    )

    print(
        f"Mann-Whitney p: "
        f"{combined_result['mann_whitney_p']:.6g}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "experiment": (
            "graph_discriminativeness"
        ),
        "dataset": "HAI_23.05",
        "feature_alignment": {
            "overlap": "window_level",
            "graph_change": (
                "transition_to_endpoint_window"
            ),
            "excluded_graph_change_indices": [
                0,
                53941,
            ],
            "excluded_reason": [
                "first_window_has_no_previous_graph",
                "test1_to_test2_boundary",
            ],
        },
        "features": {
            "dynamic_static_divergence":
                divergence_result,
            "graph_change":
                graph_change_result,
            "combined_graph_signal":
                combined_result,
        },
    }

    output_path = (
        RESULTS_DIR
        / "graph_discriminativeness.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results[
        "features"
    ].items():
        print(
            f"{name:30s} "
            f"AUROC={result['auroc']:.4f} "
            f"AUPRC={result['auprc']:.4f}"
        )

    print(
        f"\nSaved: {output_path}"
    )


if __name__ == "__main__":
    main()
