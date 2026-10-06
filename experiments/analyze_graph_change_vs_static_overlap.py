import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


TRANSITION_PATH = Path(
    "results/tables/dynamic_graph_transition_records_seed456.json"
)

OVERLAP_PATH = Path(
    "results/tables/dynamic_static_edge_overlap.npy"
)

OUTPUT_PATH = Path(
    "results/tables/graph_change_vs_static_overlap.json"
)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    if len(values) == 0:
        return {"n": 0}

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def main():
    with open(
        TRANSITION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        transitions = json.load(f)

    overlaps = np.load(OVERLAP_PATH)

    graph_changes = []
    current_overlaps = []
    transition_types = []
    transition_indices = []

    for record in transitions:
        index = int(record["transition_index"])

        graph_change = float(record["graph_change"])

        # transition i is from window i-1 -> window i.
        # Therefore the static-overlap value corresponding
        # to this transition is overlap[index].
        if index >= len(overlaps):
            raise RuntimeError(
                f"Transition index {index} exceeds "
                f"overlap array length {len(overlaps)}"
            )

        previous_label = int(record["previous_label"])
        current_label = int(record["current_label"])

        if previous_label == 0 and current_label == 0:
            transition_type = "normal_to_normal"

        elif previous_label == 0 and current_label == 1:
            transition_type = "normal_to_attack"

        elif previous_label == 1 and current_label == 1:
            transition_type = "attack_to_attack"

        else:
            transition_type = "attack_to_normal"

        graph_changes.append(graph_change)
        current_overlaps.append(float(overlaps[index]))
        transition_types.append(transition_type)
        transition_indices.append(index)

    graph_changes = np.asarray(
        graph_changes,
        dtype=np.float64,
    )

    current_overlaps = np.asarray(
        current_overlaps,
        dtype=np.float64,
    )

    transition_types = np.asarray(
        transition_types,
        dtype=object,
    )

    transition_indices = np.asarray(
        transition_indices,
        dtype=np.int64,
    )

    if len(graph_changes) != len(current_overlaps):
        raise RuntimeError(
            "Graph-change and overlap arrays are misaligned."
        )

    # --------------------------------------------------------
    # Correlation
    # --------------------------------------------------------

    pearson_r, pearson_p = pearsonr(
        graph_changes,
        current_overlaps,
    )

    spearman_rho, spearman_p = spearmanr(
        graph_changes,
        current_overlaps,
    )

    # --------------------------------------------------------
    # Transition-specific summaries
    # --------------------------------------------------------

    transition_summary = {}

    for transition_type in [
        "normal_to_normal",
        "normal_to_attack",
        "attack_to_attack",
        "attack_to_normal",
    ]:
        mask = transition_types == transition_type

        transition_summary[transition_type] = {
            "graph_change": summarize(
                graph_changes[mask]
            ),
            "static_overlap": summarize(
                current_overlaps[mask]
            ),
        }

    # --------------------------------------------------------
    # Attack-onset comparison
    #
    # For each normal -> attack transition:
    # compare its graph change and static overlap with
    # the immediately preceding normal -> normal transition.
    #
    # This uses only chronological transitions and excludes
    # the test1 -> test2 boundary naturally because that
    # transition does not exist in transition_indices.
    # --------------------------------------------------------

    record_by_index = {
        int(record["transition_index"]): record
        for record in transitions
    }

    onset_pairs = []

    for index in transition_indices:
        record = record_by_index[int(index)]

        if (
            int(record["previous_label"]) == 0
            and int(record["current_label"]) == 1
        ):
            baseline_index = int(index) - 1

            if baseline_index not in record_by_index:
                continue

            baseline = record_by_index[baseline_index]

            if not (
                int(baseline["previous_label"]) == 0
                and int(baseline["current_label"]) == 0
            ):
                continue

            onset_pairs.append(
                {
                    "onset_index": int(index),
                    "baseline_index": int(baseline_index),
                    "onset_graph_change": float(
                        record["graph_change"]
                    ),
                    "baseline_graph_change": float(
                        baseline["graph_change"]
                    ),
                    "onset_static_overlap": float(
                        overlaps[index]
                    ),
                    "baseline_static_overlap": float(
                        overlaps[baseline_index]
                    ),
                }
            )

    onset_graph_change_diff = np.asarray(
        [
            x["onset_graph_change"]
            - x["baseline_graph_change"]
            for x in onset_pairs
        ],
        dtype=np.float64,
    )

    onset_overlap_diff = np.asarray(
        [
            x["onset_static_overlap"]
            - x["baseline_static_overlap"]
            for x in onset_pairs
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "experiment": (
            "graph_change_vs_dynamic_static_edge_overlap"
        ),
        "transition_records": str(
            TRANSITION_PATH
        ),
        "overlap_array": str(
            OVERLAP_PATH
        ),
        "num_transitions": int(
            len(graph_changes)
        ),
        "excluded_test_segment_boundary": 53941,
        "correlation": {
            "pearson": {
                "r": float(pearson_r),
                "p_value": float(pearson_p),
            },
            "spearman": {
                "rho": float(spearman_rho),
                "p_value": float(spearman_p),
            },
        },
        "overall": {
            "graph_change": summarize(
                graph_changes
            ),
            "static_overlap": summarize(
                current_overlaps
            ),
        },
        "transition_summary": transition_summary,
        "attack_onset_paired_analysis": {
            "n": int(len(onset_pairs)),
            "graph_change_difference": summarize(
                onset_graph_change_diff
            ),
            "static_overlap_difference": summarize(
                onset_overlap_diff
            ),
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Graph change vs static overlap analysis"
    )
    print(
        f"Transitions: {len(graph_changes)}"
    )
    print(
        f"Pearson r: {pearson_r:.6f}"
        f"  p={pearson_p:.6g}"
    )
    print(
        f"Spearman rho: {spearman_rho:.6f}"
        f"  p={spearman_p:.6g}"
    )

    print("\nTransition summaries:")

    for transition_type, values in (
        transition_summary.items()
    ):
        print(
            f"{transition_type}: "
            f"graph_change_mean="
            f"{values['graph_change']['mean']:.6f}, "
            f"overlap_mean="
            f"{values['static_overlap']['mean']:.6f}"
        )

    print(
        "\nAttack-onset paired analysis:"
    )

    print(
        f"Events: {len(onset_pairs)}"
    )

    if len(onset_pairs) > 0:
        print(
            "Mean graph-change difference: "
            f"{np.mean(onset_graph_change_diff):.6f}"
        )

        print(
            "Mean static-overlap difference: "
            f"{np.mean(onset_overlap_diff):.6f}"
        )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()