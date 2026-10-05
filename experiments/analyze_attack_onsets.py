from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

TRANSITION_PATH = Path(
    "results/tables/"
    "dynamic_graph_transition_records_seed456.json"
)

OUTPUT_PATH = Path(
    "results/tables/"
    "dynamic_graph_attack_onset_paired_analysis_seed456.json"
)


# ============================================================
# Helpers
# ============================================================

def mean_or_none(values):
    if not values:
        return None
    return float(np.mean(values))


def median_or_none(values):
    if not values:
        return None
    return float(np.median(values))


def std_or_none(values):
    if not values:
        return None
    return float(np.std(values))


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("PAIRED ATTACK-ONSET GRAPH-CHANGE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load transition records
    # --------------------------------------------------------

    print("\nLoading transition records...")

    with open(
        TRANSITION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        records = json.load(f)

    print(
        f"Transition records loaded: "
        f"{len(records):,}"
    )

    # --------------------------------------------------------
    # Index records by transition index
    # --------------------------------------------------------

    records_by_index = {
        int(record["transition_index"]): record
        for record in records
    }

    # --------------------------------------------------------
    # Find attack onsets
    #
    # An attack onset is:
    # previous_label = 0
    # current_label  = 1
    # --------------------------------------------------------

    attack_onsets = [
        record
        for record in records
        if (
            int(record["previous_label"]) == 0
            and int(record["current_label"]) == 1
        )
    ]

    print(
        f"Attack onsets found: "
        f"{len(attack_onsets)}"
    )

    # --------------------------------------------------------
    # Paired comparison
    #
    # For onset at index i:
    #
    #   onset_change = change(i)
    #
    # The immediately preceding transition is:
    #
    #   baseline_change = change(i - 1)
    #
    # This gives one matched baseline for each attack event.
    # --------------------------------------------------------

    paired_results = []

    for onset in attack_onsets:

        onset_index = int(
            onset["transition_index"]
        )

        baseline_index = onset_index - 1

        if baseline_index not in records_by_index:
            continue

        baseline = records_by_index[
            baseline_index
        ]

        # The baseline must be a normal -> normal
        # transition immediately before the attack onset.
        if not (
            int(baseline["previous_label"]) == 0
            and int(baseline["current_label"]) == 0
        ):
            continue

        baseline_change = float(
            baseline["graph_change"]
        )

        onset_change = float(
            onset["graph_change"]
        )

        difference = (
            onset_change
            - baseline_change
        )

        ratio = None

        if baseline_change != 0:
            ratio = (
                onset_change
                / baseline_change
            )

        paired_results.append(
            {
                "attack_onset_index": onset_index,
                "baseline_transition_index": baseline_index,
                "baseline_graph_change": baseline_change,
                "onset_graph_change": onset_change,
                "difference": difference,
                "ratio": ratio,
            }
        )

    # --------------------------------------------------------
    # Extract paired differences
    # --------------------------------------------------------

    differences = [
        item["difference"]
        for item in paired_results
    ]

    baseline_changes = [
        item["baseline_graph_change"]
        for item in paired_results
    ]

    onset_changes = [
        item["onset_graph_change"]
        for item in paired_results
    ]

    ratios = [
        item["ratio"]
        for item in paired_results
        if item["ratio"] is not None
    ]

    # --------------------------------------------------------
    # Direction of change
    # --------------------------------------------------------

    onset_greater_count = sum(
        1
        for item in paired_results
        if (
            item["onset_graph_change"]
            > item["baseline_graph_change"]
        )
    )

    onset_lower_count = sum(
        1
        for item in paired_results
        if (
            item["onset_graph_change"]
            < item["baseline_graph_change"]
        )
    )

    equal_count = sum(
        1
        for item in paired_results
        if (
            item["onset_graph_change"]
            == item["baseline_graph_change"]
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "configuration": {
            "transition_file": str(
                TRANSITION_PATH
            ),
            "comparison": (
                "attack onset graph change "
                "vs immediately preceding "
                "normal-to-normal graph change"
            ),
            "attack_onset_definition": (
                "previous_label=0 and "
                "current_label=1"
            ),
            "baseline_definition": (
                "immediately preceding "
                "normal-to-normal transition"
            ),
        },

        "counts": {
            "attack_onsets_found": len(
                attack_onsets
            ),
            "valid_paired_events": len(
                paired_results
            ),
            "onset_greater_than_baseline": (
                onset_greater_count
            ),
            "onset_less_than_baseline": (
                onset_lower_count
            ),
            "equal": equal_count,
        },

        "baseline_graph_change": {
            "count": len(baseline_changes),
            "mean": mean_or_none(
                baseline_changes
            ),
            "median": median_or_none(
                baseline_changes
            ),
            "std": std_or_none(
                baseline_changes
            ),
        },

        "attack_onset_graph_change": {
            "count": len(onset_changes),
            "mean": mean_or_none(
                onset_changes
            ),
            "median": median_or_none(
                onset_changes
            ),
            "std": std_or_none(
                onset_changes
            ),
        },

        "paired_difference": {
            "definition": (
                "onset_graph_change "
                "- baseline_graph_change"
            ),
            "mean": mean_or_none(
                differences
            ),
            "median": median_or_none(
                differences
            ),
            "std": std_or_none(
                differences
            ),
            "positive_fraction": (
                float(
                    np.mean(
                        np.asarray(differences) > 0
                    )
                )
                if differences
                else None
            ),
        },

        "onset_to_baseline_ratio": {
            "count": len(ratios),
            "mean": mean_or_none(ratios),
            "median": median_or_none(ratios),
        },

        "paired_events": paired_results,
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
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nAttack onsets found: "
        f"{len(attack_onsets)}"
    )

    print(
        f"Valid paired events: "
        f"{len(paired_results)}"
    )

    print("\nBaseline graph change:")
    print(
        f"  Mean:   "
        f"{mean_or_none(baseline_changes):.6f}"
    )
    print(
        f"  Median: "
        f"{median_or_none(baseline_changes):.6f}"
    )

    print("\nAttack-onset graph change:")
    print(
        f"  Mean:   "
        f"{mean_or_none(onset_changes):.6f}"
    )
    print(
        f"  Median: "
        f"{median_or_none(onset_changes):.6f}"
    )

    print("\nPaired difference:")
    print(
        f"  Mean:   "
        f"{mean_or_none(differences):.6f}"
    )
    print(
        f"  Median: "
        f"{median_or_none(differences):.6f}"
    )

    print(
        "\nOnset > baseline: "
        f"{onset_greater_count}/"
        f"{len(paired_results)}"
    )

    print(
        "Onset < baseline: "
        f"{onset_lower_count}/"
        f"{len(paired_results)}"
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