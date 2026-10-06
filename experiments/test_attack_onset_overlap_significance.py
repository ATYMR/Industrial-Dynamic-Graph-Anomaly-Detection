import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


TRANSITION_PATH = Path(
    "results/tables/dynamic_graph_transition_records_seed456.json"
)

OVERLAP_PATH = Path(
    "results/tables/dynamic_static_edge_overlap.npy"
)

OUTPUT_PATH = Path(
    "results/tables/attack_onset_overlap_significance.json"
)

SEED = 42
BOOTSTRAP_SAMPLES = 10000


def main():
    with open(
        TRANSITION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        transitions = json.load(f)

    overlaps = np.load(OVERLAP_PATH)

    records = {
        int(record["transition_index"]): record
        for record in transitions
    }

    onset_differences = []
    onset_pairs = []

    for index, record in records.items():
        previous_label = int(record["previous_label"])
        current_label = int(record["current_label"])

        if not (
            previous_label == 0
            and current_label == 1
        ):
            continue

        baseline_index = index - 1

        if baseline_index not in records:
            continue

        baseline = records[baseline_index]

        if not (
            int(baseline["previous_label"]) == 0
            and int(baseline["current_label"]) == 0
        ):
            continue

        onset_overlap = float(overlaps[index])
        baseline_overlap = float(
            overlaps[baseline_index]
        )

        difference = (
            onset_overlap
            - baseline_overlap
        )

        onset_differences.append(difference)

        onset_pairs.append(
            {
                "onset_index": index,
                "baseline_index": baseline_index,
                "onset_overlap": onset_overlap,
                "baseline_overlap": baseline_overlap,
                "difference": difference,
            }
        )

    differences = np.asarray(
        onset_differences,
        dtype=np.float64,
    )

    if len(differences) == 0:
        raise RuntimeError(
            "No valid attack-onset pairs found."
        )

    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------

    statistic, p_value = wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
    )

    # --------------------------------------------------------
    # Cohen's dz
    # --------------------------------------------------------

    mean_difference = float(
        np.mean(differences)
    )

    std_difference = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if std_difference == 0:
        cohens_dz = None
    else:
        cohens_dz = float(
            mean_difference
            / std_difference
        )

    # --------------------------------------------------------
    # Bootstrap confidence interval for mean difference
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    bootstrap_means = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64,
    )

    for i in range(BOOTSTRAP_SAMPLES):
        sample = rng.choice(
            differences,
            size=len(differences),
            replace=True,
        )

        bootstrap_means[i] = np.mean(sample)

    ci_low, ci_high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    # --------------------------------------------------------
    # Direction counts
    # --------------------------------------------------------

    positive = int(
        np.sum(differences > 0)
    )

    negative = int(
        np.sum(differences < 0)
    )

    zero = int(
        np.sum(differences == 0)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        "experiment": (
            "attack_onset_static_overlap_significance"
        ),
        "n_attack_onsets": int(
            len(differences)
        ),
        "comparison": (
            "onset_static_overlap "
            "minus immediately_preceding_normal_transition_overlap"
        ),
        "mean_difference": mean_difference,
        "median_difference": float(
            np.median(differences)
        ),
        "std_difference": std_difference,
        "wilcoxon_signed_rank": {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "alternative": "two-sided",
        },
        "cohens_dz": cohens_dz,
        "bootstrap_mean_difference": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
            "ci_95_percentile": [
                float(ci_low),
                float(ci_high),
            ],
        },
        "direction_counts": {
            "positive": positive,
            "negative": negative,
            "zero": zero,
        },
        "paired_differences": [
            float(x)
            for x in differences
        ],
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
        "Attack-onset static-overlap significance"
    )
    print(
        f"Events: {len(differences)}"
    )
    print(
        f"Mean difference: "
        f"{mean_difference:.6f}"
    )
    print(
        f"Median difference: "
        f"{np.median(differences):.6f}"
    )
    print(
        f"Wilcoxon statistic: "
        f"{statistic:.6f}"
    )
    print(
        f"Wilcoxon p-value: "
        f"{p_value:.6g}"
    )
    print(
        f"Cohen's dz: "
        f"{cohens_dz:.6f}"
    )
    print(
        "Bootstrap 95% CI: "
        f"[{ci_low:.6f}, {ci_high:.6f}]"
    )
    print(
        f"Positive: {positive}"
    )
    print(
        f"Negative: {negative}"
    )
    print(
        f"Zero: {zero}"
    )
    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()