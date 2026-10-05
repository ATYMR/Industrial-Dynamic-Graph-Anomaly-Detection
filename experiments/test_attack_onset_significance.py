import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


INPUT_PATH = Path(
    "results/tables/dynamic_graph_attack_onset_paired_analysis_seed456.json"
)

OUTPUT_PATH = Path(
    "results/tables/dynamic_graph_attack_onset_statistical_test_seed456.json"
)

SEED = 456
N_BOOTSTRAP = 10000


# ---------------------------------------------------------
# Load paired attack-onset analysis
# ---------------------------------------------------------

with INPUT_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)


differences = np.asarray(
    [event["difference"] for event in data["paired_events"]],
    dtype=float,
)

if differences.size == 0:
    raise ValueError("No paired differences found.")


# ---------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------

wilcoxon_result = wilcoxon(
    differences,
    alternative="two-sided",
    method="auto",
)


# ---------------------------------------------------------
# Bootstrap 95% confidence interval for mean difference
# ---------------------------------------------------------

rng = np.random.default_rng(SEED)

bootstrap_means = np.empty(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    sample = rng.choice(
        differences,
        size=differences.size,
        replace=True,
    )

    bootstrap_means[i] = np.mean(sample)


ci_lower, ci_upper = np.percentile(
    bootstrap_means,
    [2.5, 97.5],
)


# ---------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------

mean_difference = float(np.mean(differences))

median_difference = float(np.median(differences))

std_difference = float(
    np.std(differences, ddof=1)
)


# ---------------------------------------------------------
# Cohen's dz
# ---------------------------------------------------------

if std_difference > 0:
    cohens_dz = mean_difference / std_difference
else:
    cohens_dz = float("nan")


# ---------------------------------------------------------
# Direction counts
# ---------------------------------------------------------

positive = int(np.sum(differences > 0))

negative = int(np.sum(differences < 0))

zero = int(np.sum(differences == 0))


# ---------------------------------------------------------
# Build result
# ---------------------------------------------------------

result = {
    "experiment": "dynamic_graph_attack_onset_statistical_test",

    "dataset": "HAI_23.05",

    "seed": SEED,

    "n_events": int(differences.size),

    "mean_paired_difference": mean_difference,

    "median_paired_difference": median_difference,

    "std_paired_difference": std_difference,

    "positive_differences": positive,

    "negative_differences": negative,

    "zero_differences": zero,

    "wilcoxon": {
        "statistic": float(wilcoxon_result.statistic),

        "p_value": float(wilcoxon_result.pvalue),

        "alternative": "two-sided",
    },

    "bootstrap_mean_difference_95ci": {
        "lower": float(ci_lower),

        "upper": float(ci_upper),

        "n_bootstrap": N_BOOTSTRAP,
    },

    "cohens_dz": float(cohens_dz),
}


# ---------------------------------------------------------
# Save result
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(
        result,
        f,
        indent=2,
    )


# ---------------------------------------------------------
# Print summary
# ---------------------------------------------------------

print(f"Events: {differences.size}")
print(f"Mean paired difference: {mean_difference:.6f}")
print(f"Median paired difference: {median_difference:.6f}")
print(f"Wilcoxon statistic: {wilcoxon_result.statistic:.6f}")
print(f"Wilcoxon p-value: {wilcoxon_result.pvalue:.6g}")
print(
    f"Bootstrap 95% CI: "
    f"[{ci_lower:.6f}, {ci_upper:.6f}]"
)
print(f"Cohen's dz: {cohens_dz:.6f}")
print(f"Positive differences: {positive}")
print(f"Negative differences: {negative}")
print(f"Zero differences: {zero}")
print(f"Saved: {OUTPUT_PATH}")
