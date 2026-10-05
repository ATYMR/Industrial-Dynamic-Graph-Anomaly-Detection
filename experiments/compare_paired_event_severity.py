import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

PATH = Path("results/tables/event_relative_severity.json")

with open(PATH) as f:
    data = json.load(f)

static = data["static"]["events"]
dynamic = data["dynamic"]["events"]

assert len(static) == 52
assert len(dynamic) == 52

static_median = np.array([x["median_relative_score"] for x in static])
dynamic_median = np.array([x["median_relative_score"] for x in dynamic])

static_p95 = np.array([
    x["max_relative_score"] for x in static
])

dynamic_p95 = np.array([
    x["max_relative_score"] for x in dynamic
])

# Note:
# The existing file does not contain event-level p95 scores;
# max_relative_score is therefore retained only as a separate
# extreme-response measure and is NOT called p95 here.

median_diff = dynamic_median - static_median

stat, p = wilcoxon(
    dynamic_median,
    static_median,
    alternative="two-sided",
)

print("PAIRED EVENT-LEVEL COMPARISON")
print("--------------------------------")
print(f"Events: {len(static)}")

print("\nMedian relative score")
print(f"Static mean:   {static_median.mean():.6f}")
print(f"Dynamic mean:  {dynamic_median.mean():.6f}")
print(f"Static median: {np.median(static_median):.6f}")
print(f"Dynamic median:{np.median(dynamic_median):.6f}")
print(f"Mean paired difference (D-S): {median_diff.mean():.6f}")
print(f"Median paired difference:      {np.median(median_diff):.6f}")
print(f"Wilcoxon statistic: {stat:.6f}")
print(f"Wilcoxon p-value:  {p:.6f}")

dynamic_higher = np.sum(dynamic_median > static_median)
static_higher = np.sum(static_median > dynamic_median)
equal = np.sum(dynamic_median == static_median)

print("\nPairwise direction")
print(f"Dynamic higher: {dynamic_higher}")
print(f"Static higher:  {static_higher}")
print(f"Equal:          {equal}")

print("\nExtreme response (maximum relative score)")
print(f"Static median of event maxima:  {np.median(static_p95):.6f}")
print(f"Dynamic median of event maxima: {np.median(dynamic_p95):.6f}")

result = {
    "n_events": 52,
    "median_relative_score": {
        "static_mean": float(static_median.mean()),
        "dynamic_mean": float(dynamic_median.mean()),
        "static_median": float(np.median(static_median)),
        "dynamic_median": float(np.median(dynamic_median)),
        "mean_paired_difference_dynamic_minus_static": float(median_diff.mean()),
        "median_paired_difference_dynamic_minus_static": float(np.median(median_diff)),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p": float(p),
        "dynamic_higher": int(dynamic_higher),
        "static_higher": int(static_higher),
        "equal": int(equal),
    },
    "event_max_relative_score": {
        "static_median": float(np.median(static_p95)),
        "dynamic_median": float(np.median(dynamic_p95)),
    },
}

output = Path("results/tables/paired_event_severity_comparison.json")

with open(output, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: {output}")
