import json
import numpy as np
from scipy.stats import wilcoxon

with open("results/tables/event_relative_severity.json") as f:
    data = json.load(f)

static = np.array([e["median_relative_score"] for e in data["static"]["events"]])
dynamic = np.array([e["median_relative_score"] for e in data["dynamic"]["events"]])

# Log transform reduces the influence of extreme multiplicative differences.
log_static = np.log1p(static)
log_dynamic = np.log1p(dynamic)

diff = log_dynamic - log_static

stat, p = wilcoxon(diff)

rng = np.random.default_rng(42)
bootstrap_means = np.empty(10000)

for i in range(10000):
    sample = rng.choice(diff, size=len(diff), replace=True)
    bootstrap_means[i] = np.mean(sample)

ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

print("ROBUST PAIRED EVENT-SEVERITY ANALYSIS")
print("--------------------------------------")
print(f"Events: {len(diff)}")
print(f"Mean log1p severity - static:  {log_static.mean():.6f}")
print(f"Mean log1p severity - dynamic: {log_dynamic.mean():.6f}")
print(f"Mean paired log difference (D-S): {diff.mean():.6f}")
print(f"Median paired log difference:       {np.median(diff):.6f}")
print(f"Wilcoxon statistic: {stat:.6f}")
print(f"Wilcoxon p-value:  {p:.8f}")
print(f"Bootstrap 95% CI for mean difference: [{ci_low:.6f}, {ci_high:.6f}]")

print("\nDirection:")
print(f"Dynamic higher: {np.sum(diff > 0)}")
print(f"Static higher:  {np.sum(diff < 0)}")
print(f"Equal:          {np.sum(diff == 0)}")

result = {
    "n_events": len(diff),
    "transformation": "log1p",
    "static_mean_log1p": float(log_static.mean()),
    "dynamic_mean_log1p": float(log_dynamic.mean()),
    "mean_paired_difference_dynamic_minus_static": float(diff.mean()),
    "median_paired_difference": float(np.median(diff)),
    "wilcoxon_statistic": float(stat),
    "wilcoxon_p": float(p),
    "bootstrap_ci_95_mean_difference": [
        float(ci_low),
        float(ci_high)
    ],
    "dynamic_higher": int(np.sum(diff > 0)),
    "static_higher": int(np.sum(diff < 0)),
    "equal": int(np.sum(diff == 0))
}

with open(
    "results/tables/robust_paired_event_severity.json",
    "w"
) as f:
    json.dump(result, f, indent=2)

print("\nSaved: results/tables/robust_paired_event_severity.json")
