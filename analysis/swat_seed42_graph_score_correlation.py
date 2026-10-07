from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

SCORE_ROOT = Path("results/tables/swat_seed42_score_analysis")
GRAPH_ROOT = Path("results/tables/swat_seed42_graph_analysis")

labels = np.load(SCORE_ROOT / "labels.npy")
static_scores = np.load(SCORE_ROOT / "static_cosine_scores.npy")
dynamic_scores = np.load(SCORE_ROOT / "dynamic_k5_scores.npy")
random_scores = np.load(SCORE_ROOT / "random_static_scores.npy")

graph_change = np.load(GRAPH_ROOT / "graph_change_rates.npy")
static_overlap = np.load(GRAPH_ROOT / "static_overlap.npy")

assert len(labels) == len(graph_change)
assert len(labels) == len(dynamic_scores)

print("=== GRAPH CHANGE vs ANOMALY SCORE ===")

groups = {
    "ALL": np.ones(len(labels), dtype=bool),
    "NORMAL": labels == 0,
    "ATTACK": labels == 1,
}

for group_name, mask in groups.items():
    print(f"\n--- {group_name} ---")
    print(f"Windows: {mask.sum()}")

    x = graph_change[mask]
    y = dynamic_scores[mask]

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)

    print("Graph change vs dynamic anomaly score:")
    print(f"  Pearson r:   {pearson_r:.6f}")
    print(f"  Pearson p:   {pearson_p:.6e}")
    print(f"  Spearman rho: {spearman_r:.6f}")
    print(f"  Spearman p:   {spearman_p:.6e}")

    x_overlap = static_overlap[mask]

    pearson_r2, pearson_p2 = pearsonr(x_overlap, y)
    spearman_r2, spearman_p2 = spearmanr(x_overlap, y)

    print("Dynamic/static overlap vs dynamic anomaly score:")
    print(f"  Pearson r:   {pearson_r2:.6f}")
    print(f"  Pearson p:   {pearson_p2:.6e}")
    print(f"  Spearman rho: {spearman_r2:.6f}")
    print(f"  Spearman p:   {spearman_p2:.6e}")

# Compare score distributions at different graph-change levels.
print("\n=== SCORE BY GRAPH-CHANGE QUANTILES ===")

quantile_edges = np.percentile(
    graph_change,
    [0, 25, 50, 75, 90, 95, 99, 100],
)

for i in range(len(quantile_edges) - 1):
    low = quantile_edges[i]
    high = quantile_edges[i + 1]

    if i == len(quantile_edges) - 2:
        mask = (graph_change >= low) & (graph_change <= high)
    else:
        mask = (graph_change >= low) & (graph_change < high)

    if mask.sum() == 0:
        continue

    print(
        f"Q{i+1}: "
        f"change [{low:.6f}, {high:.6f}] | "
        f"n={mask.sum()} | "
        f"attack_fraction={labels[mask].mean():.6f} | "
        f"dynamic_score_median={np.median(dynamic_scores[mask]):.6f} | "
        f"dynamic_score_p90={np.percentile(dynamic_scores[mask], 90):.6f}"
    )

print("\n=== DONE ===")
