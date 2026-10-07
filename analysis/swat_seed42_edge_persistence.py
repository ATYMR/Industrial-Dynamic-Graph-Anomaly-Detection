from pathlib import Path
import sys
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from swat_gnn.data.swat_temporal_dataset import build_swat_temporal_datasets
from swat_gnn.models.dynamic_graph_autoencoder import DynamicGraphAutoencoder

SEED = 42
BATCH_SIZE = 256
GRAPH_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / f"swat_dynamic_graph_autoencoder_k5_seed{SEED}_best.pt"
)

bundle = build_swat_temporal_datasets(
    root="data/raw/SWaT",
    sequence_length=60,
    train_stride=5,
    evaluation_stride=1,
)

feature_names = bundle["feature_names"]
test_datasets = bundle["test_datasets"]
labels = np.concatenate(bundle["test_labels"]).astype(np.int64)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model = DynamicGraphAutoencoder(
    sequence_length=60,
    num_nodes=len(feature_names),
    temporal_hidden_dim=32,
    embedding_dim=32,
    gat_hidden_dim=64,
    gat_heads=4,
    k=GRAPH_K,
    dropout=0.0,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

normal_counts = Counter()
attack_counts = Counter()

normal_windows = int((labels == 0).sum())
attack_windows = int((labels == 1).sum())

window_index = 0

print("=== EDGE PERSISTENCE ANALYSIS ===")

with torch.no_grad():
    for dataset in test_datasets:
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        for batch in loader:
            batch_size = len(batch)
            batch = batch.to(DEVICE, non_blocking=True)

            graphs = model.get_dynamic_graphs(batch)

            for local_idx, graph in enumerate(graphs):
                target_counter = (
                    attack_counts
                    if labels[window_index] == 1
                    else normal_counts
                )

                for source, target in graph.cpu().numpy().T:
                    target_counter[(int(source), int(target))] += 1

                window_index += 1

assert window_index == len(labels)

def summarize(counter, denominator, name):
    print(f"\n--- {name} ---")

    values = np.array(
        list(counter.values()),
        dtype=np.float64,
    )

    print(f"Unique edges observed: {len(counter)}")
    print(f"Possible directed edges: {40 * 39}")
    print(f"Edge coverage: {len(counter) / (40 * 39):.6f}")

    print("Persistence across windows:")
    print(f"  mean:   {values.mean() / denominator:.6f}")
    print(f"  median: {np.median(values) / denominator:.6f}")
    print(f"  p75:    {np.percentile(values, 75) / denominator:.6f}")
    print(f"  p90:    {np.percentile(values, 90) / denominator:.6f}")
    print(f"  p95:    {np.percentile(values, 95) / denominator:.6f}")
    print(f"  max:    {values.max() / denominator:.6f}")

    print("\nTop 20 edges:")

    for rank, ((source, target), count) in enumerate(
        counter.most_common(20),
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"{feature_names[source]} -> {feature_names[target]} "
            f"count={count} "
            f"persistence={count / denominator:.6f}"
        )

summarize(
    normal_counts,
    normal_windows,
    "NORMAL",
)

summarize(
    attack_counts,
    attack_windows,
    "ATTACK",
)

# Compare the most persistent edges between regimes.
normal_persistence = {
    edge: count / normal_windows
    for edge, count in normal_counts.items()
}

attack_persistence = {
    edge: count / attack_windows
    for edge, count in attack_counts.items()
}

all_edges = set(normal_persistence) | set(attack_persistence)

differences = []

for edge in all_edges:
    n = normal_persistence.get(edge, 0.0)
    a = attack_persistence.get(edge, 0.0)
    differences.append((abs(a - n), a - n, edge, n, a))

differences.sort(reverse=True)

print("\n=== LARGEST ATTACK-vs-NORMAL EDGE CHANGES ===")

for rank, (_, signed_diff, edge, normal_p, attack_p) in enumerate(
    differences[:20],
    start=1,
):
    source, target = edge

    print(
        f"{rank:2d}. "
        f"{feature_names[source]} -> {feature_names[target]} | "
        f"normal={normal_p:.6f} | "
        f"attack={attack_p:.6f} | "
        f"delta={signed_diff:+.6f}"
    )

print("\n=== DONE ===")
