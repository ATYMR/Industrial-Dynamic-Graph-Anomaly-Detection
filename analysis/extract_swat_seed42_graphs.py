from pathlib import Path
import sys

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "swat_seed42_graph_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading SWaT datasets...")

bundle = build_swat_temporal_datasets(
    root="data/raw/SWaT",
    sequence_length=60,
    train_stride=5,
    evaluation_stride=1,
)

feature_names = bundle["feature_names"]
test_datasets = bundle["test_datasets"]
test_labels = bundle["test_labels"]

labels = np.concatenate(test_labels).astype(np.int64)

print(f"Features: {len(feature_names)}")
print(f"Test windows: {len(labels)}")
print(f"Attack windows: {labels.sum()}")

print("\nLoading dynamic graph model...")

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

all_edges = []

print("\nExtracting dynamic graphs...")

with torch.no_grad():
    for episode_id, dataset in enumerate(test_datasets):
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        episode_edges = []

        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)

            graphs = model.get_dynamic_graphs(batch)

            for graph in graphs:
                episode_edges.append(graph.cpu().numpy())

        all_edges.extend(episode_edges)

        print(
            f"Episode {episode_id + 1}: "
            f"{len(episode_edges)} graphs extracted"
        )

assert len(all_edges) == len(labels)

# Convert every graph to a canonical directed-edge set.
edge_sets = [
    set(map(tuple, edges.T.tolist()))
    for edges in all_edges
]

# Per-window graph statistics.
edge_counts = np.array([len(edges) for edges in edge_sets])

# Compare each graph with the graph immediately before it.
change_rates = np.zeros(len(edge_sets), dtype=np.float64)

for i in range(1, len(edge_sets)):
    previous = edge_sets[i - 1]
    current = edge_sets[i]

    intersection = len(previous & current)
    union = len(previous | current)

    change_rates[i] = 1.0 - (
        intersection / union if union else 1.0
    )

normal_change = change_rates[labels == 0]
attack_change = change_rates[labels == 1]

print("\n=== GRAPH CHANGE SUMMARY ===")

print("Edge count:")
print(f"  unique edge counts: {np.unique(edge_counts)}")
print(f"  expected edges: {len(feature_names) * GRAPH_K}")

print("\nConsecutive-window graph change:")
print("Normal:")
print(f"  mean:   {normal_change.mean():.6f}")
print(f"  median: {np.median(normal_change):.6f}")
print(f"  p95:    {np.percentile(normal_change, 95):.6f}")

print("Attack:")
print(f"  mean:   {attack_change.mean():.6f}")
print(f"  median: {np.median(attack_change):.6f}")
print(f"  p95:    {np.percentile(attack_change, 95):.6f}")

print(
    "\nAttack/normal mean change ratio:",
    attack_change.mean() / normal_change.mean()
)

# Compare each dynamic graph with the fixed cosine graph.
static_graph_path = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "swat_fixed_cosine_graph.pt"
)

static_graph = torch.load(
    static_graph_path,
    map_location="cpu",
    weights_only=False,
)

if isinstance(static_graph, dict):
    if "edge_index" in static_graph:
        static_graph = static_graph["edge_index"]

static_edges = set(
    map(tuple, static_graph.numpy().T.tolist())
)

overlap = np.zeros(len(edge_sets), dtype=np.float64)

for i, dynamic_edges in enumerate(edge_sets):
    overlap[i] = len(dynamic_edges & static_edges) / len(static_edges)

normal_overlap = overlap[labels == 0]
attack_overlap = overlap[labels == 1]

print("\n=== DYNAMIC / STATIC GRAPH OVERLAP ===")

print("Normal:")
print(f"  mean:   {normal_overlap.mean():.6f}")
print(f"  median: {np.median(normal_overlap):.6f}")
print(f"  p95:    {np.percentile(normal_overlap, 95):.6f}")

print("Attack:")
print(f"  mean:   {attack_overlap.mean():.6f}")
print(f"  median: {np.median(attack_overlap):.6f}")
print(f"  p95:    {np.percentile(attack_overlap, 95):.6f}")

print(
    "\nAttack-normal overlap difference:",
    attack_overlap.mean() - normal_overlap.mean()
)

# Save compact arrays for later statistical analysis.
np.save(OUTPUT_DIR / "labels.npy", labels)
np.save(OUTPUT_DIR / "graph_change_rates.npy", change_rates)
np.save(OUTPUT_DIR / "static_overlap.npy", overlap)

print(f"\nSaved analysis arrays to: {OUTPUT_DIR}")
print("\n=== DONE ===")
