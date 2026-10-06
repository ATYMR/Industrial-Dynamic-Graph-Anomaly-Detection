from pathlib import Path

p = Path("experiments/check_dynamic_static_edge_overlap.py")

p.write_text(r'''
import json
import torch
import numpy as np
from pathlib import Path

from swat_gnn.data.hai_loader import load_hai_dataset
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.models.dynamic_graph_autoencoder import DynamicGraphAutoencoder, DynamicGraphAutoencoderConfig

# Configuration
SEQ_LEN = 60
NUM_NODES = 66
GRAPH_K = 5
BATCH_SIZE = 256

# Load data
data = load_hai_dataset(
    root="data/raw/HAI_Kaggle/hai-23.05",
    training_files=["hai-train1.csv", "hai-train2.csv", "hai-train3.csv"],
    validation_files=["hai-train4.csv"],
    test_sensor_files=["hai-test1.csv", "hai-test2.csv"],
    test_label_files=["label-test1.csv", "label-test2.csv"],
)

# Keep the same 66-variable-feature pipeline used by the experiments.
training_segments = [
    df.drop(columns=[
        "P1_PIT01_HH", "P1_PP01AD", "P1_PP01AR", "P1_PP01BD",
        "P1_PP01BR", "P1_PP02D", "P1_PP02R", "P1_SOL01D",
        "P1_SOL03D", "P1_STSP", "P2_Emerg", "P2_OnOff",
        "P2_RTR", "P2_TripEx", "P2_VTR01", "P2_VTR02",
        "P2_VTR03", "P2_VTR04", "P3_LH01", "P3_LL01"
    ])
    for df in data["training"]
]

preprocessor = Preprocessor()
preprocessor.fit(training_segments)

test_segments = [
    df.drop(columns=[
        "P1_PIT01_HH", "P1_PP01AD", "P1_PP01AR", "P1_PP01BD",
        "P1_PP01BR", "P1_PP02D", "P1_PP02R", "P1_SOL01D",
        "P1_SOL03D", "P1_STSP", "P2_Emerg", "P2_OnOff",
        "P2_RTR", "P2_TripEx", "P2_VTR01", "P2_VTR02",
        "P2_VTR03", "P2_VTR04", "P3_LH01", "P3_LL01"
    ])
    for df in data["test"]
]

test_arrays = preprocessor.transform(test_segments)
test_dataset = TemporalWindowDataset(
    test_arrays,
    sequence_length=SEQ_LEN,
    stride=1,
)

# Static graph
static_graph = torch.load(
    "results/tables/fixed_cosine_graph.pt",
    weights_only=True,
)
static_edges = set(map(tuple, static_graph.t().cpu().tolist()))

# Dynamic model
config = DynamicGraphAutoencoderConfig(
    sequence_length=SEQ_LEN,
    num_nodes=NUM_NODES,
    temporal_hidden_dim=32,
    embedding_dim=32,
    gat_hidden_dim=64,
    gat_heads=4,
    graph_k=GRAPH_K,
    dropout=0.0,
)

model = DynamicGraphAutoencoder(config)
checkpoint = torch.load(
    "results/checkpoints/dynamic_graph_autoencoder_best.pt",
    map_location="cpu",
    weights_only=True,
)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)
model.eval()

# Compare dynamic edges against static edges.
overlaps = []
normal_overlaps = []
change_values = []

loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        embeddings = model.get_sensor_embeddings(batch)
        dynamic_graphs = model.get_dynamic_graphs(batch)

        for graph in dynamic_graphs:
            dynamic_edges = set(map(tuple, graph.t().cpu().tolist()))
            overlap = len(dynamic_edges & static_edges) / len(static_edges)
            overlaps.append(overlap)

        if batch_idx % 100 == 0:
            print(f"Processed {len(overlaps)} windows")

overlaps = np.asarray(overlaps)

result = {
    "n_windows": int(len(overlaps)),
    "static_edges": int(len(static_edges)),
    "dynamic_edges_per_window": int(NUM_NODES * GRAPH_K),
    "mean_overlap": float(overlaps.mean()),
    "median_overlap": float(np.median(overlaps)),
    "std_overlap": float(overlaps.std()),
    "p05_overlap": float(np.percentile(overlaps, 5)),
    "p95_overlap": float(np.percentile(overlaps, 95)),
    "min_overlap": float(overlaps.min()),
    "max_overlap": float(overlaps.max()),
}

print("\n=== STATIC vs DYNAMIC EDGE OVERLAP ===")
for k, v in result.items():
    print(f"{k}: {v}")

Path("results/tables/dynamic_static_edge_overlap.json").write_text(
    json.dumps(result, indent=2)
)
print("\nSaved: results/tables/dynamic_static_edge_overlap.json")
''', encoding="utf-8")

print("Script written:", p)
