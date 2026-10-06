import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from swat_gnn.data.loader import load_config, load_test_data, load_training_data
from swat_gnn.data.feature_analysis import analyze_features
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.models.dynamic_graph_autoencoder import DynamicGraphAutoencoder


SEQUENCE_LENGTH = 60
BATCH_SIZE = 256
GRAPH_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = Path("results/checkpoints/dynamic_graph_autoencoder_best.pt")
STATIC_GRAPH_PATH = Path("results/tables/fixed_cosine_graph.pt")
OUTPUT_PATH = Path("results/tables/dynamic_static_edge_overlap.json")


def edge_set(edge_index):
    return set(zip(edge_index[0].tolist(), edge_index[1].tolist()))


def main():
    config = load_config("configs/data.yaml")

    training_segments = load_training_data(config)
    test_segments, test_labels = load_test_data(config)

    analysis = analyze_features(training_segments)
    variable_features = analysis["variable_features"]

    training_selected = [
        segment[["timestamp"] + variable_features].copy()
        for segment in training_segments
    ]
    test_selected = [
        segment[["timestamp"] + variable_features].copy()
        for segment in test_segments
    ]

    preprocessor = Preprocessor()
    preprocessor.fit(training_selected)
    test_arrays = preprocessor.transform(test_selected)

    test_dataset = TemporalWindowDataset(
        test_arrays,
        sequence_length=SEQUENCE_LENGTH,
        stride=1,
    )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    checkpoint_config = checkpoint["config"]

    if checkpoint_config["graph_k"] != GRAPH_K:
        raise ValueError(
            f"Expected graph_k={GRAPH_K}, "
            f"checkpoint has graph_k={checkpoint_config['graph_k']}"
        )

    model = DynamicGraphAutoencoder(
        sequence_length=checkpoint_config["sequence_length"],
        num_nodes=checkpoint_config["num_nodes"],
        temporal_hidden_dim=checkpoint_config["temporal_hidden_dim"],
        embedding_dim=checkpoint_config["embedding_dim"],
        gat_hidden_dim=checkpoint_config["gat_hidden_dim"],
        gat_heads=checkpoint_config["gat_heads"],
        k=GRAPH_K,
        dropout=0.0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    static_graph = torch.load(
        STATIC_GRAPH_PATH,
        map_location="cpu",
        weights_only=True,
    )

    static_edges = edge_set(static_graph)
    expected_edges = checkpoint_config["num_nodes"] * GRAPH_K

    if len(static_edges) != expected_edges:
        raise ValueError(
            f"Static graph has {len(static_edges)} unique edges; "
            f"expected {expected_edges}"
        )

    overlaps = []
    labels = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            dynamic_graphs = model.get_dynamic_graphs(batch)

            for graph in dynamic_graphs:
                dynamic_edges = edge_set(graph.cpu())
                overlaps.append(
                    len(dynamic_edges & static_edges) / expected_edges
                )

            for local_index in range(batch.shape[0]):
                segment_id, timestep = test_dataset.get_target_timestep(
                    offset + local_index
                )
                labels.append(
                    int(test_labels[segment_id].iloc[timestep])
                )

            offset += batch.shape[0]

    overlaps = np.asarray(overlaps, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)

    if len(overlaps) != len(labels):
        raise RuntimeError(
            f"Overlap count {len(overlaps)} != label count {len(labels)}"
        )

    def summarize(values):
        return {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    normal = overlaps[labels == 0]
    attack = overlaps[labels == 1]

    result = {
        "experiment": "dynamic_static_edge_overlap",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_seed": checkpoint_config.get("seed"),
        "graph_k": GRAPH_K,
        "num_nodes": checkpoint_config["num_nodes"],
        "edges_per_graph": expected_edges,
        "test_windows": int(len(overlaps)),
        "attack_windows": int(np.sum(labels == 1)),
        "overall": summarize(overlaps),
        "normal": summarize(normal),
        "attack": summarize(attack),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))

    print("Dynamic vs static edge overlap")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint seed: {checkpoint_config.get('seed')}")
    print(f"Windows: {len(overlaps)}")
    print(f"Overall mean overlap: {np.mean(overlaps):.6f}")
    print(f"Overall median overlap: {np.median(overlaps):.6f}")
    print(f"Normal mean overlap: {np.mean(normal):.6f}")
    print(f"Attack mean overlap: {np.mean(attack):.6f}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
