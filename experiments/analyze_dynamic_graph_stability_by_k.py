from pathlib import Path
import json
import numpy as np
import torch

from swat_gnn.models.dynamic_graph_autoencoder import DynamicGraphAutoencoder
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.data.preprocessing import Preprocessor


SEQ_LEN = 60
TEST_STRIDE = 1
NUM_NODES = 66
BATCH_SIZE = 256

GRAPH_CONFIGS = {
    3: Path("results/checkpoints/dynamic_graph_autoencoder_k3_best.pt"),
    5: Path("results/checkpoints/dynamic_graph_autoencoder_best.pt"),
    10: Path("results/checkpoints/dynamic_graph_autoencoder_k10_best.pt"),
}

DATA_ROOT = Path("data/raw/HAI_Kaggle/hai-23.05")

TRAIN_FILES = [
    DATA_ROOT / "hai-train1.csv",
    DATA_ROOT / "hai-train2.csv",
    DATA_ROOT / "hai-train3.csv",
]

TEST_FILES = [
    DATA_ROOT / "hai-test1.csv",
    DATA_ROOT / "hai-test2.csv",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph_to_edge_set(edge_index):
    edge_index = edge_index.detach().cpu().numpy()
    return set(
        (int(src), int(dst))
        for src, dst in zip(edge_index[0], edge_index[1])
    )


def jaccard_change(graph_a, graph_b):
    union = graph_a | graph_b

    if not union:
        return 0.0

    return 1.0 - len(graph_a & graph_b) / len(union)


def load_segments():
    import pandas as pd

    train_dfs = [pd.read_csv(path) for path in TRAIN_FILES]
    test_dfs = [pd.read_csv(path) for path in TEST_FILES]

    # Fit preprocessing on the original training DataFrames only.
    preprocessor = Preprocessor()
    train_segments = preprocessor.fit_transform(train_dfs)
    test_segments = preprocessor.transform(test_dfs)

    # Reproduce the established 66-variable feature selection.
    train_concat = np.concatenate(train_segments, axis=0)
    std = train_concat.std(axis=0)
    variable_mask = std != 0

    train_segments = [x[:, variable_mask] for x in train_segments]
    test_segments = [x[:, variable_mask] for x in test_segments]

    return train_segments, test_segments

def extract_graphs(model, dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    previous_graph = None
    changes = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            batch_graphs = model.get_dynamic_graphs(batch)

            for graph in batch_graphs:
                current_graph = graph_to_edge_set(graph)

                if previous_graph is not None:
                    changes.append(
                        jaccard_change(previous_graph, current_graph)
                    )

                previous_graph = current_graph

    return np.asarray(changes, dtype=np.float32)

def analyze_k(k, checkpoint_path, test_segments):
    print(f"\n{'=' * 60}")
    print(f"k = {k}")
    print(f"checkpoint = {checkpoint_path}")
    print(f"{'=' * 60}")

    model = DynamicGraphAutoencoder(
        sequence_length=SEQ_LEN,
        num_nodes=NUM_NODES,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        k=k,
        dropout=0.0,
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"checkpoint epoch = {checkpoint.get('epoch', 'unknown')}")

    changes = []
    transitions = {
        "normal_to_normal": [],
        "normal_to_attack": [],
        "attack_to_attack": [],
        "attack_to_normal": [],
    }

    # Load labels only to reproduce the existing transition categories.
    import pandas as pd

    label_files = [
        DATA_ROOT / "label-test1.csv",
        DATA_ROOT / "label-test2.csv",
    ]

    global_change_count = 0

    for segment_id, segment in enumerate(test_segments, 1):
        dataset = TemporalWindowDataset(
            [segment],
            sequence_length=SEQ_LEN,
            stride=TEST_STRIDE,
        )

        print(
            f"test segment {segment_id}: "
            f"{len(dataset)} windows"
        )

        segment_changes = extract_graphs(model, dataset)

        labels = pd.read_csv(label_files[segment_id - 1])
        labels = labels.iloc[:, -1].to_numpy().astype(int)

        # Endpoint labels align with the last timestep of each window.
        window_labels = labels[SEQ_LEN - 1:]

        # One fewer transition than windows.
        assert len(segment_changes) == len(window_labels) - 1

        for i, change in enumerate(segment_changes):
            previous_label = int(window_labels[i])
            current_label = int(window_labels[i + 1])

            if previous_label == 0 and current_label == 0:
                key = "normal_to_normal"
            elif previous_label == 0 and current_label == 1:
                key = "normal_to_attack"
            elif previous_label == 1 and current_label == 1:
                key = "attack_to_attack"
            else:
                key = "attack_to_normal"

            changes.append(float(change))
            transitions[key].append(float(change))
            global_change_count += 1


    changes = np.asarray(changes)

    summary = {
        "k": k,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "total_windows": int(sum(segment_lengths)),
        "valid_transitions": int(global_change_count),
        "mean_graph_change": float(np.mean(changes)),
        "median_graph_change": float(np.median(changes)),
        "std_graph_change": float(np.std(changes)),
        "p95_graph_change": float(np.percentile(changes, 95)),
        "transitions": {},
    }

    for key, values in transitions.items():
        values = np.asarray(values)

        summary["transitions"][key] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p95": float(np.percentile(values, 95)),
        }

    print(f"mean change   = {summary['mean_graph_change']:.6f}")
    print(f"median change = {summary['median_graph_change']:.6f}")
    print(f"p95 change    = {summary['p95_graph_change']:.6f}")

    for key, values in summary["transitions"].items():
        print(
            f"{key:20s} "
            f"n={values['count']:6d} "
            f"mean={values['mean']:.6f} "
            f"median={values['median']:.6f}"
        )

    return summary


train_segments, test_segments = load_segments()

print(f"Device: {DEVICE}")
print(f"Train segments: {[x.shape for x in train_segments]}")
print(f"Test segments: {[x.shape for x in test_segments]}")

results = {}

for k, checkpoint in GRAPH_CONFIGS.items():
    results[str(k)] = analyze_k(
        k,
        checkpoint,
        test_segments,
    )

output = Path(
    "results/tables/dynamic_graph_stability_by_k.json"
)

with open(output, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: {output}")




