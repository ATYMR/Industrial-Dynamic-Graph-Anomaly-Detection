from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_swat_static_temporal_graph as static_exp
import train_swat_dynamic_graph as dynamic_exp
import train_swat_random_static_temporal_graph as random_exp
from swat_gnn.data.swat_temporal_dataset import build_swat_temporal_datasets

SEED = 42
BATCH_SIZE = 256
GRAPH_K = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables" / "swat_seed42_score_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def load_checkpoint(path):
    return torch.load(path, map_location=DEVICE, weights_only=False)


def score_static_cosine(test_datasets):
    checkpoint = load_checkpoint(
        CHECKPOINT_DIR / "static_temporal_graph_autoencoder_best.pt"
    )

    edge_index = checkpoint["edge_index"].to(DEVICE)

    model = static_exp.StaticTemporalGraphAutoencoder(
        edge_index=edge_index,
        num_nodes=len(checkpoint["feature_names"]),
        temporal_hidden=static_exp.TEMPORAL_HIDDEN,
        embedding_dim=static_exp.EMBEDDING_DIM,
        gat_hidden=static_exp.GAT_HIDDEN,
        gat_heads=static_exp.GAT_HEADS,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_scores = []

    with torch.no_grad():
        for dataset in test_datasets:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            episode_scores = []

            for batch in loader:
                batch = batch.to(DEVICE, non_blocking=True)

                reconstruction = model(batch)
                target = batch[:, -1, :]

                scores = torch.mean(
                    (reconstruction - target) ** 2,
                    dim=1,
                )

                episode_scores.append(scores.cpu().numpy())

            all_scores.append(np.concatenate(episode_scores))

    return np.concatenate(all_scores)


def score_dynamic(test_datasets):
    checkpoint = load_checkpoint(
        CHECKPOINT_DIR / f"swat_dynamic_graph_autoencoder_k5_seed{SEED}_best.pt"
    )

    model = dynamic_exp.DynamicGraphAutoencoder(
        sequence_length=dynamic_exp.SEQUENCE_LENGTH,
        num_nodes=len(checkpoint["feature_names"]),
        temporal_hidden_dim=dynamic_exp.TEMPORAL_HIDDEN_DIM,
        embedding_dim=dynamic_exp.EMBEDDING_DIM,
        gat_hidden_dim=dynamic_exp.GAT_HIDDEN_DIM,
        gat_heads=dynamic_exp.GAT_HEADS,
        k=GRAPH_K,
        dropout=0.0,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_scores = []

    with torch.no_grad():
        for dataset in test_datasets:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            episode_scores = []

            for batch in loader:
                batch = batch.to(DEVICE, non_blocking=True)

                reconstruction = model(batch)
                target = batch[:, -1, :]

                scores = torch.mean(
                    (reconstruction - target) ** 2,
                    dim=1,
                )

                episode_scores.append(scores.cpu().numpy())

            all_scores.append(np.concatenate(episode_scores))

    return np.concatenate(all_scores)


def score_random_static(test_datasets):
    checkpoint = load_checkpoint(
        CHECKPOINT_DIR / f"swat_random_static_temporal_graph_autoencoder_seed{SEED}_best.pt"
    )

    edge_index = checkpoint["edge_index"].to(DEVICE)

    model = random_exp.StaticTemporalGraphAutoencoder(
        edge_index=edge_index,
        num_nodes=len(checkpoint["feature_names"]),
        temporal_hidden=random_exp.TEMPORAL_HIDDEN,
        embedding_dim=random_exp.EMBEDDING_DIM,
        gat_hidden=random_exp.GAT_HIDDEN,
        gat_heads=random_exp.GAT_HEADS,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_scores = []

    with torch.no_grad():
        for dataset in test_datasets:
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )

            episode_scores = []

            for batch in loader:
                batch = batch.to(DEVICE, non_blocking=True)

                reconstruction = model(batch)
                target = batch[:, -1, :]

                scores = torch.mean(
                    (reconstruction - target) ** 2,
                    dim=1,
                )

                episode_scores.append(scores.cpu().numpy())

            all_scores.append(np.concatenate(episode_scores))

    return np.concatenate(all_scores)


print("\nBuilding SWaT temporal datasets...")

dataset_bundle = build_swat_temporal_datasets(
    root="data/raw/SWaT",
    sequence_length=60,
    train_stride=5,
    evaluation_stride=1,
)

feature_names = dataset_bundle["feature_names"]
train_dataset = dataset_bundle["train_dataset"]
validation_dataset = dataset_bundle["validation_dataset"]
test_datasets = dataset_bundle["test_datasets"]
test_labels = dataset_bundle["test_labels"]

labels = np.concatenate(test_labels).astype(np.int64)

print(f"Features: {len(feature_names)}")
print(f"Windows: {len(labels)}")
print(f"Attack windows: {labels.sum()}")
print(f"Normal windows: {(labels == 0).sum()}")

print("\nScoring static cosine...")
static_scores = score_static_cosine(test_datasets)

print("\nScoring dynamic k5...")
dynamic_scores = score_dynamic(test_datasets)

print("\nScoring random static...")
random_scores = score_random_static(test_datasets)

assert len(static_scores) == len(labels)
assert len(dynamic_scores) == len(labels)
assert len(random_scores) == len(labels)


def summarize(name, scores):
    normal = scores[labels == 0]
    attack = scores[labels == 1]

    print(f"\n{name}")
    print(f"min: {scores.min()}")
    print(f"max: {scores.max()}")
    print(f"mean: {scores.mean()}")
    print(f"median: {np.median(scores)}")
    print(f"p95: {np.percentile(scores, 95)}")
    print(f"p99: {np.percentile(scores, 99)}")
    print(f"attack_mean: {attack.mean()}")
    print(f"normal_mean: {normal.mean()}")
    print(f"attack/normal_mean_ratio: {attack.mean() / normal.mean()}")


print("\n=== SEED 42 SCORE SUMMARY ===")

summarize("static_cosine", static_scores)
summarize("dynamic_k5", dynamic_scores)
summarize("random_static", random_scores)

np.save(OUTPUT_DIR / "labels.npy", labels)
np.save(OUTPUT_DIR / "static_cosine_scores.npy", static_scores)
np.save(OUTPUT_DIR / "dynamic_k5_scores.npy", dynamic_scores)
np.save(OUTPUT_DIR / "random_static_scores.npy", random_scores)

print(f"\nSaved to: {OUTPUT_DIR}")
