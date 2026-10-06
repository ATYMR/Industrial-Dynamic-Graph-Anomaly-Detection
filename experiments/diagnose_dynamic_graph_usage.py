from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
    load_test_data,
)
from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.temporal_dataset import TemporalWindowDataset
from swat_gnn.models.dynamic_graph_autoencoder import (
    DynamicGraphAutoencoder,
)


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 60
BATCH_SIZE = 256
GRAPH_K = 5

TEMPORAL_HIDDEN = 32
EMBEDDING_DIM = 32
GAT_HIDDEN = 64
GAT_HEADS = 4

SEED = 456

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "results/checkpoints/"
    "dynamic_graph_autoencoder_best.pt"
)

RESULTS_PATH = Path(
    "results/tables/"
    "dynamic_graph_usage_diagnostic.json"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Feature selection
# ============================================================

def get_variable_features(training_segments):

    feature_names = [
        column
        for column in training_segments[0].columns
        if column != "timestamp"
    ]

    training_values = np.concatenate(
        [
            df[feature_names].to_numpy(dtype=np.float64)
            for df in training_segments
        ],
        axis=0,
    )

    feature_std = training_values.std(axis=0)

    variable_features = [
        feature_names[i]
        for i, keep in enumerate(feature_std != 0)
        if keep
    ]

    return variable_features


def select_features(
    segments,
    feature_names,
):

    return [
        df[["timestamp"] + feature_names].copy()
        for df in segments
    ]


# ============================================================
# Graph utilities
# ============================================================

def make_random_graph(
    num_nodes: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a directed random k-nearest-style graph.

    Every source node has exactly k outgoing edges.
    Self-loops are excluded.
    """

    sources = []
    targets = []

    for source in range(num_nodes):

        candidates = [
            node
            for node in range(num_nodes)
            if node != source
        ]

        selected = random.sample(
            candidates,
            k,
        )

        for target in selected:
            sources.append(source)
            targets.append(target)

    return torch.tensor(
        [sources, targets],
        dtype=torch.long,
        device=device,
    )


def make_random_graph_like(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Preserve exactly the same number of directed edges
    as the supplied graph.

    Each source retains exactly the same out-degree.
    """

    device = edge_index.device

    sources = edge_index[0].detach().cpu().numpy()

    new_sources = []
    new_targets = []

    for source in range(num_nodes):

        source_count = int(
            np.sum(sources == source)
        )

        candidates = [
            node
            for node in range(num_nodes)
            if node != source
        ]

        selected = random.sample(
            candidates,
            source_count,
        )

        for target in selected:
            new_sources.append(source)
            new_targets.append(target)

    return torch.tensor(
        [new_sources, new_targets],
        dtype=torch.long,
        device=device,
    )


# ============================================================
# Fixed graph loader
# ============================================================

def load_static_graph(
    path: Path,
    num_nodes: int,
    k: int,
) -> torch.Tensor:

    edge_index = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if edge_index.ndim != 2:
        raise ValueError(
            "Static graph must have shape [2, E]."
        )

    if edge_index.shape[0] != 2:
        raise ValueError(
            "Static graph must have shape [2, E]."
        )

    if int(edge_index.max()) >= num_nodes:
        raise ValueError(
            "Static graph contains invalid node indices."
        )

    expected_edges = num_nodes * k

    if edge_index.shape[1] != expected_edges:
        raise ValueError(
            f"Expected {expected_edges} edges, "
            f"got {edge_index.shape[1]}."
        )

    return edge_index


# ============================================================
# Forward with externally supplied graph
# ============================================================

def reconstruct_with_graph(
    model: DynamicGraphAutoencoder,
    batch: torch.Tensor,
    edge_indices: list[torch.Tensor],
) -> torch.Tensor:
    """
    Run the trained dynamic model while replacing its
    internally generated graph with externally supplied graphs.

    The temporal encoder and GAT weights remain unchanged.
    """

    embeddings = model.temporal_encoder(batch)

    hidden = model._apply_gat_batched(
        model.gat_encoder_1,
        embeddings,
        edge_indices,
        activation=True,
    )

    latent = model._apply_gat_batched(
        model.gat_encoder_2,
        hidden,
        edge_indices,
        activation=False,
    )

    hidden = model._apply_gat_batched(
        model.gat_decoder_1,
        latent,
        edge_indices,
        activation=True,
    )

    reconstruction = model._apply_gat_batched(
        model.gat_decoder_2,
        hidden,
        edge_indices,
        activation=False,
    )

    return reconstruction.squeeze(-1)


# ============================================================
# Score one graph condition
# ============================================================

@torch.no_grad()
def score_condition(
    model: DynamicGraphAutoencoder,
    dataset: TemporalWindowDataset,
    condition: str,
    static_graph: torch.Tensor | None,
    device: torch.device,
    random_seed: int,
) -> np.ndarray:

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_scores = []

    random_generator = random.Random(
        random_seed
    )

    model.eval()

    for batch_idx, batch in enumerate(loader):

        batch = batch.to(
            device,
            non_blocking=True,
        )

        batch_size = batch.shape[0]

        if condition == "dynamic":

            embeddings = model.temporal_encoder(
                batch
            )

            edge_indices = (
                model.graph_builder(
                    embeddings
                )
            )

        elif condition == "static":

            edge_indices = [
                static_graph
            ] * batch_size

        elif condition == "random":

            edge_indices = []

            for _ in range(batch_size):

                sources = []
                targets = []

                for source in range(
                    model.num_nodes
                ):

                    candidates = [
                        node
                        for node in range(
                            model.num_nodes
                        )
                        if node != source
                    ]

                    selected = random_generator.sample(
                        candidates,
                        model.k,
                    )

                    for target in selected:

                        sources.append(source)
                        targets.append(target)

                edge_indices.append(
                    torch.tensor(
                        [sources, targets],
                        dtype=torch.long,
                        device=device,
                    )
                )

        else:

            raise ValueError(
                f"Unknown condition: {condition}"
            )

        reconstruction = reconstruct_with_graph(
            model=model,
            batch=batch,
            edge_indices=edge_indices,
        )

        target = batch[:, -1, :]

        scores = torch.mean(
            (reconstruction - target) ** 2,
            dim=1,
        )

        all_scores.append(
            scores.detach().cpu().numpy()
        )

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"  {condition}: "
                f"{batch_idx + 1:,}/"
                f"{len(loader):,}"
            )

    return np.concatenate(
        all_scores,
        axis=0,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("DYNAMIC GRAPH USAGE DIAGNOSTIC")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # Load data
    # ========================================================

    print("\nLoading HAI data...")

    config = load_config()

    training_segments_df = (
        load_training_data(config)
    )

    validation_segments_df = (
        load_validation_data(config)
    )

    test_segments_df, _ = (
        load_test_data(config)
    )

    variable_features = (
        get_variable_features(
            training_segments_df
        )
    )

    print(
        f"Variable features: "
        f"{len(variable_features)}"
    )

    training_segments_df = (
        select_features(
            training_segments_df,
            variable_features,
        )
    )

    validation_segments_df = (
        select_features(
            validation_segments_df,
            variable_features,
        )
    )

    test_segments_df = (
        select_features(
            test_segments_df,
            variable_features,
        )
    )

    # ========================================================
    # Train-only normalization
    # ========================================================

    preprocessor = Preprocessor()

    training_scaled = (
        preprocessor.fit_transform(
            training_segments_df
        )
    )

    validation_scaled = (
        preprocessor.transform(
            validation_segments_df
        )
    )

    test_scaled = (
        preprocessor.transform(
            test_segments_df
        )
    )

    # ========================================================
    # Validation + test datasets
    # ========================================================

    validation_dataset = TemporalWindowDataset(
        validation_scaled,
        sequence_length=SEQUENCE_LENGTH,
        stride=1,
    )

    test_dataset = TemporalWindowDataset(
        test_scaled,
        sequence_length=SEQUENCE_LENGTH,
        stride=1,
    )

    print(
        f"Validation windows: "
        f"{len(validation_dataset):,}"
    )

    print(
        f"Test windows: "
        f"{len(test_dataset):,}"
    )

    # ========================================================
    # Load model
    # ========================================================

    print("\nLoading dynamic checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    checkpoint_config = checkpoint["config"]

    print(
        f"Checkpoint config: "
        f"{checkpoint_config}"
    )

    model = DynamicGraphAutoencoder(
        sequence_length=checkpoint_config[
            "sequence_length"
        ],
        num_nodes=checkpoint_config[
            "num_nodes"
        ],
        temporal_hidden_dim=checkpoint_config[
            "temporal_hidden_dim"
        ],
        embedding_dim=checkpoint_config[
            "embedding_dim"
        ],
        gat_hidden_dim=checkpoint_config[
            "gat_hidden_dim"
        ],
        gat_heads=checkpoint_config[
            "gat_heads"
        ],
        k=checkpoint_config[
            "graph_k"
        ],
        dropout=0.0,
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    # ========================================================
    # Load static graph
    # ========================================================

    static_graph = load_static_graph(
        path=Path(
            "results/tables/"
            "fixed_cosine_graph.pt"
        ),
        num_nodes=checkpoint_config[
            "num_nodes"
        ],
        k=checkpoint_config[
            "graph_k"
        ],
    ).to(DEVICE)

    print(
        f"Static graph edges: "
        f"{static_graph.shape[1]}"
    )

    # ========================================================
    # Validation diagnostic
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION DIAGNOSTIC")
    print("=" * 70)

    print("\nDynamic graph...")

    validation_dynamic = score_condition(
        model,
        validation_dataset,
        "dynamic",
        static_graph,
        DEVICE,
        SEED,
    )

    print("\nStatic graph...")

    validation_static = score_condition(
        model,
        validation_dataset,
        "static",
        static_graph,
        DEVICE,
        SEED,
    )

    print("\nRandom graph...")

    validation_random = score_condition(
        model,
        validation_dataset,
        "random",
        static_graph,
        DEVICE,
        SEED,
    )

    # ========================================================
    # Test diagnostic
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST DIAGNOSTIC")
    print("=" * 70)

    print("\nDynamic graph...")

    test_dynamic = score_condition(
        model,
        test_dataset,
        "dynamic",
        static_graph,
        DEVICE,
        SEED,
    )

    print("\nStatic graph...")

    test_static = score_condition(
        model,
        test_dataset,
        "static",
        static_graph,
        DEVICE,
        SEED,
    )

    print("\nRandom graph...")

    test_random = score_condition(
        model,
        test_dataset,
        "random",
        static_graph,
        DEVICE,
        SEED,
    )

    # ========================================================
    # Summary
    # ========================================================

    def summarize(scores):

        return {
            "mean": float(
                np.mean(scores)
            ),
            "std": float(
                np.std(scores)
            ),
            "median": float(
                np.median(scores)
            ),
            "p95": float(
                np.percentile(scores, 95)
            ),
            "p99": float(
                np.percentile(scores, 99)
            ),
        }

    validation_summary = {
        "dynamic": summarize(
            validation_dynamic
        ),
        "static": summarize(
            validation_static
        ),
        "random": summarize(
            validation_random
        ),
    }

    test_summary = {
        "dynamic": summarize(
            test_dynamic
        ),
        "static": summarize(
            test_static
        ),
        "random": summarize(
            test_random
        ),
    }

    # ========================================================
    # Pairwise differences
    # ========================================================

    def compare(
        reference,
        alternative,
    ):

        difference = (
            alternative - reference
        )

        return {
            "mean_score_difference": float(
                np.mean(difference)
            ),
            "median_score_difference": float(
                np.median(difference)
            ),
            "mean_absolute_difference": float(
                np.mean(np.abs(difference))
            ),
            "pearson_correlation": float(
                np.corrcoef(
                    reference,
                    alternative,
                )[0, 1]
            ),
        }

    comparisons = {
        "validation": {
            "static_minus_dynamic":
                compare(
                    validation_dynamic,
                    validation_static,
                ),
            "random_minus_dynamic":
                compare(
                    validation_dynamic,
                    validation_random,
                ),
        },
        "test": {
            "static_minus_dynamic":
                compare(
                    test_dynamic,
                    test_static,
                ),
            "random_minus_dynamic":
                compare(
                    test_dynamic,
                    test_random,
                ),
        },
    }

    # ========================================================
    # Save
    # ========================================================

    output = {
        "experiment":
            "dynamic_graph_usage_diagnostic",

        "dataset":
            "HAI_23.05",

        "checkpoint":
            str(CHECKPOINT_PATH),

        "checkpoint_config":
            checkpoint_config,

        "validation_windows":
            len(validation_dataset),

        "test_windows":
            len(test_dataset),

        "conditions": [
            "dynamic",
            "static",
            "random",
        ],

        "validation_summary":
            validation_summary,

        "test_summary":
            test_summary,

        "comparisons":
            comparisons,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        import json

        json.dump(
            output,
            handle,
            indent=2,
        )

    print(
        "\nSaved:"
        f" {RESULTS_PATH}"
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()