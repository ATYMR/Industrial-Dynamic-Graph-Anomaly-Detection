import numpy as np
import torch

from swat_gnn.graph.fixed_cosine_graph import (
    build_fixed_cosine_graph,
)


def test_fixed_cosine_graph_shape():
    rng = np.random.default_rng(42)

    windows = rng.normal(
        size=(20, 60, 66)
    ).astype(np.float32)

    edge_index = build_fixed_cosine_graph(
        [windows],
        k=5,
    )

    assert edge_index.shape == (2, 330)


def test_fixed_cosine_graph_has_five_outgoing_edges_per_node():
    rng = np.random.default_rng(42)

    windows = rng.normal(
        size=(20, 60, 66)
    ).astype(np.float32)

    edge_index = build_fixed_cosine_graph(
        [windows],
        k=5,
    )

    sources = edge_index[0]

    counts = torch.bincount(
        sources,
        minlength=66,
    )

    assert torch.all(counts == 5)


def test_fixed_cosine_graph_has_no_self_loops():
    rng = np.random.default_rng(42)

    windows = rng.normal(
        size=(20, 60, 66)
    ).astype(np.float32)

    edge_index = build_fixed_cosine_graph(
        [windows],
        k=5,
    )

    assert not torch.any(
        edge_index[0] == edge_index[1]
    )


def test_fixed_cosine_graph_is_deterministic():
    rng = np.random.default_rng(42)

    windows = rng.normal(
        size=(20, 60, 66)
    ).astype(np.float32)

    graph_1 = build_fixed_cosine_graph(
        [windows],
        k=5,
    )

    graph_2 = build_fixed_cosine_graph(
        [windows],
        k=5,
    )

    assert torch.equal(graph_1, graph_2)


def test_invalid_k_is_rejected():
    rng = np.random.default_rng(42)

    windows = rng.normal(
        size=(5, 60, 66)
    ).astype(np.float32)

    try:
        build_fixed_cosine_graph(
            [windows],
            k=66,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for k >= num_nodes"
        )