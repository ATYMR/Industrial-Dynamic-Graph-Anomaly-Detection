import torch

from swat_gnn.models.dynamic_graph_autoencoder import (
    DynamicGraphAutoencoder,
)
from swat_gnn.models.static_temporal_graph_autoencoder import (
    StaticTemporalGraphAutoencoder,
)


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def test_parameter_count_matches_dynamic_model():

    dynamic = DynamicGraphAutoencoder(
        sequence_length=60,
        num_nodes=66,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        k=5,
        dropout=0.0,
    )

    static = StaticTemporalGraphAutoencoder(
        sequence_length=60,
        num_nodes=66,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        dropout=0.0,
    )

    assert (
        count_parameters(dynamic)
        == count_parameters(static)
    )


def test_static_model_forward_shape():

    model = StaticTemporalGraphAutoencoder(
        sequence_length=60,
        num_nodes=66,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        dropout=0.0,
    )

    # Same 222-undirected-edge / 444-directed-edge
    # static graph size used in the existing experiment.
    edge_index = torch.randint(
        0,
        66,
        (2, 444),
    )

    model.set_graph(edge_index)

    x = torch.randn(
        4,
        60,
        66,
    )

    reconstruction = model(x)

    assert reconstruction.shape == (
        4,
        66,
    )


def test_static_graph_is_fixed():

    model = StaticTemporalGraphAutoencoder(
        sequence_length=60,
        num_nodes=66,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )

    model.set_graph(edge_index)

    graph_a = model.get_graph().clone()
    graph_b = model.get_graph().clone()

    assert torch.equal(
        graph_a,
        graph_b,
    )


def test_static_model_rejects_missing_graph():

    model = StaticTemporalGraphAutoencoder(
        sequence_length=60,
        num_nodes=66,
    )

    x = torch.randn(
        2,
        60,
        66,
    )

    try:
        model(x)
    except RuntimeError as exc:
        assert "initialized" in str(exc)
    else:
        raise AssertionError(
            "Model should reject forward pass "
            "before graph initialization."
        )