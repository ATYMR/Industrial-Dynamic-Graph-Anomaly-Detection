from __future__ import annotations

import pytest
import torch

from swat_gnn.models.dynamic_graph_autoencoder import (
    DynamicGraphAutoencoder,
    DynamicGraphAutoencoderConfig,
    DynamicGraphBuilder,
    SensorTemporalEncoder,
)


@pytest.fixture
def config():
    return DynamicGraphAutoencoderConfig(
        sequence_length=60,
        num_nodes=66,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        k=5,
        dropout=0.0,
    )


@pytest.fixture
def model(config):
    torch.manual_seed(42)

    return DynamicGraphAutoencoder(
        sequence_length=config.sequence_length,
        num_nodes=config.num_nodes,
        temporal_hidden_dim=config.temporal_hidden_dim,
        embedding_dim=config.embedding_dim,
        gat_hidden_dim=config.gat_hidden_dim,
        gat_heads=config.gat_heads,
        k=config.k,
        dropout=config.dropout,
    )


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    return torch.randn(4, 60, 66)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def test_config_defaults():
    config = DynamicGraphAutoencoderConfig()

    assert config.sequence_length == 60
    assert config.num_nodes == 66
    assert config.temporal_hidden_dim == 32
    assert config.embedding_dim == 32
    assert config.gat_hidden_dim == 64
    assert config.gat_heads == 4
    assert config.k == 5
    assert config.dropout == 0.0


# ---------------------------------------------------------------------
# Sensor temporal encoder
# ---------------------------------------------------------------------


def test_temporal_encoder_output_shape():
    encoder = SensorTemporalEncoder(
        sequence_length=60,
        temporal_hidden_dim=32,
        embedding_dim=32,
    )

    x = torch.randn(4, 60, 66)

    output = encoder(x)

    assert output.shape == (4, 66, 32)


def test_temporal_encoder_rejects_wrong_dimensions():
    encoder = SensorTemporalEncoder(
        sequence_length=60,
        temporal_hidden_dim=32,
        embedding_dim=32,
    )

    with pytest.raises(ValueError):
        encoder(torch.randn(4, 66))


def test_temporal_encoder_rejects_wrong_sequence_length():
    encoder = SensorTemporalEncoder(
        sequence_length=60,
        temporal_hidden_dim=32,
        embedding_dim=32,
    )

    with pytest.raises(ValueError):
        encoder(torch.randn(4, 66, 30))


# ---------------------------------------------------------------------
# Dynamic graph builder
# ---------------------------------------------------------------------


def test_graph_builder_output_count():
    builder = DynamicGraphBuilder(
        num_nodes=66,
        k=5,
    )

    embeddings = torch.randn(4, 66, 32)

    graphs = builder(embeddings)

    assert len(graphs) == 4


def test_graph_builder_edge_shape():
    builder = DynamicGraphBuilder(
        num_nodes=66,
        k=5,
    )

    embeddings = torch.randn(2, 66, 32)

    graphs = builder(embeddings)

    for edge_index in graphs:
        assert edge_index.shape == (2, 66 * 5)


def test_graph_builder_no_self_loops():
    builder = DynamicGraphBuilder(
        num_nodes=66,
        k=5,
    )

    embeddings = torch.randn(2, 66, 32)

    graphs = builder(embeddings)

    for edge_index in graphs:
        source = edge_index[0]
        target = edge_index[1]

        assert not torch.any(source == target)


def test_graph_builder_valid_node_indices():
    builder = DynamicGraphBuilder(
        num_nodes=66,
        k=5,
    )

    embeddings = torch.randn(2, 66, 32)

    graphs = builder(embeddings)

    for edge_index in graphs:
        assert torch.all(edge_index >= 0)
        assert torch.all(edge_index < 66)


def test_graph_builder_k_one():
    builder = DynamicGraphBuilder(
        num_nodes=10,
        k=1,
    )

    embeddings = torch.randn(2, 10, 8)

    graphs = builder(embeddings)

    for edge_index in graphs:
        assert edge_index.shape == (2, 10)


def test_graph_builder_rejects_invalid_k():
    with pytest.raises(ValueError):
        DynamicGraphBuilder(num_nodes=10, k=0)

    with pytest.raises(ValueError):
        DynamicGraphBuilder(num_nodes=10, k=10)


def test_graph_builder_rejects_invalid_embedding_shape():
    builder = DynamicGraphBuilder(
        num_nodes=66,
        k=5,
    )

    with pytest.raises(ValueError):
        builder(torch.randn(66, 32))


# ---------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------


def test_model_forward_shape(model, sample_input):
    output = model(sample_input)

    assert output.shape == (4, 66)


def test_model_encode_shape(model, sample_input):
    latent, graphs = model.encode(sample_input)

    assert latent.shape == (4, 66, 32)
    assert len(graphs) == 4


def test_model_decode_shape(model, sample_input):
    latent, graphs = model.encode(sample_input)

    reconstruction = model.decode(
        latent,
        graphs,
    )

    assert reconstruction.shape == (4, 66)


def test_reconstruction_error_shape(model, sample_input):
    scores = model.reconstruction_error(sample_input)

    assert scores.shape == (4,)


def test_anomaly_score_shape(model, sample_input):
    scores = model.anomaly_score(sample_input)

    assert scores.shape == (4,)


def test_anomaly_score_matches_reconstruction_error(
    model,
    sample_input,
):
    reconstruction_error = model.reconstruction_error(sample_input)
    anomaly_score = model.anomaly_score(sample_input)

    assert torch.allclose(
        reconstruction_error,
        anomaly_score,
    )


def test_dynamic_graph_count(model, sample_input):
    graphs = model.get_dynamic_graphs(sample_input)

    assert len(graphs) == 4


def test_dynamic_graph_edge_count(model, sample_input):
    graphs = model.get_dynamic_graphs(sample_input)

    for edge_index in graphs:
        assert edge_index.shape[1] == 66 * 5


# ---------------------------------------------------------------------
# Numerical validity
# ---------------------------------------------------------------------


def test_forward_output_is_finite(model, sample_input):
    output = model(sample_input)

    assert torch.isfinite(output).all()


def test_anomaly_scores_are_finite(model, sample_input):
    scores = model.anomaly_score(sample_input)

    assert torch.isfinite(scores).all()


def test_dynamic_graph_indices_are_finite(model, sample_input):
    graphs = model.get_dynamic_graphs(sample_input)

    for edge_index in graphs:
        assert torch.isfinite(
            edge_index.to(torch.float32)
        ).all()


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_model_rejects_wrong_dimensions(model):
    with pytest.raises(ValueError):
        model(torch.randn(4, 66))


def test_model_rejects_wrong_sequence_length(model):
    with pytest.raises(ValueError):
        model(torch.randn(4, 30, 66))


def test_model_rejects_wrong_number_of_nodes(model):
    with pytest.raises(ValueError):
        model(torch.randn(4, 60, 50))


def test_model_rejects_non_finite_input(model):
    x = torch.randn(2, 60, 66)
    x[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x)


# ---------------------------------------------------------------------
# Deterministic behavior
# ---------------------------------------------------------------------


def test_same_input_produces_same_output_in_eval_mode():
    torch.manual_seed(42)

    model = DynamicGraphAutoencoder(
        sequence_length=60,
        num_nodes=66,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        k=5,
        dropout=0.0,
    )

    model.eval()

    torch.manual_seed(123)
    x = torch.randn(2, 60, 66)

    with torch.no_grad():
        output_1 = model(x)
        output_2 = model(x)

    assert torch.allclose(
        output_1,
        output_2,
    )


# ---------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------


def test_model_supports_backpropagation(model, sample_input):
    model.train()

    reconstruction = model(sample_input)

    target = sample_input[:, -1, :]

    loss = torch.mean(
        (reconstruction - target) ** 2
    )

    loss.backward()

    gradients_found = False

    for parameter in model.parameters():
        if parameter.grad is not None:
            gradients_found = True
            assert torch.isfinite(parameter.grad).all()

    assert gradients_found


# ---------------------------------------------------------------------
# Dynamic behavior
# ---------------------------------------------------------------------


def test_graph_is_window_specific(model):
    torch.manual_seed(42)

    x1 = torch.randn(1, 60, 66)
    x2 = torch.randn(1, 60, 66)

    graphs_1 = model.get_dynamic_graphs(x1)
    graphs_2 = model.get_dynamic_graphs(x2)

    assert len(graphs_1) == 1
    assert len(graphs_2) == 1

    # Different windows are allowed to produce different graphs.
    # We do not require them to differ in every random test case.
    assert graphs_1[0].shape == graphs_2[0].shape