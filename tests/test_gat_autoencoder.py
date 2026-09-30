from __future__ import annotations

import pytest
import torch

from swat_gnn.models.gat_autoencoder import (
    GATAutoencoder,
    GATAutoencoderConfig,
    get_device,
    set_random_seed,
)


@pytest.fixture
def graph():
    """Small undirected graph for testing."""

    # 6 nodes:
    #
    # 0 -- 1 -- 2
    # |    |
    # 3 -- 4 -- 5
    #
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 0, 3, 1, 4, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 0, 4, 1, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )

    return edge_index


@pytest.fixture
def model():
    return GATAutoencoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        heads=2,
        dropout=0.0,
    )


@pytest.fixture
def node_features():
    torch.manual_seed(42)

    return torch.randn(6, 1)


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_model_construction():
    model = GATAutoencoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        heads=2,
    )

    assert isinstance(model, GATAutoencoder)


def test_model_has_gat_layers(model):
    assert hasattr(model, "gat1")
    assert hasattr(model, "gat2")
    assert hasattr(model, "gat3")
    assert hasattr(model, "gat4")


def test_model_parameter_count_is_positive(model):
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count > 0


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------


def test_forward_shape(model, node_features, graph):
    reconstruction = model(
        node_features,
        graph,
    )

    assert reconstruction.shape == node_features.shape


def test_encode_shape(model, node_features, graph):
    latent = model.encode(
        node_features,
        graph,
    )

    assert latent.shape == (
        node_features.shape[0],
        model.latent_dim,
    )


def test_decode_shape(model, node_features, graph):
    latent = model.encode(
        node_features,
        graph,
    )

    reconstruction = model.decode(
        latent,
        graph,
    )

    assert reconstruction.shape == node_features.shape


def test_forward_output_is_finite(
    model,
    node_features,
    graph,
):
    reconstruction = model(
        node_features,
        graph,
    )

    assert torch.isfinite(reconstruction).all()


# ---------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------


def test_reconstruction_error_shape(
    model,
    node_features,
    graph,
):
    error = model.reconstruction_error(
        node_features,
        graph,
    )

    assert error.shape == (node_features.shape[0],)


def test_reconstruction_error_is_nonnegative(
    model,
    node_features,
    graph,
):
    error = model.reconstruction_error(
        node_features,
        graph,
    )

    assert torch.all(error >= 0)


def test_reconstruction_error_is_finite(
    model,
    node_features,
    graph,
):
    error = model.reconstruction_error(
        node_features,
        graph,
    )

    assert torch.isfinite(error).all()


# ---------------------------------------------------------------------
# Graph-level anomaly score
# ---------------------------------------------------------------------


def test_anomaly_score_is_scalar(
    model,
    node_features,
    graph,
):
    score = model.anomaly_score(
        node_features,
        graph,
    )

    assert score.ndim == 0


def test_anomaly_score_is_nonnegative(
    model,
    node_features,
    graph,
):
    score = model.anomaly_score(
        node_features,
        graph,
    )

    assert score >= 0


def test_anomaly_score_is_finite(
    model,
    node_features,
    graph,
):
    score = model.anomaly_score(
        node_features,
        graph,
    )

    assert torch.isfinite(score)


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------


def test_rejects_wrong_input_rank(
    model,
    graph,
):
    x = torch.randn(6)

    with pytest.raises(ValueError):
        model(x, graph)


def test_rejects_wrong_input_dimension(
    model,
    graph,
):
    x = torch.randn(6, 2)

    with pytest.raises(ValueError):
        model(x, graph)


def test_rejects_empty_input(
    model,
    graph,
):
    x = torch.empty(0, 1)

    with pytest.raises(ValueError):
        model(x, graph)


def test_rejects_nan_input(
    model,
    graph,
):
    x = torch.randn(6, 1)
    x[0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x, graph)


def test_rejects_infinite_input(
    model,
    graph,
):
    x = torch.randn(6, 1)
    x[0, 0] = float("inf")

    with pytest.raises(ValueError):
        model(x, graph)


def test_rejects_wrong_edge_index_rank(
    model,
    node_features,
):
    edge_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(node_features, edge_index)


def test_rejects_wrong_edge_index_shape(
    model,
    node_features,
):
    edge_index = torch.tensor(
        [
            [0, 1, 2],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(node_features, edge_index)


def test_rejects_non_long_edge_index(
    model,
    node_features,
):
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.int32,
    )

    with pytest.raises(TypeError):
        model(node_features, edge_index)


def test_rejects_negative_edge_index(
    model,
    node_features,
):
    edge_index = torch.tensor(
        [
            [-1, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(node_features, edge_index)


def test_rejects_out_of_range_edge_index(
    model,
    node_features,
):
    edge_index = torch.tensor(
        [
            [0, 6],
            [1, 0],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(node_features, edge_index)


def test_rejects_nan_latent(
    model,
    graph,
):
    latent = torch.randn(6, model.latent_dim)
    latent[0, 0] = float("nan")

    with pytest.raises(ValueError):
        model.decode(latent, graph)


def test_rejects_wrong_latent_dimension(
    model,
    graph,
):
    latent = torch.randn(
        6,
        model.latent_dim + 1,
    )

    with pytest.raises(ValueError):
        model.decode(latent, graph)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def test_default_config():
    config = GATAutoencoderConfig()

    assert config.input_dim == 1
    assert config.hidden_dim == 64
    assert config.latent_dim == 32
    assert config.heads == 4
    assert config.dropout == 0.0


def test_config_rejects_invalid_input_dim():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(input_dim=0)


def test_config_rejects_invalid_hidden_dim():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(hidden_dim=0)


def test_config_rejects_invalid_latent_dim():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(latent_dim=0)


def test_config_rejects_invalid_heads():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(heads=0)


def test_config_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(dropout=1.0)


def test_config_rejects_negative_dropout():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(dropout=-0.1)


def test_config_rejects_invalid_learning_rate():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(learning_rate=0)


def test_config_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(batch_size=0)


def test_config_rejects_invalid_device():
    with pytest.raises(ValueError):
        GATAutoencoderConfig(device="tpu")


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------


def test_cpu_device():
    device = get_device("cpu")

    assert device.type == "cpu"


def test_auto_device():
    device = get_device("auto")

    assert device.type in {"cpu", "cuda"}


def test_cuda_device_if_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    device = get_device("cuda")

    assert device.type == "cuda"


def test_cuda_device_rejected_if_unavailable(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError):
        get_device("cuda")


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------


def test_random_seed_reproducibility():
    set_random_seed(42)

    model1 = GATAutoencoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        heads=2,
    )

    parameters1 = [
        parameter.detach().clone()
        for parameter in model1.parameters()
    ]

    set_random_seed(42)

    model2 = GATAutoencoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        heads=2,
    )

    parameters2 = [
        parameter.detach().clone()
        for parameter in model2.parameters()
    ]

    for parameter1, parameter2 in zip(
        parameters1,
        parameters2,
    ):
        assert torch.equal(
            parameter1,
            parameter2,
        )


# ---------------------------------------------------------------------
# Small training sanity check
# ---------------------------------------------------------------------


def test_model_can_train(
    model,
    node_features,
    graph,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    initial_loss = None
    final_loss = None

    for step in range(5):
        optimizer.zero_grad()

        reconstruction = model(
            node_features,
            graph,
        )

        loss = (
            reconstruction - node_features
        ).pow(2).mean()

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss is not None
    assert torch.isfinite(
        torch.tensor(final_loss)
    )


# ---------------------------------------------------------------------
# Attention-specific checks
# ---------------------------------------------------------------------


def test_attention_heads_are_configured(model):
    assert model.gat1.heads == model.heads
    assert model.gat3.heads == model.heads


def test_attention_dropout_is_zero_by_default():
    model = GATAutoencoder()

    assert model.dropout == 0.0
    assert model.gat1.dropout == 0.0
    assert model.gat2.dropout == 0.0
    assert model.gat3.dropout == 0.0
    assert model.gat4.dropout == 0.0