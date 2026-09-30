from __future__ import annotations

import pytest
import torch

from swat_gnn.models.gcn_autoencoder import (
    GCNAutoencoder,
    GCNAutoencoderConfig,
    get_device,
    set_random_seed,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def graph_data() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Small deterministic graph for unit tests.

    6 nodes with a ring-like undirected structure.
    Each node has one scalar feature.
    """

    x = torch.tensor(
        [
            [0.1],
            [0.2],
            [0.3],
            [0.4],
            [0.5],
            [0.6],
        ],
        dtype=torch.float32,
    )

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
        ],
        dtype=torch.long,
    )

    return x, edge_index


@pytest.fixture
def model() -> GCNAutoencoder:
    return GCNAutoencoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        output_dim=1,
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------

def test_model_construction() -> None:
    model = GCNAutoencoder(
        input_dim=1,
        hidden_dim=16,
        latent_dim=8,
        output_dim=1,
    )

    assert model.input_dim == 1
    assert model.hidden_dim == 16
    assert model.latent_dim == 8
    assert model.output_dim == 1

    assert model.conv1 is not None
    assert model.conv2 is not None
    assert model.conv3 is not None
    assert model.output_layer is not None


def test_model_has_trainable_parameters(
    model: GCNAutoencoder,
) -> None:
    parameters = list(model.parameters())

    assert len(parameters) > 0
    assert all(parameter.requires_grad for parameter in parameters)


# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

def test_forward_output_shape(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    reconstruction = model(x, edge_index)

    assert reconstruction.shape == x.shape


def test_forward_output_is_finite(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    reconstruction = model(x, edge_index)

    assert torch.isfinite(reconstruction).all()


# ---------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------

def test_encoder_output_shape(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    latent = model.encode(x, edge_index)

    assert latent.shape == (
        x.shape[0],
        model.latent_dim,
    )


def test_encoder_output_is_finite(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    latent = model.encode(x, edge_index)

    assert torch.isfinite(latent).all()


# ---------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------

def test_decoder_output_shape(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    latent = model.encode(x, edge_index)

    reconstruction = model.decode(
        latent,
        edge_index,
    )

    assert reconstruction.shape == x.shape


def test_decoder_output_is_finite(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    latent = model.encode(x, edge_index)

    reconstruction = model.decode(
        latent,
        edge_index,
    )

    assert torch.isfinite(reconstruction).all()


# ---------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------

def test_reconstruction_error_shape(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    errors = model.reconstruction_error(
        x,
        edge_index,
    )

    assert errors.shape == (x.shape[0],)


def test_reconstruction_error_is_finite(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    errors = model.reconstruction_error(
        x,
        edge_index,
    )

    assert torch.isfinite(errors).all()


def test_reconstruction_error_is_nonnegative(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    errors = model.reconstruction_error(
        x,
        edge_index,
    )

    assert torch.all(errors >= 0)


# ---------------------------------------------------------------------
# Graph-level anomaly score
# ---------------------------------------------------------------------

def test_anomaly_score_is_scalar(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    score = model.anomaly_score(
        x,
        edge_index,
    )

    assert score.ndim == 0


def test_anomaly_score_is_finite(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    score = model.anomaly_score(
        x,
        edge_index,
    )

    assert torch.isfinite(score)


def test_anomaly_score_is_nonnegative(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    score = model.anomaly_score(
        x,
        edge_index,
    )

    assert score >= 0


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def test_invalid_input_rank_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    invalid_x = torch.randn(6)

    with pytest.raises(ValueError):
        model(invalid_x, edge_index)


def test_invalid_input_dimension_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    invalid_x = torch.randn(6, 2)

    with pytest.raises(ValueError):
        model(invalid_x, edge_index)


def test_nan_input_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    x = torch.randn(6, 1)
    x[0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(x, edge_index)


def test_inf_input_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    x = torch.randn(6, 1)
    x[0, 0] = float("inf")

    with pytest.raises(ValueError):
        model(x, edge_index)


def test_negative_edge_index_raises(
    model: GCNAutoencoder,
) -> None:
    x = torch.randn(6, 1)

    edge_index = torch.tensor(
        [
            [0, -1],
            [1, 2],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(x, edge_index)


def test_wrong_edge_index_shape_raises(
    model: GCNAutoencoder,
) -> None:
    x = torch.randn(6, 1)

    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(x, edge_index)


def test_non_integer_edge_index_raises(
    model: GCNAutoencoder,
) -> None:
    x = torch.randn(6, 1)

    edge_index = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(TypeError):
        model(x, edge_index)


def test_out_of_range_edge_index_raises(
    model: GCNAutoencoder,
) -> None:
    x = torch.randn(6, 1)

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 6],
        ],
        dtype=torch.long,
    )

    with pytest.raises(ValueError):
        model(x, edge_index)


# ---------------------------------------------------------------------
# Latent validation
# ---------------------------------------------------------------------

def test_invalid_latent_dimension_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    invalid_latent = torch.randn(
        6,
        model.latent_dim + 1,
    )

    with pytest.raises(ValueError):
        model.decode(
            invalid_latent,
            edge_index,
        )


def test_invalid_latent_rank_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    invalid_latent = torch.randn(
        model.latent_dim,
    )

    with pytest.raises(ValueError):
        model.decode(
            invalid_latent,
            edge_index,
        )


def test_nan_latent_raises(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    _, edge_index = graph_data

    latent = torch.randn(
        6,
        model.latent_dim,
    )

    latent[0, 0] = float("nan")

    with pytest.raises(ValueError):
        model.decode(
            latent,
            edge_index,
        )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

def test_default_config() -> None:
    config = GCNAutoencoderConfig()

    assert config.input_dim == 1
    assert config.hidden_dim == 64
    assert config.latent_dim == 32
    assert config.output_dim == 1
    assert config.learning_rate == 1e-3
    assert config.batch_size == 256
    assert config.max_epochs == 30
    assert config.patience == 5
    assert config.random_state == 42
    assert config.device == "auto"


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_dim", 0),
        ("hidden_dim", 0),
        ("latent_dim", 0),
        ("output_dim", 0),
        ("learning_rate", 0),
        ("batch_size", 0),
        ("max_epochs", 0),
        ("patience", 0),
        ("random_state", -1),
    ],
)
def test_invalid_config_values_raise(
    field: str,
    value: int | float,
) -> None:
    with pytest.raises(ValueError):
        GCNAutoencoderConfig(
            **{field: value}
        )


def test_invalid_device_config_raises() -> None:
    with pytest.raises(ValueError):
        GCNAutoencoderConfig(
            device="invalid-device"
        )


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def test_cpu_device() -> None:
    device = get_device("cpu")

    assert device.type == "cpu"


def test_auto_device() -> None:
    device = get_device("auto")

    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_cuda_device() -> None:
    if torch.cuda.is_available():
        device = get_device("cuda")
        assert device.type == "cuda"
    else:
        with pytest.raises(RuntimeError):
            get_device("cuda")


# ---------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------

def test_random_seed_reproducibility() -> None:
    set_random_seed(42)

    first = torch.randn(10)

    set_random_seed(42)

    second = torch.randn(10)

    assert torch.equal(first, second)


# ---------------------------------------------------------------------
# Training sanity
# ---------------------------------------------------------------------

def test_single_training_step(
    model: GCNAutoencoder,
    graph_data: tuple[torch.Tensor, torch.Tensor],
) -> None:
    x, edge_index = graph_data

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    reconstruction = model(
        x,
        edge_index,
    )

    loss = torch.nn.functional.mse_loss(
        reconstruction,
        x,
    )

    assert torch.isfinite(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    parameters_with_gradients = [
        parameter
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert len(parameters_with_gradients) > 0