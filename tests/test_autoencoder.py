from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


# ============================================================
# Make src/ importable when running pytest from project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


from swat_gnn.models.autoencoder import (
    AutoencoderConfig,
    FeedForwardAutoencoder,
    get_device,
    set_random_seed,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def model() -> FeedForwardAutoencoder:
    return FeedForwardAutoencoder(
        input_dim=66,
        hidden_dim_1=32,
        hidden_dim_2=16,
        latent_dim=8,
    )


@pytest.fixture
def sample_input() -> torch.Tensor:
    set_random_seed(42)

    return torch.randn(
        16,
        66,
    )


# ============================================================
# Model construction
# ============================================================

def test_model_construction():

    model = FeedForwardAutoencoder(
        input_dim=66,
        hidden_dim_1=32,
        hidden_dim_2=16,
        latent_dim=8,
    )

    assert model.input_dim == 66
    assert model.hidden_dim_1 == 32
    assert model.hidden_dim_2 == 16
    assert model.latent_dim == 8


# ============================================================
# Forward pass
# ============================================================

def test_forward_output_shape(
    model,
    sample_input,
):

    reconstruction = model(
        sample_input
    )

    assert reconstruction.shape == (
        16,
        66,
    )


# ============================================================
# Encoder
# ============================================================

def test_encoder_output_shape(
    model,
    sample_input,
):

    latent = model.encode(
        sample_input
    )

    assert latent.shape == (
        16,
        8,
    )


# ============================================================
# Decoder
# ============================================================

def test_decoder_output_shape(
    model,
    sample_input,
):

    latent = model.encode(
        sample_input
    )

    reconstruction = model.decode(
        latent
    )

    assert reconstruction.shape == (
        16,
        66,
    )


# ============================================================
# Reconstruction error
# ============================================================

def test_reconstruction_error_shape(
    model,
    sample_input,
):

    error = model.reconstruction_error(
        sample_input
    )

    assert error.shape == (
        16,
    )


def test_reconstruction_error_is_finite(
    model,
    sample_input,
):

    error = model.reconstruction_error(
        sample_input
    )

    assert torch.isfinite(
        error
    ).all()


def test_reconstruction_error_is_nonnegative(
    model,
    sample_input,
):

    error = model.reconstruction_error(
        sample_input
    )

    assert torch.all(
        error >= 0
    )


# ============================================================
# Invalid input tests
# ============================================================

def test_invalid_input_dimension(
    model,
):

    invalid_input = torch.randn(
        16,
        65,
    )

    with pytest.raises(
        ValueError,
        match="Expected 66 features",
    ):
        model(
            invalid_input
        )


def test_invalid_input_rank(
    model,
):

    invalid_input = torch.randn(
        16,
        66,
        1,
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        model(
            invalid_input
        )


def test_nan_input(
    model,
):

    x = torch.randn(
        16,
        66,
    )

    x[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        model(
            x
        )


def test_infinite_input(
    model,
):

    x = torch.randn(
        16,
        66,
    )

    x[0, 0] = float("inf")

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        model(
            x
        )


# ============================================================
# Decoder validation
# ============================================================

def test_invalid_latent_dimension(
    model,
):

    invalid_latent = torch.randn(
        16,
        7,
    )

    with pytest.raises(
        ValueError,
        match="Expected latent dimension",
    ):
        model.decode(
            invalid_latent
        )


def test_invalid_latent_rank(
    model,
):

    invalid_latent = torch.randn(
        16,
        8,
        1,
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        model.decode(
            invalid_latent
        )


# ============================================================
# Configuration tests
# ============================================================

def test_default_config():

    config = AutoencoderConfig()

    config.validate()

    assert config.input_dim == 66
    assert config.hidden_dim_1 == 32
    assert config.hidden_dim_2 == 16
    assert config.latent_dim == 8
    assert config.learning_rate == 1e-3
    assert config.batch_size == 1024


def test_invalid_learning_rate():

    config = AutoencoderConfig(
        learning_rate=0.0
    )

    with pytest.raises(
        ValueError,
        match="learning_rate",
    ):
        config.validate()


def test_invalid_batch_size():

    config = AutoencoderConfig(
        batch_size=0
    )

    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        config.validate()


def test_invalid_epoch_count():

    config = AutoencoderConfig(
        max_epochs=0
    )

    with pytest.raises(
        ValueError,
        match="max_epochs",
    ):
        config.validate()


# ============================================================
# Device
# ============================================================

def test_cpu_device():

    device = get_device(
        "cpu"
    )

    assert device.type == "cpu"


def test_auto_device():

    device = get_device(
        "auto"
    )

    assert device.type in {
        "cpu",
        "cuda",
    }


# ============================================================
# Reproducibility
# ============================================================

def test_random_seed_reproducibility():

    set_random_seed(42)

    model_1 = FeedForwardAutoencoder(
        input_dim=66,
    )

    parameters_1 = [
        parameter.detach().clone()
        for parameter in model_1.parameters()
    ]

    set_random_seed(42)

    model_2 = FeedForwardAutoencoder(
        input_dim=66,
    )

    parameters_2 = [
        parameter.detach().clone()
        for parameter in model_2.parameters()
    ]

    for p1, p2 in zip(
        parameters_1,
        parameters_2,
    ):
        assert torch.equal(
            p1,
            p2,
        )


# ============================================================
# Training sanity check
# ============================================================

def test_model_can_train_on_small_batch():

    set_random_seed(42)

    model = FeedForwardAutoencoder(
        input_dim=66,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.MSELoss()

    x = torch.randn(
        32,
        66,
    )

    model.train()

    initial_loss = None

    for _ in range(10):

        optimizer.zero_grad()

        reconstruction = model(
            x
        )

        loss = criterion(
            reconstruction,
            x,
        )

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()

        optimizer.step()

    final_loss = loss.item()

    assert np.isfinite(
        final_loss
    )

    assert final_loss < initial_loss