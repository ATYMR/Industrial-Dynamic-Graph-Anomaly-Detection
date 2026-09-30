from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swat_gnn.models.lstm_autoencoder import (
    LSTMAutoencoder,
    LSTMAutoencoderConfig,
    get_device,
    set_random_seed,
)


@pytest.fixture
def model():
    return LSTMAutoencoder(
        input_dim=66,
        hidden_dim=32,
        latent_dim=16,
    )


@pytest.fixture
def sample_input():
    return torch.randn(
        4,
        60,
        66,
    )


def test_model_construction(model):
    assert model.input_dim == 66
    assert model.hidden_dim == 32
    assert model.latent_dim == 16


def test_forward_output_shape(model, sample_input):
    output = model(sample_input)

    assert output.shape == sample_input.shape


def test_encoder_output_shape(model, sample_input):
    latent = model.encode(sample_input)

    assert latent.shape == (4, 16)


def test_decoder_output_shape(model, sample_input):
    latent = model.encode(sample_input)

    output = model.decode(
        latent,
        sequence_length=60,
    )

    assert output.shape == sample_input.shape


def test_reconstruction_error_shape(model, sample_input):
    error = model.reconstruction_error(sample_input)

    assert error.shape == (4, 60)


def test_final_timestep_error_shape(model, sample_input):
    error = model.final_timestep_error(sample_input)

    assert error.shape == (4,)


def test_reconstruction_error_is_finite(
    model,
    sample_input,
):
    error = model.reconstruction_error(sample_input)

    assert torch.isfinite(error).all()


def test_reconstruction_error_is_nonnegative(
    model,
    sample_input,
):
    error = model.reconstruction_error(sample_input)

    assert torch.all(error >= 0)


def test_invalid_input_dimension(model):
    bad_input = torch.randn(4, 60, 65)

    with pytest.raises(ValueError):
        model(bad_input)


def test_invalid_input_rank(model):
    bad_input = torch.randn(4, 66)

    with pytest.raises(ValueError):
        model(bad_input)


def test_nan_input(model):
    bad_input = torch.randn(4, 60, 66)
    bad_input[0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        model(bad_input)


def test_infinite_input(model):
    bad_input = torch.randn(4, 60, 66)
    bad_input[0, 0, 0] = float("inf")

    with pytest.raises(ValueError):
        model(bad_input)


def test_invalid_latent_dimension(model):
    latent = torch.randn(4, 15)

    with pytest.raises(ValueError):
        model.decode(
            latent,
            sequence_length=60,
        )


def test_invalid_latent_rank(model):
    latent = torch.randn(4, 16, 1)

    with pytest.raises(ValueError):
        model.decode(
            latent,
            sequence_length=60,
        )


def test_invalid_sequence_length(model, sample_input):
    latent = model.encode(sample_input)

    with pytest.raises(ValueError):
        model.decode(
            latent,
            sequence_length=0,
        )


def test_default_config():
    config = LSTMAutoencoderConfig()

    config.validate()

    assert config.input_dim == 66
    assert config.hidden_dim == 64
    assert config.latent_dim == 32
    assert config.batch_size == 256


def test_invalid_learning_rate():
    config = LSTMAutoencoderConfig(
        learning_rate=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_batch_size():
    config = LSTMAutoencoderConfig(
        batch_size=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_epoch_count():
    config = LSTMAutoencoderConfig(
        max_epochs=0
    )

    with pytest.raises(ValueError):
        config.validate()


def test_cpu_device():
    device = get_device("cpu")

    assert device.type == "cpu"


def test_auto_device():
    device = get_device("auto")

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_random_seed_reproducibility():
    set_random_seed(42)

    model_a = LSTMAutoencoder(
        input_dim=66,
        hidden_dim=32,
        latent_dim=16,
    )

    set_random_seed(42)

    model_b = LSTMAutoencoder(
        input_dim=66,
        hidden_dim=32,
        latent_dim=16,
    )

    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
    ):
        assert torch.equal(
            parameter_a,
            parameter_b,
        )


def test_model_can_train_on_small_batch(
    model,
    sample_input,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    model.train()

    initial_output = model(sample_input)

    initial_loss = torch.mean(
        (initial_output - sample_input) ** 2
    )

    optimizer.zero_grad()
    initial_loss.backward()
    optimizer.step()

    final_output = model(sample_input)

    final_loss = torch.mean(
        (final_output - sample_input) ** 2
    )

    assert torch.isfinite(final_loss)