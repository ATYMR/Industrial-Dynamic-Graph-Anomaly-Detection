from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder for multivariate
    industrial time-series anomaly detection.

    Input shape:
        (batch, sequence_length, input_dim)

    Output shape:
        (batch, sequence_length, input_dim)

    The encoder compresses the complete sequence into a latent
    representation. The decoder reconstructs the sequence from
    that representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.to_latent = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------
        self.from_latent = nn.Linear(
            latent_dim,
            hidden_dim,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            input_dim,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a sequence into a latent representation.

        Args:
            x:
                Tensor of shape
                (batch, sequence_length, input_dim).

        Returns:
            Tensor of shape
            (batch, latent_dim).
        """
        self._validate_input(x)

        _, (hidden, _) = self.encoder(x)

        # Hidden state from the final encoder layer.
        final_hidden = hidden[-1]

        latent = self.to_latent(final_hidden)

        return latent

    def decode(
        self,
        latent: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """
        Decode a latent representation into a sequence.

        Args:
            latent:
                Tensor of shape (batch, latent_dim).

            sequence_length:
                Number of timesteps to reconstruct.

        Returns:
            Tensor of shape
            (batch, sequence_length, input_dim).
        """
        if not isinstance(latent, torch.Tensor):
            raise TypeError(
                "latent must be a torch.Tensor."
            )

        if latent.ndim != 2:
            raise ValueError(
                "latent must have shape "
                "(batch, latent_dim)."
            )

        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {latent.shape[1]}."
            )

        if not torch.isfinite(latent).all():
            raise ValueError(
                "latent contains NaN or infinite values."
            )

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        # Map latent representation into decoder input space.
        decoder_input = self.from_latent(latent)

        # Repeat the latent representation across all timesteps.
        decoder_input = decoder_input.unsqueeze(1).repeat(
            1,
            sequence_length,
            1,
        )

        decoded, _ = self.decoder(
            decoder_input
        )

        reconstruction = self.output_layer(
            decoded
        )

        return reconstruction

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the complete input sequence.
        """
        self._validate_input(x)

        sequence_length = x.shape[1]

        latent = self.encode(x)

        reconstruction = self.decode(
            latent,
            sequence_length,
        )

        return reconstruction

    def reconstruction_error(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate reconstruction MSE for every timestep.

        Returns:
            Tensor of shape:

                (batch, sequence_length)

        Each value represents the mean squared reconstruction
        error across all features at that timestep.
        """
        self._validate_input(x)

        reconstruction = self.forward(x)

        error = (
            reconstruction - x
        ).pow(2).mean(dim=2)

        return error

    def final_timestep_error(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the anomaly score for the final timestep.

        This is the primary anomaly score for our causal
        temporal experiment.

        Returns:
            Tensor of shape:

                (batch,)
        """
        timestep_errors = self.reconstruction_error(x)

        return timestep_errors[:, -1]

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Validate model input.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "x must have shape "
                "(batch, sequence_length, input_dim)."
            )

        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.shape[2]}."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "x contains zero samples."
            )

        if x.shape[1] == 0:
            raise ValueError(
                "x contains zero timesteps."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "x contains NaN or infinite values."
            )


@dataclass
class LSTMAutoencoderConfig:
    """
    Configuration for the LSTM Autoencoder experiment.
    """

    input_dim: int = 66
    hidden_dim: int = 64
    latent_dim: int = 32
    num_layers: int = 1

    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 30
    patience: int = 5

    random_state: int = 42

    device: str = "auto"

    def validate(self) -> None:
        """
        Validate configuration values.
        """
        if self.input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if self.latent_dim <= 0:
            raise ValueError(
                "latent_dim must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be positive."
            )

        if self.patience <= 0:
            raise ValueError(
                "patience must be positive."
            )

        valid_devices = {
            "auto",
            "cpu",
            "cuda",
        }

        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of {valid_devices}."
            )


def get_device(
    device: str = "auto",
) -> torch.device:
    """
    Resolve the requested computation device.
    """
    if device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "device must be 'auto', 'cpu', or 'cuda'."
        )

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(
    seed: int,
) -> None:
    """
    Set PyTorch random seeds for reproducibility.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)