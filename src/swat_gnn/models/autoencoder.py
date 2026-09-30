from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class FeedForwardAutoencoder(nn.Module):
    """
    Feed-forward autoencoder for multivariate industrial
    sensor anomaly detection.

    Architecture:

        input_dim
            ↓
        hidden_dim_1
            ↓
        hidden_dim_2
            ↓
        latent_dim
            ↓
        hidden_dim_2
            ↓
        hidden_dim_1
            ↓
        input_dim

    The model is trained to reconstruct normal observations.

    During anomaly detection, reconstruction error is used as
    the anomaly score.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 32,
        hidden_dim_2: int = 16,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()

        if not isinstance(input_dim, int):
            raise TypeError(
                "input_dim must be an integer."
            )

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be greater than zero."
            )

        dimensions = {
            "hidden_dim_1": hidden_dim_1,
            "hidden_dim_2": hidden_dim_2,
            "latent_dim": latent_dim,
        }

        for name, value in dimensions.items():

            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer."
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.latent_dim = latent_dim

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim_1,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim_1,
                hidden_dim_2,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim_2,
                latent_dim,
            ),
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim_2,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim_2,
                hidden_dim_1,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim_1,
                input_dim,
            ),
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode input into the latent representation.
        """

        self._validate_input(x)

        return self.encoder(x)

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode a latent representation.
        """

        if not isinstance(
            latent,
            torch.Tensor,
        ):
            raise TypeError(
                "latent must be a torch.Tensor."
            )

        if latent.ndim != 2:
            raise ValueError(
                "latent must have shape "
                "(batch_size, latent_dim)."
            )

        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension "
                f"{self.latent_dim}, got "
                f"{latent.shape[1]}."
            )

        return self.decoder(latent)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the input.
        """

        latent = self.encode(x)

        reconstruction = self.decode(
            latent
        )

        return reconstruction

    def reconstruction_error(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate per-sample mean squared reconstruction error.

        Returns
        -------
        torch.Tensor
            Shape:

                (batch_size,)

            Each value is the anomaly score for one observation.
        """

        self._validate_input(x)

        reconstruction = self.forward(x)

        error = torch.mean(
            (x - reconstruction) ** 2,
            dim=1,
        )

        return error

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        """
        Validate model input.
        """

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                "x must have shape "
                "(batch_size, input_dim)."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, "
                f"got {x.shape[1]}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "x contains NaN or infinite values."
            )


@dataclass
class AutoencoderConfig:
    """
    Configuration for the feed-forward autoencoder.
    """

    input_dim: int = 66
    hidden_dim_1: int = 32
    hidden_dim_2: int = 16
    latent_dim: int = 8

    learning_rate: float = 1e-3
    batch_size: int = 1024

    max_epochs: int = 50
    patience: int = 7

    random_state: int = 42

    device: str = "auto"

    def validate(self) -> None:
        """
        Validate training configuration.
        """

        dimensions = {
            "input_dim": self.input_dim,
            "hidden_dim_1": self.hidden_dim_1,
            "hidden_dim_2": self.hidden_dim_2,
            "latent_dim": self.latent_dim,
        }

        for name, value in dimensions.items():

            if not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer."
                )

            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than zero."
                )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs must be greater than zero."
            )

        if self.patience <= 0:
            raise ValueError(
                "patience must be greater than zero."
            )

        if self.random_state < 0:
            raise ValueError(
                "random_state must be non-negative."
            )

        valid_devices = {
            "auto",
            "cpu",
            "cuda",
        }

        if self.device not in valid_devices:
            raise ValueError(
                f"device must be one of "
                f"{sorted(valid_devices)}."
            )


def get_device(
    requested_device: str = "auto",
) -> torch.device:
    """
    Resolve the requested PyTorch device.
    """

    if requested_device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "requested_device must be "
            "'auto', 'cpu', or 'cuda'."
        )

    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducible model initialization
    and training.
    """

    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)