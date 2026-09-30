from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNAutoencoder(nn.Module):
    """
    Static Graph Convolutional Network autoencoder.

    Each timestep is represented as a graph:
        nodes   = industrial sensor variables
        edges   = static sensor relationships
        feature = sensor value at that timestep

    Input:
        x         : [num_nodes, input_dim]
        edge_index: [2, num_edges]

    Output:
        reconstructed node features:
                   [num_nodes, output_dim]

    For the HAI experiment:
        num_nodes = 66
        input_dim = 1
        output_dim = 1

    The model reconstructs the sensor values at a single timestep.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.conv1 = GCNConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
        )

        self.conv2 = GCNConv(
            in_channels=hidden_dim,
            out_channels=latent_dim,
        )

        self.conv3 = GCNConv(
            in_channels=latent_dim,
            out_channels=hidden_dim,
        )

        self.output_layer = GCNConv(
            in_channels=hidden_dim,
            out_channels=output_dim,
        )

        self.activation = nn.ReLU()

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode node features into graph-aware latent representations.

        Parameters
        ----------
        x:
            Node features with shape [num_nodes, input_dim].

        edge_index:
            Graph connectivity with shape [2, num_edges].

        Returns
        -------
        torch.Tensor:
            Latent node representations with shape
            [num_nodes, latent_dim].
        """

        self._validate_input(x, edge_index)

        x = self.conv1(x, edge_index)
        x = self.activation(x)

        x = self.conv2(x, edge_index)

        return x

    # -----------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent graph representations into reconstructed
        sensor values.
        """

        self._validate_edge_index(edge_index)

        if not isinstance(z, torch.Tensor):
            raise TypeError("z must be a torch.Tensor.")

        if z.ndim != 2:
            raise ValueError(
                "z must be a 2D tensor with shape "
                "[num_nodes, latent_dim]."
            )

        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {z.shape[1]}."
            )

        if not torch.isfinite(z).all():
            raise ValueError(
                "z contains NaN or infinite values."
            )

        x = self.conv3(z, edge_index)
        x = self.activation(x)

        x = self.output_layer(x, edge_index)

        return x

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform graph-aware reconstruction.
        """

        latent = self.encode(
            x=x,
            edge_index=edge_index,
        )

        reconstruction = self.decode(
            z=latent,
            edge_index=edge_index,
        )

        return reconstruction

    # -----------------------------------------------------------------
    # Reconstruction error
    # -----------------------------------------------------------------

    def reconstruction_error(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate per-node reconstruction MSE.

        Returns:
            Tensor with shape [num_nodes].
        """

        reconstruction = self.forward(
            x=x,
            edge_index=edge_index,
        )

        error = (
            reconstruction - x
        ).pow(2).mean(dim=1)

        return error

    # -----------------------------------------------------------------
    # Graph-level anomaly score
    # -----------------------------------------------------------------

    def anomaly_score(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate one anomaly score for the entire graph/timestep.

        The score is the mean reconstruction error across all
        sensor nodes.
        """

        node_errors = self.reconstruction_error(
            x=x,
            edge_index=edge_index,
        )

        return node_errors.mean()

    # -----------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------

    def _validate_input(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")

        if x.ndim != 2:
            raise ValueError(
                "x must be a 2D tensor with shape "
                "[num_nodes, input_dim]."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "x contains zero nodes."
            )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {x.shape[1]}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "x contains NaN or infinite values."
            )

        self._validate_edge_index(edge_index)

        if edge_index.numel() > 0:
            max_node = int(edge_index.max().item())

            if max_node >= x.shape[0]:
                raise ValueError(
                    "edge_index references a node outside "
                    "the range of x."
                )

    @staticmethod
    def _validate_edge_index(
        edge_index: torch.Tensor,
    ) -> None:

        if not isinstance(edge_index, torch.Tensor):
            raise TypeError(
                "edge_index must be a torch.Tensor."
            )

        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must be a 2D tensor."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape "
                "[2, num_edges]."
            )

        if edge_index.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "edge_index must contain integer indices."
            )

        if edge_index.numel() > 0:
            if (edge_index < 0).any():
                raise ValueError(
                    "edge_index cannot contain negative indices."
                )


@dataclass
class GCNAutoencoderConfig:
    """
    Configuration for the static GCN autoencoder experiment.
    """

    input_dim: int = 1
    hidden_dim: int = 64
    latent_dim: int = 32
    output_dim: int = 1

    learning_rate: float = 1e-3
    batch_size: int = 256

    max_epochs: int = 30
    patience: int = 5

    random_state: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:

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

        if self.output_dim <= 0:
            raise ValueError(
                "output_dim must be positive."
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
                f"device must be one of {valid_devices}."
            )


def get_device(
    device: str = "auto",
) -> torch.device:
    """
    Resolve the requested PyTorch device.
    """

    if device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError(
            "device must be 'auto', 'cpu', or 'cuda'."
        )

    if device == "cpu":
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.
    """

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)