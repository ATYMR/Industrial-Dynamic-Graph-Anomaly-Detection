from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATAutoencoder(nn.Module):
    """
    Static Graph Attention Network autoencoder.

    Each timestep is represented as a graph where:
        - each sensor/feature is a node
        - each node has one scalar feature
        - edge_index defines the fixed sensor relationships

    Architecture:
        GATConv(1, hidden_dim, heads)
        GATConv(hidden_dim * heads, latent_dim, heads=1)
        GATConv(latent_dim, hidden_dim, heads)
        GATConv(hidden_dim * heads, 1, heads=1)

    The primary anomaly score is the mean reconstruction MSE
    across all nodes for a timestep.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        if heads <= 0:
            raise ValueError("heads must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dropout = dropout

        # Encoder
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=latent_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        # Decoder
        self.gat3 = GATConv(
            in_channels=latent_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.gat4 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=input_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features into graph-aware latent representations."""

        self._validate_input(x, edge_index)

        x = self.gat1(x, edge_index)
        x = torch.relu(x)

        x = self.gat2(x, edge_index)

        return x

    def decode(
        self,
        latent: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent graph representations into reconstructed features."""

        self._validate_edge_index(edge_index)

        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent must be a torch.Tensor.")

        if latent.ndim != 2:
            raise ValueError(
                "latent must have shape (num_nodes, latent_dim)."
            )

        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {latent.shape[1]}."
            )

        if latent.shape[0] == 0:
            raise ValueError("latent contains zero nodes.")

        if not torch.isfinite(latent).all():
            raise ValueError("latent contains NaN or infinite values.")

        x = self.gat3(latent, edge_index)
        x = torch.relu(x)

        x = self.gat4(x, edge_index)

        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct node features."""

        self._validate_input(x, edge_index)

        latent = self.encode(x, edge_index)

        reconstruction = self.decode(latent, edge_index)

        return reconstruction

    def reconstruction_error(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-node reconstruction MSE.

        Shape:
            (num_nodes,)
        """

        reconstruction = self.forward(x, edge_index)

        error = (reconstruction - x).pow(2).mean(dim=1)

        return error

    def anomaly_score(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return one graph-level anomaly score.

        The score is the mean reconstruction error
        across all nodes.
        """

        node_errors = self.reconstruction_error(x, edge_index)

        return node_errors.mean()

    def _validate_input(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")

        if x.ndim != 2:
            raise ValueError(
                "x must have shape (num_nodes, input_dim)."
            )

        if x.shape[0] == 0:
            raise ValueError("x contains zero nodes.")

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
                    "edge_index references a node outside x."
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
                "edge_index must have shape (2, num_edges)."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape (2, num_edges)."
            )

        if edge_index.dtype != torch.long:
            raise TypeError(
                "edge_index must have dtype torch.long."
            )

        if edge_index.numel() > 0 and edge_index.min().item() < 0:
            raise ValueError(
                "edge_index cannot contain negative node indices."
            )


@dataclass
class GATAutoencoderConfig:
    """Configuration for the GAT autoencoder."""

    input_dim: int = 1
    hidden_dim: int = 64
    latent_dim: int = 32
    heads: int = 4

    dropout: float = 0.0

    learning_rate: float = 1e-3
    batch_size: int = 256

    max_epochs: int = 30
    patience: int = 5

    random_state: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        if self.heads <= 0:
            raise ValueError("heads must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

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

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "device must be one of: auto, cpu, cuda."
            )


def get_device(device: str = "auto") -> torch.device:
    """Resolve the requested computation device."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(
            "device must be one of: auto, cpu, cuda."
        )

    if device == "cpu":
        return torch.device("cpu")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)