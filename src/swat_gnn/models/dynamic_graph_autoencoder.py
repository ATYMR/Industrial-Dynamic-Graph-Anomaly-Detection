from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


@dataclass
class DynamicGraphAutoencoderConfig:
    """Configuration for the dynamic graph autoencoder."""

    sequence_length: int = 60
    num_nodes: int = 66

    temporal_hidden_dim: int = 32
    embedding_dim: int = 32

    gat_hidden_dim: int = 64
    gat_heads: int = 4

    k: int = 5

    dropout: float = 0.0


class SensorTemporalEncoder(nn.Module):
    """
    Shared temporal encoder applied independently to every sensor.

    Public input:
        [batch, sequence_length, num_nodes]

    Example:
        [B, 60, 66]

    Internally, the tensor is transposed so that every sensor gets
    its own 60-step temporal sequence.

    Output:
        [batch, num_nodes, embedding_dim]
    """

    def __init__(
        self,
        sequence_length: int,
        temporal_hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if temporal_hidden_dim <= 0:
            raise ValueError(
                "temporal_hidden_dim must be positive."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.projection = nn.Linear(
            temporal_hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each sensor's temporal history.

        Input:
            [B, T, N]

        Output:
            [B, N, E]
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "[batch, sequence_length, num_nodes]."
            )

        batch_size, sequence_length, num_nodes = x.shape

        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence_length={self.sequence_length}, "
                f"got {sequence_length}."
            )

        # Project-wide convention:
        #
        # [B, T, N]
        #
        # Convert to:
        #
        # [B, N, T]
        #
        # so each sensor has its own temporal sequence.
        sensor_histories = x.transpose(1, 2)

        # Treat every sensor independently.
        #
        # [B, N, T]
        #       ↓
        # [B*N, T, 1]
        sensor_sequences = sensor_histories.reshape(
            batch_size * num_nodes,
            sequence_length,
            1,
        )

        lstm_output, _ = self.lstm(
            sensor_sequences
        )

        # Take the final temporal representation.
        #
        # [B*N, T, H]
        #       ↓
        # [B*N, H]
        final_hidden = lstm_output[:, -1, :]

        embeddings = self.projection(
            final_hidden
        )

        # [B*N, E]
        #       ↓
        # [B, N, E]
        embeddings = embeddings.reshape(
            batch_size,
            num_nodes,
            -1,
        )

        return embeddings


class DynamicGraphBuilder(nn.Module):
    """
    Construct a window-specific top-k graph from sensor embeddings.

    Input:
        [batch, num_nodes, embedding_dim]

    Output:
        List of edge_index tensors.

    One graph is produced for every item in the batch.

    Each edge_index has shape:

        [2, num_nodes * k]

    The graph is directed because each node independently selects
    its top-k neighbors.
    """

    def __init__(
        self,
        num_nodes: int,
        k: int,
    ) -> None:
        super().__init__()

        if num_nodes <= 1:
            raise ValueError(
                "num_nodes must be greater than 1."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if k >= num_nodes:
            raise ValueError(
                f"k must be smaller than num_nodes ({num_nodes})."
            )

        self.num_nodes = num_nodes
        self.k = k

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Build one dynamic graph per window.
        """

        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(
                "embeddings must be a torch.Tensor."
            )

        if embeddings.ndim != 3:
            raise ValueError(
                "Expected embeddings with shape "
                "[batch, num_nodes, embedding_dim]."
            )

        batch_size, num_nodes, _ = embeddings.shape

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"got {num_nodes}."
            )

        # Normalize node embeddings so that dot product becomes
        # cosine similarity.
        normalized = F.normalize(
            embeddings,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        # Pairwise cosine similarity.
        #
        # [B, N, E] @ [B, E, N]
        #
        # -> [B, N, N]
        similarity = torch.bmm(
            normalized,
            normalized.transpose(1, 2),
        )

        # Remove self-connections.
        diagonal_mask = torch.eye(
            num_nodes,
            device=embeddings.device,
            dtype=torch.bool,
        )

        similarity = similarity.masked_fill(
            diagonal_mask.unsqueeze(0),
            float("-inf"),
        )

        edge_indices: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            batch_similarity = similarity[
                batch_idx
            ]

            # Each node selects its k most similar
            # other nodes.
            neighbor_indices = torch.topk(
                batch_similarity,
                k=self.k,
                dim=-1,
            ).indices

            source = (
                torch.arange(
                    num_nodes,
                    device=embeddings.device,
                )
                .unsqueeze(1)
                .expand(-1, self.k)
                .reshape(-1)
            )

            target = neighbor_indices.reshape(-1)

            edge_index = torch.stack(
                [
                    source,
                    target,
                ],
                dim=0,
            )

            edge_indices.append(
                edge_index
            )

        return edge_indices


class DynamicGraphAutoencoder(nn.Module):
    """
    Dynamic graph autoencoder for multivariate industrial
    time-series anomaly detection.

    Input:
        [batch, sequence_length, num_nodes]

    Example:
        [B, 60, 66]

    Processing:

        60-second sensor window
                    ↓
        Shared sensor-wise LSTM
                    ↓
        66 sensor embeddings
                    ↓
        Window-specific similarity graph
                    ↓
        GAT encoder
                    ↓
        Latent graph representation
                    ↓
        GAT decoder
                    ↓
        Reconstructed final timestep
                    ↓
        Reconstruction error
                    ↓
        Anomaly score
    """

    def __init__(
        self,
        sequence_length: int = 60,
        num_nodes: int = 66,
        temporal_hidden_dim: int = 32,
        embedding_dim: int = 32,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
        k: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if num_nodes <= 1:
            raise ValueError(
                "num_nodes must be greater than 1."
            )

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if gat_hidden_dim <= 0:
            raise ValueError(
                "gat_hidden_dim must be positive."
            )

        if gat_heads <= 0:
            raise ValueError(
                "gat_heads must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        self.sequence_length = sequence_length
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.k = k

        # ---------------------------------------------------------
        # Temporal encoder
        # ---------------------------------------------------------

        self.temporal_encoder = SensorTemporalEncoder(
            sequence_length=sequence_length,
            temporal_hidden_dim=temporal_hidden_dim,
            embedding_dim=embedding_dim,
        )

        # ---------------------------------------------------------
        # Dynamic graph construction
        # ---------------------------------------------------------

        self.graph_builder = DynamicGraphBuilder(
            num_nodes=num_nodes,
            k=k,
        )

        # ---------------------------------------------------------
        # GAT encoder
        # ---------------------------------------------------------

        self.gat_encoder_1 = GATConv(
            in_channels=embedding_dim,
            out_channels=gat_hidden_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
        )

        self.gat_encoder_2 = GATConv(
            in_channels=gat_hidden_dim * gat_heads,
            out_channels=embedding_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        # ---------------------------------------------------------
        # GAT decoder
        # ---------------------------------------------------------

        self.gat_decoder_1 = GATConv(
            in_channels=embedding_dim,
            out_channels=gat_hidden_dim,
            heads=gat_heads,
            concat=True,
            dropout=dropout,
        )

        self.gat_decoder_2 = GATConv(
            in_channels=gat_hidden_dim * gat_heads,
            out_channels=1,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encode input windows into latent graph representations.

        Input:
            [B, T, N]

        Returns:
            latent:
                [B, N, E]

            edge_indices:
                One dynamic graph per batch item.
        """

        self._validate_input(x)

        embeddings = self.temporal_encoder(x)

        edge_indices = self.graph_builder(
            embeddings
        )

        latent_nodes = []

        for batch_idx, edge_index in enumerate(
            edge_indices
        ):
            node_embeddings = embeddings[
                batch_idx
            ]

            hidden = self.gat_encoder_1(
                node_embeddings,
                edge_index,
            )

            hidden = F.elu(hidden)

            latent = self.gat_encoder_2(
                hidden,
                edge_index,
            )

            latent_nodes.append(
                latent
            )

        latent = torch.stack(
            latent_nodes,
            dim=0,
        )

        return latent, edge_indices

    def decode(
        self,
        latent: torch.Tensor,
        edge_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode latent graph representations.

        Input:
            latent:
                [B, N, E]

        Returns:
            reconstruction:
                [B, N]
        """

        if not isinstance(latent, torch.Tensor):
            raise TypeError(
                "latent must be a torch.Tensor."
            )

        if latent.ndim != 3:
            raise ValueError(
                "Expected latent tensor with shape "
                "[batch, num_nodes, embedding_dim]."
            )

        batch_size, num_nodes, embedding_dim = (
            latent.shape
        )

        if batch_size != len(edge_indices):
            raise ValueError(
                "Number of graphs does not match "
                "batch size."
            )

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"got {num_nodes}."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"got {embedding_dim}."
            )

        reconstructed_nodes = []

        for batch_idx, edge_index in enumerate(
            edge_indices
        ):
            node_latent = latent[
                batch_idx
            ]

            hidden = self.gat_decoder_1(
                node_latent,
                edge_index,
            )

            hidden = F.elu(hidden)

            reconstruction = self.gat_decoder_2(
                hidden,
                edge_index,
            )

            reconstruction = (
                reconstruction.squeeze(-1)
            )

            reconstructed_nodes.append(
                reconstruction
            )

        return torch.stack(
            reconstructed_nodes,
            dim=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the final timestep.

        Input:
            [B, T, N]

        Output:
            [B, N]
        """

        latent, edge_indices = self.encode(x)

        reconstruction = self.decode(
            latent,
            edge_indices,
        )

        return reconstruction

    def reconstruction_error(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate per-window reconstruction MSE.

        The target is the final timestep of each
        60-second window.

        Returns:
            [B]
        """

        self._validate_input(x)

        target = x[:, -1, :]

        reconstruction = self.forward(x)

        return torch.mean(
            (reconstruction - target) ** 2,
            dim=1,
        )

    def anomaly_score(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return reconstruction error as anomaly score.
        """

        return self.reconstruction_error(x)

    def get_dynamic_graphs(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Return the graph generated for every
        input window.

        This will later be useful for graph
        visualization and explainability.
        """

        self._validate_input(x)

        embeddings = self.temporal_encoder(x)

        return self.graph_builder(
            embeddings
        )

    def get_sensor_embeddings(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return learned sensor representations.

        Output:
            [B, N, embedding_dim]
        """

        self._validate_input(x)

        return self.temporal_encoder(x)

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[batch, sequence_length, num_nodes]."
            )

        _, sequence_length, num_nodes = x.shape

        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence_length="
                f"{self.sequence_length}, "
                f"got {sequence_length}."
            )

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes="
                f"{self.num_nodes}, "
                f"got {num_nodes}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Input contains NaN or infinite values."
            )