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

    Input:
        [B, T, N]

    Output:
        [B, N, E]
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

        # [B, T, N] -> [B, N, T]
        sensor_histories = x.transpose(1, 2)

        # [B, N, T] -> [B*N, T, 1]
        sensor_sequences = sensor_histories.reshape(
            batch_size * num_nodes,
            sequence_length,
            1,
        )

        lstm_output, _ = self.lstm(
            sensor_sequences
        )

        # Final temporal representation.
        final_hidden = lstm_output[:, -1, :]

        embeddings = self.projection(
            final_hidden
        )

        # [B*N, E] -> [B, N, E]
        embeddings = embeddings.reshape(
            batch_size,
            num_nodes,
            -1,
        )

        return embeddings


class DynamicGraphBuilder(nn.Module):
    """
    Build one window-specific top-k graph per sample.

    Each graph contains:
        num_nodes * k

    directed edges.

    The graphs remain independent from one another.
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
        Input:
            [B, N, E]

        Output:
            list of B edge_index tensors.
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

        # Normalize embeddings so dot product becomes
        # cosine similarity.
        normalized = F.normalize(
            embeddings,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        # [B, N, E] @ [B, E, N]
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

        # Graph construction is inherently per-window because
        # every window has its own similarity matrix.
        for batch_idx in range(batch_size):
            batch_similarity = similarity[
                batch_idx
            ]

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
                [source, target],
                dim=0,
            )

            edge_indices.append(edge_index)

        return edge_indices


class DynamicGraphAutoencoder(nn.Module):
    """
    Dynamic graph autoencoder.

    Input:
        [B, 60, 66]

    Processing:

        sensor histories
              ↓
        shared sensor LSTM
              ↓
        sensor embeddings
              ↓
        window-specific top-k graph
              ↓
        batched disconnected GAT
              ↓
        graph decoder
              ↓
        final timestep reconstruction
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

        self.temporal_encoder = SensorTemporalEncoder(
            sequence_length=sequence_length,
            temporal_hidden_dim=temporal_hidden_dim,
            embedding_dim=embedding_dim,
        )

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

    # =================================================================
    # Graph batching
    # =================================================================

    def _batch_edge_indices(
        self,
        edge_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Combine independent graphs into one disconnected graph.

        Example:

            Graph 0 -> nodes 0..65
            Graph 1 -> nodes 66..131
            Graph 2 -> nodes 132..197

        No edges are added between graphs.

        This lets PyG process the whole batch using one GAT call
        while preserving exactly the same graph structure.
        """

        if not edge_indices:
            raise ValueError(
                "edge_indices cannot be empty."
            )

        batched_edges = []

        device = edge_indices[0].device

        for graph_idx, edge_index in enumerate(
            edge_indices
        ):
            if edge_index.ndim != 2:
                raise ValueError(
                    "Each edge_index must be 2D."
                )

            if edge_index.shape[0] != 2:
                raise ValueError(
                    "Each edge_index must have shape [2, E]."
                )

            offset = graph_idx * self.num_nodes

            batched_edge_index = (
                edge_index + offset
            )

            batched_edges.append(
                batched_edge_index
            )

        return torch.cat(
            batched_edges,
            dim=1,
        ).to(device)

    # =================================================================
    # Batched GAT helper
    # =================================================================

    def _apply_gat_batched(
        self,
        layer: GATConv,
        node_features: torch.Tensor,
        edge_indices: list[torch.Tensor],
        activation: bool = False,
    ) -> torch.Tensor:
        """
        Apply one GAT layer to all independent graphs
        simultaneously.

        Input:
            node_features:
                [B, N, F]

        Output:
            [B, N, F_out]
        """

        if node_features.ndim != 3:
            raise ValueError(
                "node_features must have shape "
                "[batch, num_nodes, features]."
            )

        batch_size, num_nodes, _ = (
            node_features.shape
        )

        if batch_size != len(edge_indices):
            raise ValueError(
                "Number of graphs must equal batch size."
            )

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"got {num_nodes}."
            )

        # [B, N, F] -> [B*N, F]
        flattened_nodes = node_features.reshape(
            batch_size * num_nodes,
            -1,
        )

        batched_edges = (
            self._batch_edge_indices(
                edge_indices
            )
        )

        output = layer(
            flattened_nodes,
            batched_edges,
        )

        if activation:
            output = F.elu(output)

        # [B*N, F_out] -> [B, N, F_out]
        output = output.reshape(
            batch_size,
            num_nodes,
            -1,
        )

        return output

    # =================================================================
    # Encoder
    # =================================================================

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encode windows into latent graph representations.

        Returns:
            latent:
                [B, N, E]

            edge_indices:
                list containing one graph per window.
        """

        self._validate_input(x)

        embeddings = self.temporal_encoder(x)

        edge_indices = self.graph_builder(
            embeddings
        )

        hidden = self._apply_gat_batched(
            self.gat_encoder_1,
            embeddings,
            edge_indices,
            activation=True,
        )

        latent = self._apply_gat_batched(
            self.gat_encoder_2,
            hidden,
            edge_indices,
            activation=False,
        )

        return latent, edge_indices

    # =================================================================
    # Decoder
    # =================================================================

    def decode(
        self,
        latent: torch.Tensor,
        edge_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode latent representations.

        Input:
            latent:
                [B, N, E]

        Output:
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
                "Number of graphs does not match batch size."
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

        hidden = self._apply_gat_batched(
            self.gat_decoder_1,
            latent,
            edge_indices,
            activation=True,
        )

        reconstruction = self._apply_gat_batched(
            self.gat_decoder_2,
            hidden,
            edge_indices,
            activation=False,
        )

        # [B, N, 1] -> [B, N]
        return reconstruction.squeeze(-1)

    # =================================================================
    # Forward
    # =================================================================

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

        return self.decode(
            latent,
            edge_indices,
        )

    # =================================================================
    # Reconstruction error
    # =================================================================

    def reconstruction_error(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-window reconstruction MSE.

        Target:
            final timestep of the window.

        Output:
            [B]
        """

        self._validate_input(x)

        target = x[:, -1, :]

        reconstruction = self.forward(x)

        return torch.mean(
            (reconstruction - target) ** 2,
            dim=1,
        )

    # =================================================================
    # Anomaly score
    # =================================================================

    def anomaly_score(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.reconstruction_error(x)

    # =================================================================
    # Dynamic graphs
    # =================================================================

    def get_dynamic_graphs(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Return one dynamic graph for each input window.
        """

        self._validate_input(x)

        embeddings = self.temporal_encoder(x)

        return self.graph_builder(
            embeddings
        )

    # =================================================================
    # Sensor embeddings
    # =================================================================

    def get_sensor_embeddings(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return learned sensor embeddings.

        Output:
            [B, N, E]
        """

        self._validate_input(x)

        return self.temporal_encoder(x)

    # =================================================================
    # Input validation
    # =================================================================

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