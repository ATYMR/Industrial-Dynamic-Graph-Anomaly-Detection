from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class StaticTemporalGraphAutoencoder(nn.Module):
    """
    Static-graph temporal autoencoder.

    This is the controlled counterpart to the dynamic graph model.

    Input:
        [B, T, N]

    Processing:
        sensor histories
            ↓
        shared sensor LSTM
            ↓
        sensor embeddings
            ↓
        FIXED graph
            ↓
        batched GAT
            ↓
        graph decoder
            ↓
        final timestep reconstruction

    The temporal encoder and GAT architecture are intentionally
    identical to the DynamicGraphAutoencoder.

    The only intended architectural difference is graph topology:
        dynamic model -> window-specific graph
        static model  -> one fixed graph
    """

    def __init__(
        self,
        sequence_length: int = 60,
        num_nodes: int = 66,
        temporal_hidden_dim: int = 32,
        embedding_dim: int = 32,
        gat_hidden_dim: int = 64,
        gat_heads: int = 4,
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

        if temporal_hidden_dim <= 0:
            raise ValueError(
                "temporal_hidden_dim must be positive."
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

        # ====================================================
        # Same temporal encoder as the dynamic model
        # ====================================================

        self.temporal_encoder = SensorTemporalEncoder(
            sequence_length=sequence_length,
            temporal_hidden_dim=temporal_hidden_dim,
            embedding_dim=embedding_dim,
        )

        # ====================================================
        # Same GAT encoder
        # ====================================================

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

        # ====================================================
        # Same GAT decoder
        # ====================================================

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

        # Fixed graph is registered after set_graph() is called.
        self.register_buffer(
            "edge_index",
            torch.empty(
                (2, 0),
                dtype=torch.long,
            ),
        )

        self.graph_initialized = False

    # ========================================================
    # Graph setup
    # ========================================================

    def set_graph(
        self,
        edge_index: torch.Tensor,
    ) -> None:
        """
        Set the single fixed graph used for every window.

        edge_index:
            [2, E]
        """

        if not isinstance(
            edge_index,
            torch.Tensor,
        ):
            raise TypeError(
                "edge_index must be a torch.Tensor."
            )

        if edge_index.ndim != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E]."
            )

        if edge_index.dtype != torch.long:
            edge_index = edge_index.long()

        if edge_index.numel() > 0:

            if edge_index.min().item() < 0:
                raise ValueError(
                    "edge_index contains negative node indices."
                )

            if edge_index.max().item() >= self.num_nodes:
                raise ValueError(
                    "edge_index contains node indices "
                    "outside the valid range."
                )

        self.edge_index = edge_index.detach().clone()

        self.graph_initialized = True

    # ========================================================
    # Graph batching
    # ========================================================

    def _batch_edge_indices(
        self,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Replicate the same fixed graph across all samples.

        Graph 0:
            nodes 0..N-1

        Graph 1:
            nodes N..2N-1

        etc.
        """

        if not self.graph_initialized:
            raise RuntimeError(
                "Graph has not been initialized. "
                "Call set_graph() first."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.edge_index.numel() == 0:
            raise RuntimeError(
                "Fixed graph contains no edges."
            )

        batched_edges = []

        for graph_idx in range(batch_size):

            offset = (
                graph_idx
                * self.num_nodes
            )

            batched_edges.append(
                self.edge_index
                + offset
            )

        return torch.cat(
            batched_edges,
            dim=1,
        )

    # ========================================================
    # Batched GAT
    # ========================================================

    def _apply_gat_batched(
        self,
        layer: GATConv,
        node_features: torch.Tensor,
        activation: bool = False,
    ) -> torch.Tensor:
        """
        Apply one GAT layer to all independent graphs.

        Input:
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

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"got {num_nodes}."
            )

        flattened_nodes = (
            node_features.reshape(
                batch_size * num_nodes,
                -1,
            )
        )

        batched_edges = (
            self._batch_edge_indices(
                batch_size
            )
        )

        output = layer(
            flattened_nodes,
            batched_edges,
        )

        if activation:
            output = F.elu(output)

        return output.reshape(
            batch_size,
            num_nodes,
            -1,
        )

    # ========================================================
    # Encoder
    # ========================================================

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode windows using the fixed graph.

        Returns:
            latent [B, N, E]
        """

        self._validate_input(x)

        if not self.graph_initialized:
            raise RuntimeError(
                "Graph has not been initialized. "
                "Call set_graph() before encode()."
            )

        embeddings = (
            self.temporal_encoder(x)
        )

        hidden = self._apply_gat_batched(
            self.gat_encoder_1,
            embeddings,
            activation=True,
        )

        latent = self._apply_gat_batched(
            self.gat_encoder_2,
            hidden,
            activation=False,
        )

        return latent

    # ========================================================
    # Decoder
    # ========================================================

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent graph representations.

        Returns:
            [B, N]
        """

        if not isinstance(
            latent,
            torch.Tensor,
        ):
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
            activation=True,
        )

        reconstruction = self._apply_gat_batched(
            self.gat_decoder_2,
            hidden,
            activation=False,
        )

        return reconstruction.squeeze(-1)

    # ========================================================
    # Forward
    # ========================================================

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

        latent = self.encode(x)

        return self.decode(latent)

    # ========================================================
    # Reconstruction error
    # ========================================================

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

    # ========================================================
    # Anomaly score
    # ========================================================

    def anomaly_score(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.reconstruction_error(x)

    # ========================================================
    # Fixed graph access
    # ========================================================

    def get_graph(
        self,
    ) -> torch.Tensor:

        if not self.graph_initialized:
            raise RuntimeError(
                "Graph has not been initialized."
            )

        return self.edge_index

    # ========================================================
    # Sensor embeddings
    # ========================================================

    def get_sensor_embeddings(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        self._validate_input(x)

        return self.temporal_encoder(x)

    # ========================================================
    # Input validation
    # ========================================================

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:

        if not isinstance(
            x,
            torch.Tensor,
        ):
            raise TypeError(
                "x must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                "Expected x with shape "
                "[batch, sequence_length, num_nodes]."
            )

        _, sequence_length, num_nodes = (
            x.shape
        )

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


class SensorTemporalEncoder(nn.Module):
    """
    Shared temporal encoder applied independently
    to every sensor.

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

        if x.ndim != 3:
            raise ValueError(
                "Expected input with shape "
                "[batch, sequence_length, num_nodes]."
            )

        batch_size, sequence_length, num_nodes = (
            x.shape
        )

        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence_length="
                f"{self.sequence_length}, "
                f"got {sequence_length}."
            )

        # [B, T, N] -> [B, N, T]
        sensor_histories = x.transpose(
            1,
            2,
        )

        # [B, N, T] -> [B*N, T, 1]
        sensor_sequences = (
            sensor_histories.reshape(
                batch_size * num_nodes,
                sequence_length,
                1,
            )
        )

        lstm_output, _ = self.lstm(
            sensor_sequences
        )

        final_hidden = (
            lstm_output[:, -1, :]
        )

        embeddings = self.projection(
            final_hidden
        )

        return embeddings.reshape(
            batch_size,
            num_nodes,
            -1,
        )