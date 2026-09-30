from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StaticCorrelationGraph:
    """
    Undirected static graph constructed from a correlation matrix.

    Parameters
    ----------
    feature_names:
        Names of the graph nodes.

    edge_index:
        NumPy array of shape (2, num_edges).
        Each column contains [source_node, target_node].

    edge_weight:
        Signed correlation value for each edge.

    k:
        Number of strongest neighbors considered for each node.
    """

    feature_names: list[str]
    edge_index: np.ndarray
    edge_weight: np.ndarray
    k: int

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.feature_names)

    @property
    def num_edges(self) -> int:
        """Return the number of undirected edges."""
        return self.edge_index.shape[1]


def build_top_k_correlation_graph(
    correlation_matrix: pd.DataFrame,
    k: int,
) -> StaticCorrelationGraph:
    """
    Construct an undirected top-k correlation graph.

    For every feature, the k strongest relationships are selected
    according to absolute correlation magnitude.

    The resulting graph is made undirected using the union of the
    selected neighborhoods.

    Absolute correlation is used only for selecting neighbors.
    The original signed correlation is retained as the edge weight.

    Parameters
    ----------
    correlation_matrix:
        Square pandas DataFrame containing the correlation matrix.

    k:
        Number of strongest neighbors selected for each node.

    Returns
    -------
    StaticCorrelationGraph
        The constructed graph.

    Notes
    -----
    The correlation matrix should be computed from training data only.
    Test data or attack labels must not be used to construct this graph.
    """

    # ---------------------------------------------------------
    # Validate input type
    # ---------------------------------------------------------

    if not isinstance(correlation_matrix, pd.DataFrame):
        raise TypeError(
            "correlation_matrix must be a pandas DataFrame."
        )

    # ---------------------------------------------------------
    # Validate matrix shape
    # ---------------------------------------------------------

    if (
        correlation_matrix.shape[0]
        != correlation_matrix.shape[1]
    ):
        raise ValueError(
            "correlation_matrix must be square."
        )

    # ---------------------------------------------------------
    # Validate node names
    # ---------------------------------------------------------

    if not correlation_matrix.index.equals(
        correlation_matrix.columns
    ):
        raise ValueError(
            "correlation_matrix index and columns must "
            "contain the same features in the same order."
        )

    # ---------------------------------------------------------
    # Validate k
    # ---------------------------------------------------------

    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer.")

    n_nodes = correlation_matrix.shape[0]

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if k >= n_nodes:
        raise ValueError(
            f"k must be smaller than the number of nodes ({n_nodes})."
        )

    # ---------------------------------------------------------
    # Validate correlation matrix
    # ---------------------------------------------------------

    if correlation_matrix.isna().any().any():
        raise ValueError(
            "correlation_matrix contains NaN values. "
            "Handle missing correlations before graph construction."
        )

    # ---------------------------------------------------------
    # Store feature names
    # ---------------------------------------------------------

    feature_names = list(correlation_matrix.index)

    # ---------------------------------------------------------
    # Convert correlation matrix to NumPy
    # ---------------------------------------------------------

    correlation = correlation_matrix.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    # ---------------------------------------------------------
    # Remove self-connections
    #
    # A correlation matrix normally has 1.0 on its diagonal.
    # We don't want a node selecting itself as a neighbor.
    # ---------------------------------------------------------

    np.fill_diagonal(correlation, 0.0)

    # ---------------------------------------------------------
    # Absolute correlation is used for neighbor selection.
    #
    # Example:
    #
    # correlation = -0.95
    # absolute correlation = 0.95
    #
    # This means a strong negative relationship is considered
    # just as strongly connected as a strong positive one.
    # ---------------------------------------------------------

    absolute_correlation = np.abs(correlation)

    # ---------------------------------------------------------
    # Construct directed top-k neighborhoods.
    #
    # Each node initially selects its own k strongest neighbors.
    # ---------------------------------------------------------

    directed_adjacency = np.zeros(
        (n_nodes, n_nodes),
        dtype=bool,
    )

    for node_idx in range(n_nodes):

        neighbor_indices = np.argpartition(
            -absolute_correlation[node_idx],
            kth=k - 1,
        )[:k]

        directed_adjacency[
            node_idx,
            neighbor_indices,
        ] = True

    # ---------------------------------------------------------
    # Convert to an undirected graph.
    #
    # If either:
    #
    # A -> B
    #
    # or:
    #
    # B -> A
    #
    # was selected, keep the undirected edge A -- B.
    # ---------------------------------------------------------

    adjacency = (
        directed_adjacency
        | directed_adjacency.T
    )

    # Explicitly remove self-loops.
    np.fill_diagonal(adjacency, False)

    # ---------------------------------------------------------
    # Extract each undirected edge only once.
    #
    # np.triu(..., k=1) keeps only the upper triangle.
    # Therefore we don't store both:
    #
    # A -> B
    # B -> A
    #
    # as separate edges.
    # ---------------------------------------------------------

    source_indices, target_indices = np.where(
        np.triu(adjacency, k=1)
    )

    # ---------------------------------------------------------
    # Preserve the ORIGINAL signed correlation as the weight.
    #
    # Important:
    #
    # Neighbor selection:
    #     abs(correlation)
    #
    # Edge weight:
    #     correlation
    #
    # Therefore a relationship of -0.8 remains -0.8.
    # ---------------------------------------------------------

    edge_weights = correlation[
        source_indices,
        target_indices,
    ]

    # ---------------------------------------------------------
    # Construct edge_index.
    #
    # Shape:
    #
    #     (2, num_edges)
    #
    # Example:
    #
    # [[0, 0, 1],
    #  [2, 4, 5]]
    #
    # means:
    #
    # 0 -- 2
    # 0 -- 4
    # 1 -- 5
    # ---------------------------------------------------------

    edge_index = np.vstack(
        [
            source_indices,
            target_indices,
        ]
    ).astype(np.int64)

    # PyTorch/PyG-compatible floating-point weights.
    edge_weights = edge_weights.astype(
        np.float32
    )

    # ---------------------------------------------------------
    # Return graph object
    # ---------------------------------------------------------

    return StaticCorrelationGraph(
        feature_names=feature_names,
        edge_index=edge_index,
        edge_weight=edge_weights,
        k=k,
    )