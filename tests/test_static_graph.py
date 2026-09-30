from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


# ============================================================
# Project imports
# ============================================================

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
)

from swat_gnn.graph.static_graph import (
    StaticCorrelationGraph,
    build_top_k_correlation_graph,
)


# ============================================================
# Helper
# ============================================================

def get_training_correlation() -> pd.DataFrame:
    """
    Build the training-only correlation matrix.

    This reproduces the same training-feature selection used
    during exploratory analysis.

    The 20 features that were constant across all training
    segments are excluded.

    The resulting matrix should therefore be:

        66 x 66
    """

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    config = load_config(
        str(config_path)
    )

    training_segments = load_training_data(
        config
    )

    # --------------------------------------------------------
    # Get feature names.
    #
    # Timestamp is metadata, not a model feature.
    # --------------------------------------------------------

    feature_names = [
        column
        for column in training_segments[0].columns
        if column != "timestamp"
    ]

    # --------------------------------------------------------
    # Find features that are constant across ALL
    # training segments.
    # --------------------------------------------------------

    variable_features = []

    for feature in feature_names:

        is_constant = all(
            df[feature].nunique(
                dropna=False
            ) <= 1
            for df in training_segments
        )

        if not is_constant:
            variable_features.append(
                feature
            )

    # --------------------------------------------------------
    # Combine training segments.
    #
    # These are still training observations only.
    # --------------------------------------------------------

    training_data = pd.concat(
        [
            df[variable_features]
            for df in training_segments
        ],
        axis=0,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Compute Pearson correlation.
    # --------------------------------------------------------

    correlation = training_data.corr()

    return correlation


# ============================================================
# Test 1
# ============================================================

def test_graph_type():
    """Graph constructor returns the expected graph object."""

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    assert isinstance(
        graph,
        StaticCorrelationGraph,
    )


# ============================================================
# Test 2
# ============================================================

def test_node_count():
    """
    The graph should contain the 66 variable features
    identified during training-data analysis.
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    assert graph.num_nodes == 66

    assert len(
        graph.feature_names
    ) == 66


# ============================================================
# Test 3
# ============================================================

def test_expected_edge_counts():
    """
    Verify the graph reproduces the edge counts observed
    during exploratory analysis.

    Expected:

        k=3   -> 144 edges
        k=5   -> 222 edges
        k=8   -> 357 edges
        k=10  -> 432 edges
    """

    correlation = get_training_correlation()

    expected_edges = {
        3: 144,
        5: 222,
        8: 357,
        10: 432,
    }

    for k, expected in expected_edges.items():

        graph = build_top_k_correlation_graph(
            correlation_matrix=correlation,
            k=k,
        )

        assert graph.num_edges == expected


# ============================================================
# Test 4
# ============================================================

def test_no_self_loops():
    """
    No node should be connected to itself.
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    source = graph.edge_index[0]
    target = graph.edge_index[1]

    assert not np.any(
        source == target
    )


# ============================================================
# Test 5
# ============================================================

def test_graph_is_undirected():
    """
    Each undirected edge should be stored only once.

    The implementation stores edges such that:

        source < target

    Therefore both directions are not duplicated.
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    source = graph.edge_index[0]
    target = graph.edge_index[1]

    assert np.all(
        source < target
    )


# ============================================================
# Test 6
# ============================================================

def test_no_isolated_nodes():
    """
    Every feature should have at least one graph connection.

    Because every node selects k neighbors and the graph is
    constructed using the union of those neighborhoods, the
    minimum degree should be at least k.
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    degrees = np.zeros(
        graph.num_nodes,
        dtype=np.int64,
    )

    # Count source endpoints.
    np.add.at(
        degrees,
        graph.edge_index[0],
        1,
    )

    # Count target endpoints.
    np.add.at(
        degrees,
        graph.edge_index[1],
        1,
    )

    assert degrees.min() >= 3

    assert np.all(
        degrees > 0
    )


# ============================================================
# Test 7
# ============================================================

def test_edge_weights_are_signed_correlations():
    """
    Neighbor selection uses absolute correlation, but the
    stored edge weight must preserve the ORIGINAL signed
    correlation.

    Example:

        correlation = -0.85

    must remain:

        edge_weight = -0.85

    rather than:

        edge_weight = +0.85
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    for edge_idx in range(
        graph.num_edges
    ):

        source = graph.edge_index[
            0,
            edge_idx,
        ]

        target = graph.edge_index[
            1,
            edge_idx,
        ]

        expected_weight = correlation.iloc[
            source,
            target,
        ]

        actual_weight = graph.edge_weight[
            edge_idx
        ]

        assert np.isclose(
            actual_weight,
            expected_weight,
            atol=1e-6,
        )


# ============================================================
# Test 8
# ============================================================

def test_feature_order_is_preserved():
    """
    The graph's node ordering must match the ordering of the
    correlation matrix.
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    assert graph.feature_names == list(
        correlation.index
    )


# ============================================================
# Test 9
# ============================================================

def test_edge_index_shape():
    """
    edge_index must have shape:

        (2, number_of_edges)
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    assert graph.edge_index.shape == (
        2,
        graph.num_edges,
    )


# ============================================================
# Test 10
# ============================================================

def test_edge_weight_shape():
    """
    There must be exactly one edge weight for every edge.
    """

    correlation = get_training_correlation()

    graph = build_top_k_correlation_graph(
        correlation_matrix=correlation,
        k=3,
    )

    assert graph.edge_weight.shape == (
        graph.num_edges,
    )


# ============================================================
# Test 11
# ============================================================

def test_invalid_k_zero():
    """
    k=0 is invalid.
    """

    correlation = get_training_correlation()

    try:

        build_top_k_correlation_graph(
            correlation_matrix=correlation,
            k=0,
        )

    except ValueError:

        return

    raise AssertionError(
        "Expected ValueError for k=0."
    )


# ============================================================
# Test 12
# ============================================================

def test_invalid_k_too_large():
    """
    k >= number of nodes is invalid because a node cannot
    select itself as a neighbor.
    """

    correlation = get_training_correlation()

    try:

        build_top_k_correlation_graph(
            correlation_matrix=correlation,
            k=len(correlation),
        )

    except ValueError:

        return

    raise AssertionError(
        "Expected ValueError when "
        "k >= number of nodes."
    )


# ============================================================
# Test 13
# ============================================================

def test_non_square_matrix():
    """
    A non-square DataFrame must be rejected.
    """

    correlation = get_training_correlation()

    invalid_matrix = correlation.iloc[
        :-1,
        :,
    ]

    try:

        build_top_k_correlation_graph(
            correlation_matrix=invalid_matrix,
            k=3,
        )

    except ValueError:

        return

    raise AssertionError(
        "Expected ValueError for "
        "non-square correlation matrix."
    )


# ============================================================
# Test 14
# ============================================================

def test_nan_correlation_matrix():
    """
    A correlation matrix containing NaN values must be rejected.
    """

    correlation = get_training_correlation().copy()

    correlation.iloc[
        0,
        1,
    ] = np.nan

    try:

        build_top_k_correlation_graph(
            correlation_matrix=correlation,
            k=3,
        )

    except ValueError:

        return

    raise AssertionError(
        "Expected ValueError for "
        "correlation matrix containing NaN."
    )