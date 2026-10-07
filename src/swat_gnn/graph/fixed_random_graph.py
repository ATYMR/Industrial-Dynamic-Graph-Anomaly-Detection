from __future__ import annotations

import numpy as np
import torch


def build_fixed_random_graph(
    num_nodes: int,
    k: int = 5,
    seed: int = 42,
) -> torch.Tensor:
    """
    Build one fixed directed random top-k graph.

    Each node receives exactly k randomly selected outgoing
    neighbors, with no self-connections.
    """

    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if k <= 0:
        raise ValueError("k must be positive.")

    if k >= num_nodes:
        raise ValueError(
            f"k must be smaller than num_nodes ({num_nodes})."
        )

    rng = np.random.default_rng(seed)

    neighbors = np.empty(
        (num_nodes, k),
        dtype=np.int64,
    )

    for source in range(num_nodes):
        candidates = np.delete(
            np.arange(num_nodes),
            source,
        )

        neighbors[source] = rng.choice(
            candidates,
            size=k,
            replace=False,
        )

    source = np.repeat(
        np.arange(num_nodes),
        k,
    )

    target = neighbors.reshape(-1)

    return torch.tensor(
        np.stack([source, target]),
        dtype=torch.long,
    )
