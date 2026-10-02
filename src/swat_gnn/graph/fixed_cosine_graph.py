from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_fixed_cosine_graph(
    training_windows: list[np.ndarray],
    k: int = 5,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Build one fixed directed top-k cosine-similarity graph
    using training windows only.
    """

    if not training_windows:
        raise ValueError("training_windows cannot be empty.")

    if k <= 0:
        raise ValueError("k must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    first = np.asarray(training_windows[0])

    if first.ndim != 3:
        raise ValueError(
            "Expected training windows with shape "
            "[num_windows, sequence_length, num_nodes]."
        )

    _, sequence_length, num_nodes = first.shape

    if k >= num_nodes:
        raise ValueError(
            f"k must be smaller than num_nodes ({num_nodes})."
        )

    similarity_sum = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float64,
    )

    total_windows = 0

    for windows in training_windows:
        windows = np.asarray(windows)

        if windows.ndim != 3:
            raise ValueError(
                "Each training array must have shape "
                "[num_windows, sequence_length, num_nodes]."
            )

        if windows.shape[1] != sequence_length:
            raise ValueError(
                "Sequence length mismatch between training segments."
            )

        if windows.shape[2] != num_nodes:
            raise ValueError(
                "Number of nodes mismatch between training segments."
            )

        if not np.isfinite(windows).all():
            raise ValueError(
                "Training windows contain NaN or infinite values."
            )

        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]

            # [W, T, N] -> [W, N, T]
            sensor_histories = np.transpose(
                batch,
                (0, 2, 1),
            )

            tensor = torch.from_numpy(
                sensor_histories.astype(np.float32)
            )

            normalized = F.normalize(
                tensor,
                p=2,
                dim=-1,
                eps=1e-8,
            )

            # [W, N, T] @ [W, T, N] -> [W, N, N]
            similarity = torch.bmm(
                normalized,
                normalized.transpose(1, 2),
            )

            similarity_sum += similarity.sum(dim=0).numpy()
            total_windows += batch.shape[0]

    mean_similarity = similarity_sum / total_windows

    # Remove self-connections.
    np.fill_diagonal(
        mean_similarity,
        -np.inf,
    )

    # Select top-k neighbors for every sensor.
    neighbor_indices = np.argpartition(
        mean_similarity,
        -k,
        axis=1,
    )[:, -k:]

    # Sort selected neighbors deterministically.
    rows = np.arange(num_nodes)[:, None]

    selected_values = mean_similarity[
        rows,
        neighbor_indices,
    ]

    order = np.argsort(
        -selected_values,
        axis=1,
    )

    neighbor_indices = np.take_along_axis(
        neighbor_indices,
        order,
        axis=1,
    )

    source = np.repeat(
        np.arange(num_nodes),
        k,
    )

    target = neighbor_indices.reshape(-1)

    return torch.tensor(
        np.stack([source, target]),
        dtype=torch.long,
    )