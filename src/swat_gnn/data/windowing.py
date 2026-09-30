from __future__ import annotations

import numpy as np


def create_windows(
    data: np.ndarray,
    sequence_length: int = 60,
    stride: int = 1,
) -> np.ndarray:
    """
    Convert a continuous time-series array into overlapping windows.

    Parameters
    ----------
    data:
        Array with shape (timesteps, features).

    sequence_length:
        Number of timesteps in each window.

    stride:
        Number of timesteps between consecutive windows.

    Returns
    -------
    np.ndarray
        Array with shape:
        (number_of_windows, sequence_length, features)
    """
    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D array (timesteps, features), got shape {data.shape}"
        )

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    n_timesteps, n_features = data.shape

    if n_timesteps < sequence_length:
        raise ValueError(
            f"Data has {n_timesteps} timesteps, "
            f"but sequence_length is {sequence_length}."
        )

    n_windows = (
        (n_timesteps - sequence_length) // stride
    ) + 1

    windows = np.empty(
        (n_windows, sequence_length, n_features),
        dtype=data.dtype,
    )

    for i in range(n_windows):
        start = i * stride
        end = start + sequence_length
        windows[i] = data[start:end]

    return windows


def create_windows_from_segments(
    segments: list[np.ndarray],
    sequence_length: int = 60,
    stride: int = 1,
) -> list[np.ndarray]:
    """
    Create windows independently for each time-series segment.

    Windows are never allowed to cross segment/file boundaries.
    """
    if not segments:
        raise ValueError("No segments provided.")

    return [
        create_windows(
            data=segment,
            sequence_length=sequence_length,
            stride=stride,
        )
        for segment in segments
    ]