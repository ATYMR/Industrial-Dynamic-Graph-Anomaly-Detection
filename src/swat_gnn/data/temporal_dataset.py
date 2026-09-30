from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SegmentInfo:
    """Metadata describing one continuous time-series segment."""

    segment_id: int
    length: int


class TemporalWindowDataset(Dataset):
    """
    Lazily generates fixed-length causal windows from continuous segments.

    A sample ending at timestep t contains:

        [t-sequence_length+1, ..., t]

    The target timestep is therefore the final timestep of the window.

    Windows never cross segment boundaries.
    """

    def __init__(
        self,
        segments: list[np.ndarray],
        sequence_length: int = 60,
        stride: int = 1,
    ) -> None:
        if not segments:
            raise ValueError("segments must not be empty.")

        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        if stride <= 0:
            raise ValueError("stride must be positive.")

        validated_segments: list[np.ndarray] = []

        for i, segment in enumerate(segments):
            array = np.asarray(segment)

            if array.ndim != 2:
                raise ValueError(
                    f"Segment {i} must be 2D "
                    f"(timesteps, features), got shape {array.shape}."
                )

            if array.shape[0] < sequence_length:
                raise ValueError(
                    f"Segment {i} has {array.shape[0]} timesteps, "
                    f"but sequence_length is {sequence_length}."
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"Segment {i} contains NaN or infinite values."
                )

            validated_segments.append(
                array.astype(np.float32, copy=False)
            )

        feature_counts = {segment.shape[1] for segment in validated_segments}

        if len(feature_counts) != 1:
            raise ValueError(
                "All segments must have the same number of features."
            )

        self.segments = validated_segments
        self.sequence_length = sequence_length
        self.stride = stride

        self.segment_info = [
            SegmentInfo(segment_id=i, length=len(segment))
            for i, segment in enumerate(self.segments)
        ]

        self._index: list[tuple[int, int]] = []

        for segment_id, segment in enumerate(self.segments):
            n_windows = (
                (len(segment) - sequence_length) // stride
            ) + 1

            for window_number in range(n_windows):
                end_index = (
                    sequence_length - 1
                    + window_number * stride
                )

                self._index.append(
                    (segment_id, end_index)
                )

    @property
    def num_features(self) -> int:
        return self.segments[0].shape[1]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset "
                f"of length {len(self)}."
            )

        segment_id, end_index = self._index[index]

        start_index = (
            end_index - self.sequence_length + 1
        )

        window = self.segments[segment_id][
            start_index : end_index + 1
        ]

        return torch.from_numpy(window)

    def get_target_timestep(
        self,
        index: int,
    ) -> tuple[int, int]:
        """
        Return (segment_id, timestep_index) for the final
        timestep represented by a dataset sample.
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset "
                f"of length {len(self)}."
            )

        return self._index[index]