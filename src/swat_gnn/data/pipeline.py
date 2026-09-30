from __future__ import annotations

import numpy as np

from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.windowing import create_windows_from_segments


class DataPipeline:
    """
    End-to-end preprocessing pipeline for HAI time-series data.

    The pipeline:
        1. Fits normalization on training data only.
        2. Transforms training and validation segments.
        3. Creates fixed-length windows independently per segment.
        4. Never allows windows to cross file boundaries.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        stride: int = 1,
    ):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.sequence_length = sequence_length
        self.stride = stride
        self.preprocessor = Preprocessor()

    def fit_transform_training(
        self,
        training_segments,
    ) -> list[np.ndarray]:
        """
        Fit the scaler on training data and create training windows.
        """
        scaled_segments = self.preprocessor.fit_transform(
            training_segments
        )

        return create_windows_from_segments(
            scaled_segments,
            sequence_length=self.sequence_length,
            stride=self.stride,
        )

    def transform_validation(
        self,
        validation_segments,
    ) -> list[np.ndarray]:
        """
        Transform validation data using training-fitted parameters
        and create validation windows.
        """
        scaled_segments = self.preprocessor.transform(
            validation_segments
        )

        return create_windows_from_segments(
            scaled_segments,
            sequence_length=self.sequence_length,
            stride=self.stride,
        )
    