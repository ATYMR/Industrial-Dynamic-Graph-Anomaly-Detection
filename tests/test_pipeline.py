from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
)
from swat_gnn.data.pipeline import DataPipeline


def main():
    config = load_config()

    training_segments = load_training_data(config)
    validation_segments = load_validation_data(config)

    pipeline = DataPipeline(
        sequence_length=config["windowing"]["sequence_length"],
        stride=config["windowing"]["stride"],
    )

    training_windows = pipeline.fit_transform_training(
        training_segments
    )

    validation_windows = pipeline.transform_validation(
        validation_segments
    )

    print("=== DATA PIPELINE VERIFICATION ===")

    print(
        "Sequence length:",
        pipeline.sequence_length,
    )

    print(
        "Stride:",
        pipeline.stride,
    )

    print(
        "Training segments:",
        len(training_windows),
    )

    print(
        "Validation segments:",
        len(validation_windows),
    )

    for i, windows in enumerate(training_windows, 1):
        print(
            f"train{i}: "
            f"shape={windows.shape}"
        )

    for i, windows in enumerate(validation_windows, 1):
        print(
            f"validation{i}: "
            f"shape={windows.shape}"
        )

    # HAI has 86 input features.
    for windows in training_windows:
        assert windows.ndim == 3
        assert windows.shape[1:] == (60, 86)

    for windows in validation_windows:
        assert windows.ndim == 3
        assert windows.shape[1:] == (60, 86)

    # Verify that the pipeline actually produced finite values.
    for windows in training_windows:
        assert np.isfinite(windows).all()

    for windows in validation_windows:
        assert np.isfinite(windows).all()

    print("\nWindow shapes: PASSED")
    print("Finite-value check: PASSED")
    print("Data pipeline verification: SUCCESS")


if __name__ == "__main__":
    main()