from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
)
from swat_gnn.data.preprocessing import Preprocessor


def main():
    config = load_config()

    train = load_training_data(config)
    validation = load_validation_data(config)

    preprocessor = Preprocessor()

    train_scaled = preprocessor.fit_transform(train)
    validation_scaled = preprocessor.transform(validation)

    print("=== PREPROCESSING VERIFICATION ===")
    print("Number of features:", len(preprocessor.feature_names))
    print("Training segments:", len(train_scaled))
    print("Validation segments:", len(validation_scaled))

    for i, data in enumerate(train_scaled, 1):
        print(
            f"train{i}: "
            f"shape={data.shape}, "
            f"mean={data.mean():.6f}, "
            f"std={data.std():.6f}"
        )

    for i, data in enumerate(validation_scaled, 1):
        print(
            f"validation{i}: "
            f"shape={data.shape}, "
            f"mean={data.mean():.6f}, "
            f"std={data.std():.6f}"
        )

    # Combine all training segments.
    combined_train = np.concatenate(train_scaled, axis=0)

    # Calculate per-feature statistics.
    feature_means = combined_train.mean(axis=0)
    feature_stds = combined_train.std(axis=0)

    # Every training feature should have mean approximately zero.
    assert np.allclose(
        feature_means,
        0.0,
        atol=1e-6,
    )

    # Identify features that were non-constant before scaling.
    raw_train = np.concatenate(
        [
            df[
                [
                    c
                    for c in df.columns
                    if c != "timestamp"
                ]
            ].to_numpy(dtype=np.float64)
            for df in train
        ],
        axis=0,
    )

    raw_feature_stds = raw_train.std(axis=0)

    non_constant = raw_feature_stds > 0
    constant = ~non_constant

    # Non-constant features should have standard deviation ~1.
    assert np.allclose(
        feature_stds[non_constant],
        1.0,
        atol=1e-6,
    )

    # Constant features should remain constant after scaling.
    assert np.allclose(
        feature_stds[constant],
        0.0,
        atol=1e-6,
    )

    print("\nTraining normalization: PASSED")
    print(
        f"Non-constant features: {non_constant.sum()}"
    )
    print(
        f"Constant features: {constant.sum()}"
    )
    print("Preprocessing verification: SUCCESS")


if __name__ == "__main__":
    main()