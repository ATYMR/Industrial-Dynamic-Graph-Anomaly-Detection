from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_gnn.data.swat_temporal_dataset import (
    build_swat_temporal_datasets,
)


def main():
    data = build_swat_temporal_datasets()

    train_dataset = data["train_dataset"]
    validation_dataset = data["validation_dataset"]
    test_datasets = data["test_datasets"]
    test_labels = data["test_labels"]
    preprocessor = data["preprocessor"]

    print("=== SWaT DATA PIPELINE VERIFICATION ===")

    print("Features:", len(data["feature_names"]))
    print("Skipped training segments:", data["skipped_train_segments"])
    print("Training windows:", len(train_dataset))
    print("Validation windows:", len(validation_dataset))
    print("Test episodes:", len(test_datasets))
    print("Test windows:", [len(dataset) for dataset in test_datasets])
    print("Attack windows:", [int(labels.sum()) for labels in test_labels])

    assert len(data["feature_names"]) == 40
    assert data["skipped_train_segments"] == [14]

    assert len(train_dataset) == 32415
    assert len(validation_dataset) == 68836

    assert len(test_datasets) == 2
    assert [len(dataset) for dataset in test_datasets] == [
        17547,
        151023,
    ]

    assert [int(labels.sum()) for labels in test_labels] == [
        0,
        5672,
    ]

    assert train_dataset.num_features == 40
    assert validation_dataset.num_features == 40

    for dataset in test_datasets:
        assert dataset.num_features == 40

    assert len(preprocessor.feature_names) == 40
    assert preprocessor.feature_names == data["feature_names"]

    assert np.isfinite(preprocessor.scaler.mean_).all()
    assert np.isfinite(preprocessor.scaler.scale_).all()

    # Check representative windows from every split.
    assert np.isfinite(train_dataset[0].numpy()).all()
    assert np.isfinite(validation_dataset[0].numpy()).all()

    for dataset in test_datasets:
        assert np.isfinite(dataset[0].numpy()).all()

    # Test labels must be binary.
    for labels in test_labels:
        assert set(np.unique(labels)).issubset({0, 1})

    print("\nFeature schema: PASSED")
    print("Window counts: PASSED")
    print("Test-label alignment: PASSED")
    print("Scaler validation: PASSED")
    print("Finite-value checks: PASSED")
    print("SWaT data pipeline verification: SUCCESS")


if __name__ == "__main__":
    main()
