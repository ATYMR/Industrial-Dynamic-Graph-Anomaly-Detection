from __future__ import annotations

import numpy as np

from swat_gnn.data.preprocessing import Preprocessor
from swat_gnn.data.swat_loader import load_swat_dataset
from swat_gnn.data.temporal_dataset import TemporalWindowDataset


def _filter_short_segments(
    segments: list[np.ndarray],
    sequence_length: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Keep only segments long enough to produce one temporal window."""
    usable = []
    skipped = []

    for segment_id, segment in enumerate(segments):
        if len(segment) < sequence_length:
            skipped.append(segment_id)
        else:
            usable.append(segment)

    if not usable:
        raise ValueError(
            "No segments are long enough for the requested sequence length."
        )

    return usable, skipped


def build_swat_temporal_datasets(
    root: str = "data/raw/SWaT",
    sequence_length: int = 60,
    train_stride: int = 5,
    evaluation_stride: int = 1,
) -> dict[str, object]:
    """
    Build leakage-safe SWaT temporal datasets.

    Normalization is fitted on training segments only.
    Windows are created independently within each continuous segment.
    Test labels are aligned to the final timestep of each window.
    """
    dataset = load_swat_dataset(root)

    preprocessor = Preprocessor()

    train_segments = dataset["train_segments"]
    validation_segments = dataset["validation_segments"]
    test_episodes = dataset["test_episodes"]

    # Fit normalization on every selected training observation.
    scaled_train_all = preprocessor.fit_transform(train_segments)

    # Only remove segments that cannot produce a single temporal window.
    scaled_train, skipped_train_segments = _filter_short_segments(
        scaled_train_all,
        sequence_length,
    )

    scaled_validation = preprocessor.transform(validation_segments)

    train_dataset = TemporalWindowDataset(
        scaled_train,
        sequence_length=sequence_length,
        stride=train_stride,
    )

    validation_dataset = TemporalWindowDataset(
        scaled_validation,
        sequence_length=sequence_length,
        stride=evaluation_stride,
    )

    scaled_test = []
    test_labels = []

    for episode_df, labels in test_episodes:
        scaled_episode = preprocessor.transform([episode_df])[0]

        if len(scaled_episode) != len(labels):
            raise ValueError(
                "SWaT test episode data/label length mismatch: "
                f"{len(scaled_episode)} != {len(labels)}"
            )

        scaled_test.append(scaled_episode)
        test_labels.append(np.asarray(labels, dtype=np.int64))

    test_datasets = [
        TemporalWindowDataset(
            [episode],
            sequence_length=sequence_length,
            stride=evaluation_stride,
        )
        for episode in scaled_test
    ]

    window_labels = []

    for temporal_dataset, labels in zip(test_datasets, test_labels):
        aligned_labels = np.asarray(
            [
                labels[timestep_index]
                for _, timestep_index in (
                    temporal_dataset.get_target_timestep(i)
                    for i in range(len(temporal_dataset))
                )
            ],
            dtype=np.int64,
        )

        window_labels.append(aligned_labels)

    return {
        "feature_names": dataset["feature_columns"],
        "preprocessor": preprocessor,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_datasets": test_datasets,
        "test_labels": window_labels,
        "skipped_train_segments": skipped_train_segments,
    }
