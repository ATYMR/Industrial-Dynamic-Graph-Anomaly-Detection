from pathlib import Path
import sys

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from swat_gnn.data.temporal_dataset import TemporalWindowDataset


def test_dataset_length():
    data = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)

    dataset = TemporalWindowDataset(
        [data],
        sequence_length=5,
        stride=1,
    )

    assert len(dataset) == 16


def test_window_shape():
    data = np.random.randn(20, 3).astype(np.float32)

    dataset = TemporalWindowDataset(
        [data],
        sequence_length=5,
        stride=1,
    )

    sample = dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (5, 3)


def test_first_window_is_correct():
    data = np.arange(10 * 2, dtype=np.float32).reshape(10, 2)

    dataset = TemporalWindowDataset(
        [data],
        sequence_length=4,
        stride=1,
    )

    expected = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(dataset[0], expected)


def test_windows_are_causal():
    data = np.arange(10, dtype=np.float32).reshape(10, 1)

    dataset = TemporalWindowDataset(
        [data],
        sequence_length=4,
        stride=1,
    )

    assert dataset.get_target_timestep(0) == (0, 3)
    assert dataset.get_target_timestep(1) == (0, 4)
    assert dataset.get_target_timestep(2) == (0, 5)


def test_stride():
    data = np.arange(20, dtype=np.float32).reshape(20, 1)

    dataset = TemporalWindowDataset(
        [data],
        sequence_length=5,
        stride=5,
    )

    assert len(dataset) == 4

    assert dataset.get_target_timestep(0) == (0, 4)
    assert dataset.get_target_timestep(1) == (0, 9)
    assert dataset.get_target_timestep(2) == (0, 14)
    assert dataset.get_target_timestep(3) == (0, 19)


def test_multiple_segments_do_not_cross_boundaries():
    segment_a = np.ones((10, 2), dtype=np.float32)
    segment_b = np.full((10, 2), 9.0, dtype=np.float32)

    dataset = TemporalWindowDataset(
        [segment_a, segment_b],
        sequence_length=5,
        stride=1,
    )

    # 6 windows from each segment.
    assert len(dataset) == 12

    first_b_window = dataset[6]

    assert torch.all(first_b_window == 9.0)

    assert dataset.get_target_timestep(5) == (0, 9)
    assert dataset.get_target_timestep(6) == (1, 4)


def test_feature_count():
    data = np.random.randn(20, 66).astype(np.float32)

    dataset = TemporalWindowDataset(
        [data],
        sequence_length=10,
    )

    assert dataset.num_features == 66


def test_empty_segments_rejected():
    with pytest.raises(ValueError):
        TemporalWindowDataset([])


def test_invalid_sequence_length():
    data = np.random.randn(20, 3).astype(np.float32)

    with pytest.raises(ValueError):
        TemporalWindowDataset(
            [data],
            sequence_length=0,
        )


def test_invalid_stride():
    data = np.random.randn(20, 3).astype(np.float32)

    with pytest.raises(ValueError):
        TemporalWindowDataset(
            [data],
            sequence_length=5,
            stride=0,
        )


def test_short_segment_rejected():
    data = np.random.randn(4, 3).astype(np.float32)

    with pytest.raises(ValueError):
        TemporalWindowDataset(
            [data],
            sequence_length=5,
        )


def test_nan_rejected():
    data = np.random.randn(20, 3).astype(np.float32)
    data[0, 0] = np.nan

    with pytest.raises(ValueError):
        TemporalWindowDataset([data])


def test_infinite_values_rejected():
    data = np.random.randn(20, 3).astype(np.float32)
    data[0, 0] = np.inf

    with pytest.raises(ValueError):
        TemporalWindowDataset([data])


def test_feature_mismatch_rejected():
    segment_a = np.random.randn(20, 3).astype(np.float32)
    segment_b = np.random.randn(20, 4).astype(np.float32)

    with pytest.raises(ValueError):
        TemporalWindowDataset(
            [segment_a, segment_b]
        )