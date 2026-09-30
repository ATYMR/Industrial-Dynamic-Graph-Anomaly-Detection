from pathlib import Path
import sys

import numpy as np
import pytest


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


# ============================================================
# Project imports
# ============================================================

from swat_gnn.evaluation.metrics import (
    apply_threshold,
    find_best_threshold,
    compute_binary_metrics,
)


# ============================================================
# Test: threshold application
# ============================================================

def test_apply_threshold():

    scores = np.array(
        [
            0.1,
            0.2,
            0.5,
            0.8,
            1.0,
        ],
        dtype=np.float64,
    )

    predictions = apply_threshold(
        scores,
        threshold=0.5,
    )

    expected = np.array(
        [
            0,
            0,
            1,
            1,
            1,
        ]
    )

    np.testing.assert_array_equal(
        predictions,
        expected,
    )


# ============================================================
# Test: threshold selection
# ============================================================

def test_find_best_threshold():

    scores = np.array(
        [
            0.10,
            0.20,
            0.30,
            0.80,
            0.90,
            1.00,
        ]
    )

    labels = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    threshold, metrics = find_best_threshold(
        scores=scores,
        labels=labels,
    )

    assert isinstance(
        threshold,
        float,
    )

    assert "f1" in metrics

    assert metrics["f1"] == pytest.approx(
        1.0
    )


# ============================================================
# Test: binary metrics
# ============================================================

def test_compute_binary_metrics():

    labels = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
        ]
    )

    predictions = np.array(
        [
            0,
            0,
            1,
            1,
            1,
            0,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.2,
            0.7,
            0.8,
            0.9,
            0.3,
        ]
    )

    metrics = compute_binary_metrics(
        labels=labels,
        predictions=predictions,
        scores=scores,
    )

    expected_keys = {
        "precision",
        "recall",
        "f1",
        "fpr",
        "auroc",
        "auprc",
    }

    assert expected_keys.issubset(
        metrics.keys()
    )

    assert metrics["precision"] == pytest.approx(
        2 / 3
    )

    assert metrics["recall"] == pytest.approx(
        2 / 3
    )

    assert metrics["f1"] == pytest.approx(
        2 / 3
    )


# ============================================================
# Test: perfect predictions
# ============================================================

def test_perfect_predictions():

    labels = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    predictions = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    scores = np.array(
        [
            0.1,
            0.2,
            0.8,
            0.9,
        ]
    )

    metrics = compute_binary_metrics(
        labels=labels,
        predictions=predictions,
        scores=scores,
    )

    assert metrics["precision"] == pytest.approx(
        1.0
    )

    assert metrics["recall"] == pytest.approx(
        1.0
    )

    assert metrics["f1"] == pytest.approx(
        1.0
    )

    assert metrics["fpr"] == pytest.approx(
        0.0
    )

    assert metrics["auroc"] == pytest.approx(
        1.0
    )

    assert metrics["auprc"] == pytest.approx(
        1.0
    )


# ============================================================
# Test: no scores
# ============================================================

def test_metrics_without_scores():

    labels = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    predictions = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    metrics = compute_binary_metrics(
        labels=labels,
        predictions=predictions,
    )

    assert metrics["precision"] == pytest.approx(
        1.0
    )

    assert metrics["recall"] == pytest.approx(
        1.0
    )

    assert metrics["f1"] == pytest.approx(
        1.0
    )

    assert metrics["fpr"] == pytest.approx(
        0.0
    )

    assert np.isnan(
        metrics["auroc"]
    )

    assert np.isnan(
        metrics["auprc"]
    )


# ============================================================
# Test: mismatched lengths
# ============================================================

def test_mismatched_lengths():

    scores = np.array(
        [
            0.1,
            0.2,
            0.3,
        ]
    )

    labels = np.array(
        [
            0,
            1,
        ]
    )

    with pytest.raises(ValueError):

        find_best_threshold(
            scores=scores,
            labels=labels,
        )


# ============================================================
# Test: invalid score values
# ============================================================

def test_invalid_scores():

    scores = np.array(
        [
            0.1,
            np.nan,
            0.3,
        ]
    )

    labels = np.array(
        [
            0,
            1,
            0,
        ]
    )

    with pytest.raises(ValueError):

        find_best_threshold(
            scores=scores,
            labels=labels,
        )


# ============================================================
# Test: invalid labels
# ============================================================

def test_invalid_labels():

    scores = np.array(
        [
            0.1,
            0.2,
            0.3,
        ]
    )

    labels = np.array(
        [
            0,
            1,
            2,
        ]
    )

    with pytest.raises(ValueError):

        find_best_threshold(
            scores=scores,
            labels=labels,
        )


# ============================================================
# Test: invalid predictions
# ============================================================

def test_invalid_predictions():

    labels = np.array(
        [
            0,
            1,
            1,
        ]
    )

    predictions = np.array(
        [
            0,
            1,
            2,
        ]
    )

    with pytest.raises(ValueError):

        compute_binary_metrics(
            labels=labels,
            predictions=predictions,
        )


# ============================================================
# Test: invalid threshold
# ============================================================

def test_invalid_threshold():

    scores = np.array(
        [
            0.1,
            0.2,
            0.3,
        ]
    )

    with pytest.raises(ValueError):

        apply_threshold(
            scores,
            threshold=np.nan,
        )


# ============================================================
# Test: empty scores
# ============================================================

def test_empty_scores():

    scores = np.array(
        [],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):

        apply_threshold(
            scores,
            threshold=0.5,
        )