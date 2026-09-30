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

from swat_gnn.models.isolation_forest import (
    IsolationForestModel,
)


# ============================================================
# Test data helpers
# ============================================================

def make_normal_data(
    n_samples: int = 500,
    n_features: int = 5,
) -> np.ndarray:
    """Generate synthetic normal observations."""

    rng = np.random.default_rng(42)

    return rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_samples, n_features),
    ).astype(np.float32)


def make_anomaly_data(
    n_samples: int = 50,
    n_features: int = 5,
) -> np.ndarray:
    """Generate synthetic anomalous observations."""

    rng = np.random.default_rng(123)

    return rng.normal(
        loc=8.0,
        scale=1.0,
        size=(n_samples, n_features),
    ).astype(np.float32)


# ============================================================
# Tests
# ============================================================

def test_model_creation():

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    assert isinstance(
        model,
        IsolationForestModel,
    )


def test_model_fit():

    X_train = make_normal_data()

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    model.fit(X_train)

    assert len(
        model.model.estimators_
    ) == 50


def test_score_shape():

    X_train = make_normal_data()
    X_test = make_normal_data(100)

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    model.fit(X_train)

    scores = model.score(X_test)

    assert scores.shape == (100,)


def test_score_is_finite():

    X_train = make_normal_data()

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    model.fit(X_train)

    scores = model.score(X_train)

    assert np.isfinite(scores).all()


def test_anomalies_have_higher_average_score():

    X_train = make_normal_data()

    X_normal = make_normal_data(100)
    X_anomaly = make_anomaly_data(100)

    model = IsolationForestModel.create(
        n_estimators=200,
        random_state=42,
    )

    model.fit(X_train)

    normal_scores = model.score(X_normal)
    anomaly_scores = model.score(X_anomaly)

    assert anomaly_scores.mean() > normal_scores.mean()


def test_prediction_shape():

    X_train = make_normal_data()
    X_test = make_normal_data(100)

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    model.fit(X_train)

    scores = model.score(X_test)

    threshold = float(
        np.median(scores)
    )

    predictions = model.predict(
        X_test,
        threshold=threshold,
    )

    assert predictions.shape == (100,)

    assert set(
        np.unique(predictions)
    ).issubset({0, 1})


def test_invalid_input_dimension():

    X_train = make_normal_data()

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    with pytest.raises(ValueError):

        model.fit(
            X_train[0]
        )


def test_nan_input_rejected():

    X_train = make_normal_data()

    X_train[0, 0] = np.nan

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    with pytest.raises(ValueError):

        model.fit(X_train)


def test_infinite_input_rejected():

    X_train = make_normal_data()

    X_train[0, 0] = np.inf

    model = IsolationForestModel.create(
        n_estimators=50,
        random_state=42,
    )

    with pytest.raises(ValueError):

        model.fit(X_train)