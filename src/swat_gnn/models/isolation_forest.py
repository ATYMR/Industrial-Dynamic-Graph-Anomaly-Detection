from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestModel:
    """
    Wrapper around sklearn's IsolationForest.

    The model is trained on normal training observations only.
    """

    model: IsolationForest

    @classmethod
    def create(
        cls,
        n_estimators: int = 300,
        max_samples: str | int | float = "auto",
        contamination: str | float = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> "IsolationForestModel":
        """
        Create an Isolation Forest model.

        Parameters
        ----------
        n_estimators:
            Number of trees.

        max_samples:
            Number of samples used to train each tree.

        contamination:
            Kept as 'auto' because the final anomaly threshold
            will be selected separately using validation data.

        random_state:
            Random seed for reproducibility.

        n_jobs:
            Number of CPU workers. -1 uses all available cores.
        """

        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        return cls(model=model)

    def fit(
        self,
        X_train: np.ndarray,
    ) -> "IsolationForestModel":
        """
        Fit the model using normal training observations.

        X_train shape:
            (n_samples, n_features)
        """

        X_train = self._validate_input(X_train)

        self.model.fit(X_train)

        return self

    def score(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return anomaly scores.

        Higher values mean more anomalous.

        sklearn's decision_function has the opposite convention,
        so its sign is inverted here.
        """

        X = self._validate_input(X)

        # sklearn:
        #     larger = more normal
        #
        # Our convention:
        #     larger = more anomalous
        #
        return -self.model.decision_function(X)

    def predict(
        self,
        X: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Convert anomaly scores into binary predictions.

        Returns
        -------
        np.ndarray
            0 = normal
            1 = anomaly
        """

        scores = self.score(X)

        return (scores >= threshold).astype(np.int64)

    @staticmethod
    def _validate_input(
        X: np.ndarray,
    ) -> np.ndarray:
        """Validate model input."""

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "Expected a 2D array with shape "
                "(n_samples, n_features)."
            )

        if X.shape[0] == 0:
            raise ValueError(
                "Input contains zero samples."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Input contains NaN or infinite values."
            )

        return X.astype(
            np.float32,
            copy=False,
        )