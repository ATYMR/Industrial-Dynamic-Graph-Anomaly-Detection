from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Preprocessor:
    scaler: StandardScaler | None = None
    feature_names: list[str] | None = None

    def fit(self, training_segments: list[pd.DataFrame]) -> "Preprocessor":
        """
        Fit preprocessing parameters using training data only.
        """
        if not training_segments:
            raise ValueError("No training segments provided.")

        self.feature_names = [
            c for c in training_segments[0].columns
            if c != "timestamp"
        ]

        # Ensure every segment has the same feature set.
        for df in training_segments:
            if list(
                c for c in df.columns if c != "timestamp"
            ) != self.feature_names:
                raise ValueError("Feature mismatch between training segments.")

        # Fit on normal training data only.
        training_values = np.concatenate(
            [
                df[self.feature_names].to_numpy(dtype=np.float64)
                for df in training_segments
            ],
            axis=0,
        )

        self.scaler = StandardScaler()
        self.scaler.fit(training_values)

        return self

    def transform(self, segments: list[pd.DataFrame]) -> list[np.ndarray]:
        """
        Transform data using parameters learned from training data.
        """
        if self.scaler is None or self.feature_names is None:
            raise RuntimeError("Preprocessor must be fitted first.")

        transformed = []

        for df in segments:
            missing = set(self.feature_names) - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing features during transformation: {sorted(missing)}"
                )

            values = df[self.feature_names].to_numpy(dtype=np.float64)
            transformed.append(self.scaler.transform(values))

        return transformed

    def fit_transform(
        self,
        training_segments: list[pd.DataFrame],
    ) -> list[np.ndarray]:
        """Fit on training data and transform it."""
        self.fit(training_segments)
        return self.transform(training_segments)
    