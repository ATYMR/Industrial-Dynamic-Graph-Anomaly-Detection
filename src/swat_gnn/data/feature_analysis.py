from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_features(
    training_segments: list[pd.DataFrame],
) -> dict:
    """
    Analyze feature variability across all training segments.

    The analysis is performed on training data only.
    """

    if not training_segments:
        raise ValueError("No training segments provided.")

    feature_names = [
        column
        for column in training_segments[0].columns
        if column != "timestamp"
    ]

    # Ensure all training segments have identical features.
    for df in training_segments:
        current_features = [
            column
            for column in df.columns
            if column != "timestamp"
        ]

        if current_features != feature_names:
            raise ValueError(
                "Feature mismatch between training segments."
            )

    # Combine training data for the audit.
    values = np.concatenate(
        [
            df[feature_names].to_numpy(dtype=np.float64)
            for df in training_segments
        ],
        axis=0,
    )

    feature_std = values.std(axis=0)
    feature_min = values.min(axis=0)
    feature_max = values.max(axis=0)

    constant_mask = feature_std == 0

    constant_features = [
        feature_names[i]
        for i, is_constant in enumerate(constant_mask)
        if is_constant
    ]

    variable_features = [
        feature_names[i]
        for i, is_constant in enumerate(constant_mask)
        if not is_constant
    ]

    statistics = pd.DataFrame(
        {
            "feature": feature_names,
            "std": feature_std,
            "min": feature_min,
            "max": feature_max,
            "constant": constant_mask,
        }
    )

    return {
        "feature_names": feature_names,
        "constant_features": constant_features,
        "variable_features": variable_features,
        "statistics": statistics,
    }