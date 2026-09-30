from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
    load_test_data,
)

from swat_gnn.models.isolation_forest import (
    IsolationForestModel,
)

from swat_gnn.evaluation.metrics import (
    apply_threshold,
    compute_binary_metrics,
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

# The primary operating point is chosen BEFORE looking at
# test labels.
#
# 99th percentile means that approximately the highest 1%
# of normal validation anomaly scores will be classified
# as anomalous.
PRIMARY_PERCENTILE = 99.0

# Additional predefined percentiles are reported only as
# threshold sensitivity information.
#
# They are NOT used to select the final model based on
# test performance.
THRESHOLD_PERCENTILES = [
    95.0,
    97.0,
    98.0,
    99.0,
    99.5,
]


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_variable_features(
    training_segments: list[pd.DataFrame],
) -> list[str]:
    """
    Identify features that vary across the complete training set.

    A feature is considered constant if it has only one unique
    value across every training segment.

    Constant features are excluded from the Isolation Forest
    baseline because they provide no information for anomaly
    detection in the training distribution.
    """

    if not training_segments:
        raise ValueError(
            "No training segments were provided."
        )

    feature_names = [
        column
        for column in training_segments[0].columns
        if column != "timestamp"
    ]

    variable_features: list[str] = []

    for feature in feature_names:

        is_constant = all(
            df[feature].nunique(
                dropna=False
            ) <= 1
            for df in training_segments
        )

        if not is_constant:
            variable_features.append(feature)

    return variable_features


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    segments: list[pd.DataFrame],
    feature_names: list[str],
) -> list[np.ndarray]:
    """
    Extract selected features from each dataframe segment.

    Each returned array has shape:

        (timesteps, features)
    """

    if not segments:
        raise ValueError(
            "No dataframe segments were provided."
        )

    if not feature_names:
        raise ValueError(
            "No feature names were provided."
        )

    arrays: list[np.ndarray] = []

    for segment_index, df in enumerate(segments):

        missing_features = [
            feature
            for feature in feature_names
            if feature not in df.columns
        ]

        if missing_features:
            raise ValueError(
                f"Segment {segment_index} is missing "
                f"features: {missing_features}"
            )

        values = df[
            feature_names
        ].to_numpy(
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Segment {segment_index} contains "
                "NaN or infinite values."
            )

        arrays.append(values)

    return arrays


# ============================================================
# CONCATENATE SEGMENTS
# ============================================================

def concatenate_segments(
    segments: list[np.ndarray],
) -> np.ndarray:
    """
    Concatenate chronological dataset segments.

    This function is used only after the segments have already
    been independently loaded. Windows are NOT created here,
    so there is no possibility of creating a temporal window
    across a file boundary.
    """

    if not segments:
        raise ValueError(
            "No arrays were provided."
        )

    return np.concatenate(
        segments,
        axis=0,
    )


# ============================================================
# NORMAL-ONLY VALIDATION THRESHOLD
# ============================================================

def select_validation_threshold(
    validation_scores: np.ndarray,
    percentile: float,
) -> tuple[float, float]:
    """
    Select an anomaly threshold using only normal validation data.

    Parameters
    ----------
    validation_scores:
        Anomaly scores from the normal validation segment.

    percentile:
        Percentile used to determine the threshold.

    Returns
    -------
    threshold:
        Anomaly-score threshold.

    validation_flag_rate:
        Fraction of validation observations whose scores exceed
        the threshold.
    """

    validation_scores = np.asarray(
        validation_scores,
        dtype=np.float64,
    )

    if validation_scores.ndim != 1:
        raise ValueError(
            "validation_scores must be a 1D array."
        )

    if validation_scores.size == 0:
        raise ValueError(
            "validation_scores cannot be empty."
        )

    if not np.isfinite(
        validation_scores
    ).all():
        raise ValueError(
            "validation_scores contain NaN or infinite values."
        )

    if not 0.0 < percentile < 100.0:
        raise ValueError(
            "percentile must be greater than 0 and less than 100."
        )

    threshold = float(
        np.percentile(
            validation_scores,
            percentile,
        )
    )

    validation_predictions = apply_threshold(
        validation_scores,
        threshold,
    )

    validation_flag_rate = float(
        validation_predictions.mean()
    )

    return (
        threshold,
        validation_flag_rate,
    )


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    training_segments: list[pd.DataFrame],
    validation_segments: list[pd.DataFrame],
    test_segments: list[pd.DataFrame],
    feature_names: list[str],
    X_train: np.ndarray,
    X_validation: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Print a concise summary of the experiment data.
    """

    print()
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"Training segments       : {len(training_segments)}"
    )

    print(
        f"Validation segments     : {len(validation_segments)}"
    )

    print(
        f"Test segments           : {len(test_segments)}"
    )

    print(
        f"Model features          : {len(feature_names)}"
    )

    print(
        f"Training observations   : {X_train.shape}"
    )

    print(
        f"Validation observations : {X_validation.shape}"
    )

    print(
        f"Test observations       : {X_test.shape}"
    )

    print(
        f"Test attack observations: {int(y_test.sum())}"
    )

    print(
        f"Test normal observations: "
        f"{int((y_test == 0).sum())}"
    )

    print("=" * 70)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # 1. Load configuration
    # --------------------------------------------------------

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "data.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    config = load_config(
        str(config_path)
    )

    print()
    print("=" * 70)
    print("HAI 23.05 — Isolation Forest Baseline")
    print("=" * 70)

    # --------------------------------------------------------
    # 2. Load datasets
    # --------------------------------------------------------

    print()
    print("Loading data...")

    training_segments = load_training_data(
        config
    )

    validation_segments = load_validation_data(
        config
    )

    test_segments, test_labels = load_test_data(
        config
    )

    # --------------------------------------------------------
    # 3. Identify variable features
    #
    # This decision uses TRAINING DATA ONLY.
    # --------------------------------------------------------

    feature_names = get_variable_features(
        training_segments
    )

    print(
        f"Total model features: {len(feature_names)}"
    )

    # --------------------------------------------------------
    # 4. Extract selected features
    # --------------------------------------------------------

    X_train_segments = extract_features(
        training_segments,
        feature_names,
    )

    X_validation_segments = extract_features(
        validation_segments,
        feature_names,
    )

    X_test_segments = extract_features(
        test_segments,
        feature_names,
    )

    # --------------------------------------------------------
    # 5. Concatenate segments
    #
    # Important:
    # This does NOT create temporal windows.
    #
    # Each CSV remains a separate temporal segment during
    # preprocessing. Concatenation here is only for the
    # point-wise Isolation Forest.
    # --------------------------------------------------------

    X_train = concatenate_segments(
        X_train_segments
    )

    X_validation = concatenate_segments(
        X_validation_segments
    )

    X_test = concatenate_segments(
        X_test_segments
    )

    # --------------------------------------------------------
    # 6. Concatenate test labels
    # --------------------------------------------------------

    y_test = np.concatenate(
        [
            labels.to_numpy(
                dtype=np.int64
            )
            for labels in test_labels
        ],
        axis=0,
    )

    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            "Test feature count does not match "
            "test label count."
        )

    # --------------------------------------------------------
    # 7. Print dataset summary
    # --------------------------------------------------------

    print_dataset_summary(
        training_segments=training_segments,
        validation_segments=validation_segments,
        test_segments=test_segments,
        feature_names=feature_names,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_test=y_test,
    )

    # --------------------------------------------------------
    # 8. Fit StandardScaler
    #
    # CRITICAL:
    # The scaler sees ONLY X_train.
    #
    # X_validation and X_test are never used to calculate
    # means or standard deviations.
    # --------------------------------------------------------

    print()
    print("Fitting StandardScaler on training data...")

    scaler = StandardScaler()

    scaler.fit(
        X_train
    )

    print(
        "StandardScaler fitted."
    )

    # --------------------------------------------------------
    # 9. Transform datasets
    # --------------------------------------------------------

    print(
        "Transforming training data..."
    )

    X_train_scaled = scaler.transform(
        X_train
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "Transforming validation data..."
    )

    X_validation_scaled = scaler.transform(
        X_validation
    ).astype(
        np.float32,
        copy=False,
    )

    print(
        "Transforming test data..."
    )

    X_test_scaled = scaler.transform(
        X_test
    ).astype(
        np.float32,
        copy=False,
    )

    # --------------------------------------------------------
    # 10. Verify preprocessing
    # --------------------------------------------------------

    if not np.isfinite(
        X_train_scaled
    ).all():
        raise ValueError(
            "Scaled training data contains "
            "NaN or infinite values."
        )

    if not np.isfinite(
        X_validation_scaled
    ).all():
        raise ValueError(
            "Scaled validation data contains "
            "NaN or infinite values."
        )

    if not np.isfinite(
        X_test_scaled
    ).all():
        raise ValueError(
            "Scaled test data contains "
            "NaN or infinite values."
        )

    print(
        "Standardization complete."
    )

    # --------------------------------------------------------
    # 11. Train Isolation Forest
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING ISOLATION FOREST")
    print("=" * 70)

    print(
        f"Number of trees : {N_ESTIMATORS}"
    )

    print(
        f"Random state    : {RANDOM_STATE}"
    )

    print(
        f"Training shape  : {X_train_scaled.shape}"
    )

    print()

    model = IsolationForestModel.create(
        n_estimators=N_ESTIMATORS,
        max_samples=256,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train_scaled
    )

    print()
    print(
        "Isolation Forest training complete."
    )

    # --------------------------------------------------------
    # 12. Score validation data
    #
    # Validation contains NORMAL data only.
    # --------------------------------------------------------

    print()
    print(
        "Scoring normal validation data..."
    )

    validation_scores = model.score(
        X_validation_scaled
    )

    print(
        "Validation scoring complete."
    )

    # --------------------------------------------------------
    # 13. Evaluate predefined threshold percentiles
    #
    # This is NOT test-set model selection.
    #
    # We simply characterize the operating point on normal
    # validation data.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold_table: list[dict[str, float]] = []

    for percentile in THRESHOLD_PERCENTILES:

        threshold, validation_flag_rate = (
            select_validation_threshold(
                validation_scores=validation_scores,
                percentile=percentile,
            )
        )

        threshold_table.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "validation_flag_rate": (
                    validation_flag_rate
                ),
            }
        )

        print(
            f"{percentile:6.1f}th percentile"
            f" | threshold = {threshold:.6f}"
            f" | validation flag rate = "
            f"{validation_flag_rate:.6%}"
        )

    # --------------------------------------------------------
    # 14. Freeze primary threshold
    #
    # The primary threshold is predefined as the 99th
    # percentile of NORMAL validation scores.
    #
    # We do NOT inspect test performance to choose it.
    # --------------------------------------------------------

    selected_percentile = PRIMARY_PERCENTILE

    threshold, validation_flag_rate = (
        select_validation_threshold(
            validation_scores=validation_scores,
            percentile=selected_percentile,
        )
    )

    print()
    print(
        f"Primary percentile: "
        f"{selected_percentile:.1f}"
    )

    print(
        f"Frozen threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation flag rate: "
        f"{validation_flag_rate:.6%}"
    )

    # --------------------------------------------------------
    # 15. Score TEST data
    #
    # The threshold is already frozen.
    # --------------------------------------------------------

    print()
    print(
        "Scoring test data..."
    )

    test_scores = model.score(
        X_test_scaled
    )

    print(
        "Test scoring complete."
    )

    # --------------------------------------------------------
    # 16. Apply frozen threshold
    # --------------------------------------------------------

    test_predictions = apply_threshold(
        test_scores,
        threshold,
    )

    # --------------------------------------------------------
    # 17. Compute final test metrics
    # --------------------------------------------------------

    metrics = compute_binary_metrics(
        labels=y_test,
        predictions=test_predictions,
        scores=test_scores,
    )

    # --------------------------------------------------------
    # 18. Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Precision : {metrics['precision']:.6f}"
    )

    print(
        f"Recall    : {metrics['recall']:.6f}"
    )

    print(
        f"F1        : {metrics['f1']:.6f}"
    )

    print(
        f"FPR       : {metrics['fpr']:.6f}"
    )

    print(
        f"AUROC     : {metrics['auroc']:.6f}"
    )

    print(
        f"AUPRC     : {metrics['auprc']:.6f}"
    )

    print()

    print(
        f"Predicted anomalies : "
        f"{int(test_predictions.sum())}"
    )

    print(
        f"Actual anomalies    : "
        f"{int(y_test.sum())}"
    )

    print(
        f"Total test samples  : "
        f"{len(y_test)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 19. Save results
    # --------------------------------------------------------

    results_dir = (
        PROJECT_ROOT
        / "results"
        / "tables"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Primary experiment result
    result_row = {
        "dataset": "HAI_23.05",
        "model": "Isolation Forest",
        "n_features": len(feature_names),
        "n_estimators": N_ESTIMATORS,
        "max_samples": 256,
        "random_state": RANDOM_STATE,
        "threshold_percentile": selected_percentile,
        "threshold": threshold,
        "validation_flag_rate": validation_flag_rate,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "fpr": metrics["fpr"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "test_samples": len(y_test),
        "test_attack_samples": int(y_test.sum()),
        "predicted_anomalies": int(
            test_predictions.sum()
        ),
    }

    results_df = pd.DataFrame(
        [result_row]
    )

    results_path = (
        results_dir
        / "isolation_forest_baseline.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # Threshold sensitivity table
    threshold_df = pd.DataFrame(
        threshold_table
    )

    threshold_path = (
        results_dir
        / "isolation_forest_thresholds.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    print()
    print(
        f"Primary results saved to:"
    )

    print(
        f"  {results_path}"
    )

    print(
        f"Threshold analysis saved to:"
    )

    print(
        f"  {threshold_path}"
    )

    print()
    print(
        "Experiment completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()