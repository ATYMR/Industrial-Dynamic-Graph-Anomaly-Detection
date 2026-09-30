from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validate_1d_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert score/label arrays.

    Parameters
    ----------
    scores:
        One-dimensional anomaly scores.

    labels:
        One-dimensional binary labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated scores and labels.
    """

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if scores.ndim != 1:
        raise ValueError(
            "scores must be a 1D array."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a 1D array."
        )

    if len(scores) != len(labels):
        raise ValueError(
            "scores and labels must have the same length."
        )

    if len(scores) == 0:
        raise ValueError(
            "scores and labels cannot be empty."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "scores contain NaN or infinite values."
        )

    unique_labels = np.unique(labels)

    if not np.isin(
        unique_labels,
        [0, 1],
    ).all():
        raise ValueError(
            "labels must contain only 0 and 1."
        )

    return (
        scores.astype(np.float64, copy=False),
        labels.astype(np.int64, copy=False),
    )


def apply_threshold(
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert anomaly scores into binary predictions.

    Parameters
    ----------
    scores:
        One-dimensional anomaly scores.
        Higher values indicate more anomalous observations.

    threshold:
        Decision threshold.

    Returns
    -------
    np.ndarray
        Binary predictions:

            0 = normal
            1 = anomaly
    """

    scores = np.asarray(scores)

    if scores.ndim != 1:
        raise ValueError(
            "scores must be a 1D array."
        )

    if len(scores) == 0:
        raise ValueError(
            "scores cannot be empty."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "scores contain NaN or infinite values."
        )

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite."
        )

    return (
        scores >= threshold
    ).astype(np.int64)


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Select an anomaly threshold using validation data.

    The threshold maximizes F1 on the supplied validation set.

    IMPORTANT
    ---------
    This function should be called only on the validation set
    during the actual experiment.

    The test set must never be used to select this threshold.

    Parameters
    ----------
    scores:
        Validation anomaly scores.

    labels:
        Validation binary labels.

    Returns
    -------
    tuple
        Selected threshold and corresponding validation metrics.
    """

    scores, labels = _validate_1d_arrays(
        scores,
        labels,
    )

    # Sort unique observed scores.
    candidate_thresholds = np.unique(
        scores
    )

    best_threshold = float(
        candidate_thresholds[0]
    )

    best_f1 = -1.0

    best_predictions = None

    for threshold in candidate_thresholds:

        predictions = apply_threshold(
            scores,
            float(threshold),
        )

        current_f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        if current_f1 > best_f1:

            best_f1 = float(
                current_f1
            )

            best_threshold = float(
                threshold
            )

            best_predictions = predictions

    # This should never happen because candidate_thresholds
    # contains at least one value, but keep the check explicit.
    if best_predictions is None:
        raise RuntimeError(
            "Failed to select a threshold."
        )

    validation_metrics = compute_binary_metrics(
        labels=labels,
        predictions=best_predictions,
        scores=scores,
    )

    return (
        best_threshold,
        validation_metrics,
    )


def compute_binary_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute binary anomaly-detection metrics.

    Parameters
    ----------
    labels:
        Ground-truth binary labels.

    predictions:
        Binary predictions.

    scores:
        Continuous anomaly scores.

        Required for AUROC and AUPRC.

    Returns
    -------
    dict[str, float]
        Precision, recall, F1, FPR, AUROC and AUPRC.
    """

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a 1D array."
        )

    if predictions.ndim != 1:
        raise ValueError(
            "predictions must be a 1D array."
        )

    if len(labels) != len(predictions):
        raise ValueError(
            "labels and predictions must have "
            "the same length."
        )

    if len(labels) == 0:
        raise ValueError(
            "labels and predictions cannot be empty."
        )

    if not np.isin(
        np.unique(labels),
        [0, 1],
    ).all():
        raise ValueError(
            "labels must contain only 0 and 1."
        )

    if not np.isin(
        np.unique(predictions),
        [0, 1],
    ).all():
        raise ValueError(
            "predictions must contain only 0 and 1."
        )

    labels = labels.astype(
        np.int64,
        copy=False,
    )

    predictions = predictions.astype(
        np.int64,
        copy=False,
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # False positive rate
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    if tn + fp > 0:
        fpr = fp / (tn + fp)
    else:
        fpr = 0.0

    # --------------------------------------------------------
    # AUROC and AUPRC
    # --------------------------------------------------------

    auroc = float("nan")
    auprc = float("nan")

    if scores is not None:

        scores = np.asarray(scores)

        if scores.ndim != 1:
            raise ValueError(
                "scores must be a 1D array."
            )

        if len(scores) != len(labels):
            raise ValueError(
                "scores and labels must have "
                "the same length."
            )

        if not np.isfinite(scores).all():
            raise ValueError(
                "scores contain NaN or infinite values."
            )

        # AUROC/AUPRC are undefined when the evaluation set
        # contains only one class.
        if len(np.unique(labels)) == 2:

            auroc = float(
                roc_auc_score(
                    labels,
                    scores,
                )
            )

            auprc = float(
                average_precision_score(
                    labels,
                    scores,
                )
            )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "auroc": auroc,
        "auprc": auprc,
    }