from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("data/raw/HAI_Kaggle/hai-23.05")

LABEL_FILES = [
    "label-test1.csv",
    "label-test2.csv",
]

RESULTS_DIR = Path("results/tables")

MODEL_FILES = {
    "static_temporal_graph":
        "static_temporal_graph_scores.npz",

    "dynamic_graph":
        "dynamic_graph_scores.npz",
}

SEQUENCE_LENGTH = 60
STRIDE = 1


# ============================================================
# Load test-label segments
# ============================================================

def load_test_label_segments() -> list[np.ndarray]:
    """
    Load each HAI test label file separately.

    We must preserve test-file boundaries because the model
    never creates temporal windows across file boundaries.
    """

    label_segments = []

    for filename in LABEL_FILES:

        path = DATA_ROOT / filename

        df = pd.read_csv(path)

        if "label" not in df.columns:
            raise ValueError(
                f"Missing 'label' column in {path}"
            )

        labels = (
            df["label"]
            .astype(int)
            .to_numpy()
        )

        label_segments.append(labels)

    return label_segments


# ============================================================
# Create window-endpoint labels
# ============================================================

def create_window_labels_from_segments(
    label_segments: list[np.ndarray],
    sequence_length: int = 60,
    stride: int = 1,
) -> np.ndarray:
    """
    Convert timestep labels into labels corresponding to the
    endpoint of each temporal window.

    Each test segment is processed independently so that no
    window crosses a test-file boundary.
    """

    if not label_segments:
        raise ValueError(
            "No label segments provided."
        )

    window_labels = []

    for segment_id, labels in enumerate(
        label_segments
    ):

        if labels.ndim != 1:
            raise ValueError(
                f"Expected 1D labels for segment "
                f"{segment_id}, got shape {labels.shape}"
            )

        if len(labels) < sequence_length:
            raise ValueError(
                f"Segment {segment_id} has "
                f"{len(labels)} timesteps, but "
                f"sequence_length={sequence_length}."
            )

        endpoint_indices = np.arange(
            sequence_length - 1,
            len(labels),
            stride,
        )

        segment_window_labels = (
            labels[endpoint_indices]
        )

        window_labels.append(
            segment_window_labels
        )

    return np.concatenate(
        window_labels
    )


# ============================================================
# Find contiguous attack events
# ============================================================

def find_events(
    labels: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Find contiguous positive intervals.

    Returns:
        [(start_index, end_index), ...]
    """

    events = []

    in_event = False
    start = None

    for i, label in enumerate(labels):

        if label == 1 and not in_event:

            start = i
            in_event = True

        elif label == 0 and in_event:

            events.append(
                (start, i - 1)
            )

            in_event = False

    if in_event:

        events.append(
            (start, len(labels) - 1)
        )

    return events


# ============================================================
# Event-level evaluation
# ============================================================

def evaluate_events(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """
    Evaluate whether each ground-truth attack event was
    detected.

    Detection latency is measured from the beginning of the
    attack event to the first predicted anomaly inside that
    event.

    Since HAI measurements are sampled once per second,
    one index corresponds to approximately one second.
    """

    true_events = find_events(labels)

    detected_events = 0

    detection_latencies = []

    missed_events = []

    for start, end in true_events:

        event_predictions = predictions[
            start:end + 1
        ]

        positive_indices = np.flatnonzero(
            event_predictions == 1
        )

        if len(positive_indices) == 0:

            missed_events.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "duration_seconds": int(
                        end - start + 1
                    ),
                }
            )

        else:

            detected_events += 1

            first_detection = (
                start + positive_indices[0]
            )

            latency = (
                first_detection - start
            )

            detection_latencies.append(
                int(latency)
            )

    total_events = len(true_events)

    detection_rate = (
        detected_events / total_events
        if total_events > 0
        else 0.0
    )

    if detection_latencies:

        mean_latency = float(
            np.mean(
                detection_latencies
            )
        )

        median_latency = float(
            np.median(
                detection_latencies
            )
        )

        max_latency = int(
            np.max(
                detection_latencies
            )
        )

        min_latency = int(
            np.min(
                detection_latencies
            )
        )

    else:

        mean_latency = None
        median_latency = None
        max_latency = None
        min_latency = None

    return {
        "total_events":
            total_events,

        "detected_events":
            detected_events,

        "missed_events":
            len(missed_events),

        "event_detection_rate":
            detection_rate,

        "mean_detection_latency_seconds":
            mean_latency,

        "median_detection_latency_seconds":
            median_latency,

        "min_detection_latency_seconds":
            min_latency,

        "max_detection_latency_seconds":
            max_latency,

        "detection_latencies_seconds":
            detection_latencies,

        "missed_event_intervals":
            missed_events,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("EVENT-LEVEL ANOMALY DETECTION ANALYSIS")
    print("=" * 70)

    # ========================================================
    # Load labels
    # ========================================================

    label_segments = (
        load_test_label_segments()
    )

    raw_timesteps = sum(
        len(labels)
        for labels in label_segments
    )

    attack_timesteps = sum(
        int(labels.sum())
        for labels in label_segments
    )

    print(
        f"\nRaw test timesteps: "
        f"{raw_timesteps:,}"
    )

    print(
        f"Attack timesteps: "
        f"{attack_timesteps:,}"
    )

    for i, labels in enumerate(
        label_segments,
        start=1,
    ):

        print(
            f"Test segment {i}: "
            f"{len(labels):,} timesteps, "
            f"{int(labels.sum()):,} attack timesteps"
        )

    # ========================================================
    # Convert timestep labels to window labels
    # ========================================================

    window_labels = (
        create_window_labels_from_segments(
            label_segments,
            sequence_length=SEQUENCE_LENGTH,
            stride=STRIDE,
        )
    )

    print(
        f"\nWindow labels: "
        f"{len(window_labels):,}"
    )

    # ========================================================
    # Find attack events
    # ========================================================

    true_events = find_events(
        window_labels
    )

    print(
        f"Attack events: "
        f"{len(true_events)}"
    )

    # ========================================================
    # Evaluate each model
    # ========================================================

    all_results = {}

    for model_name, filename in (
        MODEL_FILES.items()
    ):

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Model: {model_name}"
        )

        score_path = (
            RESULTS_DIR / filename
        )

        if not score_path.exists():

            print(
                f"Score file not found:"
            )

            print(score_path)

            continue

        # ----------------------------------------------------
        # Load saved scores
        # ----------------------------------------------------

        data = np.load(
            score_path
        )

        if "scores" not in data:

            raise ValueError(
                f"'scores' missing from "
                f"{score_path}"
            )

        if "threshold" not in data:

            raise ValueError(
                f"'threshold' missing from "
                f"{score_path}"
            )

        scores = data["scores"]

        threshold = float(
            data["threshold"]
        )

        print(
            f"Score count: "
            f"{len(scores):,}"
        )

        print(
            f"Threshold: "
            f"{threshold:.8f}"
        )

        # ----------------------------------------------------
        # Verify exact alignment
        # ----------------------------------------------------

        if len(scores) != len(
            window_labels
        ):

            raise ValueError(
                f"Length mismatch for "
                f"{model_name}: "
                f"scores={len(scores)}, "
                f"labels={len(window_labels)}"
            )

        # ----------------------------------------------------
        # Convert scores to predictions
        # ----------------------------------------------------

        predictions = (
            scores >= threshold
        ).astype(np.int64)

        print(
            f"Predicted anomalies: "
            f"{int(predictions.sum()):,}"
        )

        # ----------------------------------------------------
        # Event evaluation
        # ----------------------------------------------------

        result = evaluate_events(
            window_labels,
            predictions,
        )

        all_results[
            model_name
        ] = result

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Total events: "
            f"{result['total_events']}"
        )

        print(
            f"Detected events: "
            f"{result['detected_events']}"
        )

        print(
            f"Missed events: "
            f"{result['missed_events']}"
        )

        print(
            f"Event detection rate: "
            f"{result['event_detection_rate']:.4f}"
        )

        print(
            "Mean detection latency: "
            f"{result['mean_detection_latency_seconds']} s"
        )

        print(
            "Median detection latency: "
            f"{result['median_detection_latency_seconds']} s"
        )

        print(
            "Minimum detection latency: "
            f"{result['min_detection_latency_seconds']} s"
        )

        print(
            "Maximum detection latency: "
            f"{result['max_detection_latency_seconds']} s"
        )

    # ========================================================
    # Save results
    # ========================================================

    output_path = (
        RESULTS_DIR
        / "event_detection_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Saved event-level results:"
    )

    print(
        output_path
    )

    print(
        "\nAnalysis complete."
    )


if __name__ == "__main__":
    main()