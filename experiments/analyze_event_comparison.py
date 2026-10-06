from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path("data/raw/HAI_Kaggle/hai-23.05")
RESULTS_DIR = Path("results/tables")

LABEL_FILES = [
    "label-test1.csv",
    "label-test2.csv",
]

MODEL_FILES = {
    "static": "static_temporal_graph_scores.npz",
    "dynamic": "dynamic_graph_scores.npz",
}

SEQUENCE_LENGTH = 60
STRIDE = 1


def load_test_label_segments() -> list[np.ndarray]:
    segments = []

    for filename in LABEL_FILES:
        df = pd.read_csv(DATA_ROOT / filename)

        if "label" not in df.columns:
            raise ValueError(f"Missing label column in {filename}")

        segments.append(
            df["label"].astype(int).to_numpy()
        )

    return segments


def create_window_labels(
    label_segments: list[np.ndarray],
) -> np.ndarray:
    window_labels = []

    for labels in label_segments:
        endpoint_indices = np.arange(
            SEQUENCE_LENGTH - 1,
            len(labels),
            STRIDE,
        )

        window_labels.append(
            labels[endpoint_indices]
        )

    return np.concatenate(window_labels)


def find_events(
    labels: np.ndarray,
) -> list[tuple[int, int]]:
    events = []
    in_event = False
    start = None

    for i, label in enumerate(labels):

        if label == 1 and not in_event:
            start = i
            in_event = True

        elif label == 0 and in_event:
            events.append((start, i - 1))
            in_event = False

    if in_event:
        events.append((start, len(labels) - 1))

    return events


def evaluate_event(
    scores: np.ndarray,
    threshold: float,
    start: int,
    end: int,
) -> dict:

    event_scores = scores[start:end + 1]

    positive_indices = np.flatnonzero(
        event_scores >= threshold
    )

    detected = len(positive_indices) > 0

    if detected:
        first_detection = (
            start + positive_indices[0]
        )

        latency = first_detection - start

        max_score = float(
            np.max(event_scores)
        )

        mean_score = float(
            np.mean(event_scores)
        )

        median_score = float(
            np.median(event_scores)
        )

        return {
            "detected": True,
            "latency_seconds": int(latency),
            "max_score": max_score,
            "mean_score": mean_score,
            "median_score": median_score,
        }

    return {
        "detected": False,
        "latency_seconds": None,
        "max_score": float(np.max(event_scores)),
        "mean_score": float(np.mean(event_scores)),
        "median_score": float(np.median(event_scores)),
    }


def main():

    print("=" * 70)
    print("EVENT-LEVEL STATIC VS DYNAMIC COMPARISON")
    print("=" * 70)

    label_segments = load_test_label_segments()
    window_labels = create_window_labels(
        label_segments
    )

    events = find_events(window_labels)

    print(f"\nTotal windows: {len(window_labels):,}")
    print(f"Total attack events: {len(events)}")

    model_scores = {}

    for model_name, filename in MODEL_FILES.items():

        data = np.load(
            RESULTS_DIR / filename
        )

        scores = data["scores"]
        threshold = float(data["threshold"])

        if len(scores) != len(window_labels):
            raise ValueError(
                f"{model_name}: score/label length mismatch"
            )

        model_scores[model_name] = {
            "scores": scores,
            "threshold": threshold,
        }

        print(
            f"{model_name}: "
            f"threshold={threshold:.8f}"
        )

    rows = []

    for event_id, (start, end) in enumerate(
        events,
        start=1,
    ):

        duration = end - start + 1

        static_result = evaluate_event(
            model_scores["static"]["scores"],
            model_scores["static"]["threshold"],
            start,
            end,
        )

        dynamic_result = evaluate_event(
            model_scores["dynamic"]["scores"],
            model_scores["dynamic"]["threshold"],
            start,
            end,
        )

        if (
            static_result["detected"]
            and dynamic_result["detected"]
        ):
            category = "both_detected"

        elif (
            not static_result["detected"]
            and not dynamic_result["detected"]
        ):
            category = "both_missed"

        elif static_result["detected"]:
            category = "static_only"

        else:
            category = "dynamic_only"

        rows.append(
            {
                "event_id": event_id,
                "start_window": start,
                "end_window": end,
                "duration_seconds": duration,

                "category": category,

                "static_detected":
                    static_result["detected"],

                "static_latency_seconds":
                    static_result["latency_seconds"],

                "static_max_score":
                    static_result["max_score"],

                "static_mean_score":
                    static_result["mean_score"],

                "static_median_score":
                    static_result["median_score"],

                "dynamic_detected":
                    dynamic_result["detected"],

                "dynamic_latency_seconds":
                    dynamic_result["latency_seconds"],

                "dynamic_max_score":
                    dynamic_result["max_score"],

                "dynamic_mean_score":
                    dynamic_result["mean_score"],

                "dynamic_median_score":
                    dynamic_result["median_score"],
            }
        )

    df = pd.DataFrame(rows)

    print("\nEvent categories:")
    print(
        df["category"]
        .value_counts()
        .sort_index()
    )

    print("\nDetection rates:")
    print(
        f"Static:  "
        f"{df['static_detected'].sum()}/{len(df)}"
    )

    print(
        f"Dynamic: "
        f"{df['dynamic_detected'].sum()}/{len(df)}"
    )

    output_csv = (
        RESULTS_DIR
        / "event_comparison.csv"
    )

    output_json = (
        RESULTS_DIR
        / "event_comparison.json"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
        )

    print(
        f"\nSaved CSV: {output_csv}"
    )

    print(
        f"Saved JSON: {output_json}"
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()