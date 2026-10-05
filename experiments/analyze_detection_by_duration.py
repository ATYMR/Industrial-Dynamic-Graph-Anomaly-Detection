from pathlib import Path
import json
import numpy as np

SEQ_LEN = 60
RESULTS_DIR = Path("results/tables")

LABEL_FILES = [
    Path("data/raw/HAI_Kaggle/hai-23.05/label-test1.csv"),
    Path("data/raw/HAI_Kaggle/hai-23.05/label-test2.csv"),
]

SCORE_FILES = {
    "static": RESULTS_DIR / "static_temporal_graph_scores.npz",
    "dynamic": RESULTS_DIR / "dynamic_graph_scores.npz",
}

BINS = [
    ("<60s", 0, 59),
    ("60-119s", 60, 119),
    ("120-299s", 120, 299),
    ("300s+", 300, float("inf")),
]


def load_labels(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return (data[:, -1] > 0).astype(int)


def find_events(labels):
    events = []
    start = None

    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            events.append((start, i - 1))
            start = None

    if start is not None:
        events.append((start, len(labels) - 1))

    return events


def load_all_events():
    segments = []

    for label_file in LABEL_FILES:
        labels = load_labels(label_file)
        events = find_events(labels)
        segments.append((events, len(labels)))

    return segments


def evaluate_model(model_name, segments):
    data = np.load(SCORE_FILES[model_name])
    scores = data["scores"]
    threshold = float(data["threshold"])
    predictions = scores >= threshold

    results = []
    score_offset = 0

    for segment_id, (events, label_length) in enumerate(segments):
        score_length = label_length - SEQ_LEN + 1

        for raw_start, raw_end in events:
            duration = raw_end - raw_start + 1

            local_start = max(0, raw_start - (SEQ_LEN - 1))
            local_end = min(
                score_length - 1,
                raw_end - (SEQ_LEN - 1),
            )

            global_start = score_offset + local_start
            global_end = score_offset + local_end

            detected = bool(
                np.any(predictions[global_start:global_end + 1])
            )

            results.append({
                "segment": segment_id + 1,
                "start": raw_start,
                "end": raw_end,
                "duration_s": duration,
                "detected": detected,
            })

        score_offset += score_length

    return results


def duration_bin(duration):
    for name, low, high in BINS:
        if low <= duration <= high:
            return name
    raise ValueError(duration)


segments = load_all_events()

all_results = {}

for model_name in SCORE_FILES:
    events = evaluate_model(model_name, segments)

    summary = []

    for bin_name, _, _ in BINS:
        bin_events = [
            e for e in events
            if duration_bin(e["duration_s"]) == bin_name
        ]

        detected = sum(e["detected"] for e in bin_events)
        missed = len(bin_events) - detected

        summary.append({
            "duration_bin": bin_name,
            "total_events": len(bin_events),
            "detected": detected,
            "missed": missed,
            "detection_rate": (
                detected / len(bin_events)
                if bin_events else None
            ),
        })

    all_results[model_name] = {
        "total_events": len(events),
        "bins": summary,
    }

    print(f"\n{model_name.upper()}")

    for row in summary:
        print(
            f"{row['duration_bin']:>8} | "
            f"n={row['total_events']:2d} | "
            f"detected={row['detected']:2d} | "
            f"missed={row['missed']:2d} | "
            f"rate={row['detection_rate']:.3f}"
            if row["total_events"]
            else
            f"{row['duration_bin']:>8} | n=0"
        )

output = RESULTS_DIR / "event_detection_by_duration.json"

with open(output, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: {output}")
