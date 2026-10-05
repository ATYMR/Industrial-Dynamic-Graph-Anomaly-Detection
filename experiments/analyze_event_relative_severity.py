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


segments = []

for label_file in LABEL_FILES:
    labels = load_labels(label_file)
    segments.append((find_events(labels), len(labels)))


all_results = {}

for model_name, score_file in SCORE_FILES.items():
    data = np.load(score_file)
    scores = data["scores"]
    threshold = float(data["threshold"])

    model_events = []
    score_offset = 0

    for segment_id, (events, label_length) in enumerate(segments):
        score_length = label_length - SEQ_LEN + 1

        for event_id, (raw_start, raw_end) in enumerate(events, 1):
            local_start = max(0, raw_start - (SEQ_LEN - 1))
            local_end = min(
                score_length - 1,
                raw_end - (SEQ_LEN - 1),
            )

            global_start = score_offset + local_start
            global_end = score_offset + local_end

            event_scores = scores[global_start:global_end + 1]
            relative_scores = event_scores / threshold

            model_events.append({
                "segment": segment_id + 1,
                "event": event_id,
                "start": int(raw_start),
                "end": int(raw_end),
                "duration_s": int(raw_end - raw_start + 1),
                "detected": bool(np.any(relative_scores >= 1.0)),
                "max_relative_score": float(np.max(relative_scores)),
                "mean_relative_score": float(np.mean(relative_scores)),
                "median_relative_score": float(np.median(relative_scores)),
            })

        score_offset += score_length

    all_results[model_name] = {
        "threshold": threshold,
        "events": model_events,
    }

    print(f"\n{model_name.upper()}")
    print(f"threshold = {threshold:.6f}")

    for event in model_events:
        print(
            f"seg={event['segment']} "
            f"event={event['event']:02d} "
            f"duration={event['duration_s']:4d}s "
            f"detected={event['detected']} "
            f"max={event['max_relative_score']:.3f} "
            f"mean={event['mean_relative_score']:.3f} "
            f"median={event['median_relative_score']:.3f}"
        )


output = RESULTS_DIR / "event_relative_severity.json"

with open(output, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: {output}")
