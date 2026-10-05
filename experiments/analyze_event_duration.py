from pathlib import Path
import json
import numpy as np
from scipy.stats import mannwhitneyu

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


def evaluate_model(model_name, events_by_segment):
    data = np.load(SCORE_FILES[model_name])
    scores = data["scores"]
    threshold = float(data["threshold"])
    predictions = scores >= threshold

    segment_score_lengths = []
    total = 0

    for segment_events, label_path in events_by_segment:
        labels = load_labels(label_path)
        score_len = len(labels) - SEQ_LEN + 1
        segment_score_lengths.append(score_len)
        total += score_len

    assert total == len(scores), (
        f"{model_name}: score length mismatch: "
        f"expected {total}, got {len(scores)}"
    )

    detected_durations = []
    missed_durations = []
    detected_count = 0
    missed_count = 0

    score_offset = 0

    for segment_id, ((events, label_path), segment_score_len) in enumerate(
        zip(events_by_segment, segment_score_lengths)
    ):
        for raw_start, raw_end in events:
            # Convert raw timestep indices to this segment's
            # endpoint-label score indices.
            local_start = max(0, raw_start - (SEQ_LEN - 1))
            local_end = min(
                segment_score_len - 1,
                raw_end - (SEQ_LEN - 1),
            )

            assert local_start <= local_end, (
                f"Invalid mapping for {model_name}, segment {segment_id}: "
                f"{raw_start}-{raw_end}"
            )

            global_start = score_offset + local_start
            global_end = score_offset + local_end

            event_predictions = predictions[global_start:global_end + 1]

            duration_seconds = raw_end - raw_start + 1

            if np.any(event_predictions):
                detected_count += 1
                detected_durations.append(duration_seconds)
            else:
                missed_count += 1
                missed_durations.append(duration_seconds)

        score_offset += segment_score_len

    result = {
        "model": model_name,
        "detected_events": detected_count,
        "missed_events": missed_count,
        "total_events": detected_count + missed_count,
        "detection_rate": detected_count / (detected_count + missed_count),
        "detected_duration_mean_s": (
            float(np.mean(detected_durations))
            if detected_durations else None
        ),
        "detected_duration_median_s": (
            float(np.median(detected_durations))
            if detected_durations else None
        ),
        "missed_duration_mean_s": (
            float(np.mean(missed_durations))
            if missed_durations else None
        ),
        "missed_duration_median_s": (
            float(np.median(missed_durations))
            if missed_durations else None
        ),
    }

    if detected_durations and missed_durations:
        u_stat, p_value = mannwhitneyu(
            detected_durations,
            missed_durations,
            alternative="two-sided",
        )

        result["mann_whitney_u"] = float(u_stat)
        result["mann_whitney_p"] = float(p_value)

    return result


events_by_segment = []

for label_file in LABEL_FILES:
    labels = load_labels(label_file)
    events = find_events(labels)
    events_by_segment.append((events, label_file))

    print(
        f"{label_file.name}: "
        f"{len(labels)} rows, {len(events)} attack events"
    )

all_results = {}

for model_name in SCORE_FILES:
    result = evaluate_model(model_name, events_by_segment)
    all_results[model_name] = result

    print(f"\n{model_name.upper()}")
    print(json.dumps(result, indent=2))

    # These are the counts established by the original
    # segment-aware event analysis.
    expected = {
        "static": (39, 13),
        "dynamic": (34, 18),
    }

    assert (
        result["detected_events"],
        result["missed_events"],
    ) == expected[model_name], (
        f"{model_name}: corrected alignment does not reproduce "
        f"the established event counts."
    )

output_path = RESULTS_DIR / "event_duration_analysis_corrected.json"

with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved: {output_path}")
print("\nAlignment verification PASSED.")
