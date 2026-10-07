from pathlib import Path
import numpy as np

ROOT = Path("results/tables/swat_seed42_score_analysis")

labels = np.load(ROOT / "labels.npy")
static = np.load(ROOT / "static_cosine_scores.npy")
dynamic = np.load(ROOT / "dynamic_k5_scores.npy")

static_threshold = 6.966011047363281
dynamic_threshold = 3.41457462310791

attack_indices = np.where(labels == 1)[0]

events = []
start = attack_indices[0]
previous = attack_indices[0]

for idx in attack_indices[1:]:
    if idx != previous + 1:
        events.append((start, previous))
        start = idx
    previous = idx

events.append((start, previous))

print("=== STATIC vs DYNAMIC EVENT SCORE COMPARISON ===")
print()

print(
    "Event | Windows | "
    "Static Mean | Static Median | Static Max | "
    "Dynamic Mean | Dynamic Median | Dynamic Max | "
    "Static Det | Dynamic Det"
)

for event_id, (start, end) in enumerate(events, start=1):
    s = static[start:end + 1]
    d = dynamic[start:end + 1]

    static_detected = bool(np.any(s >= static_threshold))
    dynamic_detected = bool(np.any(d >= dynamic_threshold))

    print(
        f"{event_id:5d} | "
        f"{end-start+1:7d} | "
        f"{s.mean():11.6f} | "
        f"{np.median(s):13.6f} | "
        f"{s.max():10.6f} | "
        f"{d.mean():12.6f} | "
        f"{np.median(d):14.6f} | "
        f"{d.max():11.6f} | "
        f"{str(static_detected):10s} | "
        f"{str(dynamic_detected):11s}"
    )

print("\n=== MISSED-EVENT COMPARISON ===")

missed_events = []

for event_id, (start, end) in enumerate(events, start=1):
    if not np.any(dynamic[start:end + 1] >= dynamic_threshold):
        missed_events.append(event_id)

for event_id in missed_events:
    start, end = events[event_id - 1]

    s = static[start:end + 1]
    d = dynamic[start:end + 1]

    print(
        f"Event {event_id}: "
        f"static median={np.median(s):.6f}, "
        f"dynamic median={np.median(d):.6f}, "
        f"static max={s.max():.6f}, "
        f"dynamic max={d.max():.6f}"
    )

print("\n=== DONE ===")
