from pathlib import Path
import numpy as np

SCORE_ROOT = Path("results/tables/swat_seed42_score_analysis")
GRAPH_ROOT = Path("results/tables/swat_seed42_graph_analysis")

labels = np.load(SCORE_ROOT / "labels.npy")
dynamic_scores = np.load(SCORE_ROOT / "dynamic_k5_scores.npy")
graph_change = np.load(GRAPH_ROOT / "graph_change_rates.npy")

threshold = 3.41457462310791

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

print("=== EVENT-LEVEL GRAPH / SCORE ANALYSIS ===")
print(f"Attack events: {len(events)}")
print()

print(
    "Event | Windows | Detected | "
    "GraphChange Mean | GraphChange Median | GraphChange Max | "
    "Score Mean | Score Median | Score Max"
)

for event_id, (start, end) in enumerate(events, start=1):
    changes = graph_change[start:end + 1]
    scores = dynamic_scores[start:end + 1]

    detected = bool(np.any(scores >= threshold))

    print(
        f"{event_id:5d} | "
        f"{end-start+1:7d} | "
        f"{str(detected):8s} | "
        f"{changes.mean():16.6f} | "
        f"{np.median(changes):18.6f} | "
        f"{changes.max():15.6f} | "
        f"{scores.mean():10.6f} | "
        f"{np.median(scores):12.6f} | "
        f"{scores.max():9.6f}"
    )

# Compare detected vs missed events.
detected_changes = []
missed_changes = []
detected_scores = []
missed_scores = []

for start, end in events:
    changes = graph_change[start:end + 1]
    scores = dynamic_scores[start:end + 1]

    if np.any(scores >= threshold):
        detected_changes.append(changes.mean())
        detected_scores.append(np.median(scores))
    else:
        missed_changes.append(changes.mean())
        missed_scores.append(np.median(scores))

print("\n=== DETECTED vs MISSED EVENTS ===")

print(f"Detected events: {len(detected_changes)}")
print(f"Missed events: {len(missed_changes)}")

print("\nMean graph change:")
print(f"  detected: {np.mean(detected_changes):.6f}")
print(f"  missed:   {np.mean(missed_changes):.6f}")

print("\nMedian anomaly score:")
print(f"  detected: {np.mean(detected_scores):.6f}")
print(f"  missed:   {np.mean(missed_scores):.6f}")

print("\n=== DONE ===")
