# Dynamic Graph Experiment Specification

## 1. Research Question

Does allowing sensor relationships to change over time improve
multivariate industrial anomaly detection compared with a fixed
sensor relationship graph?

## 2. Hypothesis

H1:
A window-conditioned dynamic graph can capture changes in
relationships between industrial variables that are not represented
by a single static graph.

H0:
A dynamic graph does not provide a meaningful improvement over the
static graph baseline under the same experimental protocol.

## 3. Dataset

Dataset:
HAI 23.05

Features:
66 variable industrial sensor/control variables

Window:
60 seconds

Split:
Training = train1 + train2 + train3
Validation = train4
Test = test1 + test2

Normalization:
StandardScaler fitted only on training data.

## 4. Static Baseline

The existing GAT-AE uses a fixed top-5 training-correlation graph.

The same graph is used for every temporal window.

## 5. Dynamic Graph

For every 60-second window:

1. Encode each sensor's temporal behavior.
2. Produce a representation for each sensor.
3. Compute pairwise sensor relationships from those representations.
4. Construct a window-specific adjacency matrix.
5. Apply graph attention/message passing using that adjacency.
6. Reconstruct the input window.
7. Use reconstruction error as the anomaly score.

### 5.1 Graph Construction

Each 60-second input window contains 66 sensor variables.

Each sensor is treated as one graph node. The temporal history of
each sensor within the window is encoded into a sensor representation.

Pairwise similarity between sensor representations is then used to
construct a window-specific top-k graph.

Therefore, unlike the static GAT baseline, the adjacency matrix is
recomputed for each temporal window:

A(t) = Graph(H(t))

where H(t) contains the learned representations of the 66 sensors.

The initial dynamic graph will use a top-k similarity graph. The value
of k will be selected before the final experiment and kept fixed across
training, validation, and testing.

### 5.2 Dynamic Graph Architecture

Each 60-second window is processed sensor-by-sensor.

For each of the 66 sensors, its 60-second history is encoded using a
shared temporal encoder. The encoder produces a 32-dimensional
representation for each sensor.

This produces a node representation matrix:

H ∈ R^(66 × 32)

Pairwise cosine similarity between node representations is used to
construct a window-specific top-k graph.

The resulting graph is passed to a graph attention network.

Initial architecture:

- Temporal encoder: LSTM
- Sensor embedding dimension: 32
- Dynamic graph: top-k cosine similarity
- GAT layers: 32 → 64 → 32 → 64 → 1

The model reconstructs the sensor state at the final timestep of the
60-second window.

The mean squared reconstruction error at the final timestep is used
as the primary anomaly score.

### 5.3 Causality and Leakage Prevention

The dynamic graph for timestep t is constructed exclusively from the
60-second window ending at t.

No future observations or anomaly labels are used during graph
construction or model inference.

The test labels are used only for final evaluation.

## 6. Primary Comparison

Static GAT-AE
vs.
Dynamic Graph-AE

Everything other than graph construction should remain as consistent
as practically possible.

## 7. Primary Metrics

- F1
- AUROC
- AUPRC
- Precision
- Recall
- False-positive rate

Additional:
- Detection latency
- Event-level detection

## 8. Ablations

To be defined after the primary dynamic model works.

## 9. Explainability

Investigate whether changes in learned sensor relationships correspond
to anomalous periods.

## 10. Reproducibility

- Random seed: 42
- Record configuration
- Save model checkpoint
- Save training history
- Save anomaly scores
- Save evaluation metrics
- Save dynamic graph statistics