from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from swat_gnn.models.dynamic_graph_autoencoder import (
    DynamicGraphAutoencoder,
)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    batch_size = 256
    sequence_length = 60
    num_nodes = 66

    model = DynamicGraphAutoencoder(
        sequence_length=sequence_length,
        num_nodes=num_nodes,
        temporal_hidden_dim=32,
        embedding_dim=32,
        gat_hidden_dim=64,
        gat_heads=4,
        k=5,
        dropout=0.0,
    ).to(device)

    model.train()

    print(
        f"Trainable parameters: "
        f"{count_parameters(model):,}"
    )

    # ---------------------------------------------------------
    # Synthetic input
    # ---------------------------------------------------------

    x = torch.randn(
        batch_size,
        sequence_length,
        num_nodes,
        device=device,
    )

    print(
        f"Input shape: {tuple(x.shape)}"
    )

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    print("\nRunning warm-up...")

    for _ in range(2):
        model.zero_grad(set_to_none=True)

        reconstruction = model(x)

        loss = torch.mean(
            (reconstruction - x[:, -1, :]) ** 2
        )

        loss.backward()

    synchronize(device)

    # ---------------------------------------------------------
    # Forward benchmark
    # ---------------------------------------------------------

    print("\nBenchmarking forward pass...")

    forward_times = []

    for _ in range(5):
        synchronize(device)

        start = time.perf_counter()

        with torch.no_grad():
            reconstruction = model(x)

        synchronize(device)

        elapsed = time.perf_counter() - start
        forward_times.append(elapsed)

    mean_forward = sum(forward_times) / len(
        forward_times
    )

    print(
        f"Mean forward time: "
        f"{mean_forward:.4f} s"
    )

    print(
        f"Forward time per sample: "
        f"{mean_forward / batch_size:.4f} s"
    )

    # ---------------------------------------------------------
    # Backward benchmark
    # ---------------------------------------------------------

    print("\nBenchmarking forward + backward...")

    backward_times = []

    for _ in range(5):
        model.zero_grad(set_to_none=True)

        synchronize(device)

        start = time.perf_counter()

        reconstruction = model(x)

        loss = torch.mean(
            (reconstruction - x[:, -1, :]) ** 2
        )

        loss.backward()

        synchronize(device)

        elapsed = time.perf_counter() - start
        backward_times.append(elapsed)

    mean_backward = sum(backward_times) / len(
        backward_times
    )

    print(
        f"Mean forward + backward time: "
        f"{mean_backward:.4f} s"
    )

    print(
        f"Training time per sample: "
        f"{mean_backward / batch_size:.4f} s"
    )

    # ---------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------

    if device.type == "cuda":
        allocated = (
            torch.cuda.memory_allocated(device)
            / (1024 ** 2)
        )

        reserved = (
            torch.cuda.memory_reserved(device)
            / (1024 ** 2)
        )

        peak = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 2)
        )

        print("\nGPU memory:")
        print(
            f"Current allocated: {allocated:.2f} MB"
        )
        print(
            f"Current reserved:  {reserved:.2f} MB"
        )
        print(
            f"Peak allocated:    {peak:.2f} MB"
        )

    # ---------------------------------------------------------
    # Output validation
    # ---------------------------------------------------------

    print("\nOutput validation:")

    print(
        f"Reconstruction shape: "
        f"{tuple(reconstruction.shape)}"
    )

    expected_shape = (
        batch_size,
        num_nodes,
    )

    if tuple(reconstruction.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected reconstruction shape: "
            f"{tuple(reconstruction.shape)}, "
            f"expected {expected_shape}"
        )

    print("Reconstruction shape: OK")

    # ---------------------------------------------------------
    # Dynamic graph validation
    # ---------------------------------------------------------

    print("\nInspecting dynamic graphs...")

    model.eval()

    with torch.no_grad():
        graphs = model.get_dynamic_graphs(x)

    print(
        f"Number of graphs: {len(graphs)}"
    )

    edge_counts = [
        graph.shape[1]
        for graph in graphs
    ]

    print(
        f"Edges per graph: {edge_counts}"
    )

    expected_edges = num_nodes * 5

    if not all(
        count == expected_edges
        for count in edge_counts
    ):
        raise RuntimeError(
            "Unexpected number of graph edges."
        )

    print(
        f"Expected directed edges per graph: "
        f"{expected_edges}"
    )

    print("Graph edge count: OK")

    # ---------------------------------------------------------
    # Check whether graphs actually vary
    # ---------------------------------------------------------

    different_graphs = False

    reference = graphs[0]

    for graph in graphs[1:]:
        if not torch.equal(
            reference,
            graph,
        ):
            different_graphs = True
            break

    print(
        f"Graphs vary across windows: "
        f"{different_graphs}"
    )

    if not different_graphs:
        print(
            "WARNING: all sampled graphs are "
            "identical. Investigate graph construction."
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DYNAMIC GRAPH BENCHMARK COMPLETE")
    print("=" * 60)

    print(
        f"Parameters:        "
        f"{count_parameters(model):,}"
    )

    print(
        f"Batch size:        "
        f"{batch_size}"
    )

    print(
        f"Forward:           "
        f"{mean_forward:.4f} s"
    )

    print(
        f"Forward+backward:  "
        f"{mean_backward:.4f} s"
    )

    print(
        f"Graphs dynamic:    "
        f"{different_graphs}"
    )


if __name__ == "__main__":
    main()