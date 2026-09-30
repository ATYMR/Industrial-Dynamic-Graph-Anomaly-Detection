from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_gnn.data.windowing import (
    create_windows,
    create_windows_from_segments,
)


def main():
    print("=== WINDOWING VERIFICATION ===")

    # Simple synthetic data for deterministic testing.
    data = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)

    windows = create_windows(
        data,
        sequence_length=5,
        stride=1,
    )

    print("Input shape:", data.shape)
    print("Window shape:", windows.shape)

    # 20 timesteps, window length 5, stride 1:
    # 20 - 5 + 1 = 16 windows.
    assert windows.shape == (16, 5, 3)

    # First window must contain timesteps 0 through 4.
    assert np.array_equal(
        windows[0],
        data[0:5],
    )

    # Second window must contain timesteps 1 through 5.
    assert np.array_equal(
        windows[1],
        data[1:6],
    )

    # Test that separate segments remain separate.
    segment1 = np.ones((10, 3), dtype=np.float32)
    segment2 = np.full((10, 3), 2.0, dtype=np.float32)

    segment_windows = create_windows_from_segments(
        [segment1, segment2],
        sequence_length=5,
        stride=1,
    )

    assert len(segment_windows) == 2
    assert segment_windows[0].shape == (6, 5, 3)
    assert segment_windows[1].shape == (6, 5, 3)

    # First segment must contain only 1s.
    assert np.all(segment_windows[0] == 1.0)

    # Second segment must contain only 2s.
    assert np.all(segment_windows[1] == 2.0)

    print("\nBasic windowing: PASSED")
    print("Segment boundary protection: PASSED")
    print("Windowing verification: SUCCESS")


if __name__ == "__main__":
    main()