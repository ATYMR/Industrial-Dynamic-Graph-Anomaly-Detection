from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_gnn.data.loader import (
    load_config,
    load_training_data,
    load_validation_data,
    load_test_data,
)


def main():
    config = load_config()

    train = load_training_data(config)
    validation = load_validation_data(config)
    test_sensor, test_labels = load_test_data(config)

    print("=== TRAINING ===")
    for i, df in enumerate(train, 1):
        print(
            f"train{i}: "
            f"rows={len(df)}, "
            f"columns={len(df.columns)}, "
            f"time={df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
        )

    print("\n=== VALIDATION ===")
    for i, df in enumerate(validation, 1):
        print(
            f"validation{i}: "
            f"rows={len(df)}, "
            f"columns={len(df.columns)}, "
            f"time={df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
        )

    print("\n=== TEST ===")
    for i, (df, labels) in enumerate(zip(test_sensor, test_labels), 1):
        print(
            f"test{i}: "
            f"sensor_rows={len(df)}, "
            f"sensor_columns={len(df.columns)}, "
            f"label_rows={len(labels)}, "
            f"attack_rows={int(labels.sum())}"
        )

    print("\nLoader verification: SUCCESS")


if __name__ == "__main__":
    main()