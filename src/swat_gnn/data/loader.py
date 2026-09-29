from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "configs/data.yaml") -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sensor_file(path: Path) -> pd.DataFrame:
    """Load one HAI sensor CSV."""
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column: {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def load_training_data(config: dict) -> list[pd.DataFrame]:
    """Load the normal training segments."""
    root = Path(config["dataset"]["root"])

    return [
        load_sensor_file(root / filename)
        for filename in config["training"]["files"]
    ]


def load_validation_data(config: dict) -> list[pd.DataFrame]:
    """Load the normal validation segments."""
    root = Path(config["dataset"]["root"])

    return [
        load_sensor_file(root / filename)
        for filename in config["validation"]["files"]
    ]


def load_test_data(config: dict) -> tuple[list[pd.DataFrame], list[pd.Series]]:
    """
    Load test sensor data and corresponding labels.

    HAI test labels are aligned with sensor observations by row position.
    """
    root = Path(config["dataset"]["root"])

    sensor_files = config["test"]["sensor_files"]
    label_files = config["test"]["label_files"]

    sensors = []
    labels = []

    for sensor_file, label_file in zip(sensor_files, label_files):
        sensor_df = load_sensor_file(root / sensor_file)
        label_df = pd.read_csv(root / label_file)

        if "label" not in label_df.columns:
            raise ValueError(f"Missing label column: {label_file}")

        if len(sensor_df) != len(label_df):
            raise ValueError(
                f"Sensor/label length mismatch: "
                f"{sensor_file}={len(sensor_df)}, "
                f"{label_file}={len(label_df)}"
            )

        sensors.append(sensor_df)
        labels.append(label_df["label"].astype(int))

    return sensors, labels