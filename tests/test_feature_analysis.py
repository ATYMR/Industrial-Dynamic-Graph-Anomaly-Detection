from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swat_gnn.data.feature_analysis import analyze_features
from swat_gnn.data.loader import (
    load_config,
    load_training_data,
)


def main():
    config = load_config()

    training_segments = load_training_data(config)

    result = analyze_features(training_segments)

    constant_features = result["constant_features"]
    variable_features = result["variable_features"]

    print("=== FEATURE ANALYSIS ===")
    print("Total features:", len(result["feature_names"]))
    print("Constant features:", len(constant_features))
    print("Variable features:", len(variable_features))

    print("\n=== CONSTANT FEATURES ===")

    for feature in constant_features:
        print(feature)

    print("\n=== VERIFICATION ===")

    assert len(result["feature_names"]) == 86
    assert len(constant_features) == 20
    assert len(variable_features) == 66

    assert set(constant_features).isdisjoint(
        set(variable_features)
    )

    assert (
        len(constant_features) + len(variable_features)
        == len(result["feature_names"])
    )

    print("Feature count: PASSED")
    print("Constant/variable classification: PASSED")
    print("Feature analysis verification: SUCCESS")


if __name__ == "__main__":
    main()