from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SWAT_TIMESTAMP_COLUMN = " Timestamp"
SWAT_LABEL_COLUMN = "Normal/Attack"

TRAIN_NORMAL_SEGMENTS = tuple(range(1, 21))
VALIDATION_NORMAL_SEGMENTS = tuple(range(21, 25))
TEST_NORMAL_SEGMENTS = tuple(range(25, 38))
TEST_ATTACK_SEGMENTS = tuple(range(25, 36))

TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"


@dataclass(frozen=True)
class SwatSegment:
    """A continuous SWaT time-series segment."""

    segment_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Parse SWaT timestamps using the dataset's explicit format."""
    return pd.to_datetime(
        series.astype(str).str.strip(),
        format=TIMESTAMP_FORMAT,
    )


def _find_continuous_segments(
    timestamps: pd.Series,
) -> list[SwatSegment]:
    """
    Find contiguous segments where consecutive observations are exactly
    one second apart.
    """
    timestamps = timestamps.reset_index(drop=True)
    delta = timestamps.diff().dt.total_seconds()

    starts = [0]
    starts.extend(
        delta.index[1:][delta.iloc[1:].ne(1)].tolist()
    )

    ends = starts[1:] + [len(timestamps)]

    return [
        SwatSegment(
            segment_id=i + 1,
            start=start,
            end=end,
        )
        for i, (start, end) in enumerate(zip(starts, ends))
    ]


def _load_swat_csv(path: Path) -> pd.DataFrame:
    """Load one raw SWaT CSV and parse its timestamp column."""
    if not path.exists():
        raise FileNotFoundError(f"SWaT file not found: {path}")

    df = pd.read_csv(path)

    if SWAT_TIMESTAMP_COLUMN not in df.columns:
        raise ValueError(
            f"Missing SWaT timestamp column {SWAT_TIMESTAMP_COLUMN!r}: {path}"
        )

    df[SWAT_TIMESTAMP_COLUMN] = _parse_timestamp(
        df[SWAT_TIMESTAMP_COLUMN]
    )
    df = df.rename(columns={SWAT_TIMESTAMP_COLUMN: "timestamp"})

    return df


def _select_segments(
    df: pd.DataFrame,
    segments: list[SwatSegment],
    segment_ids: tuple[int, ...],
) -> list[pd.DataFrame]:
    """Return selected segments without crossing boundaries."""
    selected = []

    for segment_id in segment_ids:
        matches = [
            segment
            for segment in segments
            if segment.segment_id == segment_id
        ]

        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one segment with id {segment_id}, "
                f"found {len(matches)}."
            )

        segment = matches[0]
        selected.append(
            df.iloc[segment.start:segment.end].copy()
        )

    return selected


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return SWaT sensor/actuator columns, excluding metadata columns."""
    excluded = {"timestamp", SWAT_LABEL_COLUMN}
    return [column for column in df.columns if column not in excluded]


def _to_feature_array(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Convert selected SWaT features to a finite float32 array."""
    values = df[feature_columns].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(
            "Selected SWaT data contains NaN or infinite values."
        )

    return values


def _select_variable_features(
    training_segments: list[pd.DataFrame],
    feature_columns: list[str],
) -> list[str]:
    """Select features with non-zero variance in training data only."""
    training_values = np.concatenate(
        [
            df[feature_columns].to_numpy(dtype=np.float64)
            for df in training_segments
        ],
        axis=0,
    )

    std = np.std(training_values, axis=0)

    variable_features = [
        feature
        for feature, feature_std in zip(feature_columns, std)
        if feature_std > 0
    ]

    if not variable_features:
        raise ValueError("No variable SWaT features found.")

    return variable_features


def _prepare_feature_frames(
    segments: list[pd.DataFrame],
    feature_columns: list[str],
) -> list[pd.DataFrame]:
    """Keep timestamp plus selected features and reject non-finite values."""
    prepared = []

    for df in segments:
        result = df[["timestamp", *feature_columns]].copy()

        values = result[feature_columns].to_numpy(dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                "Selected SWaT data contains NaN or infinite values."
            )

        prepared.append(result)

    return prepared


def _build_test_episodes(
    test_normal_segments: list[pd.DataFrame],
    test_attack_segments: list[pd.DataFrame],
) -> list[tuple[pd.DataFrame, np.ndarray]]:
    """Build non-overlapping chronological SWaT test episodes.

    Normal segment 25 is a standalone normal episode because there is a
    timestamp gap before normal segment 26. Normal segments 26-37 and
    attack segments 25-35 form one continuous chronological episode.
    """
    if len(test_normal_segments) != 13:
        raise ValueError(
            f"Expected 13 selected normal segments, found "
            f"{len(test_normal_segments)}."
        )

    if len(test_attack_segments) != 11:
        raise ValueError(
            f"Expected 11 selected attack segments, found "
            f"{len(test_attack_segments)}."
        )

    episodes = []

    # Normal segment 25 is separated from the main test sequence by a
    # genuine timestamp gap, so keep it as its own normal-only episode.
    normal_only = test_normal_segments[0]
    normal_only_labels = np.zeros(len(normal_only), dtype=np.int64)
    episodes.append((normal_only.copy(), normal_only_labels))

    # Build the main continuous sequence:
    # normal26 -> attack25 -> normal27 -> attack26 -> ... -> attack35 -> normal37
    main_frames = []
    main_labels = []

    for i, attack in enumerate(test_attack_segments):
        previous_normal = test_normal_segments[i + 1]

        if i == 0:
            main_frames.append(previous_normal)
            main_labels.append(
                np.zeros(len(previous_normal), dtype=np.int64)
            )

        before_gap = (
            attack["timestamp"].iloc[0]
            - main_frames[-1]["timestamp"].iloc[-1]
        ).total_seconds()

        if before_gap != 1:
            raise ValueError(
                f"SWaT test attack {i + 25} is not continuous after "
                f"the preceding normal segment: gap={before_gap}s."
            )

        main_frames.append(attack)
        main_labels.append(
            np.ones(len(attack), dtype=np.int64)
        )

        next_normal = test_normal_segments[i + 2]

        after_gap = (
            next_normal["timestamp"].iloc[0]
            - attack["timestamp"].iloc[-1]
        ).total_seconds()

        if after_gap != 1:
            raise ValueError(
                f"SWaT test attack {i + 25} is not continuous before "
                f"the following normal segment: gap={after_gap}s."
            )

        main_frames.append(next_normal)
        main_labels.append(
            np.zeros(len(next_normal), dtype=np.int64)
        )

    main_episode = pd.concat(
        main_frames,
        axis=0,
        ignore_index=True,
    )
    main_labels = np.concatenate(main_labels)

    if len(main_episode) != len(main_labels):
        raise ValueError(
            f"Main SWaT episode data/label length mismatch: "
            f"{len(main_episode)} != {len(main_labels)}."
        )

    episodes.append((main_episode, main_labels))

    return episodes


def load_swat_dataset(
    root: str | Path = "data/raw/SWaT",
) -> dict[str, object]:
    """
    Load the frozen primary SWaT experiment split.

    Training:
        normal segments 1-20.

    Validation:
        normal segments 21-24.

    Test:
        normal segments 25-37 and attack segments 25-35.

    Segments 38-39 of normal.csv are intentionally excluded.
    """
    root = Path(root).resolve()

    normal_df = _load_swat_csv(root / "normal.csv")
    attack_df = _load_swat_csv(root / "attack.csv")

    normal_segments = _find_continuous_segments(
        normal_df["timestamp"]
    )
    attack_segments = _find_continuous_segments(
        attack_df["timestamp"]
    )

    if len(normal_segments) != 39:
        raise ValueError(
            f"Expected 39 normal segments, found {len(normal_segments)}."
        )

    if len(attack_segments) != 35:
        raise ValueError(
            f"Expected 35 attack segments, found {len(attack_segments)}."
        )

    feature_columns = _feature_columns(normal_df)

    if _feature_columns(attack_df) != feature_columns:
        raise ValueError(
            "Normal and attack feature schemas do not match."
        )

    train_segments = _select_segments(
        normal_df,
        normal_segments,
        TRAIN_NORMAL_SEGMENTS,
    )

    validation_segments = _select_segments(
        normal_df,
        normal_segments,
        VALIDATION_NORMAL_SEGMENTS,
    )

    test_normal_segments = _select_segments(
        normal_df,
        normal_segments,
        TEST_NORMAL_SEGMENTS,
    )

    test_attack_segments = _select_segments(
        attack_df,
        attack_segments,
        TEST_ATTACK_SEGMENTS,
    )

    variable_features = _select_variable_features(
        train_segments,
        feature_columns,
    )

    train_segments = _prepare_feature_frames(
        train_segments,
        variable_features,
    )
    validation_segments = _prepare_feature_frames(
        validation_segments,
        variable_features,
    )
    test_normal_segments = _prepare_feature_frames(
        test_normal_segments,
        variable_features,
    )
    test_attack_segments = _prepare_feature_frames(
        test_attack_segments,
        variable_features,
    )

    test_episodes = _build_test_episodes(
        test_normal_segments,
        test_attack_segments,
    )

    return {
        "feature_columns": variable_features,
        "all_feature_columns": feature_columns,
        "train_segments": train_segments,
        "validation_segments": validation_segments,
        "test_normal_segments": test_normal_segments,
        "test_attack_segments": test_attack_segments,
        "test_episodes": test_episodes,
        "normal_segments": normal_segments,
        "attack_segments": attack_segments,
    }
