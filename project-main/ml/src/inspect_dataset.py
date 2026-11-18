"""Dataset inspection report for the teaching-strategy dataset.

Run with: python -m src.inspect_dataset
Prints row/column stats needed before training (row count, unique students,
missing values, duplicates, class balance, numeric ranges, invalid values)
so that training decisions (student-level split, leakage checks) are made
from real data rather than assumptions.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "strategy_dataset.csv"

VALID_STRATEGIES = {"visual", "textual", "step-by-step", "example-based", "socratic"}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def report(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DATASET REPORT:", DATA_PATH)
    print("=" * 70)

    print(f"\nRows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print(f"\nUnique students (student_id): {df['student_id'].nunique()}")
    print(f"Unique skills (skill_id): {df['skill_id'].nunique()}")

    print("\n-- Missing values per column --")
    print(df.isna().sum().to_string())

    full_dupes = df.duplicated().sum()
    print(f"\nFully duplicate rows: {full_dupes}")
    key_cols = [c for c in df.columns if c != "student_id"]
    dupes_ignoring_student = df.duplicated(subset=key_cols).sum()
    print(f"Duplicate rows ignoring student_id: {dupes_ignoring_student}")

    print("\n-- Repeated students (row count per student) --")
    counts = df["student_id"].value_counts()
    print(f"Min rows/student: {counts.min()}  Max rows/student: {counts.max()}  Mean: {counts.mean():.2f}")
    print(f"Students with exactly 1 row: {(counts == 1).sum()}")
    print(f"Students with >1 row: {(counts > 1).sum()}")

    print("\n-- Target class distribution (next_recommended_strategy) --")
    print(df["next_recommended_strategy"].value_counts().to_string())
    print("\nAs proportions:")
    print((df["next_recommended_strategy"].value_counts(normalize=True) * 100).round(2).to_string())

    print("\n-- previous_strategy distribution --")
    print(df["previous_strategy"].value_counts().to_string())

    print("\n-- Numeric ranges --")
    numeric_cols = ["previous_accuracy", "response_time", "attempts", "hints_requested", "topic_mastery"]
    print(df[numeric_cols].describe().to_string())

    print("\n-- Invalid value checks --")
    invalid_acc = df[(df["previous_accuracy"] < 0) | (df["previous_accuracy"] > 1)]
    print(f"previous_accuracy out of [0,1]: {len(invalid_acc)}")
    invalid_mastery = df[(df["topic_mastery"] < 0) | (df["topic_mastery"] > 1)]
    print(f"topic_mastery out of [0,1]: {len(invalid_mastery)}")
    invalid_rt = df[df["response_time"] < 0]
    print(f"response_time negative: {len(invalid_rt)}")
    invalid_attempts = df[df["attempts"] < 1]
    print(f"attempts < 1: {len(invalid_attempts)}")
    invalid_hints = df[df["hints_requested"] < 0]
    print(f"hints_requested negative: {len(invalid_hints)}")
    invalid_skill = df[df["skill_id"] < 0]
    print(f"skill_id negative/sentinel (e.g. -1): {len(invalid_skill)}")

    invalid_prev_strategy = df[~df["previous_strategy"].isin(VALID_STRATEGIES)]
    print(f"previous_strategy not in the 5 valid labels: {len(invalid_prev_strategy)}")
    invalid_next_strategy = df[~df["next_recommended_strategy"].isin(VALID_STRATEGIES)]
    print(f"next_recommended_strategy not in the 5 valid labels: {len(invalid_next_strategy)}")

    print("\n-- Deterministic-target / leakage smoke check --")
    # If (previous_strategy, next_recommended_strategy) always co-occur with an
    # identical feature row, the target may be trivially derivable from a subset
    # of features rather than reflecting real variation.
    grouped = df.groupby(["previous_strategy"])["next_recommended_strategy"].value_counts(normalize=True)
    print("P(next_strategy | previous_strategy) -- if near-deterministic (~1.0), that alone predicts the target:")
    print(grouped.round(3).to_string())

    print("\n" + "=" * 70)


if __name__ == "__main__":
    report(load_raw())
