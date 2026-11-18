"""Data loading, cleaning and feature/target preparation.

Two things learned from ml/src/inspect_dataset.py drove the choices here:

1. The raw CSV is ~61% exact-duplicate rows (19630 -> 7666 unique rows), which
   looks like class-balancing via row duplication. We drop exact duplicates
   before splitting so the model is trained/evaluated on distinct
   observations rather than memorized repeats.
2. Interactions belong to students who appear multiple times, so a random
   row-level split would leak a student's other rows into both train and
   test. We split by student_id instead, so a student never appears in more
   than one split.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "strategy_dataset.csv"

STRATEGIES = ["visual", "textual", "step-by-step", "example-based", "socratic"]

NUMERIC_FEATURES = ["previous_accuracy", "response_time", "attempts", "hints_requested", "topic_mastery"]
CATEGORICAL_FEATURES = ["previous_strategy"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "next_recommended_strategy"

ROLLING_NUMERIC_FEATURES = [
    "rolling_mean_accuracy",
    "rolling_mean_response_time",
    "rolling_mean_hints",
]


def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Computes rolling averages over student history for accuracy, response_time, and hints_requested."""
    df_out = df.copy()
    if "student_id" in df_out.columns:
        df_out["rolling_mean_accuracy"] = df_out.groupby("student_id")["previous_accuracy"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df_out["rolling_mean_response_time"] = df_out.groupby("student_id")["response_time"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df_out["rolling_mean_hints"] = df_out.groupby("student_id")["hints_requested"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
    return df_out


def load_clean_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Load the CSV and drop exact duplicate rows. Returns (clean_df, stats)."""
    df = pd.read_csv(path)
    rows_raw = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    rows_clean = len(df_clean)
    stats = {
        "rows_raw": rows_raw,
        "rows_after_dedup": rows_clean,
        "duplicates_removed": rows_raw - rows_clean,
        "unique_students": int(df_clean["student_id"].nunique()),
        "unique_skills": int(df_clean["skill_id"].nunique()),
    }
    return df_clean, stats


def student_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by student_id so no student appears in more than one split."""
    students = df["student_id"].unique()
    rng = np.random.default_rng(random_state)
    shuffled = students.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))

    test_students = set(shuffled[:n_test])
    val_students = set(shuffled[n_test : n_test + n_val])
    train_students = set(shuffled[n_test + n_val :])

    train_df = df[df["student_id"].isin(train_students)].reset_index(drop=True)
    val_df = df[df["student_id"].isin(val_students)].reset_index(drop=True)
    test_df = df[df["student_id"].isin(test_students)].reset_index(drop=True)

    assert train_students.isdisjoint(val_students)
    assert train_students.isdisjoint(test_students)
    assert val_students.isdisjoint(test_students)

    return train_df, val_df, test_df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(categories=[STRATEGIES], handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
