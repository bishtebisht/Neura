from src.preprocessing import (
    STRATEGIES,
    build_preprocessor,
    get_X_y,
    load_clean_dataset,
    student_level_split,
)


def test_load_clean_dataset_deduplicates():
    df, stats = load_clean_dataset()
    assert stats["rows_after_dedup"] < stats["rows_raw"]
    assert len(df) == stats["rows_after_dedup"]
    assert not df.duplicated().any()


def test_student_level_split_has_no_overlap():
    df, _ = load_clean_dataset()
    train_df, val_df, test_df = student_level_split(df, test_size=0.2, val_size=0.1, random_state=0)

    train_students = set(train_df["student_id"])
    val_students = set(val_df["student_id"])
    test_students = set(test_df["student_id"])

    assert train_students.isdisjoint(val_students)
    assert train_students.isdisjoint(test_students)
    assert val_students.isdisjoint(test_students)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_get_X_y_shapes():
    df, _ = load_clean_dataset()
    X, y = get_X_y(df)
    assert set(X.columns) == {"previous_accuracy", "response_time", "attempts", "hints_requested", "topic_mastery", "previous_strategy"}
    assert len(X) == len(y) == len(df)
    assert set(y.unique()) <= set(STRATEGIES)


def test_preprocessor_handles_unseen_strategy():
    pre = build_preprocessor()
    df, _ = load_clean_dataset()
    X, _ = get_X_y(df.head(50))
    pre.fit(X)
    unseen = X.head(1).copy()
    unseen["previous_strategy"] = "not-a-real-strategy"
    transformed = pre.transform(unseen)
    assert transformed.shape[0] == 1


def test_add_rolling_features():
    from src.preprocessing import add_rolling_features
    df, _ = load_clean_dataset()
    df_roll = add_rolling_features(df.head(20), window=3)
    assert "rolling_mean_accuracy" in df_roll.columns
    assert "rolling_mean_response_time" in df_roll.columns
    assert "rolling_mean_hints" in df_roll.columns
    assert len(df_roll) == 20
