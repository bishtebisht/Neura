import pytest

from src.predict import predict_strategy
from src.preprocessing import STRATEGIES
from src.schemas import PredictRequest


def test_predict_returns_valid_strategy_and_confidence():
    req = PredictRequest(
        previous_accuracy=0.72,
        response_time=15,
        attempts=2,
        hints_requested=1,
        topic_mastery=0.68,
        previous_strategy="visual",
    )
    result = predict_strategy(req)
    assert result.next_strategy in STRATEGIES
    assert 0.0 <= result.confidence <= 1.0


def test_predict_is_deterministic_for_same_input():
    req = PredictRequest(
        previous_accuracy=0.4,
        response_time=25,
        attempts=3,
        hints_requested=2,
        topic_mastery=0.3,
        previous_strategy="step-by-step",
    )
    first = predict_strategy(req)
    second = predict_strategy(req)
    assert first.next_strategy == second.next_strategy
    assert first.confidence == second.confidence


def test_predict_request_rejects_invalid_values():
    with pytest.raises(ValueError):
        PredictRequest(
            previous_accuracy=1.5,
            response_time=15,
            attempts=2,
            hints_requested=1,
            topic_mastery=0.68,
            previous_strategy="visual",
        )
    with pytest.raises(ValueError):
        PredictRequest(
            previous_accuracy=0.5,
            response_time=-1,
            attempts=2,
            hints_requested=1,
            topic_mastery=0.68,
            previous_strategy="visual",
        )
    with pytest.raises(ValueError):
        PredictRequest(
            previous_accuracy=0.5,
            response_time=15,
            attempts=0,
            hints_requested=1,
            topic_mastery=0.68,
            previous_strategy="visual",
        )
    with pytest.raises(ValueError):
        PredictRequest(
            previous_accuracy=0.5,
            response_time=15,
            attempts=2,
            hints_requested=1,
            topic_mastery=0.68,
            previous_strategy="not-a-strategy",
        )
