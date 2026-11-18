from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_strategy_success():
    response = client.post(
        "/predict/strategy",
        json={
            "previous_accuracy": 0.72,
            "response_time": 15,
            "attempts": 2,
            "hints_requested": 1,
            "topic_mastery": 0.68,
            "previous_strategy": "visual",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["next_strategy"] in ["visual", "textual", "step-by-step", "example-based", "socratic"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_strategy_missing_field():
    response = client.post(
        "/predict/strategy",
        json={
            "previous_accuracy": 0.72,
            "response_time": 15,
            "attempts": 2,
            "hints_requested": 1,
            "topic_mastery": 0.68,
            # previous_strategy missing
        },
    )
    assert response.status_code == 422


def test_predict_strategy_invalid_strategy():
    response = client.post(
        "/predict/strategy",
        json={
            "previous_accuracy": 0.72,
            "response_time": 15,
            "attempts": 2,
            "hints_requested": 1,
            "topic_mastery": 0.68,
            "previous_strategy": "guessing",
        },
    )
    assert response.status_code == 422


def test_predict_strategy_negative_response_time():
    response = client.post(
        "/predict/strategy",
        json={
            "previous_accuracy": 0.72,
            "response_time": -5,
            "attempts": 2,
            "hints_requested": 1,
            "topic_mastery": 0.68,
            "previous_strategy": "visual",
        },
    )
    assert response.status_code == 422


def test_predict_strategy_invalid_attempts():
    response = client.post(
        "/predict/strategy",
        json={
            "previous_accuracy": 0.72,
            "response_time": 15,
            "attempts": 0,
            "hints_requested": 1,
            "topic_mastery": 0.68,
            "previous_strategy": "visual",
        },
    )
    assert response.status_code == 422
