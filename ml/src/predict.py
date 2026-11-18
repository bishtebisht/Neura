"""Load the trained pipeline once and expose a predict_strategy() function."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from .preprocessing import FEATURE_COLUMNS
from .schemas import PredictRequest, PredictResponse

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "strategy_model.joblib"


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `python -m src.train_strategy` first."
        )
    return joblib.load(MODEL_PATH)


def predict_strategy(request: PredictRequest) -> PredictResponse:
    model = get_model()
    row = pd.DataFrame([[getattr(request, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

    predicted_label = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    class_index = list(model.classes_).index(predicted_label)
    confidence = float(proba[class_index])

    return PredictResponse(next_strategy=predicted_label, confidence=confidence)
