"""FastAPI service exposing the teaching-strategy prediction model.

Run with: uvicorn main:app --host 0.0.0.0 --port 8001
"""

from fastapi import FastAPI, HTTPException

from src.predict import ModelNotTrainedError, predict_strategy
from src.schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(title="NEURA Strategy Prediction Service")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict/strategy", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return predict_strategy(request)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
