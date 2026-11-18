"""Pydantic request/response models for the ML service."""

from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["visual", "textual", "step-by-step", "example-based", "socratic"]


class PredictRequest(BaseModel):
    previous_accuracy: float = Field(ge=0, le=1)
    response_time: float = Field(gt=0)
    attempts: int = Field(ge=1)
    hints_requested: int = Field(ge=0)
    topic_mastery: float = Field(ge=0, le=1)
    previous_strategy: Strategy


class PredictResponse(BaseModel):
    next_strategy: Strategy
    confidence: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
