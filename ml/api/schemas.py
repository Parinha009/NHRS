from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    question: str
    answer: str
    context: Optional[str] = None


class PredictResponse(BaseModel):
    model_version: str
    score_0_100: int
    risk_level: str
    flags: list[str] = Field(default_factory=list)
    action: str
    rationale: str
    errors: list[str] = Field(default_factory=list)
