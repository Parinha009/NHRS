from pydantic import BaseModel, Field


class HeartPredictRequest(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


class HeartPredictResponse(BaseModel):
    model_version: str
    prediction: int
    probability: float = Field(ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)
