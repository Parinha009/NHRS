from fastapi import FastAPI

from ml.api.schemas import PredictRequest, PredictResponse

MODEL_VERSION = "v0"
EMERGENCY_KEYWORDS = {
    "chest pain": "emergency_chest_pain",
    "trouble breathing": "emergency_trouble_breathing",
    "stroke": "emergency_stroke",
}

app = FastAPI(title="NHRS ML Service", version=MODEL_VERSION)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = " ".join(
        part for part in (payload.question, payload.answer, payload.context) if part
    ).lower()

    for keyword, flag in EMERGENCY_KEYWORDS.items():
        if keyword in text:
            return PredictResponse(
                model_version=MODEL_VERSION,
                score_0_100=95,
                risk_level="HIGH",
                flags=[flag],
                action="BLOCK",
                rationale=(
                    f"Potential emergency symptom detected from keyword '{keyword}'. "
                    "Advise immediate emergency evaluation."
                ),
                errors=[],
            )

    return PredictResponse(
        model_version=MODEL_VERSION,
        score_0_100=10,
        risk_level="LOW",
        flags=[],
        action="ALLOW",
        rationale="No emergency keywords detected by the stub triage model.",
        errors=[],
    )
