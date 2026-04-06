from fastapi import APIRouter, HTTPException, status

from ml.api.schemas_heart import HeartPredictRequest, HeartPredictResponse
from ml.src.heart.inference import predict as heart_predict

router = APIRouter(prefix="/heart", tags=["heart"])


@router.post("/predict", response_model=HeartPredictResponse)
def predict_heart(payload: HeartPredictRequest) -> HeartPredictResponse:
    result = heart_predict(payload.model_dump())
    if result["errors"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Heart model inference is unavailable.",
                "model_version": result.get("model_version"),
                "errors": result["errors"],
            },
        )

    return HeartPredictResponse(**result)
