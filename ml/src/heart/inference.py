from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_VERSION = "heart-v1"
ML_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ML_DIR / "artifacts" / "heart" / "v1"


@lru_cache(maxsize=1)
def load_artifacts() -> dict[str, Any]:
    model_path = ARTIFACT_DIR / "model.joblib"
    preprocessor_path = ARTIFACT_DIR / "preprocessor.joblib"
    metadata_path = ARTIFACT_DIR / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preprocessor_path}"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata artifact not found at {metadata_path}")

    return {
        "model": joblib.load(model_path),
        "preprocessor": joblib.load(preprocessor_path),
        "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
    }


def predict(payload: dict) -> dict:
    result = {
        "model_version": MODEL_VERSION,
        "prediction": None,
        "probability": None,
        "errors": [],
    }

    try:
        artifacts = load_artifacts()
        model = artifacts["model"]
        preprocessor = artifacts["preprocessor"]
        metadata = artifacts["metadata"]

        feature_order = metadata.get("feature_order", [])
        if not feature_order:
            raise ValueError("Metadata is missing feature_order.")

        result["model_version"] = metadata.get("model_version", MODEL_VERSION)

        frame = pd.DataFrame(
            [{feature: payload.get(feature) for feature in feature_order}],
            columns=feature_order,
        )
        transformed = preprocessor.transform(frame)

        prediction = int(model.predict(transformed)[0])
        positive_class = 1 if 1 in model.classes_ else model.classes_[-1]
        positive_index = list(model.classes_).index(positive_class)
        probability = float(model.predict_proba(transformed)[0][positive_index])

        result["prediction"] = prediction
        result["probability"] = probability
    except Exception as exc:
        result["errors"] = [str(exc)]

    return result
