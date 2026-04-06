from fastapi.testclient import TestClient

from ml.api.main import app

client = TestClient(app)

VALID_HEART_PAYLOAD = {
    "age": 58,
    "sex": 1,
    "cp": 0,
    "trestbps": 150,
    "chol": 270,
    "fbs": 0,
    "restecg": 0,
    "thalach": 111,
    "exang": 1,
    "oldpeak": 0.8,
    "slope": 1,
    "ca": 2,
    "thal": 3,
}


def test_heart_predict_returns_prediction_payload():
    response = client.post("/heart/predict", json=VALID_HEART_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model_version", "prediction", "probability", "errors"}
    assert body["model_version"] == "heart-v1"
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["errors"] == []


def test_heart_predict_missing_field_returns_422():
    payload = VALID_HEART_PAYLOAD.copy()
    payload.pop("thal")

    response = client.post("/heart/predict", json=payload)

    assert response.status_code == 422
