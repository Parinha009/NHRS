from fastapi.testclient import TestClient

from ml.api.main import app

client = TestClient(app)


def test_health_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "v0"}


def test_predict_returns_low_risk_for_non_emergency_input():
    response = client.post(
        "/predict",
        json={
            "question": "I have a mild headache.",
            "answer": "It started after a long day at work.",
            "context": "No other symptoms.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_version": "v0",
        "score_0_100": 10,
        "risk_level": "LOW",
        "flags": [],
        "action": "ALLOW",
        "rationale": "No emergency keywords detected by the stub triage model.",
        "errors": [],
    }


def test_predict_returns_high_risk_for_emergency_keyword():
    response = client.post(
        "/predict",
        json={
            "question": "I suddenly have chest pain.",
            "answer": "It started a few minutes ago.",
            "context": None,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "model_version": "v0",
        "score_0_100": 95,
        "risk_level": "HIGH",
        "flags": ["emergency_chest_pain"],
        "action": "BLOCK",
        "rationale": (
            "Potential emergency symptom detected from keyword 'chest pain'. "
            "Advise immediate emergency evaluation."
        ),
        "errors": [],
    }
