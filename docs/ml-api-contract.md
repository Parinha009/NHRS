# NHRS ML API Contract

This service provides educational support only. It is not a medical diagnosis tool and must not be used as a substitute for licensed clinical judgment or emergency care.

## GET /health

Response:

```json
{
  "status": "ok",
  "model_version": "v0"
}
```

## POST /predict

Request:

```json
{
  "question": "I suddenly have chest pain.",
  "answer": "It started a few minutes ago.",
  "context": "Adult patient"
}
```

Response:

```json
{
  "model_version": "v0",
  "score_0_100": 95,
  "risk_level": "HIGH",
  "flags": ["emergency_chest_pain"],
  "action": "BLOCK",
  "rationale": "Potential emergency symptom detected from keyword 'chest pain'. Advise immediate emergency evaluation.",
  "errors": []
}
```

## POST /heart/predict

This endpoint is educational support only and is not a medical diagnosis.

Request:

```json
{
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
  "thal": 3
}
```

Response:

```json
{
  "model_version": "heart-v1",
  "prediction": 1,
  "probability": 0.73,
  "errors": []
}
```
