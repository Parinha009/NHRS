# NHRS ML Service

## Setup

```bash
python -m pip install -r ml/requirements.txt
```

## Run

```bash
uvicorn ml.api.main:app --reload
```

## Test

```bash
python -m pytest -p no:cacheprovider ml/tests/test_api.py
```
