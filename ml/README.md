# NHRS ML Service

## Setup

```bash
python -m pip install -r ml/requirements.txt
```

## Data

`ml/data/heart.csv` is a small placeholder dataset included so the heart pipeline works out of the box.
You can replace it with a real dataset as long as you keep the same column headers, including `target`.

## Train Heart Model

```bash
python -m ml.src.heart.train
```

## Run

```bash
uvicorn ml.api.main:app --reload
```

## Test

```bash
python -m pytest -p no:cacheprovider ml/tests/test_api.py ml/tests/test_heart_api.py
```
