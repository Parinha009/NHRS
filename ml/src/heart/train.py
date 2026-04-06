from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.src.heart.runtime_artifacts import HeartModel, HeartPreprocessor

MODEL_VERSION = "heart-v1"
TARGET_COLUMN = "target"
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]
DISCLAIMER = (
    "Educational support only. This model is not a medical diagnosis tool and "
    "must not replace licensed clinical judgment or emergency care."
)

ML_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ML_DIR / "data" / "heart.csv"
ARTIFACT_DIR = ML_DIR / "artifacts" / "heart" / "v1"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )
    return dataset


def split_dataset(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        from sklearn.model_selection import train_test_split

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except Exception:
        test_size = max(2, int(round(len(X) * 0.2)))
        positive_indices = y[y == 1].index.tolist()
        negative_indices = y[y == 0].index.tolist()

        test_positive = positive_indices[: max(1, len(positive_indices) // 5)]
        test_negative = negative_indices[: max(1, len(negative_indices) // 5)]
        test_indices = sorted((test_positive + test_negative)[:test_size])
        train_indices = [index for index in X.index if index not in test_indices]

        return (
            X.loc[train_indices],
            X.loc[test_indices],
            y.loc[train_indices],
            y.loc[test_indices],
        )


def build_sklearn_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def train_with_sklearn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, object, object, str]:
    from sklearn.linear_model import LogisticRegression

    preprocessor = build_sklearn_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_transformed, y_train)

    return preprocessor, model, X_test_transformed, "sklearn"


def fit_fallback_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series
) -> HeartModel:
    feature_names = list(X_train.columns)
    X_values = X_train.to_numpy(dtype=float)
    y_values = y_train.to_numpy(dtype=float)

    weights = np.zeros(len(feature_names), dtype=float)
    intercept = 0.0
    learning_rate = 0.1
    regularization = 0.01

    for _ in range(2000):
        logits = np.clip(X_values @ weights + intercept, -50.0, 50.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        error = probabilities - y_values

        weight_gradient = (X_values.T @ error) / len(X_values)
        weight_gradient += regularization * weights
        intercept_gradient = float(error.mean())

        weights -= learning_rate * weight_gradient
        intercept -= learning_rate * intercept_gradient

    return HeartModel(
        feature_names=feature_names,
        coefficients={
            feature: float(weight)
            for feature, weight in zip(feature_names, weights)
        },
        intercept=float(intercept),
    )


def train_with_fallback(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[HeartPreprocessor, HeartModel, pd.DataFrame, str]:
    preprocessor = HeartPreprocessor.fit(
        X_train,
        feature_order=FEATURE_COLUMNS,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    model = fit_fallback_logistic_regression(X_train_transformed, y_train)
    return preprocessor, model, X_test_transformed, "numpy_fallback"


def calculate_metrics(y_true: pd.Series, predictions: object) -> dict[str, float]:
    actual = y_true.to_numpy(dtype=int)
    predicted = np.asarray(list(predictions), dtype=int)

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())

    accuracy = float((actual == predicted).mean())
    precision = (
        float(true_positive / (true_positive + false_positive))
        if (true_positive + false_positive)
        else 0.0
    )
    recall = (
        float(true_positive / (true_positive + false_negative))
        if (true_positive + false_negative)
        else 0.0
    )
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train() -> dict:
    dataset = load_dataset()
    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    try:
        preprocessor, model, X_test_transformed, backend = train_with_sklearn(
            X_train,
            X_test,
            y_train,
        )
    except Exception:
        preprocessor, model, X_test_transformed, backend = train_with_fallback(
            X_train,
            X_test,
            y_train,
        )

    predictions = model.predict(X_test_transformed)
    metrics = calculate_metrics(y_test, predictions)

    metadata = {
        "model_version": MODEL_VERSION,
        "target_column": TARGET_COLUMN,
        "feature_order": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_labels": [int(label) for label in model.classes_],
        "metrics": metrics,
        "training_backend": backend,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    metadata = train()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
