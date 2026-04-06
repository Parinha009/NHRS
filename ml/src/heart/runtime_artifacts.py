from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def normalize_category(value: Any) -> Any:
    if pd.isna(value):
        return value
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass
class HeartPreprocessor:
    feature_order: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_fill_values: dict[str, float]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]
    categorical_fill_values: dict[str, Any]
    categorical_categories: dict[str, list[Any]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        feature_order: list[str],
        numeric_features: list[str],
        categorical_features: list[str],
    ) -> "HeartPreprocessor":
        numeric_fill_values: dict[str, float] = {}
        numeric_means: dict[str, float] = {}
        numeric_scales: dict[str, float] = {}
        categorical_fill_values: dict[str, Any] = {}
        categorical_categories: dict[str, list[Any]] = {}

        for feature in numeric_features:
            series = pd.to_numeric(frame[feature], errors="coerce")
            fill_value = float(series.median())
            filled = series.fillna(fill_value).astype(float)
            numeric_fill_values[feature] = fill_value
            numeric_means[feature] = float(filled.mean())
            scale = float(filled.std(ddof=0))
            numeric_scales[feature] = scale if scale else 1.0

        for feature in categorical_features:
            series = pd.to_numeric(frame[feature], errors="coerce")
            mode = series.mode(dropna=True)
            fill_value = normalize_category(mode.iloc[0] if not mode.empty else 0)
            filled = series.fillna(fill_value).map(normalize_category)
            categories = sorted(set(filled.tolist()))
            categorical_fill_values[feature] = fill_value
            categorical_categories[feature] = categories

        return cls(
            feature_order=feature_order,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_fill_values=numeric_fill_values,
            numeric_means=numeric_means,
            numeric_scales=numeric_scales,
            categorical_fill_values=categorical_fill_values,
            categorical_categories=categorical_categories,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=frame.index)

        for feature in self.numeric_features:
            series = pd.to_numeric(frame[feature], errors="coerce")
            filled = series.fillna(self.numeric_fill_values[feature]).astype(float)
            centered = filled - self.numeric_means[feature]
            transformed[feature] = centered / self.numeric_scales[feature]

        for feature in self.categorical_features:
            series = pd.to_numeric(frame[feature], errors="coerce")
            filled = series.fillna(self.categorical_fill_values[feature]).map(
                normalize_category
            )
            for category in self.categorical_categories[feature]:
                transformed[f"{feature}__{category}"] = (
                    filled == category
                ).astype(float)

        return transformed


@dataclass
class HeartModel:
    feature_names: list[str]
    coefficients: dict[str, float]
    intercept: float
    classes_: tuple[int, int] = (0, 1)

    def _aligned_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.reindex(columns=self.feature_names, fill_value=0.0).astype(float)

    def predict_proba(self, frame: pd.DataFrame) -> list[list[float]]:
        aligned = self._aligned_frame(frame)
        weights = np.array(
            [self.coefficients[feature] for feature in self.feature_names],
            dtype=float,
        )
        logits = np.clip(aligned.to_numpy(dtype=float) @ weights + self.intercept, -50, 50)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probabilities, probabilities]).tolist()

    def predict(self, frame: pd.DataFrame) -> list[int]:
        probabilities = self.predict_proba(frame)
        return [int(row[1] >= 0.5) for row in probabilities]
