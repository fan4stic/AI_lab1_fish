from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Собирает ColumnTransformer из числовых и категориальных признаков.

    Параметры
    ---------
    numeric_features : список имён числовых колонок
    categorical_features : список имён категориальных колонок

    Возвращает
    ----------
    ColumnTransformer, готовый для вставки в Pipeline.
    """
    transformers = []

    if len(numeric_features) > 0:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, list(numeric_features)))

    if len(categorical_features) > 0:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipe, list(categorical_features)))

    return ColumnTransformer(transformers)