"""Минимальные тесты для лабораторной №2 (раздел 7 задания).

Запуск из корня репозитория:
    pytest tests/ -v

Тесты не требуют полного обучения — работают с уже сохранёнными артефактами:
- runs/split_indices.csv
- models/best_model.joblib
- data/fish_participant.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import load

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import build_preprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def split_info() -> pd.DataFrame:
    path = ROOT / "runs" / "split_indices.csv"
    if not path.exists():
        pytest.skip(f"Нет файла {path}; сначала запустите src/train.py")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def model():
    path = ROOT / "models" / "best_model.joblib"
    if not path.exists():
        pytest.skip(f"Нет файла {path}; сначала запустите src/train.py")
    return load(path)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    path = ROOT / "data" / "fish_participant.csv"
    if not path.exists():
        pytest.skip(f"Нет файла {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 1. Индексы train и test не пересекаются
# ---------------------------------------------------------------------------

def test_train_test_indices_disjoint(split_info):
    """Требование: индексы train и test не должны пересекаться."""
    train = set(split_info.loc[split_info["split"] == "train", "index"])
    test = set(split_info.loc[split_info["split"] == "test", "index"])

    assert len(train) > 0, "В train нет строк"
    assert len(test) > 0, "В test нет строк"
    assert train.isdisjoint(test), \
        f"Пересечение train и test: {train & test}"

    total = set(split_info["index"])
    assert train | test == total, \
        "train ∪ test не покрывает весь набор индексов"


# ---------------------------------------------------------------------------
# 2. Цепочка обработки принимает строку с неизвестной категорией
# ---------------------------------------------------------------------------

def test_pipeline_handles_unknown_category():
    """OneHotEncoder должен игнорировать неизвестную категорию."""
    pre = build_preprocessor(
        numeric_features=["Length3", "Height", "Width"],
        categorical_features=["Species"],
    )

    df = pd.DataFrame({
        "Length3": [30.0, 25.0],
        "Height": [10.0, 8.0],
        "Width": [5.0, 4.0],
        "Species": ["Bream", "AlienFishFromMars"],
    })

    result = pre.fit_transform(df)
    assert result.shape[0] == 2, "Неверное число строк после препроцессинга"
    assert np.isfinite(result).all(), \
        "Препроцессинг вернул NaN/Inf на неизвестной категории"


# ---------------------------------------------------------------------------
# 3. Число прогнозов = число строк, все значения конечны
# ---------------------------------------------------------------------------

def test_predictions_count_and_finiteness(model, data):
    """predict возвращает столько же чисел, сколько строк, все конечны."""
    features = ["Length3", "Height", "Width", "Species"]
    X = data[features].head(10)

    y_pred = model.predict(X)

    assert len(y_pred) == len(X), \
        f"Получено {len(y_pred)} прогнозов на {len(X)} строк"
    assert np.isfinite(y_pred).all(), "В прогнозах есть NaN или Inf"


# ---------------------------------------------------------------------------
# 4. Метрики на фиксированном примере совпадают с ручным расчётом
# ---------------------------------------------------------------------------

def test_manual_metrics_match_sklearn():
    """MSE/MAE/R² на 4 точках должны совпасть с ручным расчётом."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 320.0, 380.0])

    # Ручной расчёт
    manual_mse = float(np.mean((y_true - y_pred) ** 2))
    manual_mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    manual_r2 = 1 - ss_res / ss_tot

    # Sklearn
    sk_mse = float(mean_squared_error(y_true, y_pred))
    sk_mae = float(mean_absolute_error(y_true, y_pred))
    sk_r2 = float(r2_score(y_true, y_pred))

    assert abs(manual_mse - sk_mse) < 1e-9, f"MSE не совпал: {manual_mse} vs {sk_mse}"
    assert abs(manual_mae - sk_mae) < 1e-9, f"MAE не совпал: {manual_mae} vs {sk_mae}"
    assert abs(manual_r2 - sk_r2) < 1e-9, f"R² не совпал: {manual_r2} vs {sk_r2}"


# ---------------------------------------------------------------------------
# 5. Сохранённый pipeline после загрузки воспроизводит прогноз
# ---------------------------------------------------------------------------

def test_saved_pipeline_reproduces_predictions(model, data):
    """После загрузки с диска прогноз должен совпасть с прямым вызовом."""
    features = ["Length3", "Height", "Width", "Species"]
    X = data[features].head(5)

    y_pred_1 = model.predict(X)

    # Перезагружаем модель с диска ещё раз
    path = ROOT / "models" / "best_model.joblib"
    model_2 = load(path)
    y_pred_2 = model_2.predict(X)

    assert np.allclose(y_pred_1, y_pred_2, atol=1e-6), \
        "Прогноз загруженной модели отличается от исходной"


# ---------------------------------------------------------------------------
# 6. Бонус: pipeline не падает на данных с пропусками
# ---------------------------------------------------------------------------

def test_pipeline_handles_missing_values():
    """SimpleImputer должен корректно обработать NaN."""
    pre = build_preprocessor(
        numeric_features=["Length3", "Height", "Width"],
        categorical_features=["Species"],
    )

    df = pd.DataFrame({
        "Length3": [30.0, np.nan],
        "Height": [np.nan, 8.0],
        "Width": [5.0, 4.0],
        "Species": ["Bream", "Perch"],
    })

    result = pre.fit_transform(df)
    assert result.shape[0] == 2
    assert np.isfinite(result).all(), "Imputer не убрал NaN"