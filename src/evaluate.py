"""Анализ ошибок лучшей модели (раздел 6 задания).

Запуск из корня репозитория:
    python src/evaluate.py

Что делает:
1. Загружает лучшую модель и то же разбиение, что использовал train.py.
2. Считает MSE/MAE/R² на замороженном тесте.
3. Находит 5 крупнейших абсолютных ошибок вместе с исходными признаками.
4. Считает MAE по видам рыб и по квартилям длины.
5. Проверяет физически невозможные прогнозы.
6. Строит графики остатков, pred-vs-true и топ-5 ошибок.
7. Выводит всё в консоль и сохраняет в runs/ и reports/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import load

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    cfg = load_config(ROOT / "configs" / "train.yaml")
    target = cfg["data"]["target"]
    numeric_features = list(cfg["data"]["numeric_features"])
    categorical_features = list(cfg["data"]["categorical_features"])

    # ------------------------------------------------------------------
    # 1. Загрузка данных и восстановление тестовой выборки
    # ------------------------------------------------------------------
    full = pd.read_csv(ROOT / cfg["data"]["path"])
    split_info = pd.read_csv(ROOT / "runs" / "split_indices.csv")

    test_indices = split_info.loc[split_info["split"] == "test", "index"].tolist()
    test_df = full.loc[test_indices].reset_index(drop=True)

    X_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    print(f"Тестовая выборка: {len(test_df)} объектов")
    print(f"Индексы теста сохранены в runs/split_indices.csv "
          f"(seed=42, test_size=0.2)")

    # ------------------------------------------------------------------
    # 2. Загрузка лучшей модели и предсказания
    # ------------------------------------------------------------------
    model = load(ROOT / "models" / "best_model.joblib")
    y_pred = model.predict(X_test)

    m = compute_metrics(y_test, y_pred)
    print("\n=== Метрики лучшей модели на тесте ===")
    print(f"MSE = {m['mse']:.2f}")
    print(f"MAE = {m['mae']:.2f}")
    print(f"R²  = {m['r2']:.4f}")

    # Собираем единую таблицу для дальнейшего анализа
    analysis = test_df.copy()
    analysis["y_true"] = y_test.values
    analysis["y_pred"] = y_pred
    analysis["residual"] = analysis["y_true"] - analysis["y_pred"]
    analysis["abs_residual"] = analysis["residual"].abs()

    # ------------------------------------------------------------------
    # 3. Топ-5 крупнейших абсолютных ошибок
    # ------------------------------------------------------------------
    print("\n=== Топ-5 крупнейших абсолютных ошибок ===")
    top5 = analysis.sort_values("abs_residual", ascending=False).head(5)
    top5_cols = (
        ["Species", "Length3", "Height", "Width", "y_true", "y_pred", "residual"]
    )
    top5_out = top5[top5_cols].round(2)
    print(top5_out.to_string(index=False))
    top5_out.to_csv(ROOT / "runs" / "top_errors.csv", index=False)
    print("\nСохранено в runs/top_errors.csv")

    # ------------------------------------------------------------------
    # 4. MAE по видам
    # ------------------------------------------------------------------
    print("\n=== MAE по видам (на тесте) ===")
    by_species = (
        analysis
        .groupby("Species")
        .apply(
            lambda g: pd.Series({
                "n": len(g),
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                "mse": mean_squared_error(g["y_true"], g["y_pred"]),
                "mean_abs_residual": g["abs_residual"].mean(),
            })
        )
        .round(2)
        .sort_values("mae")
    )
    print(by_species.to_string())
    by_species.to_csv(ROOT / "runs" / "mae_by_species.csv")
    print("\nСохранено в runs/mae_by_species.csv")

    # ------------------------------------------------------------------
    # 5. MAE по диапазонам размера (квартили Length3)
    # ------------------------------------------------------------------
    print("\n=== MAE по диапазонам размера (квартили Length3) ===")
    q1, q2 = analysis["Length3"].quantile([1 / 3, 2 / 3])

    def size_bin(x: float) -> str:
        if x <= q1:
            return "small"
        elif x <= q2:
            return "medium"
        return "large"

    analysis["size_bin"] = analysis["Length3"].apply(size_bin)
    by_size = (
        analysis
        .groupby("size_bin")
        .apply(
            lambda g: pd.Series({
                "n": len(g),
                "Length3_min": g["Length3"].min(),
                "Length3_max": g["Length3"].max(),
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                "mean_abs_residual": g["abs_residual"].mean(),
            })
        )
        .round(2)
    )
    print(f"Границы квартилей Length3: {q1:.1f}, {q2:.1f}")
    print(by_size.to_string())
    by_size.to_csv(ROOT / "runs" / "mae_by_size.csv")
    print("\nСохранено в runs/mae_by_size.csv")

    # ------------------------------------------------------------------
    # 6. Проверка физически невозможных прогнозов
    # ------------------------------------------------------------------
    print("\n=== Проверка физической осмысленности прогнозов ===")
    n = len(analysis)
    n_neg = int((analysis["y_pred"] <= 0).sum())
    n_huge = int((analysis["y_pred"] > 5000).sum())
    print(f"Всего прогнозов: {n}")
    print(f"Прогнозов ≤ 0 г: {n_neg}")
    print(f"Прогнозов > 5000 г: {n_huge}")
    if n_neg == 0 and n_huge == 0:
        print("Все прогнозы лежат в физически осмысленном диапазоне.")
    else:
        print("ВНИМАНИЕ: есть физически невозможные прогнозы — см. таблицу ниже.")
        bad = analysis[(analysis["y_pred"] <= 0) | (analysis["y_pred"] > 5000)]
        print(bad[["Species", "Length3", "y_true", "y_pred"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 7. Графики
    # ------------------------------------------------------------------
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 7.1 Остатки vs прогноз
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(analysis["y_pred"], analysis["residual"], alpha=0.7)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Прогноз, г")
    ax.set_ylabel("Остаток (y_true − y_pred), г")
    ax.set_title("Остатки лучшей модели на тесте")
    fig.tight_layout()
    fig.savefig(figures_dir / "best_model_residuals.png", dpi=120)
    plt.close(fig)

    # 7.2 Pred vs true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(analysis["y_true"], analysis["y_pred"], alpha=0.7)
    lo = min(analysis["y_true"].min(), analysis["y_pred"].min())
    hi = max(analysis["y_true"].max(), analysis["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
    ax.set_xlabel("Истинная масса, г")
    ax.set_ylabel("Прогноз, г")
    ax.set_title("Прогноз vs истина (лучшая модель)")
    fig.tight_layout()
    fig.savefig(figures_dir / "best_model_pred_vs_true.png", dpi=120)
    plt.close(fig)

    # 7.3 Топ-5 ошибок
    fig, ax = plt.subplots(figsize=(7, 4))
    top5_sorted = top5.sort_values("abs_residual", ascending=True)
    y_pos = np.arange(len(top5_sorted))
    ax.barh(y_pos, top5_sorted["abs_residual"], color="tomato")
    labels = [
        f"{sp} (L3={l3:.1f})"
        for sp, l3 in zip(top5_sorted["Species"], top5_sorted["Length3"])
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("|Остаток|, г")
    ax.set_title("Топ-5 крупнейших абсолютных ошибок")
    fig.tight_layout()
    fig.savefig(figures_dir / "best_model_top_errors.png", dpi=120)
    plt.close(fig)

    print("\nГрафики сохранены в reports/figures/:")
    print("  best_model_residuals.png")
    print("  best_model_pred_vs_true.png")
    print("  best_model_top_errors.png")

    # ------------------------------------------------------------------
    # 8. Итог для отчёта
    # ------------------------------------------------------------------
    print("\n=== Сводка для отчёта ===")
    print(f"Лучшая модель: HistGradientBoosting (см. runs/summary.csv)")
    print(f"Тестовых объектов: {n}")
    print(f"MSE = {m['mse']:.1f}, MAE = {m['mae']:.1f}, R² = {m['r2']:.4f}")
    worst = top5.iloc[0]
    print(
        f"Крупнейшая ошибка: {worst['Species']} "
        f"y_true={worst['y_true']:.0f} г, y_pred={worst['y_pred']:.0f} г, "
        f"остаток={worst['residual']:.0f} г"
    )
    print(f"Прогнозов ≤ 0: {n_neg}, прогнозов > 5000 г: {n_huge}")


if __name__ == "__main__":
    main()