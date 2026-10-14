"""Экспорт метаданных MLflow в CSV и markdown (для сдачи).

Запуск из корня репозитория:
    python src/export_mlflow.py

Создаёт:
    reports/mlflow_runs_export.csv   — полная выгрузка всех запусков
    reports/mlflow_summary.md        — человекочитаемая сводка
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import mlflow  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACKING_URI = (ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "fishgrow-lab02"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    df = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
    if df.empty:
        raise SystemExit(
            f"Эксперимент '{EXPERIMENT_NAME}' пуст. "
            f"Сначала запустите src/train.py."
        )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # --- 1. Полная выгрузка в CSV ---
    csv_path = reports_dir / "mlflow_runs_export.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}  ({len(df)} запусков, {df.shape[1]} колонок)")

    # --- 2. Человекочитаемая сводка в markdown ---
    keys_metrics = [
        "metrics.cv_mse_mean",
        "metrics.cv_mse_std",
        "metrics.test_mse",
        "metrics.test_mae",
        "metrics.test_r2",
    ]
    keys_params = [
        "params.model",
        "params.seed",
        "params.test_size",
        "params.random_state",
        "params.data_sha256",
    ]
    keys_tags = [
        "tags.git_commit",
        "tags.python_version",
        "tags.sklearn_version",
        "tags.mlflow_version",
        "tags.pandas_version",
    ]

    summary = df[[
        "tags.mlflow.runName", "run_id", *keys_metrics,
    ]].copy()
    summary.columns = [
        "run_name", "run_id",
        "cv_mse_mean", "cv_mse_std", "test_mse", "test_mae", "test_r2",
    ]
    summary = summary.sort_values("cv_mse_mean")

    lines = ["# MLflow: сводка запусков эксперимента fishgrow-lab02", ""]

    lines.append("## Таблица метрик")
    lines.append("")
    lines.append(summary.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    # Параметры и теги — из любого запуска (они одинаковы для всех)
    sample = df.iloc[0]

    lines.append("## Параметры (общие для всех запусков)")
    lines.append("")
    for k in keys_params:
        if k in df.columns:
            lines.append(f"- `{k.replace('params.', '')}`: {sample[k]}")
    lines.append("")

    lines.append("## Теги окружения (общие для всех запусков)")
    lines.append("")
    for k in keys_tags:
        if k in df.columns:
            lines.append(f"- `{k.replace('tags.', '')}`: {sample[k]}")
    lines.append("")

    lines.append("## Как посмотреть журнал")
    lines.append("")
    lines.append("```")
    lines.append('$env:MLFLOW_ALLOW_FILE_STORE = "true"')
    lines.append("mlflow ui --backend-store-uri mlruns --workers 1")
    lines.append("```")
    lines.append("")
    lines.append("Открыть в браузере `http://127.0.0.1:5000`, выбрать эксперимент `fishgrow-lab02`.")
    lines.append("")

    md_path = reports_dir / "mlflow_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown: {md_path}")

    # --- 3. Краткая информация в консоль ---
    print("\n=== Сводка ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()