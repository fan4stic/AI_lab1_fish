"""Обучение моделей E0–E4 с логированием в MLflow."""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.sklearn")


import argparse
import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import mlflow.sklearn
from joblib import dump

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import build_preprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# --- NEW: хеш данных, чтобы фиксировать «версию данных» ---
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# --- NEW: идентификатор коммита для тегов MLflow ---
def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_residuals(y_true, y_pred, path: Path, title: str) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.6)
    ax.axhline(0, color="red", linewidth=1)
    ax.set_xlabel("Прогноз, г")
    ax.set_ylabel("Остаток (y_true − y_pred), г")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pred_vs_true(y_true, y_pred, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.6)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1)
    ax.set_xlabel("Истинная масса, г")
    ax.set_ylabel("Прогноз, г")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Ручной CV-цикл (пункты 2, 3, 4)
# ---------------------------------------------------------------------------

def manual_group_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cv: GroupKFold,
) -> np.ndarray:
    scores = []
    for tr_idx, val_idx in cv.split(X_train, y_train, groups=groups_train):
        pipe = clone(pipeline)
        pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        y_pred = pipe.predict(X_train.iloc[val_idx])
        scores.append(mean_squared_error(y_train.iloc[val_idx], y_pred))
    return np.asarray(scores)


# ---------------------------------------------------------------------------
# Один эксперимент
# ---------------------------------------------------------------------------

def run_experiment(
    *,
    run_name: str,
    pipeline_or_gs,
    X_train, y_train, groups_train,
    X_test, y_test,
    cv: GroupKFold,
    params_log: dict,
    figures_dir: Path,
    preds_dir: Path,
    needs_manual_cv: bool = True,
    extra_params: dict | None = None,   # --- NEW ---
    env_tags: dict | None = None,       # --- NEW ---
) -> dict:
    if needs_manual_cv:
        cv_scores = manual_group_cv(
            pipeline_or_gs, X_train, y_train, groups_train, cv
        )
        cv_mse_mean = float(np.mean(cv_scores))
        cv_mse_std = float(np.std(cv_scores))
        model = clone(pipeline_or_gs)
        model.fit(X_train, y_train)
        best_params = {}
    else:
        gs: GridSearchCV = pipeline_or_gs
        gs.fit(X_train, y_train, groups=groups_train)
        best_idx = gs.best_index_
        cv_mse_mean = -float(gs.cv_results_["mean_test_score"][best_idx])
        cv_mse_std = float(gs.cv_results_["std_test_score"][best_idx])
        model = gs.best_estimator_
        best_params = gs.best_params_

    y_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)

    with mlflow.start_run(run_name=run_name):
        # --- NEW: собираем полный набор параметров ---
        params_to_log = {
            **params_log,
            "cv_scheme": "GroupKFold(n_splits=5, groups=Species)",
            "test_size": 0.2,
            "random_state": 42,
        }
        if extra_params:
            params_to_log.update(extra_params)
        mlflow.log_params(params_to_log)

        if best_params:
            mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})

        # --- NEW: теги с commit SHA и версиями окружения ---
        if env_tags:
            mlflow.set_tags(env_tags)

        mlflow.log_metrics({
            "cv_mse_mean": cv_mse_mean,
            "cv_mse_std": cv_mse_std,
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        })

        # --- NEW: сохраняем остатки в той же таблице ---
        pred_path = preds_dir / f"{run_name}_predictions.csv"
        pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred,
            "residual": y_test.values - y_pred,
        }).to_csv(pred_path, index=False)
        mlflow.log_artifact(str(pred_path))

        res_path = figures_dir / f"{run_name}_residuals.png"
        pvt_path = figures_dir / f"{run_name}_pred_vs_true.png"
        plot_residuals(y_test, y_pred, res_path, run_name)
        plot_pred_vs_true(y_test, y_pred, pvt_path, run_name)
        mlflow.log_artifact(str(res_path))
        mlflow.log_artifact(str(pvt_path))

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )
        run_id = mlflow.active_run().info.run_id

    return {
        "run_name": run_name,
        "run_id": run_id,
        "cv_mse_mean": cv_mse_mean,
        "cv_mse_std": cv_mse_std,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "model": model,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    fix_seed(cfg["seed"])

    df = pd.read_csv(ROOT / cfg["data"]["path"])
    target = cfg["data"]["target"]
    numeric_features: Sequence[str] = list(cfg["data"]["numeric_features"])
    categorical_features: Sequence[str] = list(cfg["data"]["categorical_features"])

    X = df[numeric_features + categorical_features]
    y = df[target]
    groups = df["Species"]

    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
    )

    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    split_df = pd.DataFrame({
        "index": list(X_train.index) + list(X_test.index),
        "split": ["train"] * len(X_train) + ["test"] * len(X_test),
        "Species": list(groups_train) + list(groups_test),
    })
    split_df.to_csv(runs_dir / "split_indices.csv", index=False)

    assert set(X_train.index).isdisjoint(set(X_test.index)), \
        "Ошибка: train и test пересекаются!"
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    print(f"Видов в train: {sorted(groups_train.unique())}")
    print(f"Видов в test:  {sorted(groups_test.unique())}")

    cv = GroupKFold(n_splits=cfg["split"]["n_splits"])

    tracking_path = (ROOT / cfg["mlflow"]["tracking_uri"]).resolve()
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    figures_dir = ROOT / "reports" / "figures"
    preds_dir = runs_dir / "predictions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # --- NEW: собираем общие параметры и теги для всех запусков ---
    data_sha = file_sha256(ROOT / cfg["data"]["path"])
    common_kwargs = {
        "extra_params": {
            "seed": cfg["seed"],
            "numeric_features": ",".join(numeric_features),
            "categorical_features": ",".join(categorical_features),
            "data_path": cfg["data"]["path"],
            "data_sha256": data_sha,
        },
        "env_tags": {
            "git_commit": get_git_commit(),
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "mlflow_version": mlflow.__version__,
            "pandas_version": pd.__version__,
        },
    }
    print(f"data_sha256: {data_sha}")
    print(f"git_commit:  {common_kwargs['env_tags']['git_commit']}")

    def make_pipeline(regressor, use_cat: bool = True):
        cat = categorical_features if use_cat else []
        pre = build_preprocessor(numeric_features, cat)
        return Pipeline([("preprocess", pre), ("regressor", regressor)])

    results = []

    for strategy in ("mean", "median"):
        results.append(run_experiment(
            run_name=f"E0_dummy_{strategy}",
            pipeline_or_gs=make_pipeline(DummyRegressor(strategy=strategy)),
            X_train=X_train, y_train=y_train, groups_train=groups_train,
            X_test=X_test, y_test=y_test,
            cv=cv,
            params_log={"model": f"DummyRegressor(strategy='{strategy}')"},
            figures_dir=figures_dir, preds_dir=preds_dir,
            **common_kwargs,
        ))

    results.append(run_experiment(
        run_name="E1_linear_log1p",
        pipeline_or_gs=make_pipeline(TransformedTargetRegressor(
            regressor=LinearRegression(**cfg["models"]["e1_linear"]),
            func=np.log1p, inverse_func=np.expm1,
        )),
        X_train=X_train, y_train=y_train, groups_train=groups_train,
        X_test=X_test, y_test=y_test,
        cv=cv,
        params_log={
            "model": "LinearRegression + log1p(target)",
            **{f"e1_linear__{k}": v for k, v in cfg["models"]["e1_linear"].items()},
        },
        figures_dir=figures_dir, preds_dir=preds_dir,
        **common_kwargs,
    ))

    ridge_gs = GridSearchCV(
        estimator=make_pipeline(TransformedTargetRegressor(
            regressor=Ridge(random_state=cfg["seed"]),
            func=np.log1p, inverse_func=np.expm1,
        )),
        param_grid={"regressor__regressor__alpha": cfg["models"]["e2_ridge"]["alphas"]},
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=1,
    )
    results.append(run_experiment(
        run_name="E2_ridge_log1p",
        pipeline_or_gs=ridge_gs,
        X_train=X_train, y_train=y_train, groups_train=groups_train,
        X_test=X_test, y_test=y_test,
        cv=cv,
        params_log={
            "model": "Ridge + log1p(target), GridSearchCV по alpha",
            "alphas": cfg["models"]["e2_ridge"]["alphas"],
        },
        figures_dir=figures_dir, preds_dir=preds_dir,
        needs_manual_cv=False,
        **common_kwargs,
    ))

    gb_gs = GridSearchCV(
        estimator=make_pipeline(TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(random_state=cfg["seed"]),
            func=np.log1p, inverse_func=np.expm1,
        )),
        param_grid={
            "regressor__regressor__max_iter": cfg["models"]["e3_gb"]["max_iter"],
            "regressor__regressor__learning_rate": cfg["models"]["e3_gb"]["learning_rate"],
            "regressor__regressor__max_depth": cfg["models"]["e3_gb"]["max_depth"],
        },
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=1,
    )
    results.append(run_experiment(
        run_name="E3_hist_gb_log1p",
        pipeline_or_gs=gb_gs,
        X_train=X_train, y_train=y_train, groups_train=groups_train,
        X_test=X_test, y_test=y_test,
        cv=cv,
        params_log={
            "model": "HistGradientBoosting + log1p(target), GridSearchCV",
            "max_iter": cfg["models"]["e3_gb"]["max_iter"],
            "learning_rate": cfg["models"]["e3_gb"]["learning_rate"],
            "max_depth": cfg["models"]["e3_gb"]["max_depth"],
        },
        figures_dir=figures_dir, preds_dir=preds_dir,
        needs_manual_cv=False,
        **common_kwargs,
    ))

    X_train_e4 = X_train[numeric_features]
    X_test_e4 = X_test[numeric_features]

    results.append(run_experiment(
        run_name="E4_linear_no_species",
        pipeline_or_gs=make_pipeline(
            TransformedTargetRegressor(
                regressor=LinearRegression(),
                func=np.log1p, inverse_func=np.expm1,
            ),
            use_cat=False,
        ),
        X_train=X_train_e4, y_train=y_train, groups_train=groups_train,
        X_test=X_test_e4, y_test=y_test,
        cv=cv,
        params_log={"model": "LinearRegression + log1p, без Species"},
        figures_dir=figures_dir, preds_dir=preds_dir,
        **common_kwargs,
    ))

    results_sorted = sorted(results, key=lambda r: r["cv_mse_mean"])
    best = results_sorted[0]

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    dump(best["model"], models_dir / "best_model.joblib")

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "model"} for r in results_sorted
    ])
    summary_df.to_csv(runs_dir / "summary.csv", index=False)

    print("\n=== Итоговая таблица E0–E4 (сортировка по cv_mse_mean) ===")
    print(summary_df.to_string(index=False))
    print(f"\nЛучшая модель: {best['run_name']}  (run_id={best['run_id']})")
    print("Сохранена в models/best_model.joblib")
    print(f"MLflow UI: mlflow ui --backend-store-uri {tracking_path}")


if __name__ == "__main__":
    main()