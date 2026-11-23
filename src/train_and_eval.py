"""Воспроизводимый запуск обучения и оценки YOLOv8 на замороженной тестовой выборке (E1 / E2)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_frozen_test() -> str:
    """Проверяет неизменность замороженной тестовой выборки по SHA-256."""
    split_report = json.loads((PROCESSED_DIR / "split_report.json").read_text(encoding="utf-8"))
    expected_sha = split_report["frozen_test_sha256"]

    test_dir = PROCESSED_DIR / "test" / "images"
    entries = []
    for p in sorted(test_dir.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            entries.append(f"{p.name}:{h}")
    actual_sha = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(f"Тестовая выборка изменена! Expected {expected_sha}, got {actual_sha}")
    return actual_sha


def make_runtime_data_yaml(base_yaml_path: Path, run_dir: Path) -> Path:
    """Создаёт временный YAML с абсолютным путём к data/processed для надёжной работы Ultralytics на любой ОС."""
    cfg = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
    raw_path = Path(cfg["path"])
    if not raw_path.is_absolute():
        raw_path = (PROJECT_ROOT / raw_path).resolve()
    cfg["path"] = str(raw_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_yaml = run_dir / "_runtime_ppe.yaml"
    runtime_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_yaml


def count_test_instances(class_names: dict[int, str]) -> dict[int, int]:
    """Точный подсчёт числа эталонных объектов каждого класса в тестовой выборке."""
    counts = {cid: 0 for cid in class_names}
    lbl_dir = PROCESSED_DIR / "test" / "labels"
    for lp in sorted(lbl_dir.glob("*.txt")):
        text = lp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                if cid in counts:
                    counts[cid] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLOv8 experiment from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Путь к YAML-конфигу (например, configs/e1_baseline.yaml)")
    args = parser.parse_args()

    cfg_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    test_sha = verify_frozen_test()
    print(f"[{cfg['experiment_id']}] Frozen test SHA-256 verified: {test_sha[:16]}...")

    device = 0 if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[{cfg['experiment_id']}] Device: {gpu_name} (device={device})")

    project_dir = (PROJECT_ROOT / cfg["project"]).resolve()
    run_name = cfg["name"]
    run_dir = project_dir / run_name
    runtime_data_yaml = make_runtime_data_yaml((PROJECT_ROOT / cfg["data"]).resolve(), run_dir)

    model_path = (PROJECT_ROOT / cfg["model"]).resolve() if (PROJECT_ROOT / cfg["model"]).exists() else cfg["model"]
    model = YOLO(str(model_path))

    # Собираем аргументы обучения
    train_kwargs = {
        "data": str(runtime_data_yaml),
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "seed": seed,
        "deterministic": bool(cfg.get("deterministic", True)),
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": bool(cfg.get("exist_ok", True)),
        "workers": int(cfg.get("workers", 0)),
        "patience": int(cfg.get("patience", 50)),
        "device": device,
        "plots": True,
    }
    # Дополнительные параметры гипотезы (если указаны в конфиге E2, например freeze, mosaic, degrees и т.д.)
    for opt_key in ("freeze", "mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v", "degrees", "scale", "fliplr", "close_mosaic", "lr0", "cos_lr"):
        if opt_key in cfg:
            train_kwargs[opt_key] = cfg[opt_key]

    print(f"[{cfg['experiment_id']}] Starting training with params: {json.dumps({k: v for k, v in train_kwargs.items() if k not in {'data', 'project'}}, ensure_ascii=False)}")
    t0 = time.perf_counter()
    model.train(**train_kwargs)
    train_duration_sec = round(time.perf_counter() - t0, 2)
    print(f"[{cfg['experiment_id']}] Training finished in {train_duration_sec} s ({train_duration_sec / 60:.2f} min)")

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Не найден файл лучших весов: {best_weights}")

    # Итоговая оценка лучших весов на замороженной тестовой выборке
    print(f"[{cfg['experiment_id']}] Evaluating best.pt on frozen TEST split...")
    best_model = YOLO(str(best_weights))
    eval_name = f"{run_name}-test"
    t_eval = time.perf_counter()
    metrics = best_model.val(
        data=str(runtime_data_yaml),
        split="test",
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        device=device,
        workers=0,
        project=str(project_dir),
        name=eval_name,
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    eval_duration_sec = round(time.perf_counter() - t_eval, 2)

    # Извлекаем метрики по классам и общие метрики
    names: dict[int, str] = best_model.names
    test_counts = count_test_instances(names)

    per_class_rows = []
    # metrics.box.ap_class_index содержит индексы классов, для которых вычислены метрики
    ap_indices = [int(i) for i in metrics.box.ap_class_index]
    for pos, cid in enumerate(ap_indices):
        p_c, r_c, ap50_c, ap_c = metrics.box.class_result(pos)
        f1_c = (2.0 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) > 0 else 0.0
        per_class_rows.append(
            {
                "class_id": cid,
                "class_name": names[cid],
                "test_instances": test_counts.get(cid, 0),
                "precision": round(float(p_c), 4),
                "recall": round(float(r_c), 4),
                "f1": round(float(f1_c), 4),
                "ap50": round(float(ap50_c), 4),
                "ap50_95": round(float(ap_c), 4),
            }
        )

    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)
    mf1 = (2.0 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    summary = {
        "experiment_id": cfg["experiment_id"],
        "name": run_name,
        "description": cfg.get("description", ""),
        "config_path": str(cfg_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_base": cfg["model"],
        "best_weights": str(best_weights.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "device": gpu_name,
        "seed": seed,
        "epochs": int(cfg["epochs"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "train_duration_sec": train_duration_sec,
        "eval_duration_sec": eval_duration_sec,
        "frozen_test_sha256": test_sha,
        "overall_test_metrics": {
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(mf1, 4),
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "speed_ms_per_image": {k: round(float(v), 3) for k, v in metrics.speed.items()},
        },
        "per_class_test_metrics": per_class_rows,
    }

    REPORTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_METRICS_DIR / f"{run_name}_test_metrics.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv = REPORTS_METRICS_DIR / f"{run_name}_per_class.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(per_class_rows)

    # Копируем ключевые графики из runs/ в reports/figures/<run_name>/ (чтобы не потерялись при игнорировании runs/)
    fig_dir = REPORTS_FIGURES_DIR / run_name
    fig_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = project_dir / eval_name
    for src_dir, patterns in [
        (run_dir, ["results.png", "results.csv"]),
        (eval_dir, ["confusion_matrix.png", "confusion_matrix_normalized.png", "*PR_curve.png", "*F1_curve.png", "*P_curve.png", "*R_curve.png"]),
    ]:
        for pat in patterns:
            for fp in src_dir.glob(pat):
                shutil.copy2(fp, fig_dir / fp.name)

    print(f"\n=== ИТОГИ {cfg['experiment_id']} ({run_name}) НА ТЕСТОВОЙ ВЫБОРКЕ ===")
    ov = summary["overall_test_metrics"]
    print(
        f"  Precision: {ov['precision']:.4f} | Recall: {ov['recall']:.4f} | F1: {ov['f1']:.4f} | "
        f"mAP@0.5: {ov['mAP50']:.4f} | mAP@0.5:0.95: {ov['mAP50_95']:.4f}"
    )
    print(f"  Train time: {train_duration_sec} s | Test speed (ms/img): {ov['speed_ms_per_image']}")
    print("\nПо классам (test):")
    for r in per_class_rows:
        print(
            f"  [{r['class_id']}] {r['class_name']:16s} (n={r['test_instances']:3d}): "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  AP@0.5={r['ap50']:.3f}  AP@0.5:0.95={r['ap50_95']:.3f}"
        )
    print(f"\nМетрики сохранены в: {out_json} и {out_csv}")
    print(f"Графики скопированы в: {fig_dir}")


if __name__ == "__main__":
    main()
