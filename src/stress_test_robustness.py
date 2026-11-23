"""Стресс-тест устойчивости лучшей модели E1 к освещению, ракурсу и частичному перекрытию (Раздел 7)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"
TEST_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "labels"
WEIGHTS = PROJECT_ROOT / "runs" / "ppe" / "e1-opora" / "weights" / "best.pt"
OUT_JSON = PROJECT_ROOT / "reports" / "metrics" / "robustness_stress_test.json"
OUT_PLOT = PROJECT_ROOT / "reports" / "figures" / "robustness_stress_test.png"

SEED = 42
CRITICAL_IDS = {2, 4}  # NO-Hardhat, NO-Safety Vest


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def load_gt(lbl_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    if not lbl_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes, cids = [], []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        cids.append(cid)
    return np.array(boxes, dtype=np.float32), np.array(cids, dtype=np.int64)


def transform_image_and_gt(
    bgr: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray, mode: str, idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    if mode == "baseline":
        return bgr, gt_boxes, gt_cls

    if mode == "low_light":
        out = np.clip(bgr.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
        return out, gt_boxes, gt_cls

    if mode == "overexposed":
        out = np.clip(bgr.astype(np.float32) * 1.55 + 25.0, 0, 255).astype(np.uint8)
        return out, gt_boxes, gt_cls

    if mode == "tilted_10deg":
        angle = 10.0 if (idx % 2 == 0) else -10.0
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        out = cv2.warpAffine(bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        if len(gt_boxes) == 0:
            return out, gt_boxes, gt_cls
        new_boxes, new_cls = [], []
        for box, cid in zip(gt_boxes, gt_cls):
            x1, y1, x2, y2 = box
            corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
            rot = corners @ M.T
            rx1 = float(np.clip(rot[:, 0].min(), 0, w))
            ry1 = float(np.clip(rot[:, 1].min(), 0, h))
            rx2 = float(np.clip(rot[:, 0].max(), 0, w))
            ry2 = float(np.clip(rot[:, 1].max(), 0, h))
            if (rx2 - rx1) >= 4.0 and (ry2 - ry1) >= 4.0:
                new_boxes.append([rx1, ry1, rx2, ry2])
                new_cls.append(int(cid))
        return out, np.array(new_boxes, dtype=np.float32).reshape(-1, 4), np.array(new_cls, dtype=np.int64)

    if mode == "partial_occlusion":
        out = bgr.copy()
        rng = np.random.default_rng(SEED + idx)
        # Имитируем балку/колонну/перекрытие: 2 непрозрачных прямоугольника (~12% кадра суммарно)
        for _ in range(2):
            bw = int(w * rng.uniform(0.14, 0.22))
            bh = int(h * rng.uniform(0.22, 0.35))
            x0 = int(rng.integers(int(w * 0.10), max(int(w * 0.11), int(w * 0.90) - bw)))
            y0 = int(rng.integers(int(h * 0.10), max(int(h * 0.11), int(h * 0.90) - bh)))
            cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (75, 78, 82), -1)
        return out, gt_boxes, gt_cls

    raise ValueError(f"Unknown mode: {mode}")


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate_scenario(model: YOLO, image_paths: list[Path], mode: str, device: int | str) -> dict:
    num_classes = len(model.names)
    gt_totals = np.zeros(num_classes, dtype=np.int64)
    # Для каждого класса храним список (conf, is_tp) по всей выборке для точного AP@0.5
    detections_per_cls: dict[int, list[tuple[float, int]]] = defaultdict(list)

    # Также считаем метрики на рабочем пороге conf=0.25
    tp_25, fp_25, fn_25 = 0, 0, 0
    crit_tp_25, crit_gt = 0, 0

    for idx, img_path in enumerate(image_paths):
        bgr = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        gt_boxes, gt_cls = load_gt(TEST_LBL_DIR / f"{img_path.stem}.txt", w, h)
        t_img, t_gt_boxes, t_gt_cls = transform_image_and_gt(bgr, gt_boxes, gt_cls, mode, idx)

        for c in t_gt_cls:
            gt_totals[int(c)] += 1
            if int(c) in CRITICAL_IDS:
                crit_gt += 1

        # Запускаем с низким порогом conf=0.01 для точного расчёта AP@0.5
        res = model.predict(source=t_img, imgsz=640, conf=0.01, iou=0.45, device=device, verbose=False)[0]
        if len(res.boxes):
            pb = res.boxes.xyxy.cpu().numpy()
            ps = res.boxes.conf.cpu().numpy()
            pc = res.boxes.cls.cpu().numpy().astype(np.int64)
            order = np.argsort(-ps)
            pb, ps, pc = pb[order], ps[order], pc[order]
        else:
            pb = np.zeros((0, 4), dtype=np.float32)
            ps = np.zeros((0,), dtype=np.float32)
            pc = np.zeros((0,), dtype=np.int64)

        iou_mat = compute_iou_matrix(pb, t_gt_boxes)
        matched_gt = np.zeros(len(t_gt_boxes), dtype=bool)
        matched_gt_25 = np.zeros(len(t_gt_boxes), dtype=bool)

        for i in range(len(pb)):
            cid = int(pc[i])
            conf = float(ps[i])
            best_j, best_iou = -1, 0.50
            for j in range(len(t_gt_boxes)):
                if not matched_gt[j] and int(t_gt_cls[j]) == cid and iou_mat[i, j] >= best_iou:
                    best_iou = float(iou_mat[i, j])
                    best_j = j
            if best_j >= 0:
                matched_gt[best_j] = True
                detections_per_cls[cid].append((conf, 1))
            else:
                detections_per_cls[cid].append((conf, 0))

            if conf >= 0.25:
                best_j25, best_iou25 = -1, 0.50
                for j in range(len(t_gt_boxes)):
                    if not matched_gt_25[j] and int(t_gt_cls[j]) == cid and iou_mat[i, j] >= best_iou25:
                        best_iou25 = float(iou_mat[i, j])
                        best_j25 = j
                if best_j25 >= 0:
                    matched_gt_25[best_j25] = True
                    tp_25 += 1
                    if cid in CRITICAL_IDS:
                        crit_tp_25 += 1
                else:
                    fp_25 += 1

        fn_25 += int((~matched_gt_25).sum())

    aps = []
    for cid in range(num_classes):
        n_gt = int(gt_totals[cid])
        if n_gt == 0:
            continue
        dets = sorted(detections_per_cls[cid], key=lambda x: -x[0])
        if not dets:
            aps.append(0.0)
            continue
        tps = np.cumsum([d[1] for d in dets], dtype=np.float64)
        fps = np.cumsum([1 - d[1] for d in dets], dtype=np.float64)
        recs = tps / n_gt
        precs = tps / np.maximum(tps + fps, 1e-9)
        aps.append(compute_ap(recs, precs))

    p25 = tp_25 / max(tp_25 + fp_25, 1)
    r25 = tp_25 / max(tp_25 + fn_25, 1)
    f1_25 = (2 * p25 * r25 / (p25 + r25)) if (p25 + r25) > 0 else 0.0
    crit_r25 = crit_tp_25 / max(crit_gt, 1)
    map50 = float(np.mean(aps)) if aps else 0.0

    return {
        "scenario": mode,
        "mAP50": round(map50, 4),
        "precision_conf025": round(p25, 4),
        "recall_conf025": round(r25, 4),
        "f1_conf025": round(f1_25, 4),
        "critical_ppe_recall_conf025": round(crit_r25, 4),
        "tp_conf025": tp_25,
        "fp_conf025": fp_25,
        "fn_conf025": fn_25,
    }


def save_plot(results: list[dict], out_path: Path) -> None:
    label_map = {
        "baseline": "Исходные\nкадры",
        "low_light": "Сумерки\n(ярк. ×0.45)",
        "overexposed": "Засветка\n(ярк. ×1.55)",
        "tilted_10deg": "Наклон\n(±10°)",
        "partial_occlusion": "Перекрытие\n(~12% кадра)",
    }
    labels = [label_map[r["scenario"]] for r in results]
    maps = [r["mAP50"] for r in results]
    recs = [r["recall_conf025"] for r in results]
    crits = [r["critical_ppe_recall_conf025"] for r in results]

    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=140)
    b1 = ax.bar(x - w, maps, w, label="mAP@0.5", color="#2b6cb0")
    b2 = ax.bar(x, recs, w, label="Recall (все, conf=0.25)", color="#38a169")
    b3 = ax.bar(x + w, crits, w, label="Recall крит. СИЗ (NO-Hardhat/Vest)", color="#e53e3e")

    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Значение метрики (0–1)")
    ax.set_title("Стресс-тест устойчивости E1 (YOLOv8n) к условиям съёмки на test (n=110)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(WEIGHTS))
    image_paths = sorted(p for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    scenarios = ["baseline", "low_light", "overexposed", "tilted_10deg", "partial_occlusion"]

    results = []
    base_map = None
    for sc in scenarios:
        r = evaluate_scenario(model, image_paths, sc, device)
        if base_map is None:
            base_map = r["mAP50"]
        r["delta_mAP50"] = round(r["mAP50"] - base_map, 4)
        results.append(r)
        print(
            f"  {sc:18s}: mAP@0.5={r['mAP50']:.4f} (Δ={r['delta_mAP50']:+.4f}) | "
            f"P@0.25={r['precision_conf025']:.3f} | R@0.25={r['recall_conf025']:.3f} | "
            f"Crit_R={r['critical_ppe_recall_conf025']:.3f}"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"weights": str(WEIGHTS.relative_to(PROJECT_ROOT)).replace("\\", "/"), "scenarios": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plot(results, OUT_PLOT)
    print(f"\nSaved: {OUT_JSON} and {OUT_PLOT}")


if __name__ == "__main__":
    main()
