"""Эксперимент E0 (Раздел 4.1): проверка предобученной модели yolov8n.pt (COCO) на 10 изображениях без обучения."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
VAL_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "val" / "images"
OUT_DIR = PROJECT_ROOT / "reports" / "e0_pretrained"

SEED = 42
NUM_IMAGES = 10
CONF_THRESHOLD = 0.25


def select_10_val_images() -> list[dict]:
    """Детерминированно выбирает 10 разнообразных изображений из val (из разных групп, с объектами СИЗ)."""
    rows = list(csv.DictReader(MANIFEST_PATH.open("r", encoding="utf-8-sig"), delimiter=";"))
    val_rows = [r for r in rows if r["split"] == "val" and int(r["num_boxes"]) > 0]

    rng = random.Random(SEED)
    rng.shuffle(val_rows)

    selected: list[dict] = []
    used_groups: set[str] = set()
    for r in val_rows:
        if r["group_id"] in used_groups:
            continue
        selected.append(r)
        used_groups.add(r["group_id"])
        if len(selected) == NUM_IMAGES:
            break
    return selected


def make_grid(image_paths: list[Path], out_path: Path, cols: int = 5, tile_w: int = 400, tile_h: int = 300) -> None:
    """Собирает компактный коллаж 2x5 из 10 предсказаний E0 для отчёта."""
    rows = (len(image_paths) + cols - 1) // cols
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 24, dtype=np.uint8)

    for idx, p in enumerate(image_paths):
        pil_im = Image.open(p).convert("RGB")
        im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        h, w = im.shape[:2]
        scale = min(tile_w / w, tile_h / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)

        r, c = divmod(idx, cols)
        y0 = r * tile_h + (tile_h - nh) // 2
        x0 = c * tile_w + (tile_w - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized

    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        out_path.write_bytes(buf.tobytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))
    coco_names = model.names
    print(f"Loaded pretrained model: {MODEL_PATH.name} | classes: {len(coco_names)} (COCO) | device: {device}")

    selected = select_10_val_images()
    results_summary = []
    total_pred_classes: Counter[str] = Counter()
    total_gt_classes: Counter[str] = Counter()
    saved_vis_paths: list[Path] = []

    for idx, row in enumerate(selected, start=1):
        img_path = VAL_IMG_DIR / row["filename"]
        # Загружаем через PIL -> numpy BGR, чтобы не зависеть от кириллицы в пути на Windows
        pil_im = Image.open(img_path).convert("RGB")
        bgr_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        preds = model.predict(source=bgr_im, imgsz=640, conf=CONF_THRESHOLD, device=device, verbose=False)
        res = preds[0]

        detections = []
        pred_counter: Counter[str] = Counter()
        for box in res.boxes:
            cls_id = int(box.cls.item())
            cls_name = coco_names[cls_id]
            conf = round(float(box.conf.item()), 4)
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            detections.append({"class_id": cls_id, "class_name": cls_name, "conf": conf, "bbox_xyxy": xyxy})
            pred_counter[cls_name] += 1
            total_pred_classes[cls_name] += 1

        # Парсим эталонные классы СИЗ из manifest
        gt_dict: dict[str, int] = {}
        if row["classes"]:
            for part in row["classes"].split(", "):
                cname, cnt = part.split(":")
                gt_dict[cname] = int(cnt)
                total_gt_classes[cname] += int(cnt)

        # Сохраняем визуализацию с предсказаниями COCO
        plotted = res.plot()
        out_img = OUT_DIR / f"e0_{idx:02d}_{Path(row['filename']).stem}.jpg"
        ok, buf = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            out_img.write_bytes(buf.tobytes())
        saved_vis_paths.append(out_img)

        results_summary.append(
            {
                "index": idx,
                "filename": row["filename"],
                "orig_name": row["orig_name"],
                "group_id": row["group_id"],
                "size": f"{row['width']}x{row['height']}",
                "gt_ppe_classes": gt_dict,
                "pred_coco_counts": dict(pred_counter),
                "num_predictions": len(detections),
                "detections": detections,
                "saved_visualization": str(out_img.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )
        print(
            f"  [{idx:02d}] {row['filename']} | GT СИЗ: {row['classes']}  -->  "
            f"COCO предск.: {dict(pred_counter) if pred_counter else '(ничего не найдено)'}"
        )

    grid_path = OUT_DIR / "e0_grid.jpg"
    make_grid(saved_vis_paths, grid_path)

    payload = {
        "experiment": "E0_pretrained_yolov8n_coco",
        "model": str(MODEL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_num_classes": len(coco_names),
        "seed": SEED,
        "conf_threshold": CONF_THRESHOLD,
        "split_used": "val (test остаётся замороженным)",
        "num_images": len(results_summary),
        "total_gt_ppe_objects": dict(total_gt_classes),
        "total_pred_coco_objects": dict(total_pred_classes),
        "images": results_summary,
    }
    out_json = OUT_DIR / "e0_predictions.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== СВОДКА E0 ===")
    print("Эталонные объекты СИЗ на 10 кадрах:", dict(total_gt_classes))
    print("Что реально предсказала yolov8n.pt (COCO):", dict(total_pred_classes))
    print(f"Сохранено: {OUT_DIR} (10 изображений + e0_grid.jpg + e0_predictions.json)")


if __name__ == "__main__":
    main()
