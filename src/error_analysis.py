"""Повторно используемый модуль анализа и визуализации 6 категорий ошибок детектора (Раздел 6)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"
TEST_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "labels"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "ppe" / "e1-opora" / "weights" / "best.pt"
OUT_DIR = PROJECT_ROOT / "reports" / "error_galleries"

CRITICAL_CLASS_IDS = {2, 4}  # NO-Hardhat, NO-Safety Vest
CONFUSABLE_PAIRS = {
    frozenset({0, 2}),  # Hardhat <-> NO-Hardhat
    frozenset({1, 3}),  # Mask <-> NO-Mask
    frozenset({4, 7}),  # NO-Safety Vest <-> Safety Vest
    frozenset({8, 9}),  # machinery <-> vehicle
}


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


def load_gt_boxes(lbl_path: Path, w: int, h: int) -> list[dict]:
    if not lbl_path.exists():
        return []
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    items = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = max(0.0, (cx - bw / 2.0) * w)
        y1 = max(0.0, (cy - bh / 2.0) * h)
        x2 = min(float(w), (cx + bw / 2.0) * w)
        y2 = min(float(h), (cy + bh / 2.0) * h)
        items.append(
            {
                "class_id": cid,
                "bbox": [x1, y1, x2, y2],
                "area_ratio": bw * bh,
                "min_side_px": min(bw * w, bh * h),
            }
        )
    return items


def draw_label_box(
    img: np.ndarray,
    bbox: list[float],
    text: str,
    color: tuple[int, int, int],
    thickness: int = 2,
    label_pos: str = "top",
) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font_scale = max(0.42, min(w, h) / 1200.0)
    t_thick = max(1, int(round(font_scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, t_thick)
    if label_pos == "bottom":
        ty1 = min(h - th - 6, y2)
    else:
        ty1 = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, ty1), (min(w - 1, x1 + tw + 6), min(h - 1, ty1 + th + 6)), color, -1)
    cv2.putText(
        img,
        text,
        (x1 + 3, min(h - 2, ty1 + th + 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        t_thick,
        cv2.LINE_AA,
    )


def render_error_overlay(
    image_bgr: np.ndarray,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    class_names: dict[int, str],
    highlight_gt_idx: int | None = None,
    highlight_pred_idx: int | None = None,
    header_text: str | None = None,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Повторно используемый визуализатор ошибок (проверяется автотестом Раздела 8).
    Зелёный: эталон (GT), синий: предсказания (Pred), красный/жёлтый: выделенная ошибка.
    Если задан output_size=(W, H), возвращает изображение ровно размера (H, W, 3).
    """
    canvas = image_bgr.copy()

    # Сначала рисуем обычные GT (зелёный) и Pred (синий)
    for idx, g in enumerate(gt_boxes):
        if idx == highlight_gt_idx:
            continue
        cname = class_names.get(g["class_id"], str(g["class_id"]))
        draw_label_box(canvas, g["bbox"], f"GT:{cname}", (46, 160, 67), thickness=2, label_pos="top")

    for idx, p in enumerate(pred_boxes):
        if idx == highlight_pred_idx:
            continue
        cname = class_names.get(p["class_id"], str(p["class_id"]))
        draw_label_box(
            canvas, p["bbox"], f"P:{cname} {p['conf']:.2f}", (210, 110, 30), thickness=2, label_pos="bottom"
        )

    # Поверх выделяем конкретную ошибку ярким толстым контуром
    if highlight_gt_idx is not None and 0 <= highlight_gt_idx < len(gt_boxes):
        g = gt_boxes[highlight_gt_idx]
        cname = class_names.get(g["class_id"], str(g["class_id"]))
        draw_label_box(canvas, g["bbox"], f"ERR-GT:{cname}", (0, 215, 255), thickness=4, label_pos="top")

    if highlight_pred_idx is not None and 0 <= highlight_pred_idx < len(pred_boxes):
        p = pred_boxes[highlight_pred_idx]
        cname = class_names.get(p["class_id"], str(p["class_id"]))
        draw_label_box(
            canvas, p["bbox"], f"ERR-P:{cname} {p['conf']:.2f}", (35, 35, 235), thickness=4, label_pos="bottom"
        )

    if output_size is not None:
        out_w, out_h = output_size
        banner_h = 34 if header_text else 0
        body_h = max(1, out_h - banner_h)
        h, w = canvas.shape[:2]
        scale = min(out_w / w, body_h / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)

        out = np.full((out_h, out_w, 3), 26, dtype=np.uint8)
        y0 = banner_h + (body_h - nh) // 2
        x0 = (out_w - nw) // 2
        out[y0 : y0 + nh, x0 : x0 + nw] = resized

        if header_text:
            cv2.rectangle(out, (0, 0), (out_w, banner_h), (18, 22, 30), -1)
            cv2.putText(
                out,
                header_text[:85],
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
        return out

    if header_text:
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (18, 22, 30), -1)
        cv2.putText(
            canvas, header_text[:100], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
        )
    return canvas


def build_gallery_grid(
    tiles: list[np.ndarray], cols: int = 2, tile_size: tuple[int, int] = (640, 440)
) -> np.ndarray:
    """Собирает единое изображение галереи строго заданного размера."""
    tile_w, tile_h = tile_size
    if not tiles:
        return np.full((tile_h, tile_w, 3), 30, dtype=np.uint8)
    rows = (len(tiles) + cols - 1) // cols
    grid = np.full((rows * tile_h, cols * tile_w, 3), 20, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        if tile.shape[1] != tile_w or tile.shape[0] != tile_h:
            tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, cols)
        grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    return grid


def save_jpg(img: np.ndarray, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        path.write_bytes(buf.tobytes())


def analyze_errors(weights_path: Path = DEFAULT_WEIGHTS, conf_thres: float = 0.25, iou_nms: float = 0.45) -> dict:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights_path))
    class_names: dict[int, str] = model.names

    categories: dict[str, list[dict]] = {
        "01_high_conf_fp": [],
        "02_missed_critical": [],
        "03_low_iou_matches": [],
        "04_class_confusion": [],
        "05_small_and_occluded": [],
        "06_suspected_gt_errors": [],
    }

    img_paths = sorted(p for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    img_cache_meta: dict[str, dict] = {}

    for img_path in img_paths:
        bgr = cv2.imdecode(np.frombuffer(img_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        gt_list = load_gt_boxes(TEST_LBL_DIR / f"{img_path.stem}.txt", w, h)

        res = model.predict(source=bgr, imgsz=640, conf=conf_thres, iou=iou_nms, device=device, verbose=False)[0]
        pred_list = []
        for b in res.boxes:
            pred_list.append(
                {
                    "class_id": int(b.cls.item()),
                    "conf": float(b.conf.item()),
                    "bbox": [float(v) for v in b.xyxy[0].tolist()],
                }
            )
        # Сортируем предсказания по убыванию уверенности
        pred_list.sort(key=lambda x: x["conf"], reverse=True)

        img_cache_meta[img_path.name] = {"path": img_path, "gt": gt_list, "pred": pred_list, "w": w, "h": h}

        gt_arr = np.array([g["bbox"] for g in gt_list], dtype=np.float32) if gt_list else np.zeros((0, 4), dtype=np.float32)
        pr_arr = np.array([p["bbox"] for p in pred_list], dtype=np.float32) if pred_list else np.zeros((0, 4), dtype=np.float32)
        iou_mat = compute_iou_matrix(pr_arr, gt_arr)
        gt_self_iou = compute_iou_matrix(gt_arr, gt_arr)

        # 1. Стандартное сопоставление TP при IoU >= 0.50 (одинаковый класс)
        matched_gt = np.zeros(len(gt_list), dtype=bool)
        matched_pred = np.zeros(len(pred_list), dtype=bool)
        for pi, p in enumerate(pred_list):
            best_gi = -1
            best_iou = 0.50
            for gi, g in enumerate(gt_list):
                if not matched_gt[gi] and g["class_id"] == p["class_id"] and iou_mat[pi, gi] >= best_iou:
                    best_iou = float(iou_mat[pi, gi])
                    best_gi = gi
            if best_gi >= 0:
                matched_gt[best_gi] = True
                matched_pred[pi] = True

        # 2. Разбираем несопоставленные предсказания (FP)
        for pi, p in enumerate(pred_list):
            if matched_pred[pi]:
                continue
            cname_p = class_names[p["class_id"]]
            max_any_iou = float(iou_mat[pi].max()) if len(gt_list) > 0 else 0.0
            best_any_gi = int(iou_mat[pi].argmax()) if len(gt_list) > 0 else -1

            # 2a. Сопоставленный прямоугольник того же класса с низким IoU (0.15 <= IoU < 0.50)
            same_cls_ious = [
                (gi, float(iou_mat[pi, gi]))
                for gi, g in enumerate(gt_list)
                if g["class_id"] == p["class_id"] and 0.15 <= iou_mat[pi, gi] < 0.50
            ]
            if same_cls_ious:
                gi_low, low_iou = max(same_cls_ious, key=lambda x: x[1])
                categories["03_low_iou_matches"].append(
                    {
                        "filename": img_path.name,
                        "pred_idx": pi,
                        "gt_idx": gi_low,
                        "pred_class": cname_p,
                        "gt_class": class_names[gt_list[gi_low]["class_id"]],
                        "conf": round(p["conf"], 3),
                        "iou": round(low_iou, 3),
                        "score": p["conf"] * (1.0 - low_iou),
                        "summary": f"{img_path.name}: {cname_p} (conf={p['conf']:.2f}, IoU={low_iou:.2f})",
                    }
                )
                continue

            # 2b. Путаница между похожими (или пересекающимися) классами (IoU >= 0.35 с другим классом)
            diff_cls_ious = [
                (gi, float(iou_mat[pi, gi]))
                for gi, g in enumerate(gt_list)
                if g["class_id"] != p["class_id"] and iou_mat[pi, gi] >= 0.35
            ]
            if diff_cls_ious:
                gi_cf, cf_iou = max(diff_cls_ious, key=lambda x: x[1])
                gt_cid = gt_list[gi_cf]["class_id"]
                is_pair = frozenset({p["class_id"], gt_cid}) in CONFUSABLE_PAIRS
                categories["04_class_confusion"].append(
                    {
                        "filename": img_path.name,
                        "pred_idx": pi,
                        "gt_idx": gi_cf,
                        "pred_class": cname_p,
                        "gt_class": class_names[gt_cid],
                        "conf": round(p["conf"], 3),
                        "iou": round(cf_iou, 3),
                        "is_semantic_pair": is_pair,
                        "score": (2.0 if is_pair else 1.0) * p["conf"] * cf_iou,
                        "summary": f"{img_path.name}: GT={class_names[gt_cid]} -> Pred={cname_p} (conf={p['conf']:.2f}, IoU={cf_iou:.2f})",
                    }
                )
                continue

            # 2c. Предположительная ошибка эталонной разметки: очень высокая уверенность (>= 0.65) и нет близкого GT того же класса
            same_cls_max = max(
                (float(iou_mat[pi, gi]) for gi, g in enumerate(gt_list) if g["class_id"] == p["class_id"]), default=0.0
            )
            if p["conf"] >= 0.65 and same_cls_max < 0.10:
                categories["06_suspected_gt_errors"].append(
                    {
                        "filename": img_path.name,
                        "pred_idx": pi,
                        "gt_idx": None,
                        "pred_class": cname_p,
                        "conf": round(p["conf"], 3),
                        "max_any_gt_iou": round(max_any_iou, 3),
                        "score": p["conf"],
                        "summary": f"{img_path.name}: unlabelled? Pred={cname_p} (conf={p['conf']:.2f})",
                    }
                )

            # 2d. Ложноположительные срабатывания с высокой уверенностью (conf >= 0.45)
            if p["conf"] >= 0.45:
                categories["01_high_conf_fp"].append(
                    {
                        "filename": img_path.name,
                        "pred_idx": pi,
                        "gt_idx": best_any_gi if max_any_iou >= 0.15 else None,
                        "pred_class": cname_p,
                        "conf": round(p["conf"], 3),
                        "max_any_gt_iou": round(max_any_iou, 3),
                        "score": p["conf"],
                        "summary": f"{img_path.name}: FP {cname_p} (conf={p['conf']:.2f})",
                    }
                )

        # 3. Разбираем пропущенные эталонные объекты (FN)
        for gi, g in enumerate(gt_list):
            if matched_gt[gi]:
                continue
            cname_g = class_names[g["class_id"]]
            is_small = g["area_ratio"] < 0.01 or g["min_side_px"] < 32.0
            max_other_gt_iou = max(
                (float(gt_self_iou[gi, j]) for j in range(len(gt_list)) if j != gi), default=0.0
            )
            is_occluded = max_other_gt_iou >= 0.25

            # 3a. Пропуск критичного класса (NO-Hardhat, NO-Safety Vest)
            if g["class_id"] in CRITICAL_CLASS_IDS:
                categories["02_missed_critical"].append(
                    {
                        "filename": img_path.name,
                        "pred_idx": None,
                        "gt_idx": gi,
                        "gt_class": cname_g,
                        "area_ratio": round(g["area_ratio"], 5),
                        "min_side_px": round(g["min_side_px"], 1),
                        "is_small": is_small,
                        "is_occluded": is_occluded,
                        # Приоритет крупным и заметным пропускам, которые модель не должна была пропустить
                        "score": g["area_ratio"],
                        "summary": f"{img_path.name}: missed {cname_g} (min={g['min_side_px']:.0f}px, area={g['area_ratio']*100:.2f}%)",
                    }
                )

            # 3b. Малые и перекрытые объекты (FN)
            if is_small or is_occluded:
                categories["05_small_and_occluded"].append(
                    {
                        "filename": img_path.name,
                        "pred_idx": None,
                        "gt_idx": gi,
                        "gt_class": cname_g,
                        "area_ratio": round(g["area_ratio"], 5),
                        "min_side_px": round(g["min_side_px"], 1),
                        "max_other_gt_iou": round(max_other_gt_iou, 3),
                        "is_small": is_small,
                        "is_occluded": is_occluded,
                        "score": (1.0 if (is_small and is_occluded) else 0.5) + max_other_gt_iou,
                        "summary": f"{img_path.name}: FN {cname_g} (small={is_small}, occ={is_occluded}, min={g['min_side_px']:.0f}px)",
                    }
                )

    # Сортируем каждую категорию, выбираем до 4 уникальных изображений в галерею (чтобы было разнообразно!)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog_summary: dict[str, dict] = {}
    selected_examples: list[dict] = []

    for cat_key, entries in categories.items():
        entries.sort(key=lambda x: x["score"], reverse=True)
        cat_dir = OUT_DIR / cat_key
        if cat_dir.exists():
            for old in cat_dir.glob("*.jpg"):
                old.unlink()
        cat_dir.mkdir(parents=True, exist_ok=True)

        chosen = []
        seen_files = set()
        for e in entries:
            if e["filename"] in seen_files:
                continue
            chosen.append(e)
            seen_files.add(e["filename"])
            if len(chosen) == 4:
                break

        tiles = []
        saved_files = []
        for idx, e in enumerate(chosen, start=1):
            meta = img_cache_meta[e["filename"]]
            bgr = cv2.imdecode(np.frombuffer(meta["path"].read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            rendered = render_error_overlay(
                image_bgr=bgr,
                gt_boxes=meta["gt"],
                pred_boxes=meta["pred"],
                class_names=class_names,
                highlight_gt_idx=e.get("gt_idx"),
                highlight_pred_idx=e.get("pred_idx"),
                header_text=f"[{cat_key}] {e['summary']}",
                output_size=(640, 440),
            )
            out_tile = cat_dir / f"{idx:02d}_{Path(e['filename']).stem}.jpg"
            save_jpg(rendered, out_tile)
            tiles.append(rendered)
            rel_tile = str(out_tile.relative_to(PROJECT_ROOT)).replace("\\", "/")
            e_copy = dict(e)
            e_copy["category"] = cat_key
            e_copy["saved_image"] = rel_tile
            saved_files.append(e_copy)
            if idx <= 2:
                selected_examples.append(e_copy)

        grid = build_gallery_grid(tiles, cols=2, tile_size=(640, 440))
        grid_path = OUT_DIR / f"{cat_key}_gallery.jpg"
        save_jpg(grid, grid_path)

        catalog_summary[cat_key] = {
            "total_found_on_test": len(entries),
            "unique_images_affected": len({e["filename"] for e in entries}),
            "gallery_image": str(grid_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "top_examples": saved_files,
        }
        print(
            f"  {cat_key:24s}: total={len(entries):3d} cases ({len({e['filename'] for e in entries}):2d} images) "
            f"-> saved {len(chosen)} tiles + {grid_path.name}"
        )

    payload = {
        "model_weights": str(weights_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "conf_threshold": conf_thres,
        "iou_nms": iou_nms,
        "num_test_images": len(img_paths),
        "categories": catalog_summary,
        "featured_12_examples": selected_examples,
    }
    out_json = OUT_DIR / "error_catalog.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved error catalog: {out_json} (featured examples: {len(selected_examples)})")
    return payload


if __name__ == "__main__":
    analyze_errors()
