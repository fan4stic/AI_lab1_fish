"""Распаковка сырого архива Roboflow, первичный аудит датасета и подготовка выборки для ручной проверки (п. 2.1)."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imagehash
import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACT_DIR = RAW_DIR / "extracted"
AUDIT_DIR = PROJECT_ROOT / "reports" / "manual_audit"

SEED = 42
MAX_FULL_PATH = 230  # безопасный запас под Windows MAX_PATH (260)


def shorten_member_name(member: str, extract_dir: Path) -> str:
    """Укорачивает имя файла, если полный путь превышает MAX_FULL_PATH, сохраняя префикс и .rf.<hash>."""
    full_len = len(str((extract_dir / member).resolve()))
    if full_len <= MAX_FULL_PATH:
        return member

    p = Path(member)
    stem = p.stem
    suffix = p.suffix
    # Сохраняем хвост .rf.<hash> если он есть, чтобы пара image/label получила ИДЕНТИЧНОЕ имя
    rf_match = re.search(r"(\.rf\.[0-9a-fA-F]+)$", stem)
    rf_tail = rf_match.group(1) if rf_match else ""
    base = stem[: -len(rf_tail)] if rf_tail else stem

    # Сколько символов нужно убрать
    excess = full_len - MAX_FULL_PATH
    keep_len = max(24, len(base) - excess - 10)
    short_hash = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    new_stem = f"{base[:keep_len]}__{short_hash}{rf_tail}"
    return str(p.with_name(new_stem + suffix)).replace("\\", "/")


def unpack_archive() -> tuple[Path, dict[str, str]]:
    """Распаковывает первый ZIP-архив из data/raw в data/raw/extracted."""
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"ZIP-архив не найден в {RAW_DIR}")
    zip_path = zips[0]

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    renamed: dict[str, str] = {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            target_rel = shorten_member_name(info.filename, EXTRACT_DIR)
            if target_rel != info.filename:
                renamed[target_rel] = info.filename
            target_path = EXTRACT_DIR / target_rel
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                dst.write(src.read())

    with open(EXTRACT_DIR / "renamed_long_paths.json", "w", encoding="utf-8") as f:
        json.dump(renamed, f, ensure_ascii=False, indent=2)

    return zip_path, renamed


def compute_iou_xyxy(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def extract_group_id(original_name: str) -> str:
    """Извлекает предполагаемый идентификатор источника/видео/сцены из исходного имени файла Roboflow."""
    stem = Path(original_name).name
    # Убираем .rf.<hash>.<ext>
    stem = re.sub(r"\.rf\.[0-9a-fA-F]+\.[^.]+$", "", stem)
    # Убираем хвост _jpg / _png / _jpeg
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem, flags=re.IGNORECASE)

    # 1. Кадры из видео вида ..._mp4-123 или ..._mp4_123
    m = re.match(r"^(.+?_mp4)[-_]\d+$", stem, flags=re.IGNORECASE)
    if m:
        return f"video:{m.group(1)}"

    # 2. Кадры вида youtube-123 -> группа youtube (проверим, единый ли ролик)
    m = re.match(r"^(youtube)[-_]\d+$", stem, flags=re.IGNORECASE)
    if m:
        return f"series:{m.group(1)}"

    # 3. Кадры вида <prefix>_frame_123 или <prefix>-frame-123
    m = re.match(r"^(.+?)[-_]frame[-_]?\d+$", stem, flags=re.IGNORECASE)
    if m:
        return f"video:{m.group(1)}"

    # 4. Кадры вида PPE_Part1_... или Construction-..._001
    m = re.match(r"^([A-Za-z][A-Za-z0-9_-]{4,}?)[-_]\d{1,4}$", stem)
    if m:
        return f"prefix:{m.group(1)}"

    return f"single:{stem}"


def load_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def draw_annotations(
    img_path: Path,
    boxes: list[tuple[int, float, float, float, float]],
    class_names: list[str],
    out_path: Path,
) -> None:
    """Рисует эталонные рамки и подписи классов на изображении для ручной проверки."""
    # Используем PIL -> numpy для надёжной работы с путями с кириллицей на Windows
    pil_img = Image.open(img_path).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    # Стабильная палитра по классам
    rng = np.random.default_rng(12345)
    palette = rng.integers(40, 235, size=(max(len(class_names), 30), 3)).tolist()

    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int(round((cx - bw / 2.0) * w))
        y1 = int(round((cy - bh / 2.0) * h))
        x2 = int(round((cx + bw / 2.0) * w))
        y2 = int(round((cy + bh / 2.0) * h))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = tuple(int(c) for c in palette[cls_id % len(palette)])
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"cls_{cls_id}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        font_scale = max(0.45, min(w, h) / 1100.0)
        thickness = max(1, int(round(font_scale * 2)))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, ty1), (min(w - 1, x1 + tw + 6), ty1 + th + 6), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 3, ty1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ok:
        out_path.write_bytes(buf.tobytes())


def main() -> None:
    zip_path, renamed = unpack_archive()
    print(f"Unpacked: {zip_path.name} (shortened long paths: {len(renamed)})")

    data_yaml = yaml.safe_load((EXTRACT_DIR / "data.yaml").read_text(encoding="utf-8"))
    class_names: list[str] = data_yaml["names"]

    records = []
    class_box_counts: Counter[int] = Counter()
    class_img_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    invalid_boxes = 0
    empty_images = 0
    size_counter: Counter[tuple[int, int]] = Counter()
    sha256_map: dict[str, list[str]] = defaultdict(list)
    phash_map: dict[str, list[str]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)

    for split in ("train", "valid", "test"):
        img_dir = EXTRACT_DIR / split / "images"
        lbl_dir = EXTRACT_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            split_counts[split] += 1
            rel_img = str(img_path.relative_to(EXTRACT_DIR)).replace("\\", "/")
            orig_rel = renamed.get(rel_img, rel_img)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            raw_bytes = img_path.read_bytes()
            sha = hashlib.sha256(raw_bytes).hexdigest()
            sha256_map[sha].append(rel_img)

            with Image.open(img_path) as im:
                w, h = im.size
                ph = str(imagehash.phash(im))
            size_counter[(w, h)] += 1
            phash_map[ph].append(rel_img)

            boxes = load_yolo_labels(lbl_path)
            if not boxes:
                empty_images += 1

            seen_cls = set()
            small_cnt = 0
            xyxy_boxes = []
            for cls_id, cx, cy, bw, bh in boxes:
                if not (0 <= cls_id < len(class_names)) or not (
                    0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0
                ):
                    invalid_boxes += 1
                class_box_counts[cls_id] += 1
                seen_cls.add(cls_id)

                area_ratio = bw * bh
                min_px = min(bw * w, bh * h)
                if area_ratio < 0.01 or min_px < 28:
                    small_cnt += 1
                xyxy_boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

            for c in seen_cls:
                class_img_counts[c] += 1

            overlap_pairs = 0
            for i in range(len(xyxy_boxes)):
                for j in range(i + 1, len(xyxy_boxes)):
                    if compute_iou_xyxy(xyxy_boxes[i], xyxy_boxes[j]) >= 0.25:
                        overlap_pairs += 1

            group_id = extract_group_id(orig_rel)
            group_splits[group_id].add(split)

            records.append(
                {
                    "split_orig": split,
                    "rel_path": rel_img,
                    "orig_name": Path(orig_rel).name,
                    "group_id": group_id,
                    "width": w,
                    "height": h,
                    "num_boxes": len(boxes),
                    "small_boxes": small_cnt,
                    "overlap_pairs": overlap_pairs,
                    "sha256": sha,
                    "phash": ph,
                    "boxes": boxes,
                    "img_path": img_path,
                }
            )

    exact_dups = {k: v for k, v in sha256_map.items() if len(v) > 1}
    phash_dups = {k: v for k, v in phash_map.items() if len(v) > 1}
    leaked_groups = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}

    summary = {
        "total_images": len(records),
        "orig_split_counts": dict(split_counts),
        "total_boxes": sum(class_box_counts.values()),
        "empty_images": empty_images,
        "invalid_boxes": invalid_boxes,
        "exact_sha256_dup_groups": len(exact_dups),
        "exact_phash_dup_groups": len(phash_dups),
        "total_unique_groups": len(group_splits),
        "leaked_groups_in_orig_split": len(leaked_groups),
        "leaked_groups_examples": dict(list(leaked_groups.items())[:15]),
        "top_image_sizes": [{"size": f"{w}x{h}", "count": c} for (w, h), c in size_counter.most_common(10)],
        "classes": [
            {
                "id": i,
                "name": name,
                "boxes": class_box_counts[i],
                "images": class_img_counts[i],
            }
            for i, name in enumerate(class_names)
        ],
    }

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = AUDIT_DIR / "raw_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Подготовка выборки для ручной проверки (п. 2.1): 30 случайных + 10 с малыми/перекрытыми объектами
    rng = random.Random(SEED)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    random_30 = [records[i] for i in indices[:30]]
    random_set = {r["rel_path"] for r in random_30}

    # 10 сложных кадров (малые + перекрытые объекты), не пересекающихся с random_30
    hard_candidates = [
        r for r in records if r["rel_path"] not in random_set and (r["small_boxes"] >= 2 or r["overlap_pairs"] >= 2)
    ]
    hard_candidates.sort(key=lambda r: (r["small_boxes"] + r["overlap_pairs"] * 2, r["num_boxes"]), reverse=True)
    # Берём из разных групп, чтобы выборка была разнообразной
    hard_10 = []
    used_groups = set()
    for r in hard_candidates:
        if r["group_id"] in used_groups:
            continue
        hard_10.append(r)
        used_groups.add(r["group_id"])
        if len(hard_10) == 10:
            break

    # Отрисовываем аннотированные копии и создаём чек-лист
    for sub in ("random_30", "hard_10"):
        d = AUDIT_DIR / sub
        if d.exists():
            for old in d.glob("*.jpg"):
                old.unlink()
        d.mkdir(parents=True, exist_ok=True)

    checklist_rows = []
    for subset_name, subset_list in (("random_30", random_30), ("hard_10", hard_10)):
        for idx, r in enumerate(subset_list, start=1):
            out_name = f"{idx:02d}_{Path(r['rel_path']).stem[:45]}.jpg"
            out_path = AUDIT_DIR / subset_name / out_name
            draw_annotations(r["img_path"], r["boxes"], class_names, out_path)
            cls_summary = ", ".join(
                f"{class_names[c]}:{cnt}" for c, cnt in Counter(b[0] for b in r["boxes"]).most_common()
            )
            checklist_rows.append(
                {
                    "subset": subset_name,
                    "audit_file": f"{subset_name}/{out_name}",
                    "orig_split": r["split_orig"],
                    "orig_name": r["orig_name"],
                    "group_id": r["group_id"],
                    "size": f"{r['width']}x{r['height']}",
                    "num_boxes": r["num_boxes"],
                    "small_boxes": r["small_boxes"],
                    "overlap_pairs": r["overlap_pairs"],
                    "classes_present": cls_summary or "(нет рамок / фон)",
                    "missed_or_extra": "",
                    "wrong_class": "",
                    "loose_or_tight_box": "",
                    "truncated_or_occluded": "",
                    "inconsistent_rules": "",
                    "duplicate_or_neighbor": "",
                    "comment": "",
                }
            )

    checklist_path = AUDIT_DIR / "audit_checklist.csv"
    with open(checklist_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(checklist_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(checklist_rows)

    print("\n=== SUMMARY ===")
    print(f"Total images: {summary['total_images']} (orig splits: {summary['orig_split_counts']})")
    print(f"Total boxes: {summary['total_boxes']}, empty (null) images: {empty_images}, invalid boxes: {invalid_boxes}")
    print(f"Exact SHA-256 dup groups: {len(exact_dups)}, exact pHash dup groups: {len(phash_dups)}")
    print(f"Unique groups: {len(group_splits)}, leaked groups across orig splits: {len(leaked_groups)}")
    print("\nClasses distribution (boxes / images):")
    for c in summary["classes"]:
        print(f"  [{c['id']:2d}] {c['name']:20s}: {c['boxes']:5d} boxes | {c['images']:4d} images")
    print(f"\nAudit files saved to: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
