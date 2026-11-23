"""Минимальные автоматические проверки пайплайна обнаружения СИЗ (Раздел 8)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.error_analysis import build_gallery_grid, render_error_overlay
from src.predict import predict_image

PPE_YAML = PROJECT_ROOT / "configs" / "ppe.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_CSV = PROCESSED_DIR / "manifest.csv"
SPLIT_REPORT = PROCESSED_DIR / "split_report.json"


def _load_yaml_and_root() -> tuple[dict, Path]:
    assert PPE_YAML.exists(), f"Не найден конфиг датасета: {PPE_YAML}"
    cfg = yaml.safe_load(PPE_YAML.read_text(encoding="utf-8"))
    data_root = Path(cfg["path"])
    if not data_root.is_absolute():
        data_root = (PROJECT_ROOT / data_root).resolve()
    return cfg, data_root


def test_yaml_paths_exist_and_labels_readable() -> None:
    """Тест 1 (Раздел 8): все пути к изображениям из YAML существуют, файлы разметки читаются."""
    cfg, data_root = _load_yaml_and_root()
    assert data_root.exists(), f"Каталог датасета не существует: {data_root}"

    total_images = 0
    for split_key in ("train", "val", "test"):
        img_dir = data_root / cfg[split_key]
        assert img_dir.exists() and img_dir.is_dir(), f"Не найден каталог {split_key}: {img_dir}"
        lbl_dir = img_dir.parent / "labels"
        assert lbl_dir.exists() and lbl_dir.is_dir(), f"Не найден каталог labels для {split_key}: {lbl_dir}"

        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        assert len(imgs) > 0, f"В {split_key} нет изображений"
        total_images += len(imgs)

        for img_p in imgs:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            assert lbl_p.exists(), f"Отсутствует файл разметки для {img_p.name}"
            # Проверяем, что файл разметки читается как UTF-8 без ошибок
            _ = lbl_p.read_text(encoding="utf-8")

    assert total_images >= 100, f"В наборе меньше 100 изображений: {total_images}"


def test_class_ids_and_normalized_coordinates_valid() -> None:
    """Тест 2 (Раздел 8): номера классов в допустимом диапазоне [0, nc-1], координаты в [0, 1]."""
    cfg, data_root = _load_yaml_and_root()
    nc = int(cfg["nc"])
    assert nc == len(cfg["names"]), "Несоответствие nc и длины словаря names"

    checked_boxes = 0
    for split_key in ("train", "val", "test"):
        lbl_dir = (data_root / cfg[split_key]).parent / "labels"
        for lbl_p in lbl_dir.glob("*.txt"):
            text = lbl_p.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for line_no, line in enumerate(text.splitlines(), start=1):
                parts = line.strip().split()
                assert len(parts) == 5, f"Ожидалось 5 чисел в {lbl_p.name}:{line_no}, получено {len(parts)}"
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                assert 0 <= cid < nc, f"Недопустимый class_id={cid} в {lbl_p.name}:{line_no}"
                assert 0.0 <= cx <= 1.0, f"cx={cx} вне [0, 1] в {lbl_p.name}:{line_no}"
                assert 0.0 <= cy <= 1.0, f"cy={cy} вне [0, 1] в {lbl_p.name}:{line_no}"
                assert 0.0 < bw <= 1.0, f"w={bw} вне (0, 1] в {lbl_p.name}:{line_no}"
                assert 0.0 < bh <= 1.0, f"h={bh} вне (0, 1] в {lbl_p.name}:{line_no}"
                checked_boxes += 1

    assert checked_boxes > 0, "Не найдено ни одной размеченной рамки"


def test_no_group_or_hash_leakage_between_splits() -> None:
    """Тест 3 (Раздел 8): между частями выборки нет пересечения имён, group_id и контрольных хэшей."""
    assert MANIFEST_CSV.exists(), f"Не найден манифест разбиения: {MANIFEST_CSV}"
    rows = list(csv.DictReader(MANIFEST_CSV.open("r", encoding="utf-8-sig"), delimiter=";"))
    by_sp = {sp: [r for r in rows if r["split"] == sp] for sp in ("train", "val", "test")}

    for sp in ("train", "val", "test"):
        assert len(by_sp[sp]) > 0, f"Пустой сплит: {sp}"

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not ({r["filename"] for r in by_sp[a]} & {r["filename"] for r in by_sp[b]}), f"Общие имена: {a}/{b}"
        assert not ({r["group_id"] for r in by_sp[a]} & {r["group_id"] for r in by_sp[b]}), f"Утечка group_id: {a}/{b}"
        assert not ({r["sha256"] for r in by_sp[a]} & {r["sha256"] for r in by_sp[b]}), f"Утечка SHA-256: {a}/{b}"
        assert not ({r["phash"] for r in by_sp[a]} & {r["phash"] for r in by_sp[b]}), f"Утечка pHash: {a}/{b}"


def test_frozen_test_split_integrity() -> None:
    """Доп. проверка (Раздел 3): контрольная сумма замороженной тестовой выборки не изменилась."""
    report = json.loads(SPLIT_REPORT.read_text(encoding="utf-8"))
    expected = report["frozen_test_sha256"]
    test_dir = PROCESSED_DIR / "test" / "images"
    entries = [
        f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}"
        for p in sorted(test_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    actual = hashlib.sha256("\n".join(sorted(entries)).encode("utf-8")).hexdigest()
    assert actual == expected, "Замороженная тестовая выборка была изменена!"


def test_error_visualizer_returns_expected_size() -> None:
    """Тест 4 (Раздел 8): визуализатор возвращает изображение строго ожидаемого размера."""
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    gt = [{"class_id": 0, "bbox": [50.0, 40.0, 150.0, 140.0]}]
    pred = [{"class_id": 2, "conf": 0.85, "bbox": [55.0, 45.0, 155.0, 145.0]}]
    names = {0: "Hardhat", 2: "NO-Hardhat"}

    # Режим без ресайза: размер равен исходному (480, 640, 3)
    out_same = render_error_overlay(dummy_img, gt, pred, names, highlight_gt_idx=0, highlight_pred_idx=0)
    assert isinstance(out_same, np.ndarray)
    assert out_same.dtype == np.uint8
    assert out_same.shape == (480, 640, 3)

    # Режим с фиксированным output_size=(W=500, H=320) -> shape=(320, 500, 3)
    out_sized = render_error_overlay(
        dummy_img, gt, pred, names, header_text="Test header", output_size=(500, 320)
    )
    assert out_sized.shape == (320, 500, 3)

    # Сборка сетки 2x2 из 3 плиток (500x320) -> (640, 1000, 3)
    grid = build_gallery_grid([out_sized, out_sized, out_sized], cols=2, tile_size=(500, 320))
    assert grid.shape == (640, 1000, 3)


def test_single_image_inference_schema() -> None:
    """Тест 5 (Раздел 8): применение модели к одному контрольному изображению выдаёт список с допустимой схемой."""
    test_imgs = sorted((PROCESSED_DIR / "test" / "images").glob("*.jpg"))
    assert len(test_imgs) > 0
    sample_img = test_imgs[0]

    result = predict_image(sample_img, conf=0.25)
    assert isinstance(result, dict)
    assert result["image"] == sample_img.name
    assert result["width"] > 0 and result["height"] > 0
    assert isinstance(result["detections"], list)

    nc = int(result["num_classes"])
    w, h = float(result["width"]), float(result["height"])

    for det in result["detections"]:
        assert set(det.keys()) == {"class_id", "class_name", "confidence", "bbox_xyxy"}
        assert isinstance(det["class_id"], int) and 0 <= det["class_id"] < nc
        assert isinstance(det["class_name"], str) and len(det["class_name"]) > 0
        assert isinstance(det["confidence"], float) and 0.0 <= det["confidence"] <= 1.0
        bbox = det["bbox_xyxy"]
        assert isinstance(bbox, list) and len(bbox) == 4
        x1, y1, x2, y2 = bbox
        assert 0.0 <= x1 <= x2 <= w + 1e-3
        assert 0.0 <= y1 <= y2 <= h + 1e-3
