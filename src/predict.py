"""Скрипт инференса обученной модели YOLOv8 на одном изображении (Разделы 8 и 9)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "ppe" / "e1-opora" / "weights" / "best.pt"
FALLBACK_WEIGHTS = PROJECT_ROOT / "models" / "yolov8n.pt"


def resolve_weights(weights_path: str | Path | None = None) -> Path:
    if weights_path is not None:
        p = Path(weights_path)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    if DEFAULT_WEIGHTS.exists():
        return DEFAULT_WEIGHTS
    return FALLBACK_WEIGHTS


def predict_image(
    image_path: str | Path,
    weights_path: str | Path | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    save_path: str | Path | None = None,
) -> dict:
    """
    Применяет модель YOLOv8 к одному изображению и возвращает структурированный словарь
    с метаданными кадра и списком обнаруженных объектов по фиксированной схеме.
    """
    img_p = Path(image_path)
    if not img_p.is_absolute():
        img_p = (PROJECT_ROOT / img_p).resolve()
    if not img_p.exists():
        raise FileNotFoundError(f"Изображение не найдено: {img_p}")

    w_p = resolve_weights(weights_path)
    if not w_p.exists():
        raise FileNotFoundError(f"Файл весов не найден: {w_p}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(w_p))

    t0 = time.perf_counter()
    bgr = cv2.imdecode(np.frombuffer(img_p.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Не удалось декодировать изображение: {img_p}")
    h, w = bgr.shape[:2]

    res = model.predict(source=bgr, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    detections: list[dict] = []
    for box in res.boxes:
        cid = int(box.cls.item())
        score = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        x1 = max(0.0, min(x1, float(w)))
        y1 = max(0.0, min(y1, float(h)))
        x2 = max(0.0, min(x2, float(w)))
        y2 = max(0.0, min(y2, float(h)))
        detections.append(
            {
                "class_id": cid,
                "class_name": str(model.names[cid]),
                "confidence": round(score, 4),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    # Проверка правила передачи оператору (зона неуверенности 0.25..0.55)
    uncertain = [d for d in detections if 0.25 <= d["confidence"] < 0.55]
    needs_operator_review = len(uncertain) > 0

    if save_path is not None:
        sp = Path(save_path)
        if not sp.is_absolute():
            sp = (PROJECT_ROOT / sp).resolve()
        sp.parent.mkdir(parents=True, exist_ok=True)
        plotted = res.plot()
        ok, buf = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if ok:
            sp.write_bytes(buf.tobytes())

    return {
        "image": str(img_p.name),
        "width": w,
        "height": h,
        "weights": w_p.name,
        "num_classes": len(model.names),
        "elapsed_ms": elapsed_ms,
        "needs_operator_review": needs_operator_review,
        "detections": detections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск предсказания YOLOv8 СИЗ на одном изображении.")
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    parser.add_argument("--weights", type=str, default=None, help="Путь к .pt весам (по умолчанию E1 best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Порог уверенности")
    parser.add_argument("--save", type=str, default=None, help="Куда сохранить картинку с рамками (опционально)")
    args = parser.parse_args()

    out = predict_image(args.image, weights_path=args.weights, conf=args.conf, save_path=args.save)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
