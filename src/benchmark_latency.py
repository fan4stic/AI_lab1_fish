"""Замер полной задержки (p50, p95), разбивки по этапам, FPS, памяти и ложных тревог (Разделы 5 и 7)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import psutil
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "images"
TEST_LBL_DIR = PROJECT_ROOT / "data" / "processed" / "test" / "labels"
OUT_JSON = PROJECT_ROOT / "reports" / "metrics" / "latency_benchmark.json"
OUT_PLOT = PROJECT_ROOT / "reports" / "figures" / "latency_breakdown.png"

WARMUP_RUNS = 15
IMG_SIZE = 640
CONF_THRES = 0.25
IOU_NMS = 0.45


def summarize(values_ms: list[float]) -> dict[str, float]:
    arr = np.array(values_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(np.min(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
    }


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU матрица (N, M) для рамок в формате xyxy."""
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


def load_gt_xyxy(lbl_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    if not lbl_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes, classes = [], []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        boxes.append([x1, y1, x2, y2])
        classes.append(cid)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64)


def count_false_alarms(
    pred_boxes: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    gt_cls: np.ndarray,
    conf_thr: float,
    iou_thr: float = 0.50,
) -> tuple[int, dict[int, int]]:
    """Считает FP (ложные тревоги) при заданном пороге уверенности."""
    keep = pred_conf >= conf_thr
    pb, pc, ps = pred_boxes[keep], pred_cls[keep], pred_conf[keep]
    order = np.argsort(-ps)
    pb, pc = pb[order], pc[order]

    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    fp_total = 0
    fp_by_cls: dict[int, int] = {}

    if len(pb) == 0:
        return 0, {}

    iou_mat = compute_iou_matrix(pb, gt_boxes) if len(gt_boxes) > 0 else np.zeros((len(pb), 0))
    for i in range(len(pb)):
        cid = int(pc[i])
        best_j = -1
        best_iou = iou_thr
        for j in range(len(gt_boxes)):
            if not matched_gt[j] and int(gt_cls[j]) == cid and iou_mat[i, j] >= best_iou:
                best_iou = float(iou_mat[i, j])
                best_j = j
        if best_j >= 0:
            matched_gt[best_j] = True
        else:
            fp_total += 1
            fp_by_cls[cid] = fp_by_cls.get(cid, 0) + 1
    return fp_total, fp_by_cls


def benchmark_model(name: str, weights_path: Path, image_paths: list[Path]) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    process = psutil.Process()

    yolo = YOLO(str(weights_path))
    yolo.fuse()
    nn_model = yolo.model.to(device).eval()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
    class_names: dict[int, str] = yolo.names
    letterbox = LetterBox((IMG_SIZE, IMG_SIZE), auto=False, stride=32)

    # 1. Прогрев (warmup) — прогреваем и ядро нейросети, и CUDA-ядро NMS
    dummy = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            w_out = nn_model(dummy)
            _ = nms.non_max_suppression(w_out, conf_thres=0.001, iou_thres=IOU_NMS, max_det=10)
        if use_cuda:
            torch.cuda.synchronize()

    decode_ms: list[float] = []
    resize_pre_ms: list[float] = []
    infer_ms: list[float] = []
    post_ms: list[float] = []
    e2e_ms: list[float] = []
    ultralytics_predict_ms: list[float] = []

    # Статистика ложных тревог на 100 кадров при conf=0.25 и conf=0.50
    fa_stats = {
        "conf_0.25": {"total_fp": 0, "by_cls": {}},
        "conf_0.50": {"total_fp": 0, "by_cls": {}},
    }

    with torch.no_grad():
        for img_path in image_paths:
            # --- Полная сквозная цепочка по 4 этапам ---
            t_start = time.perf_counter()

            # Этап 1: чтение и декодирование JPEG
            raw = img_path.read_bytes()
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Не удалось декодировать {img_path}")
            orig_h, orig_w = bgr.shape[:2]
            t_dec = time.perf_counter()

            # Этап 2: изменение размера (letterbox) + препроцессинг + перенос на GPU
            lb = letterbox(image=bgr)
            rgb = lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
            chw = np.ascontiguousarray(rgb)
            tensor = torch.from_numpy(chw).to(device=device, dtype=torch.float32) / 255.0
            tensor = tensor.unsqueeze(0)
            if use_cuda:
                torch.cuda.synchronize()
            t_pre = time.perf_counter()

            # Этап 3: прямой прогон ядра нейросети (inference)
            raw_out = nn_model(tensor)
            if use_cuda:
                torch.cuda.synchronize()
            t_inf = time.perf_counter()

            # Этап 4: постобработка (NMS + масштабирование координат + перенос на CPU)
            preds = nms.non_max_suppression(raw_out, conf_thres=CONF_THRES, iou_thres=IOU_NMS, max_det=300)
            det = preds[0]
            if len(det):
                det[:, :4] = ops.scale_boxes(tensor.shape[2:], det[:, :4], (orig_h, orig_w)).round()
                det_np = det.detach().cpu().numpy()
                p_boxes = det_np[:, :4]
                p_conf = det_np[:, 4]
                p_cls = det_np[:, 5].astype(np.int64)
            else:
                p_boxes = np.zeros((0, 4), dtype=np.float32)
                p_conf = np.zeros((0,), dtype=np.float32)
                p_cls = np.zeros((0,), dtype=np.int64)
            t_post = time.perf_counter()

            decode_ms.append((t_dec - t_start) * 1000.0)
            resize_pre_ms.append((t_pre - t_dec) * 1000.0)
            infer_ms.append((t_inf - t_pre) * 1000.0)
            post_ms.append((t_post - t_inf) * 1000.0)
            e2e_ms.append((t_post - t_start) * 1000.0)

            # Считаем ложные тревоги на этом кадре
            gt_boxes, gt_cls = load_gt_xyxy(TEST_LBL_DIR / f"{img_path.stem}.txt", orig_w, orig_h)
            for key, c_thr in (("conf_0.25", 0.25), ("conf_0.50", 0.50)):
                fp_cnt, fp_cls = count_false_alarms(p_boxes, p_cls, p_conf, gt_boxes, gt_cls, conf_thr=c_thr)
                fa_stats[key]["total_fp"] += fp_cnt
                for cid, cnt in fp_cls.items():
                    cname = class_names[cid]
                    fa_stats[key]["by_cls"][cname] = fa_stats[key]["by_cls"].get(cname, 0) + cnt

            # Дополнительно: полный вызов через высокоуровневый API yolo.predict(bgr)
            t_u0 = time.perf_counter()
            _ = yolo.predict(source=bgr, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_NMS, device=0 if use_cuda else "cpu", verbose=False)
            if use_cuda:
                torch.cuda.synchronize()
            ultralytics_predict_ms.append((time.perf_counter() - t_u0) * 1000.0)

    n_frames = len(image_paths)
    ram_mb = round(process.memory_info().rss / (1024 * 1024), 1)
    vram_alloc_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1) if use_cuda else 0.0
    vram_res_mb = round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 1) if use_cuda else 0.0

    # Нормируем ложные тревоги на 100 кадров и на 1 минуту видео при 5 FPS (300 кадров/мин)
    false_alarms_report = {}
    for key, st in fa_stats.items():
        crit_no_hardhat = st["by_cls"].get("NO-Hardhat", 0)
        crit_no_vest = st["by_cls"].get("NO-Safety Vest", 0)
        crit_total = crit_no_hardhat + crit_no_vest
        false_alarms_report[key] = {
            "total_fp_on_test": st["total_fp"],
            "fp_per_100_frames_all_classes": round(st["total_fp"] * 100.0 / n_frames, 2),
            "fp_per_100_frames_critical_ppe": round(crit_total * 100.0 / n_frames, 2),
            "fp_per_100_frames_no_hardhat": round(crit_no_hardhat * 100.0 / n_frames, 2),
            "fp_per_100_frames_no_safety_vest": round(crit_no_vest * 100.0 / n_frames, 2),
            "fp_per_min_at_5fps_critical_ppe": round(crit_total * 300.0 / n_frames, 1),
            "fp_by_class_on_test": dict(sorted(st["by_cls"].items(), key=lambda x: -x[1])),
        }

    e2e_sum = summarize(e2e_ms)
    inf_sum = summarize(infer_ms)
    return {
        "model_name": name,
        "weights": str(weights_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "device": torch.cuda.get_device_name(0) if use_cuda else "CPU",
        "imgsz": IMG_SIZE,
        "num_frames": n_frames,
        "warmup_runs": WARMUP_RUNS,
        "stages": {
            "decode_jpeg": summarize(decode_ms),
            "resize_preprocess": summarize(resize_pre_ms),
            "model_inference": inf_sum,
            "postprocess_nms": summarize(post_ms),
            "end_to_end_full_pipeline": e2e_sum,
            "ultralytics_predict_api_no_decode": summarize(ultralytics_predict_ms),
        },
        "throughput_fps": {
            "model_core_only_fps": round(1000.0 / inf_sum["mean_ms"], 1),
            "end_to_end_pipeline_fps": round(1000.0 / e2e_sum["mean_ms"], 1),
            "end_to_end_p95_guaranteed_fps": round(1000.0 / e2e_sum["p95_ms"], 1),
        },
        "memory": {
            "process_ram_mb": ram_mb,
            "gpu_vram_peak_allocated_mb": vram_alloc_mb,
            "gpu_vram_peak_reserved_mb": vram_res_mb,
        },
        "false_alarms": false_alarms_report,
    }


def save_latency_plot(e1_res: dict, out_path: Path) -> None:
    """Строит понятный график разбивки задержки по этапам (p50 и p95) для E1."""
    stages = [
        ("Декодирование\nJPEG", e1_res["stages"]["decode_jpeg"]),
        ("Resize +\nпрепроцессинг", e1_res["stages"]["resize_preprocess"]),
        ("Инференс\nYOLOv8n (GPU)", e1_res["stages"]["model_inference"]),
        ("Постобработка\n(NMS)", e1_res["stages"]["postprocess_nms"]),
        ("Полная цепочка\n(End-to-End)", e1_res["stages"]["end_to_end_full_pipeline"]),
    ]
    labels = [s[0] for s in stages]
    p50s = [s[1]["p50_ms"] for s in stages]
    p95s = [s[1]["p95_ms"] for s in stages]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140)
    b1 = ax.bar(x - width / 2, p50s, width, label="Медиана (p50)", color="#2b6cb0")
    b2 = ax.bar(x + width / 2, p95s, width, label="95-й процентиль (p95)", color="#dd6b20")

    ax.set_ylabel("Задержка, мс (меньше — лучше)")
    ax.set_title(f"Задержка полной цепочки обработки E1 (YOLOv8n, 640×640, {e1_res['device']}, n={e1_res['num_frames']})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    image_paths = sorted(p for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    print(f"Benchmarking on {len(image_paths)} frozen test images...")

    models_to_bench = [
        ("e1-opora", PROJECT_ROOT / "runs" / "ppe" / "e1-opora" / "weights" / "best.pt"),
        ("e2-epochs100", PROJECT_ROOT / "runs" / "ppe" / "e2-epochs100" / "weights" / "best.pt"),
    ]

    all_results = {}
    for mname, wpath in models_to_bench:
        res = benchmark_model(mname, wpath, image_paths)
        all_results[mname] = res
        st = res["stages"]
        tp = res["throughput_fps"]
        mem = res["memory"]
        fa25 = res["false_alarms"]["conf_0.25"]
        fa50 = res["false_alarms"]["conf_0.50"]
        print(f"\n=== BENCHMARK: {mname} ({res['device']}) ===")
        for stage_key in ("decode_jpeg", "resize_preprocess", "model_inference", "postprocess_nms", "end_to_end_full_pipeline"):
            s = st[stage_key]
            print(f"  {stage_key:26s}: mean={s['mean_ms']:6.2f} ms | p50={s['p50_ms']:6.2f} ms | p95={s['p95_ms']:6.2f} ms")
        print(
            f"  Throughput: core_only={tp['model_core_only_fps']} FPS | "
            f"end_to_end={tp['end_to_end_pipeline_fps']} FPS | p95_guaranteed={tp['end_to_end_p95_guaranteed_fps']} FPS"
        )
        print(
            f"  Memory: RAM={mem['process_ram_mb']} MB | "
            f"VRAM_alloc={mem['gpu_vram_peak_allocated_mb']} MB | VRAM_reserved={mem['gpu_vram_peak_reserved_mb']} MB"
        )
        print(
            f"  False alarms (conf=0.25): all={fa25['fp_per_100_frames_all_classes']}/100f | "
            f"crit(NO-Hardhat+NO-Vest)={fa25['fp_per_100_frames_critical_ppe']}/100f"
        )
        print(
            f"  False alarms (conf=0.50): all={fa50['fp_per_100_frames_all_classes']}/100f | "
            f"crit(NO-Hardhat+NO-Vest)={fa50['fp_per_100_frames_critical_ppe']}/100f"
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_latency_plot(all_results["e1-opora"], OUT_PLOT)
    print(f"\nSaved JSON: {OUT_JSON}")
    print(f"Saved Plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()
