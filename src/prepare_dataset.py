"""Подготовка 10-классового датасета СИЗ и групповое разбиение 70/15/15 без утечки (Раздел 3)."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "configs" / "ppe.yaml"

SEED = 42
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
PHASH_GROUP_THRESHOLD = 10  # порог Хэмминга для объединения похожих сцен в одну группу
PHASH_DUP_THRESHOLD = 0     # точный визуальный дубликат (или <=12 при одинаковом base_name)

TARGET_CLASSES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

# Детерминированное правило сведения 25 классов Roboflow v30 -> 10 канонических классов проекта
RAW_TO_TARGET_NAME: dict[str, str | None] = {
    "Hardhat": "Hardhat",
    "Mask": "Mask",
    "NO-Hardhat": "NO-Hardhat",
    "NO-Mask": "NO-Mask",
    "NO-Safety Vest": "NO-Safety Vest",
    "Person": "Person",
    "Safety Cone": "Safety Cone",
    "Safety Vest": "Safety Vest",
    # Строительная техника -> machinery
    "machinery": "machinery",
    "Excavator": "machinery",
    "dump truck": "machinery",
    "wheel loader": "machinery",
    # Транспорт -> vehicle
    "vehicle": "vehicle",
    "SUV": "vehicle",
    "bus": "vehicle",
    "mini-van": "vehicle",
    "sedan": "vehicle",
    "semi": "vehicle",
    "trailer": "vehicle",
    "truck": "vehicle",
    "truck and trailer": "vehicle",
    "van": "vehicle",
    # Исключённые (редкие / непоследовательно размеченные / не относящиеся к СИЗ)
    "Gloves": None,
    "Ladder": None,
    "fire hydrant": None,
}


def load_and_remap_labels(
    label_path: Path, raw_names: list[str], target_name_to_id: dict[str, int]
) -> list[tuple[int, float, float, float, float]]:
    """Читает YOLO-разметку v30 и переводит в 10 целевых классов с проверкой границ [0, 1]."""
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    remapped: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        raw_cls_id = int(float(parts[0]))
        raw_name = raw_names[raw_cls_id]
        target_name = RAW_TO_TARGET_NAME.get(raw_name)
        if target_name is None:
            continue
        new_id = target_name_to_id[target_name]
        cx, cy, bw, bh = map(float, parts[1:5])
        # Безопасное приведение к строгому отрезку [0, 1] на случай погрешности округления
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 1e-6), 1.0)
        bh = min(max(bh, 1e-6), 1.0)
        remapped.append((new_id, cx, cy, bw, bh))
    return remapped


def get_base_name(orig_name: str) -> str:
    """Убирает суффикс .rf.<hash>.<ext>, оставляя исходное имя кадра."""
    return re.sub(r"\.rf\.[0-9a-fA-F]+\.[^.]+$", "", orig_name)


def get_video_or_series_key(base_name: str) -> str | None:
    """Извлекает ключ видеоролика или серии фото для группировки без утечки."""
    b = re.sub(r"_(jpg|jpeg|png)$", "", base_name, flags=re.IGNORECASE)

    # Видео: *_mp4-123, *_MOV-39, *_mov-11, *_MP4-5
    m = re.match(r"^(.+?)[-_](mp4|mov)[-_]\d+$", b, flags=re.IGNORECASE)
    if m:
        return f"video:{m.group(1).lower()}"

    # Серия фотографий с одного фотоаппарата/смартфона (например, IMG_5846..5851)
    m2 = re.match(r"^(IMG)_(\d{4})$", b)
    if m2:
        return f"img_series:{int(m2.group(2)) // 20}"

    # Кадры из одного видео с шаблоном class1_123 / class2_145 (не склеиваем общие image_NNN)
    m3 = re.match(r"^(class\d+)_\d+$", b, flags=re.IGNORECASE)
    if m3:
        return f"series:{m3.group(1).lower()}"

    return None


def get_youtube_frame_num(base_name: str) -> int | None:
    b = re.sub(r"_(jpg|jpeg|png)$", "", base_name, flags=re.IGNORECASE)
    m = re.match(r"^youtube[-_](\d+)$", b, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def collect_raw_items() -> tuple[list[dict], list[str]]:
    raw_yaml = yaml.safe_load((EXTRACT_DIR / "data.yaml").read_text(encoding="utf-8"))
    raw_names: list[str] = raw_yaml["names"]
    target_name_to_id = {name: i for i, name in enumerate(TARGET_CLASSES)}

    renamed_path = EXTRACT_DIR / "renamed_long_paths.json"
    renamed = json.loads(renamed_path.read_text(encoding="utf-8")) if renamed_path.exists() else {}

    items: list[dict] = []
    for sp in ("train", "valid", "test"):
        img_dir = EXTRACT_DIR / sp / "images"
        lbl_dir = EXTRACT_DIR / sp / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            rel = str(img_path.relative_to(EXTRACT_DIR)).replace("\\", "/")
            orig_name = Path(renamed.get(rel, rel)).name
            base_name = get_base_name(orig_name)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            raw_bytes = img_path.read_bytes()
            sha = hashlib.sha256(raw_bytes).hexdigest()
            with Image.open(img_path) as im:
                w, h = im.size
                ph = imagehash.phash(im)

            boxes = load_and_remap_labels(lbl_path, raw_names, target_name_to_id)
            # Точечное исправление подтверждённой ошибки разметки из ручного аудита (random_30/01):
            # на кадре IMG_3103_mp4-17_jpg.rf.640170b6a0f2eeac5f69dd46ebbe2e00.jpg на голове наушники без каски,
            # однако в исходной разметке стоял класс Hardhat (0) вместо NO-Hardhat (2).
            if orig_name == "IMG_3103_mp4-17_jpg.rf.640170b6a0f2eeac5f69dd46ebbe2e00.jpg":
                boxes = [(2 if cid == 0 else cid, cx, cy, bw, bh) for cid, cx, cy, bw, bh in boxes]
            items.append(
                {
                    "orig_split": sp,
                    "rel_path": rel,
                    "orig_name": orig_name,
                    "base_name": base_name,
                    "img_path": img_path,
                    "sha256": sha,
                    "phash_obj": ph,
                    "phash": str(ph),
                    "width": w,
                    "height": h,
                    "boxes": boxes,
                    "class_counts": Counter(b[0] for b in boxes),
                }
            )
    return items, raw_names


def deduplicate_items(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Удаляет точные и подтверждённые визуальные дубликаты (включая конфликт IMG_3103_mp4-17)."""
    n = len(items)
    uf = UnionFind(n)

    # 1. Точное совпадение SHA-256 или точного pHash (Hamming == 0)
    by_sha: dict[str, list[int]] = defaultdict(list)
    by_ph: dict[str, list[int]] = defaultdict(list)
    by_base: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_sha[it["sha256"]].append(i)
        by_ph[it["phash"]].append(i)
        by_base[it["base_name"]].append(i)

    for idxs in by_sha.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    for idxs in by_ph.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)
    # 2. Одинаковое исходное имя кадра до .rf. И визуальная близость (pHash <= 12)
    for idxs in by_base.values():
        if len(idxs) > 1:
            for a_idx in range(len(idxs)):
                for b_idx in range(a_idx + 1, len(idxs)):
                    i, j = idxs[a_idx], idxs[b_idx]
                    if (items[i]["phash_obj"] - items[j]["phash_obj"]) <= 12:
                        uf.union(i, j)

    dup_groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        dup_groups[uf.find(i)].append(i)

    # Приоритет при выборе лучшего представителя дубликата:
    # для IMG_3103_mp4-17 явно предпочитаем копию из valid (с верной меткой NO-Hardhat, без ошибочной Hardhat)
    split_priority = {"valid": 2, "test": 1, "train": 0}
    kept: list[dict] = []
    removed: list[dict] = []

    for root in sorted(dup_groups.keys()):
        idxs = dup_groups[root]
        if len(idxs) == 1:
            kept.append(items[idxs[0]])
            continue

        def rank(idx: int) -> tuple:
            it = items[idx]
            # Явная проверка дефекта из ручного аудита (random_30/01 vs random_30/11)
            is_known_bad = "IMG_3103_mp4-17" in it["orig_name"] and (0 in it["class_counts"])
            return (
                0 if is_known_bad else 1,
                split_priority.get(it["orig_split"], 0),
                len(it["boxes"]),
                it["orig_name"],
            )

        idxs_sorted = sorted(idxs, key=rank, reverse=True)
        winner = items[idxs_sorted[0]]
        kept.append(winner)
        for loser_idx in idxs_sorted[1:]:
            loser = items[loser_idx]
            removed.append(
                {
                    "removed_orig_name": loser["orig_name"],
                    "removed_orig_split": loser["orig_split"],
                    "kept_orig_name": winner["orig_name"],
                    "kept_orig_split": winner["orig_split"],
                    "phash_distance": int(loser["phash_obj"] - winner["phash_obj"]),
                    "same_sha256": loser["sha256"] == winner["sha256"],
                    "same_base_name": loser["base_name"] == winner["base_name"],
                }
            )

    return kept, removed


def assign_groups(items: list[dict]) -> dict[str, list[dict]]:
    """Объединяет кадры в независимые группы без утечки (видео + серии + соседние YouTube + pHash <= 10)."""
    n = len(items)
    uf = UnionFind(n)

    # 1. Один видеоролик / серия фото
    by_key: dict[str, list[int]] = defaultdict(list)
    yt_frames: list[tuple[int, int]] = []  # (frame_num, item_idx)

    for i, it in enumerate(items):
        key = get_video_or_series_key(it["base_name"])
        if key:
            by_key[key].append(i)
        yt_num = get_youtube_frame_num(it["base_name"])
        if yt_num is not None:
            yt_frames.append((yt_num, i))

    for idxs in by_key.values():
        for j in idxs[1:]:
            uf.union(idxs[0], j)

    # 2. Соседние кадры YouTube (разница номеров <= 3)
    yt_frames.sort()
    for k in range(len(yt_frames) - 1):
        num_a, idx_a = yt_frames[k]
        num_b, idx_b = yt_frames[k + 1]
        if (num_b - num_a) <= 3:
            uf.union(idx_a, idx_b)

    # 3. Визуально похожие кадры (pHash Hamming <= PHASH_GROUP_THRESHOLD)
    for i in range(n):
        phi = items[i]["phash_obj"]
        for j in range(i + 1, n):
            if (phi - items[j]["phash_obj"]) <= PHASH_GROUP_THRESHOLD:
                uf.union(i, j)

    raw_clusters: dict[int, list[dict]] = defaultdict(list)
    for i, it in enumerate(items):
        raw_clusters[uf.find(i)].append(it)

    groups: dict[str, list[dict]] = {}
    for g_idx, root in enumerate(sorted(raw_clusters.keys()), start=1):
        members = raw_clusters[root]
        # Читаемое имя группы
        vk = next((get_video_or_series_key(m["base_name"]) for m in members if get_video_or_series_key(m["base_name"])), None)
        if vk:
            gid = f"grp_{g_idx:03d}_{vk.replace(':', '_')}"
        elif any(get_youtube_frame_num(m["base_name"]) is not None for m in members):
            nums = sorted(n for m in members if (n := get_youtube_frame_num(m["base_name"])) is not None)
            gid = f"grp_{g_idx:03d}_youtube_{nums[0]}_{nums[-1]}" if len(nums) > 1 else f"grp_{g_idx:03d}_youtube_{nums[0]}"
        else:
            gid = f"grp_{g_idx:03d}_scene"
        for m in members:
            m["group_id"] = gid
        groups[gid] = members

    return groups


def stratified_group_split(groups: dict[str, list[dict]], seed: int = SEED) -> dict[str, str]:
    """Стратифицированное разбиение групп на train/val/test (~70/15/15) с гарантией всех 10 классов."""
    total_imgs = sum(len(v) for v in groups.values())
    total_cls = Counter()
    group_cls: dict[str, Counter] = {}
    for gid, members in groups.items():
        c = Counter()
        for m in members:
            c.update(m["class_counts"])
        group_cls[gid] = c
        total_cls.update(c)

    target_imgs = {sp: total_imgs * r for sp, r in TARGET_RATIOS.items()}
    target_cls = {sp: {cid: total_cls[cid] * r for cid in range(len(TARGET_CLASSES))} for sp, r in TARGET_RATIOS.items()}

    # Перебираем 200 детерминированных перестановок и выбираем лучшее по балансу 70/15/15 и всех 10 классов
    best_assign: dict[str, str] | None = None
    best_score = float("inf")

    for trial in range(200):
        rng = random.Random(seed + trial)
        gids = list(groups.keys())
        rng.shuffle(gids)
        # Сортируем крупные и насыщенные объектами группы вперёд (с небольшим случайным шумом для разнообразия)
        gids.sort(key=lambda g: (len(groups[g]) + rng.random() * 2.0, sum(group_cls[g].values())), reverse=True)

        assign: dict[str, str] = {}
        cur_imgs = {sp: 0 for sp in TARGET_RATIOS}
        cur_cls = {sp: Counter() for sp in TARGET_RATIOS}

        for gid in gids:
            g_size = len(groups[gid])
            g_c = group_cls[gid]
            present_cids = [cid for cid, cnt in g_c.items() if cnt > 0]

            best_sp = None
            best_cost = float("inf")
            for sp in ("train", "val", "test"):
                # Текущая доля заполнения сплита после добавления группы (цель -> 1.0 равномерно во всех сплитах)
                fill_img = (cur_imgs[sp] + g_size) / max(target_imgs[sp], 1.0)
                # Сильный штраф за переполнение выше 1.03
                overflow_img = max(0.0, fill_img - 1.0) ** 2 * 25.0

                if present_cids:
                    cls_fills = [
                        (cur_cls[sp][cid] + g_c[cid]) / max(target_cls[sp][cid], 1.0) for cid in present_cids
                    ]
                    fill_cls = sum(cls_fills) / len(cls_fills)
                    overflow_cls = sum(max(0.0, f - 1.0) ** 2 for f in cls_fills) * 4.0
                    # Бонус за закрытие ещё пустых классов в сплите
                    zero_bonus = -0.35 * sum(1 for cid in present_cids if cur_cls[sp][cid] == 0)
                else:
                    fill_cls = fill_img
                    overflow_cls = 0.0
                    zero_bonus = 0.0

                cost = fill_img * 2.5 + overflow_img + fill_cls * 1.5 + overflow_cls + zero_bonus
                if cost < best_cost:
                    best_cost = cost
                    best_sp = sp

            assert best_sp is not None
            assign[gid] = best_sp
            cur_imgs[best_sp] += g_size
            cur_cls[best_sp].update(g_c)

        # Жёсткое условие: все 10 классов присутствуют во всех трёх сплитах (минимум 10 рамок на класс в val/test)
        all_present = all(
            cur_cls[sp][cid] >= 10 for sp in ("train", "val", "test") for cid in range(len(TARGET_CLASSES))
        )
        if not all_present:
            continue

        # Итоговая оценка отклонения от идеала 70/15/15 по числу изображений и по каждому классу
        img_dev = sum(abs(cur_imgs[sp] / total_imgs - TARGET_RATIOS[sp]) for sp in ("train", "val", "test"))
        cls_dev = sum(
            abs(cur_cls[sp][cid] / max(total_cls[cid], 1) - TARGET_RATIOS[sp])
            for sp in ("train", "val", "test")
            for cid in range(len(TARGET_CLASSES))
        ) / len(TARGET_CLASSES)
        score = img_dev * 4.0 + cls_dev
        if score < best_score:
            best_score = score
            best_assign = assign

    if best_assign is None:
        raise RuntimeError("Не удалось построить разбиение с присутствием всех 10 классов в каждом сплите.")
    return best_assign


def verify_no_leakage(manifest_rows: list[dict], phash_by_split: dict[str, list[imagehash.ImageHash]]) -> dict:
    """Реализует строгие проверки отсутствия утечки между train, val и test (Раздел 3)."""
    splits = ("train", "val", "test")
    by_sp = {sp: [r for r in manifest_rows if r["split"] == sp] for sp in splits}

    checks: dict[str, object] = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    # 1. Отсутствие одинаковых имён (новых и исходных base_name)
    name_overlaps = {}
    base_overlaps = {}
    sha_overlaps = {}
    phash_overlaps = {}
    group_overlaps = {}
    min_phash_dist = {}

    for a, b in pairs:
        pair_key = f"{a}_vs_{b}"
        name_overlaps[pair_key] = len({r["filename"] for r in by_sp[a]} & {r["filename"] for r in by_sp[b]})
        base_overlaps[pair_key] = len({r["base_name"] for r in by_sp[a]} & {r["base_name"] for r in by_sp[b]})
        sha_overlaps[pair_key] = len({r["sha256"] for r in by_sp[a]} & {r["sha256"] for r in by_sp[b]})
        phash_overlaps[pair_key] = len({r["phash"] for r in by_sp[a]} & {r["phash"] for r in by_sp[b]})
        group_overlaps[pair_key] = len({r["group_id"] for r in by_sp[a]} & {r["group_id"] for r in by_sp[b]})

        # Минимальное расстояние Хэмминга между pHash разных сплитов
        md = 64
        for ha in phash_by_split[a]:
            for hb in phash_by_split[b]:
                d = int(ha - hb)
                if d < md:
                    md = d
        min_phash_dist[pair_key] = md

    checks["filename_overlaps"] = name_overlaps
    checks["orig_base_name_overlaps"] = base_overlaps
    checks["sha256_overlaps"] = sha_overlaps
    checks["exact_phash_overlaps"] = phash_overlaps
    checks["group_id_overlaps"] = group_overlaps
    checks["min_cross_split_phash_hamming"] = min_phash_dist

    assert all(v == 0 for v in name_overlaps.values()), f"Найдены общие имена файлов: {name_overlaps}"
    assert all(v == 0 for v in sha_overlaps.values()), f"Найдены общие SHA-256: {sha_overlaps}"
    assert all(v == 0 for v in phash_overlaps.values()), f"Найдены общие pHash: {phash_overlaps}"
    assert all(v == 0 for v in group_overlaps.values()), f"Найдены общие group_id: {group_overlaps}"
    assert all(v > PHASH_GROUP_THRESHOLD for v in min_phash_dist.values()), (
        f"Найдены слишком близкие кадры между сплитами: {min_phash_dist}"
    )

    checks["passed"] = True
    return checks


def main() -> None:
    print("1. Читаю сырые данные и привожу 25 классов -> 10 канонических классов...")
    raw_items, _ = collect_raw_items()
    print(f"   Сырых изображений: {len(raw_items)}")

    print("2. Удаляю подтверждённые дубликаты (включая конфликт IMG_3103_mp4-17)...")
    items, removed_dups = deduplicate_items(raw_items)
    print(f"   Осталось уникальных изображений: {len(items)} (удалено дубликатов: {len(removed_dups)})")

    print("3. Формирую группы без утечки (видео + серии + соседние YouTube + pHash <= 10)...")
    groups = assign_groups(items)
    group_sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"   Получено независимых групп: {len(groups)} (крупнейшие: {group_sizes[:10]})")

    print(f"4. Выполняю стратифицированное групповое разбиение 70/15/15 (seed={SEED})...")
    group_to_split = stratified_group_split(groups, seed=SEED)

    # Очищаем и пересоздаём data/processed
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    for sp in ("train", "val", "test"):
        (PROCESSED_DIR / sp / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_DIR / sp / "labels").mkdir(parents=True, exist_ok=True)

    # Копируем изображения и сохраняем перемапленные .txt разметки с чистыми именами ppe_0001.jpg
    items_sorted = sorted(items, key=lambda x: (group_to_split[x["group_id"]], x["group_id"], x["orig_name"]))
    manifest_rows: list[dict] = []
    phash_by_split: dict[str, list[imagehash.ImageHash]] = {"train": [], "val": [], "test": []}
    split_img_counts = Counter()
    split_box_counts: dict[str, Counter] = {sp: Counter() for sp in ("train", "val", "test")}
    split_group_sets: dict[str, set[str]] = {sp: set() for sp in ("train", "val", "test")}

    for idx, it in enumerate(items_sorted, start=1):
        sp = group_to_split[it["group_id"]]
        ext = it["img_path"].suffix.lower()
        new_stem = f"ppe_{idx:04d}"
        new_img_name = f"{new_stem}{ext}"
        new_lbl_name = f"{new_stem}.txt"

        shutil.copy2(it["img_path"], PROCESSED_DIR / sp / "images" / new_img_name)

        lbl_lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in it["boxes"]]
        (PROCESSED_DIR / sp / "labels" / new_lbl_name).write_text("\n".join(lbl_lines), encoding="utf-8")

        split_img_counts[sp] += 1
        split_group_sets[sp].add(it["group_id"])
        split_box_counts[sp].update(it["class_counts"])
        phash_by_split[sp].append(it["phash_obj"])

        cls_str = ", ".join(f"{TARGET_CLASSES[c]}:{cnt}" for c, cnt in sorted(it["class_counts"].items()))
        manifest_rows.append(
            {
                "filename": new_img_name,
                "split": sp,
                "group_id": it["group_id"],
                "orig_split": it["orig_split"],
                "orig_name": it["orig_name"],
                "base_name": it["base_name"],
                "sha256": it["sha256"],
                "phash": it["phash"],
                "width": it["width"],
                "height": it["height"],
                "num_boxes": len(it["boxes"]),
                "classes": cls_str,
            }
        )

    print("5. Запускаю автоматическую проверку отсутствия утечки между сплитами...")
    leak_checks = verify_no_leakage(manifest_rows, phash_by_split)

    # Контрольная сумма замороженной тестовой выборки (для гарантии неизменности до выбора конфигурации)
    test_fingerprint = hashlib.sha256(
        "\n".join(sorted(f"{r['filename']}:{r['sha256']}" for r in manifest_rows if r["split"] == "test")).encode("utf-8")
    ).hexdigest()

    # Сохраняем manifest.csv, removed_duplicates.json, split_report.json
    manifest_path = PROCESSED_DIR / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(manifest_rows)

    (PROCESSED_DIR / "removed_duplicates.json").write_text(
        json.dumps(removed_dups, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_n = len(manifest_rows)
    split_report = {
        "seed": SEED,
        "phash_group_threshold": PHASH_GROUP_THRESHOLD,
        "total_raw_images": len(raw_items),
        "removed_duplicates_count": len(removed_dups),
        "total_processed_images": total_n,
        "total_groups": len(groups),
        "splits": {
            sp: {
                "images": split_img_counts[sp],
                "share_pct": round(100.0 * split_img_counts[sp] / total_n, 2),
                "groups": len(split_group_sets[sp]),
                "total_boxes": sum(split_box_counts[sp].values()),
                "empty_images": sum(1 for r in manifest_rows if r["split"] == sp and r["num_boxes"] == 0),
                "per_class_boxes": {TARGET_CLASSES[c]: split_box_counts[sp][c] for c in range(len(TARGET_CLASSES))},
            }
            for sp in ("train", "val", "test")
        },
        "frozen_test_sha256": test_fingerprint,
        "leakage_checks": leak_checks,
    }
    (PROCESSED_DIR / "split_report.json").write_text(
        json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Сохраняем configs/ppe.yaml (относительный путь от корня проекта — воспроизводимо на любой машине)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ppe_yaml = {
        "path": "data/processed",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(TARGET_CLASSES),
        "names": {i: name for i, name in enumerate(TARGET_CLASSES)},
    }
    CONFIG_PATH.write_text(yaml.safe_dump(ppe_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Итоговый вывод
    print("\n=== ИТОГИ РАЗБИЕНИЯ БЕЗ УТЕЧКИ ===")
    for sp in ("train", "val", "test"):
        s = split_report["splits"][sp]
        print(
            f"  {sp:5s}: {s['images']:3d} images ({s['share_pct']:5.2f}%) | "
            f"{s['groups']:3d} groups | {s['total_boxes']:4d} boxes | {s['empty_images']:2d} empty"
        )
    print("\nРаспределение объектов по 10 классам (train / val / test):")
    for cid, cname in enumerate(TARGET_CLASSES):
        tr = split_box_counts["train"][cid]
        va = split_box_counts["val"][cid]
        te = split_box_counts["test"][cid]
        print(f"  [{cid}] {cname:16s}: {tr:4d} / {va:3d} / {te:3d}  (total={tr+va+te})")
    print("\nПроверки утечки:", json.dumps(leak_checks, ensure_ascii=False))
    print(f"Frozen test SHA-256: {test_fingerprint}")
    print(f"Saved config: {CONFIG_PATH}")


if __name__ == "__main__":
    main()
