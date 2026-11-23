# Обнаружение средств индивидуальной защиты (СИЗ) на стройплощадке с YOLOv8 (Лабораторная работа №3)

Адаптированный и воспроизводимый проект обнаружения СИЗ (касок, жилетов, масок, нарушений и спецтехники) на базе исходного репозитория `Construction-Site-Safety-PPE-Detection` и архитектуры **YOLOv8**.

> **Полные материалы сдачи:**
> - Итоговый отчёт с таблицами, графиками, разбором 12 ошибок, системной оценкой и разделом «Использование ИИ»: **[reports/lab03-report.md](reports/lab03-report.md)**
> - Паспорт данных, контрольная сумма архива, правила разметки и итоги ручного аудита (30+10 кадров): **[data/README.md](data/README.md)**

---

## Оборудование и зафиксированное окружение

Все характеристики зафиксированы в [configs/environment.json](configs/environment.json), [requirements.txt](requirements.txt) и [requirements-lock.txt](requirements-lock.txt):

- **ОС:** Windows 10 (`10.0.19044`)
- **Python:** `3.12.10`
- **GPU:** NVIDIA GeForce RTX 5070 (`12227 MiB VRAM`, архитектура Blackwell `compute capability 12.0`, драйвер `617.14`, CUDA драйвера `13.4`)
- **Ключевые библиотеки:** `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `ultralytics==8.4.216`, `opencv-python==5.1.0.95`, `numpy==2.5.4`, `ImageHash==4.3.2`, `pytest==9.2.0`
- **Начальное значение генератора (`seed`):** `42` (`deterministic=True`)

---

## Структура проекта

```text
Construction-Site-Safety-PPE-Detection/
├── README.md                        # Инструкция запуска, оборудование и сводка результатов
├── requirements.txt                 # Основные зависимости проекта (с CUDA 12.8)
├── requirements-lock.txt            # Полный снимок окружения (pip freeze)
├── configs/
│   ├── environment.json             # Паспорт оборудования и версий ПО
│   ├── ppe.yaml                     # Конфигурация 10-классового датасета для YOLOv8
│   ├── e1_baseline.yaml             # Конфигурация E1 (база: yolov8n, 50 эпох, 640)
│   └── e2_epochs100.yaml            # Конфигурация E2 (гипотеза: epochs 50 -> 100)
├── data/
│   ├── README.md                    # Паспорт данных, лицензия CC BY 4.0, аудит 30+10 фото
│   └── .gitignore                   # Исключает тяжёлые изображения и архивы из Git
├── src/
│   ├── unpack_and_inspect.py        # Распаковка Roboflow v30 и подготовка выборки аудита (30+10)
│   ├── prepare_dataset.py           # Маппинг 25->10 классов, дедупликация, групповое разбиение ~70/15/15
│   ├── run_e0_pretrained.py         # Эксперимент E0: предобученная yolov8n.pt на 10 кадрах без обучения
│   ├── train_and_eval.py            # Воспроизводимое обучение и оценка на замороженном тесте (E1/E2)
│   ├── benchmark_latency.py         # Замер задержки (p50/p95) по этапам, FPS, памяти и ложных тревог
│   ├── error_analysis.py            # Повторно используемый визуализатор и 6 галерей ошибок
│   ├── stress_test_robustness.py    # Стресс-тест устойчивости (освещение, наклон, перекрытие)
│   └── predict.py                   # Быстрый инференс на одном изображении (с правилом передачи оператору)
├── tests/
│   └── test_ppe_pipeline.py         # 6 быстрых автотестов (YAML, разметка, утечки, визуализатор, инференс)
├── reports/
│   ├── lab03-report.md              # Итоговый отчёт по лабораторной работе №3
│   ├── manual_audit/                # Журнал и сводка ручной проверки 40 кадров
│   ├── e0_pretrained/               # Предсказания и коллаж E0
│   ├── metrics/                     # JSON/CSV метрики E1, E2, бенчмарка и стресс-теста
│   ├── figures/                     # Матрицы ошибок, PR/F1-кривые, графики задержки и устойчивости
│   └── error_galleries/             # 6 галерей ошибок (JPG) и каталог error_catalog.json
├── models/                          # Исходные предобученные веса yolov8n.pt (и исходный best.pt автора)
└── runs/                            # Временные логи и веса обучения (в .gitignore)
```

---

## Быстрый старт и воспроизведение

### 1. Создание виртуального окружения
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Получение и подготовка данных
Скачайте архив **Construction Site Safety v30 (`raw-images_latestversion`)** в формате YOLOv8 с Roboflow Universe (подробная инструкция и `SHA-256` указаны в [data/README.md](data/README.md)), положите `.zip` в `data/raw/` и выполните:
```powershell
.\.venv\Scripts\python.exe src\unpack_and_inspect.py
.\.venv\Scripts\python.exe src\prepare_dataset.py
```

### 3. Запуск автоматических тестов (`pytest`)
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
Ожидаемый результат: `6 passed` (~6 с) — проверка путей YAML, корректности координат `[0, 1]` и классов `[0, 9]`, нулевой утечки (`group_id`, `SHA-256`, `pHash`), неизменности замороженного теста, размера выхода визуализатора и схемы предсказания модели.

### 4. Воспроизведение всех экспериментов и аналитики
```powershell
# E0: проверка предобученной yolov8n.pt без обучения на 10 кадрах val
.\.venv\Scripts\python.exe src\run_e0_pretrained.py

# E1: базовое решение (50 эпох, imgsz=640, batch=16, seed=42)
.\.venv\Scripts\python.exe src\train_and_eval.py --config configs/e1_baseline.yaml

# E2: проверка гипотезы (epochs: 50 -> 100)
.\.venv\Scripts\python.exe src\train_and_eval.py --config configs/e2_epochs100.yaml

# Замер задержки по этапам (p50/p95), FPS, памяти и ложных тревог
.\.venv\Scripts\python.exe src\benchmark_latency.py

# Построение 6 галерей ошибок на тестовой выборке
.\.venv\Scripts\python.exe src\error_analysis.py

# Стресс-тест устойчивости (сумерки, засветка, наклон, перекрытие)
.\.venv\Scripts\python.exe src\stress_test_robustness.py
```

### 5. Быстрый сценарий предсказания на одном фото (для демонстрации на защите)
```powershell
.\.venv\Scripts\python.exe src\predict.py --image data/processed/test/images/ppe_0063.jpg --save reports/figures/demo_pred_ppe_0063.jpg
```

---

## Сводка результатов (замороженный тест: 110 фото, 711 объектов, 0 утечек)

| Эксперимент | Конфигурация | `Precision` | `Recall` | `mAP@0.5` | `mAP@0.5:0.95` | Задержка `p50 / p95` (End-to-End) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **E0** | `yolov8n.pt` (COCO, без обучения) | — | 0% по СИЗ | 0.000 | 0.000 | — |
| **E1 (лучшая)** | `yolov8n`, `epochs=50`, `imgsz=640` | 0.7084 | **0.5540** | **0.5946** | **0.3575** | **14.75 / 23.72 мс (62.6 FPS)** |
| **E2** | `yolov8n`, `epochs=100`, `imgsz=640` | **0.7461** | 0.5391 | 0.5849 | 0.3477 | 15.26 / 21.67 мс (61.6 FPS) |
