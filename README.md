# Fish Weight Prediction Model:

![Fish](images/Fishlength.jpg)

## Requirements

Hello! Welcome to the famous Tsukiji fish market of Tokyo, Japan! We came here to collect data on some of the fish they have here 
but we didn't wake up at 5am for the tuna auction and by the time we showed up they were only left with a few species of fish. 
We got to work and gathered measurements from a few different species of fish and want you to train a regression model to predict
the weight of a fish using some of the features we were able to measure. We have no idea which features will be good predictors. 

We will hold out 30% of the data before we hand it to you and we will use that csv for scoring.

Here's what we need from you:
1. A function that accepts a csv path and returns the predictions of your regression model using our csv. The csv we use will contain all the columns. 
2. Use a pipenv and scikit learn to submit the final model. You may use R for model selection. 

With this function and it's output, we will rank the students by how well their model performed on predicting weight based on naive data. 
Your grade will be determined by ranking according to Mean-squared Error.
If your function does not return a list of predictions or we cannot compute the accuracy of your model that it will be an automatic F. 

## Used

*   Linear Regression
*   Lasso Regression
*   Random Forest
*   Scikit Learn
*   Python
*   Joblib
*   Matplotlib
*   Pipenv

# FishGrow: прогностическая модель массы рыбы

**Лабораторная работа №2** по дисциплине «Введение в искусственный интеллект».  
ПетрГУ, 2026.  
Ветка: `lab/02-regression`.

Построение воспроизводимой регрессионной модели, которая по морфологическим признакам рыбы предсказывает её массу, сравнение линейного и нелинейного подходов, логирование экспериментов в MLflow.

## Содержание

- [Постановка задачи](#постановка-задачи)
- [Структура репозитория](#структура-репозитория)
- [Быстрый старт](#быстрый-старт)
- [Запуск экспериментов](#запуск-экспериментов)
- [Анализ ошибок](#анализ-ошибок)
- [Тесты](#тесты)
- [Просмотр MLflow](#просмотр-mlflow)
- [Воспроизводимость](#воспроизводимость)
- [Отчёт и паспорта](#отчёт-и-паспорта)
- [Использование ИИ](#использование-ии)

## Постановка задачи

**Целевая переменная:** `Weight` — масса рыбы в граммах.  
**Признаки:** `Length3`, `Height`, `Width` (числовые), `Species` (категориальный).  
**Исключены:** `Length1`, `Length2` — коллинеарны с `Length3` (corr > 0.99).

**Критерии успеха:**

- test R² ≥ 0.85;
- MAE ≤ 100 г;
- модель превосходит наивный baseline;
- все прогнозы физически осмысленны (> 0 г и < 5000 г);
- эксперимент воспроизводим при одинаковом seed и разбиении.

Подробно — в `reports/lab02-report.md`.

## Структура репозитория

```
.
├── configs/               # конфигурация экспериментов
│   └── train.yaml
├── data/                  # исходные данные и их паспорт
│   ├── README.md
│   ├── fish_participant.csv
│   └── fish_holdout_demo.csv
├── images/                # иллюстрация для README
├── models/                # сохранённая модель и её паспорт
│   ├── best_model.joblib
│   └── model-card.md
├── notebooks/             # разведочный анализ данных (только EDA)
│   └── data_analysis_and_model_selection.ipynb
├── reports/               # отчёт, графики, экспорт MLflow
│   ├── lab02-report.md
│   ├── mlflow_runs_export.csv
│   ├── mlflow_summary.md
│   ├── figures/
│   └── screenshots/
├── runs/                  # артефакты запусков
│   ├── summary.csv
│   ├── split_indices.csv
│   ├── top_errors.csv
│   ├── mae_by_species.csv
│   ├── mae_by_size.csv
│   └── predictions/
├── src/                   # исходный код
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_mlflow.py
├── tests/                 # минимальные тесты
│   └── test_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Быстрый старт

Требуется **Python 3.10+**.

```powershell
# 1. Клонировать репозиторий и перейти в него
git clone <URL репозитория>
cd fish_weight_prediction_machine_learning-master
git checkout lab/02-regression

# 2. Создать виртуальное окружение
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # Linux / macOS

# 3. Установить зависимости
pip install -r requirements.txt
```

## Запуск экспериментов

```powershell
python src/train.py
```

Скрипт:

1. Загружает `configs/train.yaml`.
2. Замораживает тестовую выборку (`train_test_split(test_size=0.2, random_state=42)`).
3. Обучает E0–E4 с кросс-валидацией `GroupKFold(n_splits=5)` по `Species`.
4. Считает MSE, MAE, R² на замороженном тесте.
5. Логирует параметры, метрики и артефакты в MLflow (`mlruns/`).
6. Сохраняет лучшую модель в `models/best_model.joblib`.
7. Пишет сводную таблицу в `runs/summary.csv`.

Время работы: 1–3 минуты.

## Анализ ошибок

```powershell
python src/evaluate.py
```

Сохраняет:

- `runs/top_errors.csv` — топ-5 крупнейших ошибок;
- `runs/mae_by_species.csv` — MAE по видам;
- `runs/mae_by_size.csv` — MAE по диапазонам длины;
- `reports/figures/best_model_residuals.png`, `best_model_pred_vs_true.png`, `best_model_top_errors.png`.

## Тесты

```powershell
pytest tests/ -v
```

6 тестов, покрывают раздел 7 задания: непересечение train и test, обработка неизвестной категории, число и конечность прогнозов, ручная проверка метрик, воспроизводимость загруженного pipeline, обработка пропусков.

## Просмотр MLflow

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri mlruns --workers 1
```

Открыть в браузере `http://127.0.0.1:5000`, выбрать эксперимент `fishgrow-lab02`.

Если UI не запускается, экспорт метаданных и снимок структуры запусков лежат в:

- `reports/mlflow_runs_export.csv` — полная выгрузка всех запусков;
- `reports/mlflow_summary.md` — человекочитаемая сводка;
- `reports/screenshots/` — скриншоты UI.

## Воспроизводимость

- **Seed:** 42 (Python, NumPy, sklearn).
- **Разбиение:** `train_test_split(test_size=0.2, random_state=42)`; индексы сохранены в `runs/split_indices.csv`.
- **Кросс-валидация:** `GroupKFold(n_splits=5)` по `Species`.
- **Версия данных:** фиксируется хешем SHA-256 файла `data/fish_participant.csv` и логируется в параметрах MLflow (`data_sha256`).
- **Версия окружения:** `python_version`, `sklearn_version`, `mlflow_version`, `pandas_version` логируются в тегах MLflow.
- **Версия кода:** `git_commit` логируется в тегах MLflow.

Повторный запуск `python src/train.py` на тех же данных даёт идентичные метрики.

## Отчёт и паспорта

- `reports/lab02-report.md` — полный отчёт по лабораторной №2;
- `models/model-card.md` — паспорт модели;
- `data/README.md` — паспорт данных.
