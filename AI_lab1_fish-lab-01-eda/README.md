# Регрессионный анализ параметров рыб (Fish Measurements)

Лабораторная работа №1 по введению в ИИ.

## Структура

```text
.
├── data/
│   └── raw/
│       └── Fish.csv              # Исходный датасет
├── notebooks/
│   └── 01-eda.ipynb              # Ноутбук с EDA, обработкой выбросов и гипотезами
├── reports/
│   ├── figures/                  # Сгенерированные графики с подписями
│   │   ├── 01_target_distribution.png
│   │   ├── 02_features_distribution.png
│   │   ├── 03_species_count.png
│   │   ├── 04_target_vs_features.png
│   │   ├── 05_pairplot_by_species.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_weight_by_species_boxplot.png
│   │   └── 08_hypothesis_validation.png
│   └── lab01-report.md           # Итоговый отчет (паспорт, графики, выводы, утечки)
├── README.md                     # Описание проекта и инструкции по запуску
└── requirements.txt              # Список зависимостей


## Клонирование репрезитория

git clone <https://github.com/fan4stic/AI_lab1_fish.git>
cd <НАЗВАНИЕ_ПАПКИ_ПРОЕКТА>

## Создание виртуального окружения (Linux)
python3 -m venv venv
source venv/bin/activate

## Создание виртуального окружения (Windows)
python -m venv venv
.\venv\Scripts\Activate.ps1

## Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt

## Запуск работы
jupyter lab
