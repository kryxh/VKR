# Propensity Model — Документация

## Обзор

Propensity Model — это система скоринга клиентов для банковско-инвестиционного контекста. Модель оценивает вероятность покупки инвестиционного инструмента в ближайшие 30 дней и формирует приоритизированный список клиентов для контакта со стороны финансового советника.

**Ключевая бизнес-задача:** не «кто уже является крупным клиентом», а «кто с высокой вероятностью совершит покупку или вернётся к активности в ближайшее время».

**Данные:** датасет [FAR-Trans](https://github.com/sanzcruz/FAR-Trans) — транзакционная история инвестиционных операций ~28 000 клиентов с 2018 по 2022 год.

---

## Архитектура

Финальная версия реализована как **двухмодельная segment-aware система**:

```
Все клиенты
    │
    ├── Warm (≤ персонального порога дней без покупки)
    │       └── CatBoost Timing → вероятность покупки в 30 дней
    │
    └── Dormant (> персонального порога)
            └── CatBoost Reactivation → вероятность возврата
    │
    └── Isotonic Calibration (раздельная для каждой модели)
    │
    └── Unified Scoring → единый ранжированный список
            └── hot_customers_{DATE}.csv  (топ-20%)
```

Обе модели обучаются независимо, калибруются раздельно через Isotonic Regression, после чего их скоры объединяются в единый ранжированный список с дополнительным весом для dormant-сегмента.

---

## Сегментация клиентов

### Персональный порог покоя

Граница между warm и dormant рассчитывается индивидуально для каждого клиента:

```
personal_dormancy_threshold = max(1.5 × median_buy_interval, 60 дней)
```

Клиент считается **dormant**, если `days_since_last_buy > personal_dormancy_threshold`.

Клиент считается **warm**, если пауза не превышает его личного порога. Для новых клиентов без истории интервалов применяется минимальный порог в 60 дней.

**Обоснование:** EDA показал, что персональный порог (reactivation rate = 3.8%) значительно точнее фиксированного в 90 дней (2.8%) или 180 дней (2.0%) — при меньшем размере dormant-пула.

---

## Признаки

### Timing-модель (warm-клиенты)

| Группа | Признаки |
|---|---|
| Профиль (категориальные) | `customerType`, `riskLevel`, `investmentCapacity` |
| RFM | `num_buys_30d`, `num_buys_90d`, `sum_value_30d`, `days_since_last_buy`, `avg_ticket` |
| Временная динамика | `activity_trend`, `is_new_customer`, `customer_tenure_days` |
| Портфель | `share_stocks`, `share_bonds`, `share_funds`, `num_unique_assets` |
| Рыночный контекст | `market_return_7d`, `market_return_30d`, `market_volatility_30d`, `market_drawdown` |
| Взаимодействия | `aggressive_x_recency`, `stocks_x_market_correction` |
| Аномальные периоды | `snapshot_in_anomaly`, `pause_includes_anomaly`, `history_includes_anomaly` |

**Признаки аномальных периодов** маркируют срезы и паузы, попадающие на периоды рыночных аномалий (COVID-кризис и др.), предотвращая искажение агрегатов.

### Reactivation-модель (dormant-клиенты)

Для спящих клиентов текущая активность отсутствует — модель опирается на исторические паттерны:

| Группа | Признаки |
|---|---|
| Профиль | `customerType`, `riskLevel`, `customer_tenure_days` |
| Портфель | `share_stocks`, `share_bonds`, `share_funds`, `num_unique_assets` |
| Cadence | `median_buy_interval`, `std_buy_interval`, `max_historical_gap`, `has_cadence_data` |
| Пауза | `current_pause_days`, `personal_dormancy_threshold`, `pause_zscore`, `pause_near_personal_max` |
| История активности | `total_buys_lifetime`, `buys_last_year`, `tenure_buy_days`, `long_term_activity_trend` |
| Возврат после паузы | `has_returned_after_90d_gap`, `survived_similar_pause` |
| BG/NBD | `p_alive` — вероятность, что клиент ещё активен (модель Fader, Hardie & Lee, 2005) |
| Сезонность | `bought_same_month_last_year`, `bought_same_quarter_last_year`, `num_buys_same_month_hist` |
| Рыночный контекст | `market_return_7d`, `market_return_30d`, `market_volatility_30d`, `market_drawdown` |
| Взаимодействия | `value_buyer_x_correction`, `snapshot_in_anomaly`, `pause_includes_anomaly`, `history_includes_anomaly` |

**Признак `p_alive`** вычисляется через BG/NBD модель (библиотека `lifetimes`). Для клиентов с менее чем 2 покупками принимает значение NaN — CatBoost обрабатывает пропуски нативно.

**Признак `has_cadence_data`** — бинарный флаг наличия данных о ритме покупок. Без него нулевые дефолты для cadence-признаков маскируют отсутствие сигнала.

---

## Параметры моделей

### CatBoost Timing

| Параметр | Значение |
|---|---|
| `iterations` | 300 |
| `learning_rate` | 0.05 |
| `depth` | 6 |
| `auto_class_weights` | `Balanced` |
| `eval_metric` | AUC |
| `early_stopping_rounds` | 50 |

### CatBoost Reactivation

| Параметр | Значение |
|---|---|
| `iterations` | 200 |
| `learning_rate` | 0.03 |
| `depth` | 6 |
| `auto_class_weights` | `Balanced` |
| `eval_metric` | AUC |
| `early_stopping_rounds` | 40 |

Гиперпараметры подобраны grid search по сетке `depth ∈ {4, 5, 6}` × `learning_rate ∈ {0.03, 0.05, 0.1}`. Оба набора параметров зафиксированы в `config.py`.

**`auto_class_weights='Balanced'`** применяется из-за дисбаланса классов. Доля положительного класса на train: timing — 36%, reactivation — 15%; на test: timing — 23%, reactivation — 13%. Автоматическое взвешивание обратно пропорционально частоте класса.

---

## Временной сплит

Сплит вычисляется динамически от конца данных:

| Период | Граница |
|---|---|
| Test | последние 14 месяцев от конца данных |
| Validation | предшествующие 9 месяцев |
| Train | всё, что до validation |

При текущем датасете FAR-Trans (конец ≈ октябрь 2022):

**Timing:**
- Train: до 2020-12-01 (77 466 строк, label=1: 36.01%)
- Validation: 2020-12-01 — 2021-09-01 (68 276 строк, label=1: 33.01%)
- Test: с 2021-09-01 (117 246 строк, label=1: 23.25%)

**Reactivation:**
- Train: до 2020-12-01 (12 981 строк, label=1: 14.81%)
- Validation: 2020-12-01 — 2021-09-01 (12 049 строк, label=1: 15.08%)
- Test: с 2021-09-01 (27 937 строк, label=1: 12.87%)

Сплит по времени (а не случайный) исключает утечку будущего в обучающую выборку.

---

## Калибровка

После обучения обе модели проходят **Isotonic Regression калибровку** на validation-выборке:

```python
cal = IsotonicRegression(out_of_bounds="clip")
cal.fit(raw_scores, y_valid)
final_score = cal.transform(raw_score)
```

Калибровка обеспечивает:
1. Сопоставимость скоров между timing и reactivation моделями.
2. Интерпретируемость скоров как вероятностей (Brier score ↓).
3. Корректное ранжирование при объединении в единый список.

Калибраторы сохраняются как `calibrator_timing.pkl` и `calibrator_reactivation.pkl`.

---

## Инференс и объединение скоров

Функция `score_customers_unified()` в `predict.py`:

1. Определяет для каждого клиента сегмент (warm/dormant) на основе персонального порога.
2. Вычисляет признаки раздельно для каждого сегмента (`mode="timing"` или `"reactivation"`).
3. Скорит через соответствующую модель и калибратор.
4. Применяет `reactivation_weight` к dormant-скорам (бизнес-параметр, контролирует долю спящих в топе).
5. Объединяет оба списка и сортирует по итоговому скору.

**Контроль качества выхода:** если доля dormant в топ-20% ниже 15%, pipeline логирует предупреждение о необходимости настройки `reactivation_weight`. При последнем прогоне (snapshot 2022-10-31): отобрано 1 132 клиента из 5 662 (top 20%, min_score=0.05) — Warm: 931, Dormant: 201, доля dormant в топе: 17.8% (предупреждение не сработало).

### Выходные файлы

**`all_scores_{DATE}.csv`** — полный список всех проскоренных клиентов:

| Колонка | Описание |
|---|---|
| `customerID` | ID клиента |
| `propensity_score` | откалиброванная вероятность покупки [0, 1] |
| `segment` | `warm` или `dormant` |
| `days_since_last_buy` | дней с последней покупки |
| `rank` | ранг (1 = наивысший приоритет) |
| `snapshot_date` | дата скоринга |

**`hot_customers_{DATE}.csv`** — топ-20% по рангу с `propensity_score ≥ 0.05`. Является входным файлом для Recommender модели.

---

## Оценка качества

### Метрики

| Метрика | Описание |
|---|---|
| ROC-AUC | Качество ранжирования |
| PR-AUC | Основная метрика при дисбалансе классов |
| Brier Score | Калиброванность вероятностей |
| Recall@top20% | Доля реальных покупателей в топ-20% списке |
| Lift@top20% | Кратность превышения случайного отбора |
| Precision@k | Точность для топ-k клиентов |

**PR-AUC** используется как основная метрика оценки, поскольку при дисбалансе ROC-AUC оптимистично завышен.

### Результаты на тестовой выборке

| Модель | ROC-AUC | PR-AUC | Recall@top20% | Lift@top20% |
|---|---|---|---|---|
| Timing (CatBoost) | 0.8242 | 0.6413 | 0.532 | 2.66x |
| Reactivation (CatBoost) | 0.6132 | 0.1851 | 0.318 | 1.59x |

Timing модель обучилась за 146 итераций (early stop, validation ROC-AUC=0.7956). Reactivation — за 76 итераций (early stop, validation ROC-AUC=0.6361).

### Baseline

В режиме `--train --with-baseline` обучается **Logistic Regression** (отдельно для timing и reactivation) как слабый baseline для сравнения с CatBoost.

### Визуализации

Функция `run_evaluation()` сохраняет в `outputs/evaluation/`:
- ROC-кривые и PR-кривые для обеих моделей
- Cumulative Gain и Lift кривые
- Feature importance (с цветовой разметкой по группам признаков)
- Calibration plot
- Таблица сравнения метрик

---

## Режимы запуска

```bash
# Inference (сервисный режим — дефолт):
python propensity_model/main.py --snapshot-date 2022-10-31
python propensity_model/main.py --predict-only --snapshot-date 2022-10-31

# Полное обучение:
python propensity_model/main.py --train

# Обучение + LogReg baseline для сравнения:
python propensity_model/main.py --train --with-baseline

# Обучение с пропуском пересборки датасета (если уже собран):
python propensity_model/main.py --train --skip-build

# Только оценка (загружает сохранённые модели):
python propensity_model/main.py --eval-only --eval-stage test
python propensity_model/main.py --eval-only --eval-stage validation --with-baseline

# Grid search гиперпараметров CatBoost:
python propensity_model/main.py --tune
```

| Флаг | Действие |
|---|---|
| *(нет флагов)* / `--predict-only` | Загружает сохранённые модели, строит признаки на `snapshot_date`, скорит всех клиентов |
| `--train` | Полный пайплайн: сборка датасета → обучение CatBoost → калибровка → оценка |
| `--train --with-baseline` | То же + обучение LogReg baseline для сравнения |
| `--eval-only` | Загружает сохранённые модели, оценивает на заданном split |
| `--tune` | Grid search по гиперпараметрам CatBoost (depth × learning_rate), результаты → `outputs/grid_search_*.json` |
| `--skip-build` | Пропустить пересборку датасетов, использовать кэш из `.parquet` |
| `--snapshot-date` | Дата inference в формате `YYYY-MM-DD`. По умолчанию — последняя дата в транзакциях |

---

## Структура файлов

```
propensity_model/
├── main.py                       # Точка входа, CLI, оркестрация этапов
├── src/
│   ├── config.py                 # Константы, пути, списки признаков, гиперпараметры
│   ├── data_loader.py            # Загрузка и базовая очистка данных
│   ├── feature_engineering.py   # Построение всех признаков (9 групп)
│   ├── dataset_builder.py        # Snapshot-логика, построение таргета, time_split
│   ├── train.py                  # Обучение CatBoost / LogReg, калибровка, grid search
│   ├── evaluate.py               # Метрики и визуализации
│   └── predict.py                # Инференс, unified scoring, select_hot_customers
├── models/
│   ├── catboost_timing.cbm
│   ├── catboost_reactivation.cbm
│   ├── calibrator_timing.pkl
│   ├── calibrator_reactivation.pkl
│   ├── logreg_baseline.pkl            # только при --with-baseline
│   └── logreg_reactivation_baseline.pkl
└── outputs/
    ├── dataset.parquet                # кэш timing датасета
    ├── reactivation_dataset.parquet   # кэш reactivation датасета
    ├── pipeline.log
    ├── predictions/
    │   ├── all_scores_{DATE}.csv
    │   └── hot_customers_{DATE}.csv   # → вход для Recommender
    └── evaluation/
        └── *.png, all_metrics.json
```

---

## Зависимости

```
catboost
scikit-learn
pandas
numpy
lifetimes       # BG/NBD модель (p_alive признак)
matplotlib
pyarrow         # parquet кэш
```

Минимальная дата для корректного inference: **2019-08-01** (требуется не менее 365 дней истории транзакций до snapshot_date).

---

## Связь с Recommender моделью

Выходной файл `hot_customers_{DATE}.csv` является входом для Recommender модели. Оба скрипта должны запускаться на **одну и ту же дату**:

```bash
# Шаг 1 — propensity inference
python propensity_model/main.py --predict-only --snapshot-date 2022-10-31

# Шаг 2 — recommender inference
python recommender/main.py --snapshot-date 2022-10-31 --train
```
