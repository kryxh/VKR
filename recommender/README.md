# Recommender Model — Документация

## Обзор

Recommender Model генерирует персонализированные рекомендации инвестиционных инструментов (топ-3 ISIN) для каждого клиента из списка `hot_customers`, сформированного Propensity Model. Рекомендации передаются финансовому советнику вместе с текстовым обоснованием для каждой позиции.

**Ключевая задача:** не просто ранжировать инструменты, а предложить те, которые с наибольшей вероятностью соответствуют текущим предпочтениям клиента — с учётом его истории, поведения похожих клиентов и диверсификации портфеля.

**Данные:** датасет [FAR-Trans](https://github.com/sanzcruz/FAR-Trans) — транзакционная история ~28 000 клиентов, 279 уникальных ISIN (Stock, Bond, MTF).

---

## Архитектура

```
hot_customers_{DATE}.csv (от Propensity Model)
    │
    ├── Для клиентов с историей ≥ 3 уникальных ISIN в item-space
    │       └── EASE (основная модель)
    │               └── score(u, i) = (1/|H_u|) × Σ_{j∈H_u} W[j, i]
    │
    ├── Для клиентов вне EASE item-space → ALS (fallback)
    │       └── score = α × s_item + (1-α) × s_user  [percentile rank]
    │
    └── Для клиентов без достаточной истории
            └── Popularity-based fallback (по preferred_category)
    │
    └── Diversity корректировка + фильтр недавно купленных
    │
    └── recommendations_{DATE}.csv  (топ-3 ISIN на клиента)
```

---

## Основная модель: EASE

**EASE** (Embarrassingly Shallow Autoencoders for Sparse Data, Steck 2019) — item-item коллаборативная фильтрация с аналитическим закрытым решением:

```
W = I − P × diag(1 / diag(P))
P = (X^T X + λI)^{-1}
```

где `X` — user-item матрица взаимодействий, `λ` — единственный гиперпараметр регуляризации.

**Скор для пользователя `u` по инструменту `i`:**
```
score(u, i) = (1/|H_u|) × Σ_{j ∈ H_u} W[j, i]
```
где `H_u` — история покупок клиента в пространстве item-space.

**Размер матрицы W:** 158 × 158 (число ISIN с поддержкой ≥ 5 покупателей). Вычисляется аналитически за секунды.

**Оптимальный λ:** подбирается grid search по 5 значениям на LOO-валидации. Финальное значение: `λ = 500`.

**Переобучение:** EASE переобучается при каждом новом snapshot, поскольку скользящее 365-дневное окно транзакций смещается.

---

## Fallback модель: ALS

**ALS** (Alternating Least Squares, Hu et al. 2008) используется как fallback при недоступности EASE или отсутствии клиента в item-space матрицы:

- User-факторы `U` (N × k) и Item-факторы `V` (M × k), k = 20
- **Item-based score:** `s_item(u, i) = (1/|H_u|) × Σ_{j∈H_u} (v_j · v_i)`
- **User-based score:** `s_user(u, i) = u_u · v_i`
- **Fusion:** `score(u, i) = α × rank_norm(s_item) + (1−α) × rank_norm(s_user)`

Нормализация через **percentile rank** — устойчива к выбросам, не зависит от абсолютных значений скоров.

**Оптимальный FUSION_ALPHA = 1.0** (100% item-based, 0% user-based): обусловлен тем, что лишь ~1.6% hot customers имеют user-эмбеддинги в тренировочной матрице — user-компонент фактически не вносит вклада, и модель вырождается в чисто item-based скоринг.

---

## Interaction Matrix

### Параметры построения

| Параметр | Значение | Обоснование |
|---|---|---|
| `INTERACTION_WINDOW_DAYS` | 365 | Плотность матрицы 1.759% vs 0.92% для всей истории; ablation: +54% NDCG |
| `MIN_ITEM_SUPPORT` | 5 | Удаляет 20% ISIN, теряет только 0.2% транзакций; эмбеддинги редких ISIN ненадёжны |
| `EXCLUDE_RECENT_DAYS` | 30 | 62.3% повторных покупок одного ISIN в ≤ 30 дней — без фильтра модель рекомендует уже купленное |
| `MIN_TX_REC` | 3 | EDA: warm-клиенты с ≥ 3 уникальными ISIN = 77.1% → ALS является основным путём |

### Confidence weighting с temporal decay

```
c_ui = 1 + CONFIDENCE_ALPHA × log(1 + count_ui) × exp(−DECAY_LAMBDA × days_ago / 365)
```

- Основа: формула из Hu et al. (2008) для implicit feedback
- Расширение: множитель temporal decay по Koren (2010, TimeSVD++)
- `CONFIDENCE_ALPHA = 80`, `DECAY_LAMBDA = 1.0` — подобраны grid search
- При `DECAY_LAMBDA = 1.0`: транзакция годичной давности весит `e^{-1} ≈ 37%` от сегодняшней

---

## Grid Search

### Этап 1 — гиперпараметры ALS (135 комбинаций)

| Параметр | Grid | Лучшее значение |
|---|---|---|
| k (латентные факторы) | {12, 16, 20, 24, 32} | 20 |
| λ (L2-регуляризация) | {0.01, 0.05, 0.1} | 0.1 |
| `CONFIDENCE_ALPHA` | {40, 60, 80} | 80 |
| `DECAY_LAMBDA` | {0.5, 1.0, 2.0} | 1.0 |

**Метрика:** NDCG@3 на LOO-валидации (Leave-One-Out).

### Этап 2 — FUSION_ALPHA (11 значений)

После фиксации лучших ALS-параметров перебираются веса fusion: `FUSION_ALPHA ∈ {0.0, 0.1, ..., 1.0}`. Лучшее: **0.9**.

### Валидация EASE (5 значений)

Grid search по `λ ∈ {50, 200, 500, 1000, 2000}` на тех же LOO-парах. Лучшее: **λ = 500**.

---

## LOO-валидация

**Протокол:** для каждого eligible клиента (≥ 3 уникальных ISIN в окне) скрываем последнюю по времени транзакцию в validation-периоде и проверяем попадание скрытого ISIN в топ-3 рекомендации.

Число eligible клиентов: **926** (validation-окно) / **2 223** (test-окно).

**Важно для EASE:** матрица обучается без LOO-пар (`build_tx_without_loo_pairs()`), поскольку EASE — аналитическое решение, и скрытая пара оставляет прямой след в матрице `W`. Для ALS с k=20 leakage несущественен — один пример размывается по 17 470 взаимодействиям.

---

## Расширение истории (ступенчатое окно)

Для каждого клиента история `H_u` строится ступенчато:

```
1. H_u = ISINs в [snapshot − 365d, snapshot) ∩ ALS item-space
   if |unique(H_u)| ≥ 3 → ALS/EASE path

2. Расширяем до 730d
   if |unique(H_u)| ≥ 3 → ALS path (window_used="730d")

3. Расширяем до полной истории (3650d)
   if |unique(H_u)| ≥ 3 → ALS path (window_used="full")

4. → Fallback path (window_used="none")
```

**Критически важно:** `H_u` фильтруется по ALS item-space — ISIN из более старой истории могут отсутствовать в 365d-матрице и вызвать KeyError без фильтрации.

---

## Diversity корректировка

### Portfolio boost (активируется при n_cat ≥ 2, охват 4.4% базы)

Мягкий boost × 1.2 к скорам инструментов из категорий с меньшей долей в портфеле клиента. Обоснование: портфельная диверсификация как бизнес-цель.

### Intra-category diversity (для 95.6% клиентов с одной категорией)

```
Stock (с данными о секторе):
    → топ-1 ISIN из каждого из топ-3 секторов по popularity

MTF / Bond без данных о секторе:
    → weighted random sampling с весом log(n_buyers)
```

### Фильтр недавно купленных

Исключаются ISIN, купленные клиентом в последние 30 дней (`EXCLUDE_RECENT_DAYS`).

---

## Поведенческий флаг (без hard-filter)

```python
outside_historical_behavior = (category_of_i not in historical_categories_of_u)
```

Флаг передаётся советнику как информация, **не** исключает инструмент из рекомендаций. Обоснование из EDA: клиенты с профилем "Conservative" реально покупают 76.2% Stock — hard-filter по riskLevel исключил бы 82% их реальных предпочтений.

---

## Fallback: popularity-based

При недостаточной истории клиент получает рекомендации на основе популярности (число уникальных покупателей) внутри его preferred_category (определяется по истории или risk profile).

**Обоснование `rank_N_justification`:**
- `"risk_profile"` — соответствует риск-профилю клиента
- `"similar_clients"` — покупали клиенты с похожим портфелем
- `"portfolio_complement"` — дополняет текущую структуру портфеля
- `"popularity"` — для fallback пути

---

## Метрики оценки

| Метрика | Описание |
|---|---|
| NDCG@3 | Основная метрика: позиционно взвешенная точность |
| Precision@3 | Доля клиентов с ≥ 1 совпадением в топ-3 |
| NDCG@3 popularity baseline | Обязательная точка сравнения |
| CF lift над baseline | (CF_NDCG − pop_NDCG) / pop_NDCG × 100% |
| HitRate@3 (fallback) | Хотя бы 1 совпадение для fallback-клиентов |
| Coverage | % уникальных ISIN, хотя бы раз попавших в рекомендации |
| Behavioral Consistency Rate | % рекомендаций в пределах исторических предпочтений |
| NDCG@3 по месяцам | Диагностика temporal degradation |

**Результаты на тестовой и validation выборках:**

| Модель | Validation NDCG@3 | Test NDCG@3 | Test lift |
|---|---|---|---|
| Popularity baseline | 0.1704 | 0.2329 | — |
| ALS | 0.1678 | 0.2528 | +8.6% |
| EASE | 0.2962 | 0.2867 | +23.1% |

**Примечание о производительности:** на validation-окне ALS незначительно уступает popularity baseline (-1.5%) — скользящее validation-окно (90 дней) короткое, ALS с k=20 менее стабилен на нём. EASE стабильно лучше на обоих split. Относительно слабый прирост ALS над popularity является структурной особенностью датасета FAR-Trans: коэффициент Gini = 0.80 (высокая концентрация), 95.6% клиентов держат только одну категорию активов. В такой среде CF-рекомендации через ALS неизбежно сходятся к популярным инструментам. EASE за счёт точных item-item весов без компрессии добавляет значимый прирост в +23.1%.

---

## Режимы запуска

```bash
# Обучение EASE + inference (еженедельный сервисный запуск):
python recommender/main.py --snapshot-date 2022-10-31 --train

# Только inference (EASE уже обучена):
python recommender/main.py --snapshot-date 2022-10-31
python recommender/main.py --snapshot-date 2022-10-31 --predict-only

# Полный grid search (ALS 135 + EASE 5 комбинаций):
python recommender/main.py --snapshot-date 2022-10-31 --tune

# Оценка (EASE + ALS baseline):
python recommender/main.py --snapshot-date 2022-10-31 --eval-only --eval-stage validation
python recommender/main.py --snapshot-date 2022-10-31 --eval-only --eval-stage test

# Обучение EASE + ALS baseline (для eval):
python recommender/main.py --snapshot-date 2022-10-31 --train --with-baseline
```

| Флаг | Действие |
|---|---|
| *(нет флагов)* / `--predict-only` | Загружает `ease_model_{DATE}.npz`, строит imat, генерирует рекомендации |
| `--train` | Строит interaction matrix, обучает EASE (grid search по λ), сохраняет модель |
| `--train --with-baseline` | То же + обучает ALS с лучшими параметрами из config |
| `--tune` | Полный ALS grid search (135 комб.) + EASE grid search, результаты → лог |
| `--eval-only` | Обучает ALS baseline + загружает EASE, сравнивает NDCG@3 |
| `--snapshot-date` | Дата в формате `YYYY-MM-DD`. По умолчанию — последняя дата в транзакциях |
| `--eval-stage` | `validation` (по умолчанию) или `test` |
| `--with-baseline` | Дополнительно обучить / использовать ALS при `--train` / `--eval-only` |
| `--no-save` | Не сохранять выходной CSV рекомендаций |

---

## Выходной файл

**`recommendations_{DATE}.csv`** — одна строка на клиента:

| Колонка | Тип | Описание |
|---|---|---|
| `customerID` | str | ID клиента |
| `segment` | str | `warm` / `dormant` |
| `propensity_score` | float | скор из propensity модели |
| `snapshot_date` | datetime | дата скоринга |
| `rec_type` | str | `ease` или `fallback` |
| `window_used` | str | `365d` / `730d` / `full` / `none` |
| `risk_profile_verified` | bool | подтверждён ли риск-профиль клиента |
| `n_history_isins` | int | число уникальных ISIN в истории |
| `rank_1_isin` | str | ISIN рекомендации №1 |
| `rank_1_category` | str | категория: `Stock` / `Bond` / `MTF` |
| `rank_1_score` | float | скор рекомендации |
| `rank_1_justification` | str | текстовое обоснование |
| `rank_1_outside_hist` | bool | за пределами исторических предпочтений |
| `rank_2_isin` ... `rank_3_*` | | аналогично для позиций 2 и 3 |

---

## Структура файлов

```
recommender/
├── main.py                     # Точка входа, CLI, оркестрация
├── src/
│   ├── config.py               # Константы, гиперпараметры, grid search пространства
│   ├── data_loader.py          # Загрузка данных, build_item_index, get_hot_customers
│   ├── matrix_builder.py       # Sparse matrix, confidence weighting, temporal decay
│   ├── train.py                # ALS, EASE, двухэтапный grid search, LOO evaluation
│   ├── evaluate.py             # NDCG@3, Precision@3, coverage, monthly breakdown
│   └── predict.py              # Inference, ступенчатое окно, diversity, fallback
├── models/
│   ├── ease_model_{DATE}.npz       # EASE W-матрица (158×158)
│   ├── ease_weights_{DATE}.json    # λ и метаданные EASE
│   ├── als_model_{DATE}.npz        # ALS user/item factors (fallback)
│   ├── als_maps_{DATE}.pkl         # user/item маппинги ALS
│   └── rec_weights_{DATE}.json     # все оптимальные параметры + история grid search
└── outputs/
    └── recommendations/
        └── recommendations_{DATE}.csv
```

---

## Зависимости

```
implicit        # ALS
numpy
scipy           # sparse матрицы
pandas
scikit-learn
matplotlib
```

Минимальная дата для корректного inference: **2019-08-01** (требуется не менее 365 дней истории транзакций). Входной файл `hot_customers_{DATE}.csv` должен быть создан Propensity Model на ту же дату.

---

## Связь с Propensity моделью

```
propensity_model/main.py --predict-only --snapshot-date DATE
    → propensity_model/outputs/predictions/hot_customers_DATE.csv

recommender/main.py --snapshot-date DATE --train
    → читает hot_customers_DATE.csv
    → recommender/outputs/recommendations/recommendations_DATE.csv
```

Дата в именах файлов должна точно совпадать. EASE переобучается при каждом запуске; ALS (fallback) можно обучать реже.
