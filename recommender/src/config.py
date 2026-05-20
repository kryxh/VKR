from pathlib import Path


# Пути
ROOT_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT_DIR.parent / "data"
SRC_DIR    = ROOT_DIR / "src"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR  = ROOT_DIR / "models"

PROPENSITY_PREDICT_DIR = ROOT_DIR.parent / "propensity_model" / "outputs" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "recommendations").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "evaluation").mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    "transactions": DATA_DIR / "transactions.csv",
    "customers":    DATA_DIR / "customer_information.csv",
    "assets":       DATA_DIR / "asset_information.csv",
    "prices":       DATA_DIR / "close_prices.csv",
    "markets":      DATA_DIR / "markets.csv",
}


# Временной сплит
TEST_MONTHS  = 14
VALID_MONTHS = 9


# Параметры user-item матрицы
# Окно истории для построения ALS-матрицы (до snapshot_date)
# EDA: density при 365d = 1.68% vs 0.92% для всей истории
# repeat-rate: 42.1% (365d) vs 47.5% (all) — потеря 5.4 п.п., выигрыш 82% плотности
INTERACTION_WINDOW_DAYS = 365

# Расширенное окно для клиентов без истории в основном окне
# (reactivation-клиенты с последней покупкой до 2020 года)
EXTENDED_WINDOW_DAYS = 730

# Минимальное число уникальных ISINs у клиента для ALS-пути (не fallback)
# EDA: warm-клиенты (≤60d) с ≥3 ISINs = 77.1% → ALS — основной путь
MIN_TX_REC = 3

# Минимальное число уникальных покупателей для ISIN, чтобы включить в ALS
# EDA: min_support=5 удаляет 20% ISINs, но теряет только 0.2% транзакций
MIN_ITEM_SUPPORT = 5

# Окно исключения недавно купленных ISIN из кандидатов
# EDA: 62.3% повторных покупок одного ISIN происходят в ≤30 дней
# Медиана интервала: 16 дней → без фильтра ALS рекомендует уже купленное
EXCLUDE_RECENT_DAYS = 30


# Confidence weighting (Hu et al., 2008 + temporal decay)
# c_ui = 1 + CONFIDENCE_ALPHA * log(1 + count_ui) * exp(-DECAY_LAMBDA * days_ago/365)
#
# CONFIDENCE_ALPHA: масштаб веса повторных покупок
# EDA: 73.8% пар (клиент, ISIN) куплены ровно 1 раз → умеренный вес повторов
# Hu et al. (2008): alpha=40 для бинарных данных; для log-варианта — меньше
#
# DECAY_LAMBDA: скорость затухания доверия к старым транзакциям
# Koren (2010, TimeSVD++): temporal decay улучшает качество в нестационарных данных
# При DECAY_LAMBDA=0 → классический Hu et al. без decay


# Grid search — гиперпараметры ALS
# Этап 1: joint grid по k, λ, CONFIDENCE_ALPHA, DECAY_LAMBDA
# Метрика: NDCG@3 на LOO validation (2,816 eligible клиентов)

ALS_FACTORS_GRID        = [12, 16, 20, 24, 32]    # k: число латентных факторов
                                                  # k_max теор. ≈ sqrt(223)*0.7 ≈ 10.5
ALS_REGULARIZATION_GRID = [0.01, 0.05, 0.1]     # λ: L2-регуляризация
CONFIDENCE_ALPHA_GRID   = [40, 60, 80]           # масштаб confidence
DECAY_LAMBDA_GRID       = [0.5, 1.0, 2.0]        # temporal decay
ALS_ITERATIONS          = 20                     # число итераций ALS (фиксировано)


# Лучшие параметры из --tune (обновлять вручную после каждого --tune)
ALS_BEST_FACTORS          = 20
ALS_BEST_REGULARIZATION   = 0.1
ALS_BEST_CONFIDENCE_ALPHA = 80
ALS_BEST_DECAY_LAMBDA     = 1.0
ALS_BEST_FUSION_ALPHA     = 1.0
EASE_BEST_REGULARIZATION  = 500



# Grid search — score fusion
# Этап 2: после выбора лучших ALS-параметров
# score(u,i) = FUSION_ALPHA * s_item + (1 - FUSION_ALPHA) * s_user
# Обоснование data-driven весов: Burke (2002), Das et al. (2007)
FUSION_ALPHA_GRID = [round(x * 0.1, 1) for x in range(0, 11)]  # [0.0, 0.1, ..., 1.0]


# EASE (Embarrassingly Shallow Autoencoders, Steck 2019)
# Закрытое решение: W = I - P × diag(1/diag(P)), P = (X^T X + λI)^{-1}
# Для 154 items матрица 154×154 — вычисляется мгновенно.
# λ — единственный гиперпараметр (нет k, нет итераций).
EASE_REGULARIZATION_GRID = [50, 200, 500, 1000, 2000]


# LOO validation
# Минимум уникальных ISINs в train-истории для LOO evaluation
# EDA: 2,816 eligible клиентов при MIN_TX_EVAL=3
MIN_TX_EVAL = 3


# Portfolio diversity boost
# Применяется ТОЛЬКО при n_cat_in_history >= 2 (клиент покупал ≥2 категорий)
# EDA: 95.6% клиентов имеют single-category портфель → boost для них нерелевантен
PORTFOLIO_BOOST_GAMMA        = 1.2
PORTFOLIO_BOOST_MIN_CATS     = 2


# Fallback: маппинг riskLevel → preferred_category (по данным EDA)
# Используется при пустой истории клиента
# EDA: Conservative: 23.3% MTF (наибольший non-Stock сигнал)
#      Income/Balanced/Aggressive: 91-95% Stock
RISK_FALLBACK_CATEGORY = {
    "Conservative": "MTF",
    "Income":       "Stock",
    "Balanced":     "Stock",
    "Aggressive":   "Stock",
    "Not_Available": "Stock",
}


# Топ-3 рекомендации
TOP_K = 3


# Имена колонок в выходном файле
OUTPUT_COLS = [
    "customerID",
    "segment",
    "propensity_score",
    "snapshot_date",
    "rec_type",
    "window_used",
    "risk_profile_verified",
    "n_history_isins",
    "rank_1_isin", "rank_1_category", "rank_1_score",
    "rank_1_justification", "rank_1_outside_hist",
    "rank_2_isin", "rank_2_category", "rank_2_score",
    "rank_2_justification", "rank_2_outside_hist",
    "rank_3_isin", "rank_3_category", "rank_3_score",
    "rank_3_justification", "rank_3_outside_hist",
]
