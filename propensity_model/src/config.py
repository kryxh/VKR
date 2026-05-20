from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR.parent / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR  = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    "transactions": DATA_DIR / "transactions.csv",
    "customers": DATA_DIR / "customer_information.csv",
    "assets": DATA_DIR / "asset_information.csv",
    "prices": DATA_DIR / "close_prices.csv",
    "markets": DATA_DIR / "markets.csv",
}


# Окно предсказания: label = 1 если покупка в интервале (T, T + HORIZON_DAYS]
HORIZON_DAYS = 30

# Gap между признаками и целевым окном для timing-модели
# Убирает тривиальный прокси "купил вчера → купит завтра"
PREDICTION_GAP_DAYS = 7

# Граница между warm и dormant сегментами
DORMANCY_THRESHOLD_DAYS = 90

# Окно предсказания для reactivation модели — шире, спящие возвращаются медленнее
REACTIVATION_HORIZON_DAYS = 30

# Персональный порог дормантности: клиент dormant если пауза > multiplier × median_interval
# Fallback для клиентов без cadence-данных (<2 покупок) — фиксированный MIN_DORMANCY_DAYS
DORMANCY_MULTIPLIER = 2.0
DORMANCY_MIN_DAYS = 90   # минимальный персональный порог

# клиент должен иметь ≥1 транзакцию за ACTIVE_WINDOW_DAYS до даты T
# иначе исключаем (неактивные не подходят для рекомендаций)
ACTIVE_WINDOW_DAYS = 365

# клиенты с возрастом < порога считаются cold-start
COLD_START_THRESHOLD_DAYS = 60

# минимальное кол-во транзакций для reactivation датасета
MIN_TX_REACT = 10

# time-based split (без рандома)
# train: до 2021-01, valid: до 2021-10, далее test
# границы выбраны из EDA (достаточно позитивов, учитываем рост 2020)
#   train: [data_start, TRAIN_END)
#   valid: [TRAIN_END, VALID_END)
#   test: [VALID_END, data_end]
TEST_MONTHS  = 14
VALID_MONTHS = 9

# первая дата снапшота после периода с синтетическими транзакциями
# RFM и так очищен от синтетических транзакций, берем дату с запасмо чтобы исключить влияние на ACTIVE_WINDOW = 180 дней
SNAPSHOT_START = "2018-08-01"

# Период рыночной аномалии (COVID)
# Используется для флага в cadence-признаках
ANOMALY_PERIODS = [("2020-02-01", "2020-12-31")]

# частота снапшотов
SNAPSHOT_FREQ = "2W"


# TIMING модель — warm клиенты
TIMING_CAT_FEATURES = [
    "customerType",
    "riskLevel",
    "investmentCapacity",
]

TIMING_NUMERIC_FEATURES = [
    # Профиль
    "is_capacity_missing",

    # RFM
    "days_since_last_buy",
    "num_buys_30d",
    "num_buys_90d",
    "sum_value_30d",
    "avg_ticket_30d",
    "log1p_avg_ticket_30d",
    "num_sells_30d",

    # Динамика
    "activity_trend",
    "month",
    "is_new_customer",
    "days_since_last_tx",
    "customer_tenure_days",

    # Портфель
    "num_unique_assets",
    "share_stocks",
    "share_bonds",
    "share_funds",

    # Рыночный контекст (новое)
    "market_return_7d",
    "market_return_30d",
    "market_volatility_30d",
    "market_drawdown",

    # Взаимодействия профиль × recency (новое)
    # Снижают доминирование статичных профильных признаков
    "aggressive_x_recency",
    "stocks_x_market_correction",
    "snapshot_in_anomaly",
    "pause_includes_anomaly",
    "history_includes_anomaly",

]

TIMING_ALL_FEATURES = TIMING_CAT_FEATURES + TIMING_NUMERIC_FEATURES

# Признаки reactivation-модели (dormant клиенты, >60 дней без покупки)
# Нет признаков текущей активности — они равны нулю для спящих
REACTIVATION_CAT_FEATURES = [
    "customerType",
    "riskLevel",
]

REACTIVATION_NUMERIC_FEATURES = [
    # Профиль качества
    "customer_tenure_days",

    # Портфель (контекст)
    "share_stocks",
    "share_bonds",
    "share_funds",
    "num_unique_assets",

    "has_cadence_data",
    "median_buy_interval",
    "std_buy_interval",
    "max_historical_gap",
    "total_buys_lifetime",
    "current_pause_days",
    "personal_dormancy_threshold",
    "buys_last_year",
    "tenure_buy_days"
    "has_returned_after_90d_gap",
    "long_term_activity_trend",

    # Cadence derived (NaN для клиентов без gap-данных — CatBoost обработает)
    "pause_zscore",
    "survived_similar_pause",
    "pause_near_personal_max",
    "p_alive",

    # Сезонность
    "bought_same_month_last_year",
    "bought_same_quarter_last_year",
    "num_buys_same_month_hist",

    # Рыночный контекст
    "market_return_7d",
    "market_return_30d",
    "market_volatility_30d",
    "market_drawdown",

    # Взаимодействия личный паттерн × рынок
    "value_buyer_x_correction",
    "snapshot_in_anomaly",
    "pause_includes_anomaly",
    "history_includes_anomaly",
]

REACTIVATION_ALL_FEATURES = REACTIVATION_CAT_FEATURES + REACTIVATION_NUMERIC_FEATURES

TARGET_COL = "label"
ID_COLS    = ["customerID", "snapshot_date"]

# Model hyperparameters (подобрано с grid search)
CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
    "random_seed": 42,
    "verbose": 100,
}

# Для reactivation модели
REACTIVATION_CATBOOST_PARAMS = {
    "iterations": 200,
    "learning_rate": 0.03,
    "depth": 6,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "early_stopping_rounds": 40,
    "random_seed": 42,
    "verbose": 100,
}

LOGREG_PARAMS = {
    "C": 0.01,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
}

LOGREG_REACT_PARAMS = {
    "C": 0.01,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "solver": "lbfgs",
}


RISK_LEVEL_CANONICAL = {
    "Conservative":           "Conservative",
    "Income":                 "Income",
    "Balanced":               "Balanced",
    "Aggressive":             "Aggressive",
    "Predicted_Conservative": "Conservative",
    "Predicted_Income":       "Income",
    "Predicted_Balanced":     "Balanced",
    "Predicted_Aggressive":   "Aggressive",
    "Not_Available":          "Not_Available",
}

INVESTMENT_CAPACITY_CANONICAL = {
    "CAP_LT_30K":              "CAP_LT_30K",
    "CAP_30K_80K":             "CAP_30K_80K",
    "CAP_80K_300K":            "CAP_80K_300K",
    "CAP_GT300K":              "CAP_GT300K",
    "Predicted_CAP_LT_30K":    "CAP_LT_30K",
    "Predicted_CAP_30K_80K":   "CAP_30K_80K",
    "Predicted_CAP_80K_300K":  "CAP_80K_300K",
    "Predicted_GT300K":        "CAP_GT300K",
    "Not_Available":           "Not_Available",
}
