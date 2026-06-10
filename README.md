# Advisory Dashboard — Система рекомендаций для финансового советника

Дашборд на основе предсказаний ML модели для приоритизации клиентов и рекомендации инвестиционных инструментов.

**Стек:** FastAPI, PostgreSQL 15, Streamlit, Docker Compose  
**ML модели:** CatBoost (propensity scoring), EASE (recommender)  
**Данные:** FAR-Trans (European financial institution)

---

## Структура проекта

```
project_root/
├── data/                     # FAR-Trans CSV файлы (не в репо — слишком большие)
├── propensity_model/         # скоринг клиентов: кто купит в ближайшие 30 дней
│   ├── src/                  # pipeline: feature engineering, train, evaluate, predict
│   ├── models/               # обученные модели CatBoost + калибраторы
│   ├── outputs/predictions/  # hot_customers_YYYYMMDD.csv (нужен сервису)
│   └── main.py               # точка входа
├── recommender/              # рекомендации активов для клиента
│   ├── src/                  # EASE model: train, predict
│   ├── models/               # обученные веса
│   └── main.py               # точка входа
└── service/                  # веб-сервис для советника
    ├── backend/              # FastAPI + PostgreSQL
    ├── frontend/             # Streamlit dashboard
    └── docker-compose.yml
```

---

## Быстрый старт (Docker Compose)

**Требования:** Docker 24.0 и выше, Docker Compose v2, запущенный ML пайплайн

```bash
# 1. Запустить ML пайплайн (если нет готовых predictions)
python propensity_model/main.py --predict-only --snapshot-date 2022-11-30
python recommender/main.py --snapshot-date 2022-11-30 --train

# 2. Настроить .env
cp service/.env.example service/.env
# Установить SNAPSHOT_DATE=2022-11-30 (должна совпадать с датой файлов)

# 3. Запустить сервис
cd service
docker compose up --build

# 4. Открыть дашборд
# Streamlit UI: http://localhost:8501
# FastAPI docs: http://localhost:8000/docs
```

Загрузка данных при первом старте занимает ~5-10 минут. Следить за прогрессом:
```bash
docker compose logs backend -f
# Ищите: ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО
```

---

## Архитектура

```
┌──────────────┐   ┌──────────────┐   ┌─────────┐
│  PostgreSQL  │◄──│   FastAPI    │◄──│Streamlit│
│    :5432     │   │    :8000     │   │  :8501  │
└──────────────┘   └──────────────┘   └─────────┘
```

- **Propensity model** — inference на фиксированной обученной модели CatBoost
- **Recommender** — retrain EASE при каждом новом snapshot

