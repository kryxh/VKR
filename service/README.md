# Advisory Dashboard — Сервис финансового советника

ML-powered дашборд для приоритизации клиентов и рекомендации инвестиционных инструментов.

**Стек:** FastAPI · PostgreSQL 15 · Streamlit · Docker Compose  
**ML модели:** CatBoost (propensity) · EASE (recommender)  
**Данные:** FAR-Trans

---

## Структура проекта

```
project_root/
├── data/                              ← FAR-Trans CSV файлы (общая папка)
├── propensity_model/                  ← ML скоринг клиентов
│   └── outputs/predictions/
│       ├── all_scores_20221130.csv    ← все клиенты
│       └── hot_customers_20221130.csv ← топ-20% для звонка
├── recommender/                       ← ML рекомендации
│   └── outputs/recommendations/
│       └── recommendations_20221130.csv
└── service/                           ← этот сервис
    ├── backend/
    ├── frontend/
    ├── docker-compose.yml
    └── .env
```

---

## Быстрый старт (Docker Compose)

### Шаг 0. Предварительные требования

- Docker ≥ 24.0 + Docker Compose v2
- Запущенный ML пайплайн (результаты в `outputs/`)
- FAR-Trans данные в `project_root/data/`

Проверить:
```bash
docker --version       # Docker version 24+
docker compose version # Docker Compose version v2+
```

---

### Шаг 1. Запустить ML пайплайн (если ещё не запускался)

```bash
cd project_root

# 1a. Propensity: inference для выбранной даты
python propensity_model/main.py --predict-only --snapshot-date 2022-11-30

# Проверить что файлы созданы:
ls propensity_model/outputs/predictions/
# → all_scores_20221130.csv
# → hot_customers_20221130.csv

# 1b. Recommender: обучение EASE + inference
python recommender/main.py --snapshot-date 2022-11-30 --train

# Проверить:
ls recommender/outputs/recommendations/
# → recommendations_20221130.csv
```

> ⚠️ Дата в имени файла должна точно совпадать с `SNAPSHOT_DATE` в `.env`.
> Формат имени файла: `hot_customers_YYYYMMDD.csv` (без дефисов).

---

### Шаг 2. Настроить переменные окружения

```bash
cd service

# Проверить/отредактировать .env
cat .env
```

Содержимое `.env`:
```
DB_USER=advisory_user
DB_PASSWORD=secure_password
SNAPSHOT_DATE=2022-11-30   ← должно совпадать с датой файлов из Шага 1
```

---

### Шаг 3. Запустить сервис

```bash
cd service

# Сборка образов и запуск (первый раз ~3-5 мин из-за сборки)
docker compose up --build

# Или в фоне:
docker compose up --build -d
```

**Что происходит при старте:**
1. `db` — запускается PostgreSQL 15, ждёт healthcheck
2. `backend` — собирается образ → запускается `python db/loader.py` (загрузка CSV в БД, ~5-10 мин: транзакций 350k строк) → запускается `uvicorn`
3. `frontend` — запускается Streamlit, ждёт пока backend healthy

**Логи загрузки данных:**
```bash
docker compose logs backend -f
```
Ищите строку: `ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО`

---

### Шаг 4. Открыть дашборд

| Сервис | URL |
|---|---|
| **Streamlit UI** | http://localhost:8501 |
| **FastAPI Swagger** | http://localhost:8000/docs |
| **FastAPI ReDoc** | http://localhost:8000/redoc |
| **Health check** | http://localhost:8000/health |

---

### Шаг 5. Остановка

```bash
# Остановить контейнеры (данные в БД сохранятся)
docker compose down

# Остановить и удалить данные БД (полный сброс)
docker compose down -v
```

---

## Локальный запуск без Docker (разработка)

Удобно для быстрой итерации при разработке фронтенда или API.

### PostgreSQL локально

```bash
# Вариант 1: Docker только для БД
docker run -d \
  --name advisory_pg \
  -e POSTGRES_DB=advisory_db \
  -e POSTGRES_USER=advisory_user \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  postgres:15
```

### Backend

```bash
cd service/backend

pip install -r requirements.txt

# Установить переменные окружения
export DATABASE_URL="postgresql://advisory_user:secure_password@localhost:5432/advisory_db"
export SNAPSHOT_DATE="2022-11-30"

# Загрузить данные
python db/loader.py

# Запустить API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd service/frontend

pip install -r requirements.txt

export API_URL="http://localhost:8000"

streamlit run app.py
```

---

## Обновление данных (новый snapshot)

Когда нужно запустить пайплайн на новую дату:

```bash
# 1. Запустить ML пайплайн с новой датой
python propensity_model/main.py --predict-only --snapshot-date 2023-01-15
python recommender/main.py --snapshot-date 2023-01-15 --train

# Проверить что файлы созданы:
ls propensity_model/outputs/predictions/
# → all_scores_20230115.csv
# → hot_customers_20230115.csv

ls recommender/outputs/recommendations/
# → recommendations_20230115.csv

# 2. Обновить SNAPSHOT_DATE в .env
# Открыть service/.env и изменить строку:
SNAPSHOT_DATE=2023-01-15

# 3. Полный перезапуск сервиса с пересозданием БД
cd service
docker compose down -v          # останавливаем контейнеры и удаляем данные БД
docker compose up --build       # поднимаем заново — loader загрузит новые файлы

# 4. Следить за загрузкой (~5-10 мин)
docker compose logs backend -f
# Ищите строку: ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО
```

> ⚠️ `docker compose down -v` удаляет все данные PostgreSQL — это необходимо,
> так как loader использует upsert и должен загрузить новый snapshot с чистого листа.
> Данные всегда восстанавливаются из CSV при следующем запуске.

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                     │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────┐  │
│  │  PostgreSQL  │   │   FastAPI    │   │Streamlit│  │
│  │     :5432    │◄──│   :8000      │◄──│  :8501  │  │
│  └──────────────┘   └──────────────┘   └─────────┘  │
│         ▲                  ▲                         │
│         │                  │                         │
│   postgres_data       CSV volumes                    │
│   (persistent)     (read-only mounts)                │
└─────────────────────────────────────────────────────┘
```

**Жизненный цикл моделей:**
- **Propensity (CatBoost)** — фиксированная обученная модель, только inference при каждом новом snapshot. Пересчёт признаков + скоринг на новой дате.
- **Recommender (EASE)** — retrain при каждом новом snapshot, т.к. interaction matrix обновляется с новыми транзакциями.

---

## API эндпоинты

| Метод | Путь | Описание |
|---|---|---|
| GET | `/api/clients` | Список hot_customers с фильтрами и пагинацией |
| GET | `/api/clients/stats` | Агрегаты для фронтенда (макс. дней, итого клиентов) |
| GET | `/api/clients/{id}` | Карточка клиента |
| GET | `/api/advisors` | Список советников для фильтра |
| GET | `/api/snapshot-date` | Текущая дата ML пайплайна |
| POST | `/api/pipeline/run` | Заглушка (запуск через CLI) |
| GET | `/health` | Health check |

### Фильтры для `/api/clients`

| Параметр | Тип | Описание |
|---|---|---|
| `advisor_name` | str | Имя советника для фильтрации клиентов |
| `search_id` | str | Частичный поиск по customer_id |
| `segment` | str | `warm` или `dormant` |
| `score_min` | float | Минимальный propensity_score |
| `score_max` | float | Максимальный propensity_score |
| `days_min` | int | Минимум дней без покупки |
| `days_max` | int | Максимум дней без покупки |
| `page` | int | Страница (дефолт: 1) |
| `page_size` | int | Размер страницы (дефолт: 50, макс: 200) |


---

## Деплой на Render.com

1. Запушить репозиторий на GitHub
2. В Render создать **PostgreSQL** managed database → скопировать `DATABASE_URL`
3. Создать **Web Service** для backend:
   - Root Directory: `service/backend`
   - Environment: Docker
   - ENV: `DATABASE_URL=...`, `SNAPSHOT_DATE=2022-11-30`
   - **Важно:** данные CSV должны быть включены в образ или загружены заранее
4. Создать **Web Service** для frontend:
   - Root Directory: `service/frontend`
   - ENV: `API_URL=https://your-backend.onrender.com`

> ⚠️ **Ограничение Render бесплатного тарифа:** volume mounts для CSV файлов не поддерживаются. Для демо — либо включить CSV в Docker образ, либо предзагрузить данные в БД локально и использовать `pg_dump/restore`.

---

## Возможные проблемы

| Проблема | Причина | Решение |
|---|---|---|
| `FileNotFoundError: hot_customers_*.csv` | ML пайплайн не запущен | Шаг 1 выше |
| Backend не стартует, ошибка БД | PostgreSQL ещё не готов | `docker compose restart backend` |
| Streamlit: "Не удалось подключиться к API" | Backend ещё грузит данные | Подождать 2-5 мин, обновить страницу |
| Пустой список клиентов | Неверная `SNAPSHOT_DATE` в `.env` | Проверить что дата совпадает с файлами |
| Loader падает на transactions | Нет таблицы customers/assets | Порядок загрузки фиксирован, перезапустить |

---

## Что не входит в scope сервиса

- ❌ Реальный запуск ML пайплайна через API (только pre-computed результаты)
- ❌ История нескольких snapshot-дат в UI (только последний snapshot)  
- ❌ Авторизация / аутентификация — фильтр по советнику есть, но без логина (выбор вручную)
- ❌ Realtime обновление — только batch при старте контейнера

---

*Сервис разработан как часть выпускной работы. Версия 1.0.*
