from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.session import check_connection, create_tables
from routers import clients, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not check_connection():
        raise RuntimeError(
            "Не удалось подключиться к PostgreSQL. "
            "Проверьте DATABASE_URL и что контейнер db запущен."
        )
    create_tables()
    print("[App] Подключение к БД: OK")
    yield
    print("[App] Остановка приложения")



app = FastAPI(
    title="Investment Bank Advisory Service",
    description=(
        "ML сервис для финансовых советников: "
        "приоритизация клиентов + рекомендации активов"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(clients.router, prefix="/api", tags=["Clients"])
app.include_router(pipeline.router, prefix="/api", tags=["Pipeline"])



@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Investment Bank Advisory Service",
        "docs": "/docs",
        "health": "/health",
    }
