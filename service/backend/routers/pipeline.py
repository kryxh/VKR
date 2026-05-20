from fastapi import APIRouter

from schemas import PipelineRunRequest, PipelineRunResponse

router = APIRouter()


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(body: PipelineRunRequest):
    date_info = f" (дата: {body.snapshot_date})" if body.snapshot_date else ""
    return PipelineRunResponse(
        status="not_implemented",
        message=(
            f"Запуск пайплайна{date_info} через API не реализован в демо-режиме. "
            "Запустите пайплайн вручную через CLI и перезапустите бэкенд: "
            "'docker compose restart backend'. "
            "Подробнее см. README.md раздел 'Запуск пайплайна'."
        ),
    )
