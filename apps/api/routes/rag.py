# apps/api/routes/rag.py
from __future__ import annotations

"""
API роуты для RAG (документация → ответ с цитатами).

Основной эндпоинт:
- POST /rag/qa — отвечает на вопрос по индексу документации (Qdrant + LLM)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

# Используем готовый инструмент из вашего кода
# (см. tools/rag/qa/qa_tool.py)
from tools.rag.qa.qa_tool import answer_from_docs, get_qa_instance  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    """Запрос к RAG."""
    query: str = Field(..., min_length=1, max_length=2000,
                       description="Вопрос пользователя")
    # Необязательные параметры:
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Фильтры для поиска (например, {'lang':'en'} или {'product':'transformers'})"
    )
    # Лёгкие оверрайды ретривера (совпадают с именами в qa_tool)
    top_k_parents: Optional[int] = Field(default=None, ge=1)
    top_k_children_per_parent: Optional[int] = Field(default=None, ge=1)
    min_score: Optional[float] = Field(default=None, ge=0, le=1)
    alpha_parent_child: Optional[float] = Field(default=None, ge=0, le=1)

    # Оверрайды генерации
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_output_tokens: Optional[int] = Field(default=None, ge=16, le=8192)


class Citation(BaseModel):
    """Мини-модель для ссылки/цитаты (гибко, т.к. структура может меняться)."""
    source: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class RAGResponse(BaseModel):
    """Ответ RAG."""
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    usage: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    timestamp: Optional[str] = None
    retrieved_chunks: Optional[int] = None
    query: Optional[str] = None
    provider: Optional[str] = None


@router.post(
    "/qa",
    response_model=RAGResponse,
    summary="Ответить на вопрос по документации (RAG)",
    description="""
Ищет релевантные фрагменты в Qdrant и генерирует ответ с цитатами через LLM.
Можно передавать лёгкие оверрайды ретривера и генерации (top_k, temperature и т.п.).
""",
    response_description="Ответ, цитаты и служебные поля (модель, usage, и т.п.)"
)
def rag_qa(request: RAGRequest) -> RAGResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )

    try:
        # Явно инициализируем QA (лениво, один раз)
        get_qa_instance()

        kwargs: Dict[str, Any] = {}
        # Пробрасываем только те оверрайды, которые заданы
        for key in [
            "top_k_parents",
            "top_k_children_per_parent",
            "min_score",
            "alpha_parent_child",
            "temperature",
            "max_output_tokens",
        ]:
            val = getattr(request, key)
            if val is not None:
                kwargs[key] = val

        result: Dict[str, Any] = answer_from_docs(
            query=request.query,
            filters=request.filters or {},
            **kwargs
        )
        # result уже в нужной структуре — просто валидируем в pydantic-модель
        return RAGResponse(**result)

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("RAG internal error: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during RAG processing"
        ) from error
