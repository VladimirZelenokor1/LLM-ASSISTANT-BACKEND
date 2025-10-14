# apps/api/routes/sql_qa.py
from __future__ import annotations

"""
API роуты для SQL Question-Answering системы.

Предоставляет эндпоинты для преобразования естественно-языковых запросов
в SQL команды и выполнения их в базе данных с системой защитных ограничений.

Основной эндпоинт:
- POST /sql-qa - преобразование вопросов в SQL и выполнение запросов
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from tools.sql.query_tool import answer_from_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sql", tags=["SQL QA"])


class SQLQARequest(BaseModel):
    """Модель запроса для SQL Question-Answering системы.

    Attributes:
        question: Вопрос на естественном языке для преобразования в SQL
        dry_run: Флаг режима только генерации SQL без выполнения
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Вопрос на естественном языке для преобразования в SQL запрос",
    )

    dry_run: bool = Field(
        default=False,
        description="Если True - только сгенерировать SQL без выполнения в БД"
    )


class SQLQAResponse(BaseModel):
    """Модель ответа от SQL Question-Answering системы.

    Attributes:
        answer: Текстовое объяснение результата запроса
        data: Сырые данные результата выполнения SQL запроса
        sql: Сгенерированный SQL запрос
        guarded: Флаг срабатывания защитных ограничений
    """

    answer: str = Field(
        ...,
        description="Человекочитаемое объяснение результата SQL запроса"
    )

    data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Результаты выполнения SQL запроса в виде списка словарей"
    )

    sql: str = Field(
        ...,
        description="Сгенерированный SQL запрос для выполнения в базе данных"
    )

    guarded: bool = Field(
        default=False,
        description="Флаг указывающий на срабатывание защитных ограничений"
    )


@router.post(
    "/qa",
    response_model=SQLQAResponse,
    summary="Выполнить SQL QA запрос",
    description="""
    Преобразует вопрос на естественном языке в SQL запрос, 
    проверяет его через систему защитных ограничений и при необходимости
    выполняет в базе данных, возвращая результат.
    """,
    response_description="Результат выполнения SQL QA запроса"
)
def execute_sql_qa_query(request: SQLQARequest) -> SQLQAResponse:
    """Выполняет SQL Question-Answering запрос с преобразованием естественного языка.

    Args:
        request: Запрос с вопросом и флагом dry_run

    Returns:
        Ответ с результатами выполнения SQL запроса

    Raises:
        HTTPException: При ошибках валидации или выполнения запроса
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty"
        )

    try:
        logger.info(
            "Processing SQL QA request. Question: '%s', Dry run: %s",
            request.question, request.dry_run
        )

        result = answer_from_db(request.question, dry_run=request.dry_run)

        logger.info(
            "SQL QA request processed successfully. "
            "Query length: %d, Result rows: %d, Guarded: %s",
            len(result.sql), len(result.data), result.guarded
        )

        return SQLQAResponse(
            answer=result.answer,
            data=result.data,
            sql=result.sql,
            guarded=result.guarded,
        )

    except ValueError as error:
        # Ошибки валидации и guardrails
        logger.warning("SQL QA validation error: %s", error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        ) from error

    except Exception as error:
        # Внутренние ошибки сервера
        logger.error("SQL QA internal error: %s", error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during SQL query processing"
        ) from error
