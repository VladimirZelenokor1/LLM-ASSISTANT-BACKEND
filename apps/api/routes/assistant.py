# apps/api/routes/assistant.py
from __future__ import annotations

"""
API роуты для системы ассистента с маршрутизацией запросов.

Содержит эндпоинты для обработки пользовательских запросов через
систему маршрутизации на основе правил и LLM-классификации.

Основной эндпоинт:
- POST /assistant - обработка запросов с маршрутизацией по инструментам
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, status

from agents.router import Agent
from agents.schemas import AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["Assistant"])

# Модульная переменная для кэширования экземпляра агента
_agent_instance: Optional[Agent] = None


def get_agent() -> Agent:
    """Возвращает синглтон экземпляр агента с ленивой инициализацией.

    Returns:
        Инициализированный экземпляр агента-роутера

    Raises:
        RuntimeError: При ошибках инициализации агента
    """
    global _agent_instance

    if _agent_instance is None:
        try:
            _agent_instance = Agent("configs/agent.yaml")
            logger.info("Agent instance initialized successfully")
        except Exception as error:
            logger.error("Failed to initialize agent: %s", error)
            raise RuntimeError(f"Agent initialization failed: {error}") from error

    return _agent_instance


@router.post(
    "",
    response_model=AgentResponse,
    summary="Обработать запрос ассистента",
    description="""
    Обрабатывает пользовательский запрос через систему маршрутизации,
    определяя наиболее подходящие инструменты (документация, SQL, веб-поиск)
    и возвращая сводный ответ.
    """,
    response_description="Ответ агента с маршрутизацией и результатами инструментов"
)
def process_assistant_request(request: AgentRequest) -> AgentResponse:
    """Обрабатывает запрос к ассистенту с маршрутизацией по инструментам.

    Args:
        request: Запрос с текстом вопроса и опциональными подсказками маршрутизации

    Returns:
        Ответ агента с финальным ответом и деталями маршрутизации

    Raises:
        HTTPException: При внутренних ошибках обработки запроса
    """
    try:
        agent = get_agent()
        response = agent.route_query(request)

        logger.info(
            "Request processed successfully. Route: %s, Tools used: %d, Latency: %dms",
            response.route,
            len(response.tool_results),
            response.latency_ms
        )

        return response

    except Exception as error:
        logger.exception(
            "Error processing assistant request: %s. Query: '%s'",
            error, request.query
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(error)}"
        ) from error
