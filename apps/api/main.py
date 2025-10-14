# apps/api/main.py
from __future__ import annotations

"""
Точка входа FastAPI приложения системы LLM-ассистента.

Основные responsibilities:
- Инициализация и управление жизненным циклом LLM-клиента
- Конфигурация FastAPI приложения с middleware
- Подключение API роутеров для различных функциональностей

Модуль обеспечивает корректную инициализацию и очистку ресурсов
при запуске и остановке приложения.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.settings import get_llm_settings
from core.llm_provider import get_llm, reset_llm
from apps.api.routes.assistant import router as assistant_router
from apps.api.routes.sql_qa import router as sql_qa_router
from apps.api.routes.rag import router as rag_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Управляет жизненным циклом приложения и ресурсов LLM.

    Args:
        app: Экземпляр FastAPI приложения

    Yields:
        None: Контекст выполняется до завершения работы приложения

    Raises:
        RuntimeError: При критических ошибках инициализации LLM
    """
    logger.info("Initializing LLM application...")

    try:
        # Предварительная загрузка настроек и клиента LLM
        settings = get_llm_settings()
        llm_client = get_llm()

        logger.info(
            "LLM client initialized with provider: %s, model: %s",
            settings.provider, settings.model
        )

        yield

    except Exception as error:
        logger.critical("Failed to initialize LLM application: %s", error)
        raise RuntimeError(f"LLM initialization failed: {error}") from error

    finally:
        logger.info("Shutting down LLM application...")
        try:
            reset_llm()
            logger.info("LLM client successfully reset")
        except Exception as error:
            logger.warning("Failed to reset LLM client: %s", error)


def create_app() -> FastAPI:
    """Создает и конфигурирует экземпляр FastAPI приложения.

    Returns:
        Настроенный экземпляр FastAPI с подключенными роутерами и middleware
    """
    app = FastAPI(
        title="LLM Assistant API",
        description="Backend система для маршрутизации запросов и SQL-QA функциональности",
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Конфигурация CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # В продакшене следует ограничить
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Подключение API роутеров
    app.include_router(assistant_router)
    app.include_router(sql_qa_router)
    app.include_router(rag_router)

    logger.info("FastAPI application configured successfully")
    return app

app = create_app()
