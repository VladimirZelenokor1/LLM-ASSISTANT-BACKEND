from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import yaml
import os

from .query_tool import answer_from_db, QAResult

logger = logging.getLogger(__name__)


class SQLQATool:
    """
    SQL QA инструмент для выполнения запросов к организационным данным.

    Обеспечивает естественно-языковой интерфейс к базе данных команды и модулей.
    """

    # Индикаторы отсутствия данных в ответах
    NEGATIVE_INDICATORS = [
        "не найд", "нет данных", "не существует", "отсутств",
        "no data", "not found", "does not exist", "empty",
        "unknown", "ничего", "nothing"
    ]

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Инициализация SQL QA инструмента.

        Args:
            config_path: Путь к YAML файлу конфигурации
            config_dict: Словарь с конфигурацией (имеет приоритет над config_path)
        """
        self.config = self._load_config(config_path, config_dict)
        logger.info("SQL QA tool initialized")

    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Загружает конфигурацию из файла или словаря.

        Args:
            config_path: Путь к файлу конфигурации
            config_dict: Готовый словарь конфигурации

        Returns:
            Словарь с конфигурацией
        """
        if config_dict:
            return config_dict

        config_path = config_path or os.getenv('SQL_CONFIG', 'configs/sql.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file) or {}
                logger.info("Configuration loaded from %s", config_path)
                return config
        except FileNotFoundError:
            logger.warning("SQL config file not found at %s, using defaults", config_path)
            return {}
        except Exception as e:
            logger.error("Error loading config from %s: %s", config_path, e)
            return {}

    def answer_from_sql(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Основной метод для получения ответов из SQL базы данных.

        Args:
            query: Вопрос на естественном языке
            filters: Дополнительные фильтры (резерв для будущего использования)
            **kwargs: Дополнительные параметры

        Returns:
            Словарь с ответом, данными и метаинформацией
        """
        logger.info("Processing SQL query: %s", query)

        try:
            # Вызов существующей SQL QA функции
            # filters пока не используется в answer_from_db, но принимаем для совместимости
            result: QAResult = answer_from_db(question=query, dry_run=False)

            # Конвертация в формат, совместимый с RAG инструментом
            response = {
                "answer": result.answer,
                "confidence": self._calculate_confidence(result),
                "citations": self._format_citations(result.data),
                "data": result.data,
                "sql": result.sql,
                "guarded": result.guarded,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "filters": filters or {},
                "success": True
            }

            logger.info("SQL query processed successfully, confidence: %.2f", response["confidence"])
            return response

        except Exception as error:
            logger.error("Error processing SQL query '%s': %s", query, error)
            return self._format_error_response(query, str(error), filters)

    def _calculate_confidence(self, result: QAResult) -> float:
        """
        Вычисляет уверенность в ответе на основе качества SQL результата.

        Args:
            result: Результат выполнения SQL запроса

        Returns:
            Уровень уверенности от 0.0 до 1.0
        """
        # Нет данных - очень низкая уверенность
        if not result.data:
            return 0.1

        answer_lower = result.answer.lower()

        # Проверка на негативные индикаторы в ответе
        if any(indicator in answer_lower for indicator in self.NEGATIVE_INDICATORS):
            return 0.2

        # Проверка на пустые или нулевые значения в данных
        if self._has_empty_or_null_data(result.data):
            return 0.3

        # Проверка на ограничение LIMIT (возможно, данные усечены)
        if "limit" in result.sql.lower() and len(result.data) >= 100:
            return 0.7

        # Высокая уверенность для конкретных непустых результатов
        return 0.9

    def _has_empty_or_null_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        Проверяет, содержат ли данные пустые или нулевые значения.

        Args:
            data: Список строк данных

        Returns:
            True если есть пустые или нулевые значения
        """
        if not data:
            return True

        for row in data:
            for value in row.values():
                if value is None:
                    return True
                if value == 0 or value == "0":
                    return True
                if isinstance(value, str) and value.strip().lower() in ["", "none", "null", "n/a"]:
                    return True

        return False

    def _format_citations(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Форматирует SQL данные как цитаты для ответа.

        Args:
            data: Список словарей с данными из БД

        Returns:
            Список цитат в стандартном формате
        """
        citations = []

        for index, row in enumerate(data, 1):
            if isinstance(row, dict):
                citation_text = ", ".join([f"{key}: {value}" for key, value in row.items()])
                citations.append({
                    "source": f"database_result_{index}",
                    "text": citation_text,
                    "metadata": row
                })

        return citations

    def _format_error_response(self, query: str, error: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Форматирует ответ об ошибке.

        Args:
            query: Исходный запрос
            error: Текст ошибки
            filters: Фильтры запроса

        Returns:
            Стандартизированный ответ об ошибке
        """
        return {
            "answer": f"Извините, при запросе к базе данных произошла ошибка: {error}",
            "confidence": 0.0,
            "citations": [],
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "filters": filters or {},
            "success": False
        }


# Глобальный экземпляр для удобного использования
_global_sql: Optional[SQLQATool] = None


def answer_from_sql(query: str, filters: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """
    Удобная функция для использования глобального экземпляра SQL.

    Args:
        query: Вопрос на естественном языке
        filters: Дополнительные фильтры
        **kwargs: Дополнительные параметры

    Returns:
        Словарь с ответом и метаинформацией
    """
    global _global_sql
    if _global_sql is None:
        _global_sql = SQLQATool()
    return _global_sql.answer_from_sql(query, filters, **kwargs)


def get_sql_instance() -> SQLQATool:
    """
    Возвращает глобальный экземпляр SQL QA инструмента.

    Returns:
        Глобальный экземпляр SQLQATool
    """
    global _global_sql
    if _global_sql is None:
        _global_sql = SQLQATool()
    return _global_sql


# Псевдоним для обратной совместимости
SQLQA = SQLQATool
