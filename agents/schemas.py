# agents/schemas.py
from __future__ import annotations

"""
Схемы данных для обмена между агентом, инструментами и внешним API.

Содержит датаклассы для типизированного представления запросов, ответов
и результатов работы инструментов в системе маршрутизации.

Публичные сущности:
- ToolName: тип-обёртка для имен инструментов
- ToolResult: результат работы отдельного инструмента  
- AgentRequest: входной запрос для агента-роутера
- AgentResponse: финальный ответ агента со сводной информацией
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, ClassVar

__all__ = ["ToolName", "ToolResult", "AgentRequest", "AgentResponse"]


@dataclass(frozen=True, slots=True)
class ToolName:
    """
    Имя инструмента как иммутабельный тип-обёртка над строкой.

    Обеспечивает типобезопасность при работе с именами инструментов и
    централизованную нормализацию значений.

    Attributes:
        value: Нормализованное имя инструмента в нижнем регистре

    Examples:
        >>> tool = ToolName("Docs")
        >>> tool.value
        'docs'
        >>> tool == "docs"
        True
    """

    #: Рекомендованные имена базовых инструментов системы
    RECOMMENDED: ClassVar[tuple[str, str, str]] = ("docs", "sql", "web")

    value: str

    def __post_init__(self) -> None:
        """Валидирует и нормализует имя инструмента после инициализации."""
        normalized_value = (self.value or "").strip().lower()
        if not normalized_value:
            raise ValueError("Tool name cannot be empty")
        object.__setattr__(self, 'value', normalized_value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"ToolName('{self.value}')"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ToolName):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other.lower()
        return False

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(slots=True)
class ToolResult:
    """
    Результат работы одного инструмента обработки запроса.

    Attributes:
        tool: Имя инструмента, выполнившего работу
        answer: Текст ответа инструмента (сырые данные или человекочитаемое резюме)
        confidence: Оценка уверенности инструмента в диапазоне [0.0, 1.0]
        citations: Список ссылок/цитаций/метаданных источников
        meta: Служебные метаданные (фильтры, модель, usage и т.п.)

    Examples:
        >>> result = ToolResult(
        ...     tool=ToolName("docs"),
        ...     answer="Для использования BertTokenizer вызовите from_pretrained()",
        ...     confidence=0.85,
        ...     citations=[{"source": "huggingface.co/docs"}],
        ...     meta={"model": "gpt-4", "tokens_used": 150}
        ... )
    """

    tool: ToolName
    answer: str
    confidence: float
    citations: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидирует данные результата после инициализации."""
        if not isinstance(self.tool, ToolName):
            self.tool = ToolName(str(self.tool))

        if not self.answer or not self.answer.strip():
            raise ValueError("Answer cannot be empty")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.citations is None:
            self.citations = []

        if self.meta is None:
            self.meta = {}


@dataclass(slots=True)
class AgentRequest:
    """
    Входной запрос к агенту-роутеру для обработки.

    Attributes:
        query: Исходный текст запроса пользователя
        routing_hints: Необязательные подсказки маршрутизации

    Examples:
        >>> request = AgentRequest(
        ...     query="Как использовать BertTokenizer?",
        ...     routing_hints={"boost": {"docs": 0.9, "web": 0.3}}
        ... )
    """

    query: str
    routing_hints: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Валидирует запрос после инициализации."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")

        if self.routing_hints is None:
            self.routing_hints = {}


@dataclass(slots=True)
class AgentResponse:
    """
    Финальный ответ агента-роутера после обработки запроса.

    Содержит итоговый ответ, информацию о маршрутизации и трассировку
    для отладки и анализа работы системы.

    Attributes:
        final_answer: Итоговый ответ после сведения результатов инструментов
        route: Фактически выбранный маршрут (последовательность инструментов)
        tool_results: Частичные результаты отдельных инструментов
        decisions: Отладочная/трассировочная информация
        latency_ms: Полная латентность обработки запроса в миллисекундах

    Examples:
        >>> response = AgentResponse(
        ...     final_answer="Для использования BertTokenizer... [1]",
        ...     route=[ToolName("docs")],
        ...     tool_results=[tool_result],
        ...     decisions={"mode": "hybrid", "confidence": 0.8},
        ...     latency_ms=1450
        ... )
    """

    final_answer: str
    route: List[ToolName]
    tool_results: List[ToolResult]
    decisions: Dict[str, Any]
    latency_ms: int

    def __post_init__(self) -> None:
        """Валидирует ответ после инициализации."""
        if not self.final_answer or not self.final_answer.strip():
            raise ValueError("Final answer cannot be empty")

        if not self.route:
            raise ValueError("Route cannot be empty")

        # Нормализуем инструменты в маршруте
        normalized_route = []
        for tool in self.route:
            if not isinstance(tool, ToolName):
                tool = ToolName(str(tool))
            normalized_route.append(tool)
        object.__setattr__(self, 'route', normalized_route)

        if self.tool_results is None:
            self.tool_results = []

        if self.decisions is None:
            self.decisions = {}

        if self.latency_ms < 0:
            raise ValueError(f"Latency cannot be negative, got {self.latency_ms}")

    @property
    def primary_tool(self) -> ToolName:
        """Возвращает основной инструмент, использованный для ответа."""
        return self.route[0] if self.route else ToolName("docs")

    @property
    def total_confidence(self) -> float:
        """Вычисляет агрегированную уверенность на основе результатов инструментов."""
        if not self.tool_results:
            return 0.0

        confidences = [result.confidence for result in self.tool_results]
        return max(confidences) if confidences else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Сериализует ответ в словарь для API."""
        return {
            "final_answer": self.final_answer,
            "route": [str(tool) for tool in self.route],
            "tool_results": [
                {
                    "tool": str(result.tool),
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "citations": result.citations,
                    "meta": result.meta
                }
                for result in self.tool_results
            ],
            "decisions": self.decisions,
            "latency_ms": self.latency_ms,
            "primary_tool": str(self.primary_tool),
            "total_confidence": self.total_confidence
        }
