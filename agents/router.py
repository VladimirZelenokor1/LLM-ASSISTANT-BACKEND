# agents/router.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from core.settings import get_llm_settings
from core.llm_provider import get_llm

from agents.schemas import AgentRequest, AgentResponse, ToolResult, ToolName
from agents.classifier import RulesEngine, LLMClassifier
from agents.utils import (
    choose_route_by_priorities,
    combine_scores,
    now_ms,
    gen_trace_id,
)

logger = logging.getLogger(__name__)

# =============================================================================
# ИНСТРУМЕНТЫ С БЕЗОПАСНЫМИ ЗАГЛУШКАМИ
# =============================================================================

DOCS_AVAILABLE = False
SQL_AVAILABLE = False

try:
    from tools.rag.qa.qa_tool import answer_from_docs as _answer_from_docs

    DOCS_AVAILABLE = True
    logger.info("RAG/Docs tool successfully imported")
except ImportError as e:
    logger.warning("RAG/Docs tool is not available: %s", e)
except Exception as e:
    logger.error("Unexpected error importing RAG tool: %s", e)

try:
    from tools.sql.sql_tool import answer_from_sql as _answer_from_sql

    SQL_AVAILABLE = True
    logger.info("SQL tool successfully imported")
except ImportError as e:
    logger.warning("SQL tool is not available: %s", e)
except Exception as e:
    logger.error("Unexpected error importing SQL tool: %s", e)


def _create_fallback_response(tool_name: str, message: str) -> Dict[str, Any]:
    """Создает безопасный fallback-ответ для недоступных инструментов."""
    return {
        "answer": message,
        "confidence": 0.25,
        "citations": [],
        "meta": {"fallback": True, "tool": tool_name},
    }


def _answer_from_sql_fallback(
        query: str,
        filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Заглушка SQL-инструмента при недоступности."""
    return _create_fallback_response(
        "sql",
        "SQL-поиск временно недоступен. Пожалуйста, попробуйте позже."
    )


def _answer_from_web(
        query: str,
        filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Заглушка WEB-инструмента."""
    return _create_fallback_response(
        "web",
        "WEB-поиск в настоящее время не реализован."
    )


# Переопределяем функцию если SQL недоступен
if not SQL_AVAILABLE:
    _answer_from_sql = _answer_from_sql_fallback


# =============================================================================
# КЛАСС АГЕНТА-РОУТЕРА
# =============================================================================

class Agent:
    """Агент-роутер для маршрутизации запросов к инструментам обработки.

    Поддерживает три режима работы:
    - "rules": только rule-based классификатор
    - "llm": только LLM-классификатор
    - "hybrid": комбинированный подход

    Attributes:
        mode: Режим работы агента
        allow_combine: Разрешить комбинирование инструментов
        tools_priorities: Приоритеты инструментов
        min_confidence: Минимальная уверенность для использования инструмента
    """

    DEFAULT_CONFIG = {
        "agent": {
            "mode": "rules",
            "allow_combine": True,
            "tools_priorities": ["docs", "sql", "web"],
            "min_confidence": 0.55,
            "routing_filters": {},
            "deep_search": {"enabled": False},
        }
    }

    def __init__(self, cfg_path: Optional[Union[str, Path]] = None) -> None:
        """Инициализирует агента с конфигурацией.

        Args:
            cfg_path: Путь к YAML-файлу конфигурации
        """
        self.cfg_path = Path(cfg_path) if cfg_path else Path("configs/agent.yaml")
        self.config = self._load_config(self.cfg_path)

        agent_cfg = self.config.get("agent", {})
        self.mode: str = agent_cfg.get("mode", "rules")
        self.allow_combine: bool = bool(agent_cfg.get("allow_combine", True))
        self.tools_priorities: List[str] = list(
            agent_cfg.get("tools_priorities", ["docs", "sql", "web"])
        )
        self.min_confidence: float = float(agent_cfg.get("min_confidence", 0.55))
        self.routing_filters: Dict[str, Any] = dict(
            agent_cfg.get("routing_filters", {})
        )
        self.deep_search_cfg: Dict[str, Any] = dict(
            agent_cfg.get("deep_search", {"enabled": False})
        )

        # Инициализация классификаторов
        self.rules_engine = RulesEngine()
        self.llm_classifier = LLMClassifier()

        # LLM для финального синтеза
        self.settings = get_llm_settings()
        self.llm_client = get_llm()

        logger.info(
            "Agent initialized with mode='%s', min_confidence=%.2f, tools_priorities=%s",
            self.mode, self.min_confidence, self.tools_priorities
        )

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        """Загружает YAML-конфигурацию агента.

        Args:
            path: Путь к файлу конфигурации

        Returns:
            Словарь с конфигурацией или значения по умолчанию
        """
        if not path.exists():
            logger.info("Config file %s not found, using defaults", path)
            return Agent.DEFAULT_CONFIG

        try:
            with path.open("r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
                logger.info("Config loaded successfully from %s", path)
                return config
        except (yaml.YAMLError, IOError) as e:
            logger.error("Failed to load config from %s: %s", path, e)
            return Agent.DEFAULT_CONFIG

    def route_query(self, request: AgentRequest) -> AgentResponse:
        """Выполняет маршрутизацию запроса и возвращает ответ.

        Args:
            request: Запрос агента с текстом и подсказками маршрутизации

        Returns:
            Ответ агента с финальным ответом и трассировкой
        """
        start_time = now_ms()
        trace_id = gen_trace_id()

        query: str = request.query
        hints: Dict[str, Any] = request.routing_hints or {}

        logger.debug("Processing query: '%s' with trace_id: %s", query, trace_id)

        # 1. Классификация запроса
        decisions = self._classify_query(query, trace_id)

        # 2. Определение финального маршрута
        route, confidence = self._determine_route(decisions, hints)
        used_tools: List[ToolName] = [
            ToolName(tool) if isinstance(tool, str) else tool
            for tool in route
        ]

        # 3. Выполнение инструментов
        tool_results = self._execute_tools(query, route, confidence)

        # 4. Синтез финального ответа
        final_answer = self._synthesize_final_answer(query, tool_results)

        latency = now_ms() - start_time

        logger.info(
            "Query processed in %d ms, route: %s, confidence: %.2f",
            latency, used_tools, confidence
        )

        return AgentResponse(
            final_answer=final_answer,
            route=used_tools,
            tool_results=tool_results,
            decisions=decisions,
            latency_ms=latency,
        )

    def _classify_query(self, query: str, trace_id: str) -> Dict[str, Any]:
        """Классифицирует запрос с помощью доступных классификаторов.

        Args:
            query: Текст запроса
            trace_id: Идентификатор трассировки

        Returns:
            Словарь с результатами классификации
        """
        decisions: Dict[str, Any] = {
            "trace_id": trace_id,
            "mode": self.mode,
        }

        # Rule-based классификация
        if self.mode in ("rules", "hybrid"):
            rules_result = self.rules_engine.classify(query)
            decisions["rules"] = rules_result
            logger.debug("Rules classification: %s", rules_result.get("scores", {}))

        # LLM классификация
        if self.mode in ("llm", "hybrid"):
            llm_result = self.llm_classifier.classify(query)
            decisions["llm"] = llm_result
            logger.debug("LLM classification: %s", llm_result)

        return decisions

    def _determine_route(
            self,
            decisions: Dict[str, Any],
            hints: Dict[str, Any]
    ) -> Tuple[List[str], float]:
        """Определяет финальный маршрут обработки запроса.

        Args:
            decisions: Результаты классификации
            hints: Подсказки маршрутизации

        Returns:
            Кортеж (маршрут, уверенность)
        """
        if self.mode == "rules":
            rules_scores = decisions.get("rules", {}).get("scores", {})
            return self._route_from_rules(rules_scores)

        elif self.mode == "llm":
            llm_result = decisions.get("llm", {})
            return self._route_from_llm(llm_result)

        else:  # hybrid
            rules_scores = decisions.get("rules", {}).get("scores", {})
            llm_result = decisions.get("llm", {})
            return self._route_from_hybrid(rules_scores, llm_result, hints)

    def _route_from_rules(self, rules_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """Определяет маршрут на основе rule-based классификации."""
        candidates = [
            tool for tool, score in rules_scores.items()
            if score >= self.min_confidence
        ]
        candidates = sorted(candidates, key=rules_scores.get, reverse=True)

        if not candidates:
            candidates = ["docs"]
            logger.debug("No candidates from rules, using fallback to 'docs'")

        route = choose_route_by_priorities(
            candidates, self.tools_priorities, self.allow_combine
        )
        confidence = max(rules_scores.values()) if rules_scores else 0.6

        return route, confidence

    def _route_from_llm(self, llm_result: Dict[str, Any]) -> Tuple[List[str], float]:
        """Определяет маршрут на основе LLM классификации."""
        llm_route = llm_result.get("route", [])
        llm_confidence = float(llm_result.get("confidence", 0.6))

        if llm_confidence >= self.min_confidence:
            route = choose_route_by_priorities(
                llm_route, self.tools_priorities, self.allow_combine
            )
        else:
            route = ["docs"]
            logger.debug("LLM confidence too low, using fallback to 'docs'")

        return route, llm_confidence

    def _route_from_hybrid(
            self,
            rules_scores: Dict[str, float],
            llm_result: Dict[str, Any],
            hints: Dict[str, Any]
    ) -> Tuple[List[str], float]:
        """Определяет маршрут на основе комбинированного подхода."""
        combined_scores = rules_scores.copy()

        # Учет LLM уверенности
        llm_confidence = float(llm_result.get("confidence", 0.6))
        for tool in llm_result.get("route", []):
            current_score = combined_scores.get(tool, 0.0)
            combined_scores[tool] = max(current_score, llm_confidence * 0.8)

        # Учет подсказок маршрутизации
        if hints:
            for tool, boost in hints.get("boost", {}).items():
                current_score = combined_scores.get(tool, 0.0)
                combined_scores[tool] = max(current_score, float(boost))

        # Выбор кандидатов
        candidates = [
            tool for tool, score in combined_scores.items()
            if score >= self.min_confidence
        ]
        candidates = sorted(candidates, key=combined_scores.get, reverse=True)

        if not candidates:
            candidates = ["docs"]
            logger.debug("No candidates from hybrid, using fallback to 'docs'")

        route = choose_route_by_priorities(
            candidates, self.tools_priorities, self.allow_combine
        )
        confidence = max(combined_scores.values()) if combined_scores else 0.6

        return route, confidence

    def _execute_tools(
            self,
            query: str,
            route: List[str],
            confidence: float
    ) -> List[ToolResult]:
        """Выполняет инструменты по маршруту с ранним выходом при высокой уверенности."""
        tool_results: List[ToolResult] = []

        for tool in route:
            result = self._execute_single_tool(tool, query)
            tool_results.append(result)

            # Ранний выход при достаточной уверенности
            if result.confidence >= max(self.min_confidence, confidence):
                logger.debug(
                    "Early exit: tool '%s' confidence %.2f >= required %.2f",
                    tool, result.confidence, max(self.min_confidence, confidence)
                )
                break

        return tool_results

    def _execute_single_tool(self, tool: str, query: str) -> ToolResult:
        """Выполняет один инструмент и возвращает результат."""
        try:
            if tool == "docs":
                return self._run_docs_tool(query)
            elif tool == "sql":
                return self._run_sql_tool(query)
            elif tool == "web":
                return self._run_web_tool(query)
            else:
                logger.warning("Unknown tool '%s' in route", tool)
                return ToolResult(
                    tool=tool,
                    answer=f"Инструмент '{tool}' недоступен.",
                    confidence=0.2,
                    citations=[],
                    meta={},
                )
        except Exception as e:
            logger.error("Unexpected error executing tool '%s': %s", tool, e)
            return ToolResult(
                tool=tool,
                answer=f"Ошибка при выполнении инструмента '{tool}'.",
                confidence=0.1,
                citations=[],
                meta={"error": str(e)},
            )

    def _run_docs_tool(self, query: str) -> ToolResult:
        """Выполняет RAG/Docs инструмент."""
        if not DOCS_AVAILABLE:
            return ToolResult(
                tool="docs",
                answer="RAG: модуль недоступен.",
                confidence=0.2,
                citations=[],
                meta={},
            )

        try:
            output = _answer_from_docs(query, stream=False)
            return self._normalize_tool_output("docs", output)
        except Exception as e:
            logger.error("Docs tool execution failed: %s", e)
            return ToolResult(
                tool="docs",
                answer="Не удалось получить ответ из документации.",
                confidence=0.3,
                citations=[],
                meta={"error": str(e)},
            )

    def _run_sql_tool(self, query: str) -> ToolResult:
        """Выполняет SQL инструмент."""
        try:
            output = _answer_from_sql(query, self.routing_filters)
            return self._normalize_tool_output("sql", output)
        except Exception as e:
            logger.error("SQL tool execution failed: %s", e)
            return ToolResult(
                tool="sql",
                answer="Не удалось получить ответ из базы данных.",
                confidence=0.3,
                citations=[],
                meta={"error": str(e)},
            )

    def _run_web_tool(self, query: str) -> ToolResult:
        """Выполняет WEB инструмент."""
        try:
            output = _answer_from_web(query, self.routing_filters)
            return self._normalize_tool_output("web", output)
        except Exception as e:
            logger.error("WEB tool execution failed: %s", e)
            return ToolResult(
                tool="web",
                answer="Не удалось выполнить веб-поиск.",
                confidence=0.3,
                citations=[],
                meta={"error": str(e)},
            )

    def _normalize_tool_output(self, tool: str, output: Any) -> ToolResult:
        """Нормализует вывод инструмента к единому формату."""
        answer_text = ""
        confidence = 0.5
        citations = []
        meta = {"filters": self.routing_filters}

        if isinstance(output, dict):
            # Извлечение ответа
            answer_data = output.get("answer", output)
            if isinstance(answer_data, dict):
                answer_text = (
                        answer_data.get("text") or
                        answer_data.get("answer") or
                        answer_data.get("final") or
                        self._safe_json_dumps(answer_data)
                )
                # Дополнительные метаданные
                meta.update({
                    k: v for k, v in answer_data.items()
                    if k in ("usage", "model", "sources")
                })
            elif isinstance(answer_data, str):
                answer_text = answer_data

            # Извлечение цитат и уверенности
            citations = output.get("citations") or output.get("sources") or []
            confidence = float(output.get("confidence", 0.5))

        elif isinstance(output, str):
            answer_text = output

        # Fallback для пустого ответа
        if not answer_text.strip():
            answer_text = f"Инструмент {tool} не вернул содержательного ответа."
            confidence = max(0.3, confidence * 0.7)

        return ToolResult(
            tool=tool,
            answer=answer_text,
            confidence=confidence,
            citations=citations,
            meta=meta,
        )

    def _synthesize_final_answer(self, query: str, tool_results: List[ToolResult]) -> str:
        """Синтезирует финальный ответ из результатов инструментов."""
        if not tool_results:
            return "Не удалось получить ответ от доступных инструментов."

        # Лучший результат как fallback
        best_result = max(tool_results, key=lambda r: r.confidence)

        try:
            return self._synthesize_with_llm(query, tool_results)
        except Exception as e:
            logger.warning("LLM synthesis failed, using best result: %s", e)
            return best_result.answer

    def _synthesize_with_llm(self, query: str, tool_results: List[ToolResult]) -> str:
        """Синтезирует ответ с помощью LLM."""
        system_prompt = (
            "Ты — помощник, который объединяет ответы из разных инструментов.\n"
            "Отвечай кратко и точно, ссылайся на источники в квадратных скобках [1], [2] при наличии.\n"
            "Если данных недостаточно — скажи об этом явно.\n"
            "Никогда не выдумывай факты."
        )

        user_content_parts = [f"Вопрос: {query}", "Доступные результаты:"]

        for i, result in enumerate(tool_results, 1):
            citation_info = ""
            if result.citations:
                sources = [str(c.get("source", c)) for c in result.citations]
                citation_info = " | источники: " + "; ".join(sources)

            user_content_parts.append(
                f"[{i}] tool={result.tool}; conf={result.confidence:.2f}: "
                f"{result.answer}{citation_info}"
            )

        user_content_parts.append("\nСобери финальный ответ, перечисли номера источников в тексте.")
        user_content = "\n".join(user_content_parts)

        response = self.llm_client.responses.create(
            model=self.settings.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            top_p=1.0,
            max_tokens=512,
        )

        final_text = getattr(response, "text", "").strip()
        return final_text if final_text else tool_results[0].answer

    @staticmethod
    def _safe_json_dumps(obj: Any, max_length: int = 2000) -> str:
        """Безопасно сериализует объект в JSON строку."""
        try:
            result = json.dumps(obj, ensure_ascii=False, default=str)
            return result[:max_length]
        except (TypeError, ValueError) as e:
            logger.warning("JSON serialization failed: %s", e)
            return str(obj)[:max_length]
