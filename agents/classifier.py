# agents/classifier.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from core.settings import get_llm_settings
from core.llm_provider import get_llm

from .policies import (
    DOCS_KEYWORDS,
    SQL_KEYWORDS,
    WEB_KEYWORDS,
    DOCS_PATTERNS,
    WEB_PATTERNS,
)
from .utils import normalize_text, clamp01

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuleRationale:
    """Структурированное обоснование результатов RulesEngine."""

    hits: Dict[str, List[str]] = field(
        default_factory=lambda: {"docs": [], "sql": [], "web": []}
    )


class RulesEngine:
    """Rule-based классификатор для определения типа запроса.

    Возвращает нормализованные уверенности по трём направлениям:
    docs, sql, web. Использует только ключевые слова и паттерны.
    """

    LABELS: Tuple[str, str, str] = ("docs", "sql", "web")

    def classify(self, text: str) -> Dict[str, Any]:
        """Классифицирует запрос на основе правил и ключевых слов.

        Args:
            text: Исходный текст запроса пользователя

        Returns:
            Словарь с результатами классификации:
            - scores: уверенности по каждому label
            - rationale: обоснование с найденными ключевыми словами
        """
        normalized_text = normalize_text(text)
        scores: Dict[str, float] = {label: 0.0 for label in self.LABELS}
        rationale = RuleRationale()

        def bump(label: str, key: str, weight: float = 1.0) -> None:
            """Увеличивает счетчик для указанного label."""
            scores[label] += weight
            rationale.hits[label].append(key)

        # Обработка ключевых слов
        for keyword in DOCS_KEYWORDS:
            if keyword in normalized_text:
                bump("docs", keyword)

        for keyword in SQL_KEYWORDS:
            if keyword in normalized_text:
                bump("sql", keyword)

        for keyword in WEB_KEYWORDS:
            if keyword in normalized_text:
                bump("web", keyword, 1.2)

        # Обработка regex паттернов
        for pattern in DOCS_PATTERNS:
            if pattern.search(normalized_text):
                bump("docs", f"re:{pattern.pattern}", 1.5)

        for pattern in WEB_PATTERNS:
            if pattern.search(normalized_text):
                bump("web", f"re:{pattern.pattern}", 1.5)

        # Нормализация результатов
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            for label in scores:
                scores[label] = clamp01(scores[label] / max_score)

        return {
            "scores": scores,
            "rationale": {"hits": rationale.hits}
        }


class LLMClassifier:
    """LLM-классификатор для определения инструментов обработки запросов.

    Использует языковую модель для классификации запросов по трём направлениям:
    документация, SQL база данных, веб-поиск.
    """

    SYSTEM_PROMPT = (
        "Ты — эксперт по библиотеке transformers от Hugging Face. "
        "Определи инструмент(ы) для ответа на вопрос пользователя.\n"
        "Инструменты:\n"
        " - (A) Документация transformers (label: \"docs\") — для вопросов про модели, API, использование библиотеки\n"
        " - (B) База данных/SQL (label: \"sql\") — для организационных данных, контактов\n"
        " - (C) Веб-поиск (label: \"web\") — для актуальных новостей, обсуждений, issues\n\n"
        "Верни ровно JSON без пояснений в формате:\n"
        "{\"route\": [\"docs\"|\"sql\"|\"web\", ...], \"confidence\": 0..1, \"rationale\": \"...\"}\n"
        "По умолчанию для технических вопросов выбирай \"docs\"."
    )

    FEW_SHOT_EXAMPLES: List[Tuple[str, Dict[str, Any]]] = [
        (
            "Как использовать BertTokenizer для токенизации текста?",
            {
                "route": ["docs"],
                "confidence": 0.95,
                "rationale": "Вопрос про API токенизатора — документация transformers.",
            },
        ),
        (
            "Как сделать fine-tuning модели BERT на собственном датасете?",
            {
                "route": ["docs"],
                "confidence": 0.9,
                "rationale": "Вопрос про обучение моделей — документация transformers.",
            },
        ),
        (
            "Какие параметры есть у класса AutoModel?",
            {
                "route": ["docs"],
                "confidence": 0.95,
                "rationale": "Вопрос про API класса — документация transformers.",
            },
        ),
        (
            "Где найти пример использования pipeline из transformers?",
            {
                "route": ["docs"],
                "confidence": 0.9,
                "rationale": "Вопрос про примеры кода — документация transformers.",
            },
        ),
        (
            "Как решить проблему с загрузкой предобученной модели?",
            {
                "route": ["docs", "web"],
                "confidence": 0.8,
                "rationale": "Сначала документация, затем поиск похожих issues.",
            },
        ),
        (
            "Какая последняя версия transformers и что в ней нового?",
            {
                "route": ["web", "docs"],
                "confidence": 0.85,
                "rationale": "Версия и новости — веб, затем документация.",
            },
        ),
        (
            "Как использовать Whisper для транскрипции аудио?",
            {
                "route": ["docs"],
                "confidence": 0.9,
                "rationale": "Использование конкретной модели — документация.",
            },
        ),
        (
            "Какие есть альтернативы модели GPT-2?",
            {
                "route": ["docs", "web"],
                "confidence": 0.8,
                "rationale": "Список доступных моделей в доках, затем веб-поиск.",
            },
        ),
        (
            "Как настроить обучение на нескольких GPU?",
            {
                "route": ["docs"],
                "confidence": 0.9,
                "rationale": "Технический вопрос об обучении — документация.",
            },
        ),
        (
            "Где найти ответы на частые вопросы по transformers?",
            {
                "route": ["docs", "web"],
                "confidence": 0.75,
                "rationale": "FAQ может быть в документации и на форумах.",
            },
        ),
    ]

    ALLOWED_LABELS: Tuple[str, str, str] = ("docs", "sql", "web")

    def __init__(self) -> None:
        """Инициализирует классификатор с настройками LLM."""
        settings = get_llm_settings()
        self.client = get_llm()
        self.model = settings.model

    def _build_messages(self, question: str) -> List[Dict[str, str]]:
        """Формирует список сообщений для LLM с few-shot примерами.

        Args:
            question: Вопрос пользователя для классификации

        Returns:
            Список сообщений в формате чата
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        for user_question, assistant_answer in self.FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": user_question})
            messages.append({
                "role": "assistant",
                "content": json.dumps(assistant_answer, ensure_ascii=False)
            })

        messages.append({"role": "user", "content": question})
        return messages

    def classify(self, question: str) -> Dict[str, Any]:
        """Классифицирует запрос с помощью LLM.

        Args:
            question: Текст запроса для классификации

        Returns:
            Словарь с результатами классификации:
            - route: список выбранных инструментов
            - confidence: уверенность классификации
            - rationale: обоснование решения
        """
        messages = self._build_messages(question)

        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                top_p=1.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            raw_text = getattr(response, "text", None)
            if not raw_text:
                logger.warning("Получен пустой ответ от LLM для вопроса: %s", question)
                raise ValueError("Пустой ответ LLM")

            data = json.loads(raw_text)
            route = [label for label in data.get("route", []) if label in self.ALLOWED_LABELS]

            if not route:
                route = ["docs"]
                logger.info("Не найдено разрешенных labels, использован fallback 'docs'")

            confidence = clamp01(float(data.get("confidence", 0.6)))
            rationale = data.get("rationale", "")

            return {
                "route": route,
                "confidence": confidence,
                "rationale": rationale
            }

        except (ValueError, json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Ошибка классификации LLM: %s. Вопрос: '%s'",
                exc, question
            )
            return {
                "route": ["docs"],
                "confidence": 0.6,
                "rationale": "fallback_json_parse"
            }


__all__ = ["RulesEngine", "LLMClassifier"]
