# tools/rag/prompt.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Evidence:
    """Контейнер для доказательств с метаданными."""
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    page: int | None = None

    def __post_init__(self) -> None:
        """Валидация входных данных."""
        if not isinstance(self.text, str):
            raise TypeError("text должен быть строкой")
        if not isinstance(self.score, (int, float)):
            raise TypeError("score должен быть числом")


class PromptBuilder:
    """Строитель промптов для LLM с контекстом и цитированием."""

    def __init__(self, citation_style: str = "number") -> None:
        """Инициализация строителя промптов.

        Args:
            citation_style: Стиль цитирования (по умолчанию "number")
        """
        self.citation_style = citation_style

    def build_system_prompt(self, policy: str | None = None) -> str:
        """Создает системный промпт с инструкциями.

        Args:
            policy: Дополнительная политика для добавления в промпт

        Returns:
            Строка с системным промптом
        """
        base_prompt = """You are a helpful AI assistant that answers questions based on the provided context documents.

Instructions:
1. Answer the question using ONLY the provided context
2. If the context doesn't contain relevant information, say so
3. Be precise and factual
4. Cite sources using [number] notation where number corresponds to the source reference
5. If multiple sources support your answer, cite all relevant ones [1][2]
6. Don't make up information or use external knowledge
"""

        if policy:
            base_prompt += f"\nAdditional policy: {policy}"

        return base_prompt

    def build_user_prompt(
            self,
            query: str,
            evidences: List[Evidence],
            total_tokens: int | None = None
    ) -> str:
        """Создает пользовательский промпт с контекстом и вопросом.

        Args:
            query: Вопрос пользователя
            evidences: Список доказательств для контекста
            total_tokens: Общее количество токенов (опционально)

        Returns:
            Сформированный промпт для пользователя
        """
        if not evidences:
            return f"QUESTION: {query}\n\nNo relevant context found. Please respond accordingly."

        context_parts = []
        for i, evidence in enumerate(evidences, 1):
            source_info = self._format_source_info(evidence.metadata, i)
            context_parts.append(f"[Source {i}] {source_info}\n{evidence.text}")

        context = "\n\n".join(context_parts)

        prompt_parts = [
            "Based on the following documents, please answer the question.",
            f"CONTEXT DOCUMENTS:\n{context}",
            f"QUESTION: {query}",
            "Please provide a comprehensive answer with citations where appropriate."
        ]

        if total_tokens:
            prompt_parts.append(f"Total context tokens: {total_tokens}")

        return "\n\n".join(prompt_parts)

    def _format_source_info(self, metadata: Dict[str, Any], index: int) -> str:
        """Форматирует информацию об источнике из метаданных.

        Args:
            metadata: Словарь с метаданными
            index: Номер источника

        Returns:
            Отформатированная строка с информацией об источнике
        """
        parts = []
        source_key = "source" if "source" in metadata else "file_name"

        if source_key in metadata:
            parts.append(f"Source: {metadata[source_key]}")

        for key in ["page", "section"]:
            if key in metadata:
                parts.append(f"{key.capitalize()}: {metadata[key]}")

        return " | ".join(parts) if parts else f"Reference {index}"

    def extract_citations_from_evidences(self, evidences: List[Evidence]) -> List[Dict[str, Any]]:
        """Извлекает информацию для цитирования из доказательств.

        Args:
            evidences: Список доказательств

        Returns:
            Список словарей с информацией для цитирования
        """
        citations = []
        for i, evidence in enumerate(evidences, 1):
            citation = {
                "id": i,
                "text_preview": evidence.text[:200] + "..." if len(evidence.text) > 200 else evidence.text,
                "score": round(evidence.score, 3),
                "metadata": evidence.metadata
            }
            citations.append(citation)
        return citations
