# tools/rag/qa/retriever.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.settings import get_llm_settings
from ..embeddings.providers import EmbeddingProvider
from ..stores.base import ChildHit, ParentHit, VectorReader

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Извлеченный фрагмент документа с метаданными."""
    text: str
    score: float
    metadata: Dict[str, Any]
    chunk_type: str = "child"
    parent_id: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass
class RetrievalResult:
    """Результат поиска релевантных фрагментов."""
    chunks: List[RetrievedChunk]
    query: str
    total_chunks: int


class Retriever:
    """Ретривер, возвращающий только дочерние фрагменты для точного контекста."""

    def __init__(
            self,
            vector_reader: VectorReader,
            embedding_provider: EmbeddingProvider,
            top_k_parents: int = 3,
            top_k_children_per_parent: int = 4,
            min_score: float = 0.3,
            enable_rescoring: bool = True,
            alpha_parent_child: float = 0.7
    ) -> None:
        """Инициализация ретривера.

        Args:
            vector_reader: Векторный ридер для поиска
            embedding_provider: Провайдер эмбеддингов
            top_k_parents: Количество родительских документов для поиска
            top_k_children_per_parent: Количество дочерних фрагментов на родителя
            min_score: Минимальный порог релевантности
            enable_rescoring: Включить пересчет скоринга
            alpha_parent_child: Коэффициент для объединения скоринга
        """
        self.vector_reader = vector_reader
        self.embedding_provider = embedding_provider
        self.top_k_parents = top_k_parents
        self.top_k_children_per_parent = top_k_children_per_parent
        self.min_score = min_score
        self.enable_rescoring = enable_rescoring
        self.alpha_parent_child = alpha_parent_child

    def _rescore_child(self, child_score: float, parent_score: float) -> float:
        """Объединяет скоринг ребенка и родителя для лучшего ранжирования.

        Args:
            child_score: Скор дочернего фрагмента
            parent_score: Скор родительского документа

        Returns:
            Объединенный скор
        """
        return (
                self.alpha_parent_child * parent_score +
                (1 - self.alpha_parent_child) * child_score
        )

    def retrieve(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Извлекает только дочерние фрагменты - более точно и без дублирования.

        Args:
            query: Запрос пользователя
            filters: Фильтры для поиска

        Returns:
            Результат поиска с фрагментами
        """
        logger.info(f"Starting retrieval for query: {query}")

        query_embedding = self.embedding_provider.embed_queries([query])[0]

        # Шаг 1: Находим релевантных родителей (грубая фильтрация)
        parent_hits = self.vector_reader.search_parents(
            query_embedding, self.top_k_parents, filters
        )

        all_child_chunks: List[RetrievedChunk] = []

        # Шаг 2: Для каждого родителя находим наиболее релевантных детей
        if parent_hits:
            child_hits_by_parent = self.vector_reader.search_children(
                parent_hits, query_embedding, self.top_k_children_per_parent
            )

            for parent in parent_hits:
                children = child_hits_by_parent.get(parent.id, [])
                for child in children:
                    if child.score >= self.min_score:
                        final_score = self._calculate_final_score(child.score, parent.score)

                        chunk = RetrievedChunk(
                            text=child.text,
                            score=final_score,
                            metadata=child.meta,
                            parent_id=parent.id,
                            chunk_id=child.id
                        )
                        all_child_chunks.append(chunk)

        # Шаг 3: Удаляем дубликаты и сортируем по релевантности
        unique_chunks = self._deduplicate_and_sort_chunks(all_child_chunks)

        # Шаг 4: Обрезаем по бюджету токенов
        final_chunks = self._clip_chunks_to_token_budget(unique_chunks, query)

        logger.info(
            f"Retrieved {len(final_chunks)} unique child chunks "
            f"(from {len(all_child_chunks)} total) for query: {query}"
        )

        return RetrievalResult(
            chunks=list(final_chunks),
            query=query,
            total_chunks=len(final_chunks)
        )

    def _calculate_final_score(self, child_score: float, parent_score: float) -> float:
        """Вычисляет финальный скор с учетом родительского рейтинга."""
        if self.enable_rescoring:
            return self._rescore_child(child_score, parent_score)
        return child_score

    def _deduplicate_and_sort_chunks(
            self,
            chunks: List[RetrievedChunk]
    ) -> List[RetrievedChunk]:
        """Удаляет дубликаты и сортирует фрагменты по релевантности."""
        seen_texts: set[int] = set()
        unique_chunks: List[RetrievedChunk] = []

        for chunk in sorted(chunks, key=lambda x: x.score, reverse=True):
            # Интеллектуальная дедупликация - используем хэш текста
            text_hash = hash(chunk.text[:500])  # Хэшируем первые 500 символов

            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def _clip_chunks_to_token_budget(
            self,
            chunks: List[RetrievedChunk],
            query: str
    ) -> List[RetrievedChunk]:
        """Обрезает фрагменты по бюджету токенов."""
        max_context_tokens = getattr(self, "max_context_tokens", 4000)
        model_hint = get_llm_settings().model

        return clip_chunks_to_budget(
            chunks=chunks,
            question=query,
            system="",
            model_hint=model_hint,
            max_context_tokens=max_context_tokens,
        )


def _approx_tokens(text: str) -> int:
    """Простая эвристика: ~4 символа на токен.

    Args:
        text: Текст для оценки

    Returns:
        Приблизительное количество токенов
    """
    return max(1, len(text) // 4)


def _count_tokens(text: str, model_hint: str | None = None) -> int:
    """Подсчитывает количество токенов в тексте.

    Пытается использовать точный подсчет (tiktoken для openai-*),
    иначе использует эвристику.

    Args:
        text: Текст для подсчета
        model_hint: Модель для выбора энкодера

    Returns:
        Количество токенов
    """
    try:
        if model_hint and ("gpt-" in model_hint or "o3" in model_hint):
            import tiktoken
            encoder = tiktoken.encoding_for_model(model_hint)
            return len(encoder.encode(text))
    except (ImportError, KeyError, Exception):
        logger.debug("tiktoken not available, using approximate token count")

    return _approx_tokens(text)


def clip_chunks_to_budget(
        chunks: List[RetrievedChunk],
        question: str,
        system: str,
        model_hint: str | None,
        max_context_tokens: int,
        safety_margin: int = 256,
) -> List[RetrievedChunk]:
    """Обрезает список фрагментов по бюджету токенов.

    System + question + context <= max_context_tokens - safety_margin.
    Предполагается, что фрагменты уже отсортированы по скорингу.

    Args:
        chunks: Список фрагментов для обрезки
        question: Вопрос пользователя
        system: Системный промпт
        model_hint: Модель для подсчета токенов
        max_context_tokens: Максимальное количество токенов контекста
        safety_margin: Запас токенов для безопасности

    Returns:
        Обрезанный список фрагментов
    """
    used_tokens = (
            _count_tokens(system, model_hint) +
            _count_tokens(question, model_hint)
    )
    token_limit = max(128, max_context_tokens - safety_margin)

    selected_chunks: List[RetrievedChunk] = []

    for chunk in chunks:
        chunk_tokens = _count_tokens(chunk.text, model_hint)

        if used_tokens + chunk_tokens > token_limit:
            logger.debug(
                f"Token limit reached: {used_tokens} + {chunk_tokens} > {token_limit}"
            )
            break

        selected_chunks.append(chunk)
        used_tokens += chunk_tokens

    logger.info(
        f"Selected {len(selected_chunks)} chunks using {used_tokens} tokens "
        f"(limit: {token_limit})"
    )

    return selected_chunks
