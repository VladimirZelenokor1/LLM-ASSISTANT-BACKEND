# path: tools/rag/stores/base.py
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParentHit:
    """Результат поиска родительского документа в векторном хранилище."""

    id: str
    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация данных после инициализации."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("id must be a non-empty string")

        if not self.text or not isinstance(self.text, str):
            raise ValueError("text must be a non-empty string")

        if not isinstance(self.score, (int, float)) or not (0.0 <= self.score <= 1.0):
            raise ValueError("score must be a float between 0.0 and 1.0")

        if not isinstance(self.meta, dict):
            raise ValueError("meta must be a dictionary")

    @property
    def truncated_text(self) -> str:
        """Возвращает укороченную версию текста для отображения.

        Returns:
            Текст, обрезанный до 200 символов с многоточием
        """
        if len(self.text) <= 200:
            return self.text
        return self.text[:197] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует объект в словарь.

        Returns:
            Словарь с данными объекта
        """
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "meta": self.meta
        }

    def __str__(self) -> str:
        """Строковое представление объекта."""
        return f"ParentHit(id={self.id}, score={self.score:.3f}, text='{self.truncated_text}')"


@dataclass
class ChildHit:
    """Результат поиска дочернего документа в векторном хранилище."""

    id: str
    text: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Валидация данных после инициализации."""
        if not self.id or not isinstance(self.id, str):
            raise ValueError("id must be a non-empty string")

        if not self.text or not isinstance(self.text, str):
            raise ValueError("text must be a non-empty string")

        if not isinstance(self.score, (int, float)) or not (0.0 <= self.score <= 1.0):
            raise ValueError("score must be a float between 0.0 and 1.0")

        if not isinstance(self.meta, dict):
            raise ValueError("meta must be a dictionary")

        if self.parent_id and not isinstance(self.parent_id, str):
            raise ValueError("parent_id must be a string or None")

    @property
    def truncated_text(self) -> str:
        """Возвращает укороченную версию текста для отображения.

        Returns:
            Текст, обрезанный до 200 символов с многоточием
        """
        if len(self.text) <= 200:
            return self.text
        return self.text[:197] + "..."

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует объект в словарь.

        Returns:
            Словарь с данными объекта
        """
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "meta": self.meta,
            "parent_id": self.parent_id
        }

    def __str__(self) -> str:
        """Строковое представление объекта."""
        parent_info = f", parent={self.parent_id}" if self.parent_id else ""
        return f"ChildHit(id={self.id}, score={self.score:.3f}{parent_info}, text='{self.truncated_text}')"


class VectorReader(ABC):
    """Абстрактный интерфейс для ридеров векторных хранилищ.

    Определяет контракт для поиска родительских и дочерних документов
    в иерархических векторных хранилищах.
    """

    @abstractmethod
    def search_parents(
            self,
            query_embedding: List[float],
            top_k: int,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[ParentHit]:
        """Ищет родительские документы по векторному запросу.

        Args:
            query_embedding: Векторное представление запроса
            top_k: Количество возвращаемых результатов
            filters: Дополнительные фильтры для поиска

        Returns:
            Список объектов ParentHit, отсортированных по релевантности

        Raises:
            ValueError: Если query_embedding пустой или top_k <= 0
            ConnectionError: Если не удалось подключиться к хранилищу
        """
        pass

    @abstractmethod
    def search_children(
            self,
            parent_hits: List[ParentHit],
            query_embedding: List[float],
            top_k_each: int
    ) -> Dict[str, List[ChildHit]]:
        """Ищет дочерние документы для заданных родительских ID.

        Args:
            parent_hits: Список родительских документов
            query_embedding: Векторное представление запроса
            top_k_each: Количество дочерних документов на каждого родителя

        Returns:
            Словарь, где ключи - parent_id, значения - списки ChildHit

        Raises:
            ValueError: Если parent_hits пустой или top_k_each <= 0
            ConnectionError: Если не удалось подключиться к хранилищу
        """
        pass

    def validate_embedding(self, embedding: List[float]) -> bool:
        """Валидирует векторное представление.

        Args:
            embedding: Вектор для валидации

        Returns:
            True если вектор валиден, иначе False
        """
        if not embedding or not isinstance(embedding, list):
            return False

        if not all(isinstance(x, (int, float)) for x in embedding):
            return False

        return True

    def get_search_stats(self) -> Dict[str, Any]:
        """Возвращает статистику использования ридера.

        Returns:
            Словарь со статистикой поисковых операций
        """
        return {
            "reader_type": self.__class__.__name__,
            "description": "Abstract vector reader - statistics not implemented"
        }

    def health_check(self) -> bool:
        """Проверяет доступность векторного хранилища.

        Returns:
            True если хранилище доступно, иначе False
        """
        try:
            # Базовая проверка - попытка выполнить простой запрос
            test_embedding = [0.0] * 384  # Стандартный размер для теста
            results = self.search_parents(test_embedding, top_k=1)
            return isinstance(results, list)
        except Exception as error:
            logger.error(f"Health check failed: {error}")
            return False


class VectorSearchResult:
    """Контейнер для результатов векторного поиска."""

    def __init__(
            self,
            parent_hits: List[ParentHit],
            child_hits: Dict[str, List[ChildHit]]
    ) -> None:
        """Инициализация контейнера результатов.

        Args:
            parent_hits: Список найденных родительских документов
            child_hits: Словарь дочерних документов по parent_id
        """
        self.parent_hits = parent_hits
        self.child_hits = child_hits

    @property
    def total_parents(self) -> int:
        """Общее количество родительских документов."""
        return len(self.parent_hits)

    @property
    def total_children(self) -> int:
        """Общее количество дочерних документов."""
        return sum(len(children) for children in self.child_hits.values())

    @property
    def all_children(self) -> List[ChildHit]:
        """Все дочерние документы в одном списке."""
        return [
            child
            for children in self.child_hits.values()
            for child in children
        ]

    def get_children_for_parent(self, parent_id: str) -> List[ChildHit]:
        """Получает дочерние документы для указанного родителя.

        Args:
            parent_id: ID родительского документа

        Returns:
            Список дочерних документов или пустой список если не найдено
        """
        return self.child_hits.get(parent_id, [])

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует результаты в словарь.

        Returns:
            Словарь с результатами поиска
        """
        return {
            "parent_hits": [hit.to_dict() for hit in self.parent_hits],
            "child_hits": {
                parent_id: [child.to_dict() for child in children]
                for parent_id, children in self.child_hits.items()
            },
            "stats": {
                "total_parents": self.total_parents,
                "total_children": self.total_children
            }
        }

    def __str__(self) -> str:
        """Строковое представление результатов."""
        return (
            f"VectorSearchResult(parents={self.total_parents}, "
            f"children={self.total_children})"
        )
