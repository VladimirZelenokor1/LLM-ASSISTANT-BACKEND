from __future__ import annotations

import json
import dataclasses
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ChunkingReport:
    """Отчет о процессе чанкинга."""
    chunks_total: int = 0
    chunks_with_code: int = 0
    chunks_with_table: int = 0
    avg_len_chars: float = 0.0


class PayloadSerializer:
    """Сериализатор payload для различных типов чанков."""

    @staticmethod
    def _get_attribute(obj: Any, attr: str, default: Any = None) -> Any:
        """Безопасно извлекает атрибут из объекта или словаря."""
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    @staticmethod
    def _serialize_metadata(meta: Any) -> Dict[str, Any]:
        """Сериализует метаданные в словарь."""
        if meta is None:
            return {}

        if isinstance(meta, dict):
            return meta.copy()

        if dataclasses.is_dataclass(meta):
            return dataclasses.asdict(meta)

        # Попытка сериализации Pydantic модели
        try:
            if hasattr(meta, 'model_dump'):
                return meta.model_dump()
            elif hasattr(meta, 'dict'):
                return meta.dict()
        except Exception as e:
            logger.debug(f"Failed to serialize metadata with model methods: {e}")

        # Fallback: используем __dict__ или создаем пустой словарь
        return getattr(meta, "__dict__", {}).copy() if hasattr(meta, "__dict__") else {}

    def serialize(self, chunk: Any) -> Dict[str, Any]:
        """Сериализует чанк в стандартный формат payload."""
        if isinstance(chunk, dict):
            # Убедимся, что словарь имеет правильную структуру
            return {
                "text": chunk.get("text", ""),
                "meta": chunk.get("meta", {})
            }

        if dataclasses.is_dataclass(chunk):
            result = dataclasses.asdict(chunk)
            # Сохраняем дополнительные атрибуты, не попавшие в dataclass
            if hasattr(chunk, '__dict__'):
                for key, value in chunk.__dict__.items():
                    if key not in result and not key.startswith('_'):
                        result[key] = value
            return result

        # Для объектных чанков извлекаем текст и метаданные
        text = self._get_attribute(chunk, "text", "")
        meta = self._get_attribute(chunk, "meta", {})

        return {
            "text": text,
            "meta": self._serialize_metadata(meta)
        }


class BaseWriter(ABC):
    """Абстрактный базовый класс для записи чанков."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.serializer = PayloadSerializer()

    @abstractmethod
    def write(self, chunks: List[Any]) -> ChunkingReport:
        """Записывает чанки и возвращает отчет."""
        pass

    def _analyze_chunks(self, chunks: List[Any]) -> Dict[str, Any]:
        """Анализирует чанки и собирает статистику."""
        total_length = 0
        chunks_with_code = 0
        chunks_with_table = 0

        for chunk in chunks:
            # Извлекаем метаданные
            if isinstance(chunk, dict):
                meta = chunk.get("meta", {})
            else:
                meta = getattr(chunk, "meta", {})

            # Проверяем наличие кода и таблиц
            has_code = self.serializer._get_attribute(meta, "has_code", False)
            has_table = self.serializer._get_attribute(meta, "has_table", False)

            if has_code:
                chunks_with_code += 1
            if has_table:
                chunks_with_table += 1

            # Считаем длину текста
            text = self.serializer._get_attribute(chunk, "text", "")
            total_length += len(text)

        return {
            "total_length": total_length,
            "chunks_with_code": chunks_with_code,
            "chunks_with_table": chunks_with_table
        }


class JsonlWriter(BaseWriter):
    """Реализация writer для формата JSONL."""

    def write(self, chunks: List[Any]) -> ChunkingReport:
        """
        Записывает чанки в JSONL файл.

        Args:
            chunks: Список чанков для записи

        Returns:
            ChunkingReport: Отчет о процессе записи
        """
        if not chunks:
            logger.warning(f"No chunks to write to {self.path}")
            return ChunkingReport()

        # Анализируем чанки перед записью
        analysis = self._analyze_chunks(chunks)

        try:
            # Создаем директорию если необходимо
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.path, "w", encoding="utf-8") as file:
                for chunk in chunks:
                    payload = self.serializer.serialize(chunk)
                    json_line = json.dumps(payload, ensure_ascii=False)
                    file.write(json_line + "\n")

            logger.info(f"Successfully wrote {len(chunks)} chunks to {self.path}")

        except IOError as e:
            logger.error(f"Failed to write chunks to {self.path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while writing to {self.path}: {e}")
            raise

        # Создаем отчет
        return ChunkingReport(
            chunks_total=len(chunks),
            chunks_with_code=analysis["chunks_with_code"],
            chunks_with_table=analysis["chunks_with_table"],
            avg_len_chars=analysis["total_length"] / len(chunks) if chunks else 0.0
        )


# Альтернативная реализация для обратной совместимости
def create_jsonl_writer(path: str) -> JsonlWriter:
    """Фабричная функция для создания JsonlWriter (обратная совместимость)."""
    return JsonlWriter(path)


# Fallback для случаев когда модели недоступны
try:
    from .models import Chunk, ChunkingReport as ModelsChunkingReport
except ImportError:
    logger.warning("Models import failed, using fallback implementations")

    # Типы для type checking
    Chunk = Any

    # Используем нашу реализацию ChunkingReport
    ModelsChunkingReport = ChunkingReport
