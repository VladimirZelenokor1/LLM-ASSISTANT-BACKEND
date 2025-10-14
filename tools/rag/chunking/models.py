from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class ChunkUnit(str, Enum):
    """Единицы измерения размера чанка."""
    TOKENS = "tokens"
    CHARACTERS = "chars"


class SplitterType(str, Enum):
    """Типы сплиттеров."""
    RULE_AWARE = "rule_aware"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"


# Собственный namespace для стабильных UUIDv5
_CHUNK_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-assistant/rag/chunk")


def generate_short_hash(text: str, length: int = 16) -> str:
    """
    Генерирует короткий хеш из текста.

    Args:
        text: Исходный текст для хеширования
        length: Длина хеша (по умолчанию 16 символов)

    Returns:
        Хеш в виде hex-строки
    """
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def build_chunk_id(meta: Dict[str, Any], text_fingerprint: Optional[str] = None) -> str:
    """
    Генерирует детерминированный UUIDv5 на основе метаданных.

    Args:
        meta: Метаданные чанка
        text_fingerprint: Опциональный отпечаток текста для случаев без seq

    Returns:
        UUIDv5 в виде строки
    """

    def _normalize_h_path(h_path: Any) -> str:
        """Нормализует путь заголовков."""
        if isinstance(h_path, list):
            return ">".join(str(x) for x in h_path if x)
        return str(h_path or "")

    # Собираем стабильные части для генерации ID
    stable_parts = [
        str(meta.get("product") or ""),
        str(meta.get("product_code") or ""),
        str(meta.get("version") or ""),
        str(meta.get("lang") or ""),
        str(meta.get("repo_path") or meta.get("canonical_url") or meta.get("url") or ""),
        str(meta.get("section") or ""),
        _normalize_h_path(meta.get("h_path")),
    ]

    # Обрабатываем последовательный номер
    sequence_number = meta.get("seq")
    if sequence_number is None or str(sequence_number).strip() == "":
        # Fallback для чанков без seq
        preview_text = str(meta.get("_preview") or "")
        fingerprint = text_fingerprint or generate_short_hash(preview_text)
        stable_parts.extend(["no-seq", fingerprint])
    else:
        stable_parts.append(str(sequence_number))

    # Генерируем UUIDv5
    base_string = "|".join(stable_parts)
    return str(uuid.uuid5(_CHUNK_NAMESPACE, base_string))


class ChunkMeta(BaseModel):
    """
    Метаданные чанка текста.

    Attributes:
        seq: Порядковый номер чанка в документе (начиная с 1)
        unit: Единица измерения бюджета ("tokens" | "chars")
        chunk_size: Фактический размер чанка в выбранных единицах
        char_start: Начальная позиция в исходном тексте (символы)
        char_end: Конечная позиция в исходном тексте (символы)
        sent_start_idx: Индекс начального предложения (опционально)
        sent_end_idx: Индекс конечного предложения (опционально)
        splitter_type: Тип использованного сплиттера
        h_path: Иерархический путь заголовков для навигации
        has_code: Признак наличия программного кода
        has_table: Признак наличия таблиц
        extra: Дополнительные поля для расширения метаданных
    """
    model_config = ConfigDict(
        extra="allow",  # Разрешаем extra-поля для документных метаданных
        validate_assignment=True,
        str_strip_whitespace=True
    )

    # Основные метаданные
    seq: int = Field(default=0, ge=0, description="Порядковый номер чанка")
    unit: str = Field(default=ChunkUnit.TOKENS, description="Единица измерения")
    chunk_size: int = Field(default=0, ge=0, description="Размер чанка в единицах")

    # Позиция в тексте
    char_start: int = Field(default=0, ge=0, description="Начальная позиция в символах")
    char_end: int = Field(default=0, ge=0, description="Конечная позиция в символах")

    # Индексы предложений (опционально)
    sent_start_idx: Optional[int] = Field(default=None, ge=0, description="Индекс начального предложения")
    sent_end_idx: Optional[int] = Field(default=None, ge=0, description="Индекс конечного предложения")

    # Тип сплиттера и структура
    splitter_type: str = Field(default=SplitterType.RULE_AWARE, description="Тип сплиттера")
    h_path: List[str] = Field(default_factory=list, description="Иерархический путь заголовков")

    # Признаки содержимого
    has_code: bool = Field(default=False, description="Содержит программный код")
    has_table: bool = Field(default=False, description="Содержит таблицы")

    # Дополнительные поля
    extra: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")

    def get_extra_value(self, key: str, default: Any = None) -> Any:
        """Безопасно извлекает значение из extra."""
        return self.extra.get(key, default)

    def set_extra_value(self, key: str, value: Any) -> None:
        """Устанавливает значение в extra."""
        self.extra[key] = value

    def update_extra(self, updates: Dict[str, Any]) -> None:
        """Обновляет multiple values в extra."""
        self.extra.update(updates)


class ChunkRecord(BaseModel):
    """
    Модель чанка текста с метаданными.

    Attributes:
        text: Текст чанка
        meta: Метаданные чанка
    """
    model_config = ConfigDict(
        validate_assignment=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False
    )

    text: str = Field(..., min_length=1, description="Текст чанка")
    meta: ChunkMeta = Field(default_factory=ChunkMeta, description="Метаданные чанка")

    @property
    def chunk_id(self) -> Optional[str]:
        """Быстрый доступ к chunk_id из extra."""
        return self.meta.get_extra_value("chunk_id")

    @property
    def document_id(self) -> Optional[str]:
        """Быстрый доступ к document_id из extra."""
        return self.meta.get_extra_value("document_id")

    def calculate_text_hash(self, hash_length: int = 16) -> str:
        """Вычисляет хеш текста чанка."""
        return generate_short_hash(self.text, hash_length)

    def to_serializable_dict(self) -> Dict[str, Any]:
        """
        Конвертирует чанк в сериализуемый словарь.

        Returns:
            Словарь, готовый для JSON сериализации
        """
        return {
            "text": self.text,
            "meta": self.meta.model_dump(mode="json")
        }


# Псевдоним для обратной совместимости
Chunk = ChunkRecord


@dataclass
class ChunkingStatistics:
    """Статистика процесса чанкинга."""
    documents_total: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    chunks_total: int = 0
    chunks_with_code: int = 0
    chunks_with_table: int = 0
    total_chars_processed: int = 0

    @property
    def success_rate(self) -> float:
        """Процент успешно обработанных документов."""
        if self.documents_total == 0:
            return 0.0
        return (self.documents_processed / self.documents_total) * 100

    @property
    def avg_chars_per_chunk(self) -> float:
        """Среднее количество символов на чанк."""
        if self.chunks_total == 0:
            return 0.0
        return self.total_chars_processed / self.chunks_total

    @property
    def code_chunk_ratio(self) -> float:
        """Доля чанков с кодом."""
        if self.chunks_total == 0:
            return 0.0
        return self.chunks_with_code / self.chunks_total

    @property
    def table_chunk_ratio(self) -> float:
        """Доля чанков с таблицами."""
        if self.chunks_total == 0:
            return 0.0
        return self.chunks_with_table / self.chunks_total


class ChunkingReport(BaseModel):
    """
    Отчет о процессе чанкинга документов.

    Attributes:
        statistics: Детальная статистика процесса
        unit: Единица измерения (для обратной совместимости)
        notes: Дополнительные заметки или ошибки
        processing_time_seconds: Время обработки в секундах
    """
    statistics: ChunkingStatistics = Field(default_factory=ChunkingStatistics)
    unit: str = Field(default=ChunkUnit.TOKENS, description="Единица измерения")
    notes: Optional[str] = Field(default=None, description="Дополнительные заметки")
    processing_time_seconds: Optional[float] = Field(default=None, description="Время обработки")

    # Свойства для обратной совместимости
    @property
    def documents_total(self) -> int:
        return self.statistics.documents_total

    @documents_total.setter
    def documents_total(self, value: int) -> None:
        self.statistics.documents_total = value

    @property
    def documents_processed(self) -> int:
        return self.statistics.documents_processed

    @documents_processed.setter
    def documents_processed(self, value: int) -> None:
        self.statistics.documents_processed = value

    @property
    def chunks_total(self) -> int:
        return self.statistics.chunks_total

    @chunks_total.setter
    def chunks_total(self, value: int) -> None:
        self.statistics.chunks_total = value

    @property
    def chunks_with_code(self) -> int:
        return self.statistics.chunks_with_code

    @chunks_with_code.setter
    def chunks_with_code(self, value: int) -> None:
        self.statistics.chunks_with_code = value

    @property
    def chunks_with_table(self) -> int:
        return self.statistics.chunks_with_table

    @chunks_with_table.setter
    def chunks_with_table(self, value: int) -> None:
        self.statistics.chunks_with_table = value

    @property
    def avg_len_chars(self) -> float:
        return self.statistics.avg_chars_per_chunk

    @avg_len_chars.setter
    def avg_len_chars(self, value: float) -> None:
        self.statistics.total_chars_processed = int(value * self.statistics.chunks_total)

    def as_markdown(self) -> str:
        """
        Форматирует отчет в Markdown.

        Returns:
            Строка в формате Markdown
        """
        stats = self.statistics
        return (
            "## Chunking Report\n\n"
            f"- **Documents Processed**: {stats.documents_processed}/{stats.documents_total} "
            f"({stats.success_rate:.1f}%)\n"
            f"- **Total Chunks**: {stats.chunks_total}\n"
            f"- **Chunks with Code**: {stats.chunks_with_code} ({stats.code_chunk_ratio:.1%})\n"
            f"- **Chunks with Tables**: {stats.chunks_with_table} ({stats.table_chunk_ratio:.1%})\n"
            f"- **Average Length**: {stats.avg_chars_per_chunk:.1f} chars\n"
            f"- **Unit**: {self.unit}\n"
            f"{f'- **Processing Time**: {self.processing_time_seconds:.2f}s' if self.processing_time_seconds else ''}"
            f"{f'- **Notes**: {self.notes}' if self.notes else ''}"
        )

    def as_dict(self) -> Dict[str, Any]:
        """Конвертирует отчет в словарь."""
        return {
            "statistics": {
                "documents_total": self.statistics.documents_total,
                "documents_processed": self.statistics.documents_processed,
                "documents_failed": self.statistics.documents_failed,
                "chunks_total": self.statistics.chunks_total,
                "chunks_with_code": self.statistics.chunks_with_code,
                "chunks_with_table": self.statistics.chunks_with_table,
                "total_chars_processed": self.statistics.total_chars_processed,
                "avg_chars_per_chunk": self.statistics.avg_chars_per_chunk,
                "success_rate": self.statistics.success_rate,
                "code_chunk_ratio": self.statistics.code_chunk_ratio,
                "table_chunk_ratio": self.statistics.table_chunk_ratio,
            },
            "unit": self.unit,
            "notes": self.notes,
            "processing_time_seconds": self.processing_time_seconds,
        }


# Утилитарные функции для обратной совместимости
def create_chunk_meta(**kwargs) -> ChunkMeta:
    """Фабричная функция для создания ChunkMeta."""
    return ChunkMeta(**kwargs)


def create_chunk_record(text: str, **meta_kwargs) -> ChunkRecord:
    """Фабричная функция для создания ChunkRecord."""
    return ChunkRecord(text=text, meta=ChunkMeta(**meta_kwargs))
