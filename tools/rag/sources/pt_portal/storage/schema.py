# path: tools/rag/sources/pt_portal/storage/schema.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class PTMeta(BaseModel):
    """Метаданные документа для портала PT.

    Содержит информацию об идентификации, источниках, контенте и статистике документа.
    """

    # Основная идентификация
    product: str = Field(..., description="Название продукта")
    product_code: str = Field(..., description="Код продукта для группировки")
    version: str = Field(..., description="Версия документации")
    lang: str = Field(..., description="Язык документации")

    # Адреса и ссылки
    url: Optional[str] = Field(None, description="URL исходного документа")
    canonical_url: Optional[str] = Field(None, description="Канонический URL документа")

    # Заголовки и навигация
    title: Optional[str] = Field(None, description="Заголовок документа")
    section: Optional[str] = Field(None, description="Секция документа (например: '/model_doc/', '/tasks/')")
    anchor: Optional[str] = Field(None, description="Якорь для внутренней навигации")

    # Тип контента и источник
    content_type: Literal["md", "text"] = Field("md", description="Тип контента")
    source_format: Literal["markdown"] = Field("markdown", description="Формат исходного документа")
    source_kind: Literal["repo"] = Field("repo", description="Тип источника")

    # Метаданные репозитория
    repo_remote: Optional[str] = Field(None, description="URL удаленного репозитория")
    repo_commit: Optional[str] = Field(None, description="Хэш коммита репозитория")
    repo_path: Optional[str] = Field(None, description="Относительный путь в репозитории")

    # Хэши и временные метки
    hash_raw: Optional[str] = Field(None, description="Хэш исходного контента")
    hash_text: Optional[str] = Field(None, description="Хэш обработанного текста")
    fetched_at: Optional[datetime] = Field(None, description="Время получения документа")
    normalized_at: Optional[datetime] = Field(None, description="Время нормализации документа")

    # Статистика контента
    content_length_chars: Optional[int] = Field(None, description="Длина текста в символах")
    content_words: Optional[int] = Field(None, description="Количество слов в тексте")
    has_tables: bool = Field(False, description="Содержит ли документ таблицы")
    has_code_blocks: bool = Field(False, description="Содержит ли документ блоки кода")

    # Дополнительные метаданные
    extra: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные метаданные")

    @validator("product", "product_code", "version", "lang")
    def validate_required_fields(cls, value: str) -> str:
        """Валидирует обязательные текстовые поля."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return value.strip()

    @validator("content_length_chars", "content_words")
    def validate_positive_integers(cls, value: Optional[int]) -> Optional[int]:
        """Валидирует, что числовые значения положительные."""
        if value is not None and value < 0:
            raise ValueError("Value must be non-negative")
        return value

    def update_timestamp(self, field: str = "normalized_at") -> None:
        """Обновляет временную метку указанного поля.

        Args:
            field: Название поля для обновления (fetched_at, normalized_at)
        """
        if hasattr(self, field):
            setattr(self, field, datetime.now())
        else:
            logger.warning(f"Unknown timestamp field: {field}")

    def add_extra_field(self, key: str, value: Any) -> None:
        """Добавляет поле в дополнительные метаданные.

        Args:
            key: Ключ для сохранения
            value: Значение для сохранения
        """
        self.extra[key] = value

    def get_extra_field(self, key: str, default: Any = None) -> Any:
        """Получает значение из дополнительных метаданных.

        Args:
            key: Ключ для поиска
            default: Значение по умолчанию если ключ не найден

        Returns:
            Значение из extra или default
        """
        return self.extra.get(key, default)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class PTDocument(BaseModel):
    """Документ портала PT с текстом и метаданными."""

    text: str = Field(..., description="Текст документа")
    meta: PTMeta = Field(..., description="Метаданные документа")
    source_ref: Optional[str] = Field(None, description="Ссылка на исходный источник")

    @validator("text")
    def validate_text(cls, value: str) -> str:
        """Валидирует, что текст не пустой."""
        if not value or not value.strip():
            raise ValueError("Document text cannot be empty")
        return value.strip()

    @property
    def id(self) -> str:
        """Генерирует идентификатор документа на основе метаданных.

        Returns:
            Уникальный идентификатор документа
        """
        components = [
            self.meta.product_code,
            self.meta.version,
            self.meta.lang,
            self.meta.repo_path or "unknown",
            self.meta.hash_text or hash(self.text) % 1000000
        ]
        return "_".join(str(c) for c in components if c)

    def get_content_stats(self) -> Dict[str, Any]:
        """Возвращает статистику содержимого документа.

        Returns:
            Словарь со статистикой
        """
        return {
            "characters": len(self.text),
            "words": len(self.text.split()),
            "lines": self.text.count('\n') + 1,
            "has_tables": self.meta.has_tables,
            "has_code_blocks": self.meta.has_code_blocks,
        }

    def to_dict(self, include_text: bool = True) -> Dict[str, Any]:
        """Конвертирует документ в словарь.

        Args:
            include_text: Включать ли текст в результат

        Returns:
            Словарь с данными документа
        """
        data = {
            "meta": self.meta.dict(),
            "source_ref": self.source_ref,
            "id": self.id
        }

        if include_text:
            data["text"] = self.text

        return data

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class IngestionError(BaseModel):
    """Ошибка процесса ингрессии документа."""

    url: Optional[str] = Field(None, description="URL связанного документа")
    stage: str = Field(..., description="Стадия процесса, на которой произошла ошибка")
    message: str = Field(..., description="Сообщение об ошибке")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время возникновения ошибки")
    details: Dict[str, Any] = Field(default_factory=dict, description="Дополнительные детали ошибки")

    @validator("stage", "message")
    def validate_required_strings(cls, value: str) -> str:
        """Валидирует обязательные строковые поля."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value.strip()

    def to_log_dict(self) -> Dict[str, Any]:
        """Конвертирует ошибку в словарь для логирования.

        Returns:
            Словарь с данными ошибки
        """
        return {
            "stage": self.stage,
            "message": self.message,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }


class IngestionStats(BaseModel):
    """Статистика процесса ингрессии документов."""

    sources_total: int = Field(0, description="Общее количество источников")
    processed_ok: int = Field(0, description="Успешно обработанных документов")
    processed_err: int = Field(0, description="Документов с ошибками обработки")

    # Статистика репозиториев
    repo_files_found: int = Field(0, description="Найдено файлов в репозиториях")
    repo_files_parsed: int = Field(0, description="Распарсено файлов из репозиториев")

    # Статистика дедупликации и сохранения
    dedup_dropped_near: int = Field(0, description="Отброшено почти дубликатов")
    persisted_processed: int = Field(0, description="Сохранено обработанных документов")

    errors: List[IngestionError] = Field(default_factory=list, description="Список ошибок")

    @property
    def success_rate(self) -> float:
        """Вычисляет процент успешной обработки.

        Returns:
            Процент успешной обработки от 0.0 до 1.0
        """
        total = self.processed_ok + self.processed_err
        return self.processed_ok / total if total > 0 else 0.0

    @property
    def repo_parse_rate(self) -> float:
        """Вычисляет процент успешного парсинга репозиториев.

        Returns:
            Процент успешного парсинга от 0.0 до 1.0
        """
        if self.repo_files_found == 0:
            return 0.0
        return self.repo_files_parsed / self.repo_files_found

    def add_error(
            self,
            stage: str,
            message: str,
            url: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Добавляет ошибку в статистику.

        Args:
            stage: Стадия процесса
            message: Сообщение об ошибке
            url: URL документа (опционально)
            details: Дополнительные детали (опционально)
        """
        error = IngestionError(
            stage=stage,
            message=message,
            url=url,
            details=details or {}
        )
        self.errors.append(error)
        logger.error(f"Ingestion error at {stage}: {message}")

    def merge(self, other: IngestionStats) -> None:
        """Объединяет статистику с другой статистикой.

        Args:
            other: Другая статистика для объединения
        """
        self.sources_total += other.sources_total
        self.processed_ok += other.processed_ok
        self.processed_err += other.processed_err
        self.repo_files_found += other.repo_files_found
        self.repo_files_parsed += other.repo_files_parsed
        self.dedup_dropped_near += other.dedup_dropped_near
        self.persisted_processed += other.persisted_processed
        self.errors.extend(other.errors)

    def as_markdown(self) -> str:
        """Форматирует статистику в виде Markdown.

        Returns:
            Строка Markdown со статистикой
        """
        lines = [
            "# Ingestion Statistics",
            "",
            f"- **Total Sources**: {self.sources_total}",
            f"- **Repository Files**: found={self.repo_files_found}, parsed={self.repo_files_parsed}",
            f"- **Processing**: OK={self.processed_ok}, Errors={self.processed_err}",
            f"- **Success Rate**: {self.success_rate:.1%}",
            f"- **Near Duplicates Dropped**: {self.dedup_dropped_near}",
            f"- **Persisted Documents**: {self.persisted_processed}",
            f"- **Repository Parse Rate**: {self.repo_parse_rate:.1%}",
        ]

        if self.errors:
            lines.extend([
                "",
                "## Errors",
                *[f"- {error.stage}: {error.message}" for error in self.errors[:10]],
                f"... and {len(self.errors) - 10} more" if len(self.errors) > 10 else ""
            ])

        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        """Конвертирует статистику в словарь.

        Returns:
            Словарь со статистикой
        """
        return {
            "sources_total": self.sources_total,
            "processed_ok": self.processed_ok,
            "processed_err": self.processed_err,
            "repo_files_found": self.repo_files_found,
            "repo_files_parsed": self.repo_files_parsed,
            "dedup_dropped_near": self.dedup_dropped_near,
            "persisted_processed": self.persisted_processed,
            "success_rate": self.success_rate,
            "repo_parse_rate": self.repo_parse_rate,
            "error_count": len(self.errors),
            "errors": [error.dict() for error in self.errors]
        }

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
