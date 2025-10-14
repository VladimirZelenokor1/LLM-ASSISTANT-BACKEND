# path: tools/rag/sources/pt_portal/normalize/strategies.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from slugify import slugify

from ..storage.schema import PTDocument

logger = logging.getLogger(__name__)

# Регулярные выражения для нормализации
_WHITESPACE_RX = re.compile(r"[ \t]+")
_MULTIPLE_NEWLINES_RX = re.compile(r"\n{3,}")
_HEADING_RX = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
_WORD_COUNT_RX = re.compile(r"\w+")

# Константы для разделов по умолчанию
_DEFAULT_SECTION_PREFIX = "/"
_UNKNOWN_SECTION = "/unknown/"


@dataclass
class GenericNormalizer:
    """Универсальный нормализатор для markdown и PDF документов.

    Выполняет:
    - Очистку пробелов и множественных переносов строк
    - Генерацию anchor по последнему заголовку H2/H3
    - Определение секции на основе repo_path
    - Подсчет статистики содержимого
    """

    def _guess_anchor(self, text: str) -> Optional[str]:
        """Определяет anchor на основе последнего заголовка H2 или H3.

        Args:
            text: Текст документа для анализа

        Returns:
            Строка anchor или None если заголовки не найдены
        """
        if not text or not isinstance(text, str):
            return None

        last_heading = None
        for match in _HEADING_RX.finditer(text):
            last_heading = match

        if not last_heading:
            logger.debug("No H2/H3 headings found for anchor generation")
            return None

        heading_text = last_heading.group(2).strip()
        anchor = slugify(heading_text)

        if not anchor:
            logger.warning(f"Failed to generate anchor from heading: '{heading_text}'")
            return None

        logger.debug(f"Generated anchor '{anchor}' from heading: '{heading_text}'")
        return anchor

    def _section_from_repo_path(self, repo_path: Optional[str]) -> Optional[str]:
        """Извлекает секцию из пути репозитория.

        Args:
            repo_path: Путь к файлу в репозитории

        Returns:
            Нормализованный путь секции или None
        """
        if not repo_path:
            return None

        # Нормализуем разделители путей
        normalized_path = repo_path.replace("\\", "/")
        parts = [part for part in normalized_path.split("/") if part]

        # Ищем первую не-скрытую директорию (не начинающуюся с _)
        for part in parts:
            if part and not part.startswith("_"):
                section = f"{_DEFAULT_SECTION_PREFIX}{part}{_DEFAULT_SECTION_PREFIX}"
                logger.debug(f"Extracted section '{section}' from repo_path: {repo_path}")
                return section

        logger.warning(f"Could not extract section from repo_path: {repo_path}")
        return _UNKNOWN_SECTION

    def _clean_text(self, text: str) -> str:
        """Очищает текст от лишних пробелов и переносов строк.

        Args:
            text: Исходный текст

        Returns:
            Очищенный текст
        """
        if not text:
            return ""

        # Заменяем множественные пробелы и табы на один пробел
        cleaned = _WHITESPACE_RX.sub(" ", text)
        # Заменяем 3+ переноса строк на двойные
        cleaned = _MULTIPLE_NEWLINES_RX.sub("\n\n", cleaned)

        return cleaned.strip()

    def _calculate_content_stats(self, text: str) -> tuple[int, int]:
        """Вычисляет статистику содержимого.

        Args:
            text: Текст для анализа

        Returns:
            Кортеж (количество_символов, количество_слов)
        """
        char_count = len(text) if text else 0
        word_count = len(_WORD_COUNT_RX.findall(text)) if text else 0

        return char_count, word_count

    def normalize(self, doc: PTDocument) -> PTDocument:
        """Нормализует документ, применяя все стратегии очистки.

        Args:
            doc: Документ для нормализации

        Returns:
            Нормализованный документ

        Raises:
            ValueError: Если документ или его текст невалидны
        """
        if not doc or not isinstance(doc, PTDocument):
            raise ValueError("Invalid document: must be PTDocument instance")

        logger.info(f"Normalizing document: {doc.meta.repo_path or 'unknown'}")

        # Очистка текста
        original_text = doc.text or ""
        cleaned_text = self._clean_text(original_text)

        if not cleaned_text:
            logger.warning("Document text is empty after cleaning")
            doc.text = ""
        else:
            doc.text = cleaned_text

        # Генерация anchor если не установлен
        if not doc.meta.anchor and cleaned_text:
            anchor = self._guess_anchor(cleaned_text)
            if anchor:
                doc.meta.anchor = anchor
                logger.debug(f"Set anchor to: {anchor}")

        # Определение секции если не установлена
        if not doc.meta.section:
            section = self._section_from_repo_path(doc.meta.repo_path)
            if section:
                doc.meta.section = section
                logger.debug(f"Set section to: {section}")

        # Обновление статистики содержимого
        char_count, word_count = self._calculate_content_stats(doc.text)
        doc.meta.content_length_chars = char_count
        doc.meta.content_words = word_count

        logger.info(
            f"Normalization complete: {word_count} words, {char_count} chars, "
            f"anchor: {doc.meta.anchor or 'none'}, section: {doc.meta.section or 'none'}"
        )

        return doc

    def normalize_batch(self, documents: list[PTDocument]) -> list[PTDocument]:
        """Нормализует пакет документов.

        Args:
            documents: Список документов для нормализации

        Returns:
            Список нормализованных документов
        """
        logger.info(f"Normalizing batch of {len(documents)} documents")

        normalized_docs = []
        for i, doc in enumerate(documents):
            try:
                normalized_doc = self.normalize(doc)
                normalized_docs.append(normalized_doc)
            except Exception as error:
                logger.error(f"Failed to normalize document {i}: {error}")
                # Сохраняем оригинальный документ в случае ошибки
                normalized_docs.append(doc)

        logger.info(f"Successfully normalized {len(normalized_docs)} documents")
        return normalized_docs
