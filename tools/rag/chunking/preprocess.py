from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple, Pattern, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Конфигурация препроцессинга текста."""
    remove_license_header: bool = True
    remove_leading_h1: bool = True
    normalize_whitespace: bool = True
    fix_word_hyphenation: bool = True
    remove_emojis: bool = True
    remove_control_chars: bool = True
    normalize_list_bullets: bool = True
    ensure_trailing_newline: bool = True


class TextPreprocessor:
    """
    Препроцессор для очистки и нормализации Markdown/MDX текста.

    Выполняет:
    - Удаление лицензионных заголовков
    - Нормализацию пробелов и переносов строк
    - Склейку переносов слов
    - Удаление эмодзи и контрольных символов
    - Нормализацию маркеров списков
    """

    # Регулярные выражения для обработки текста
    _HEAD_LICENSE_RE: Pattern = re.compile(r"^\s*<!--.*?-->\s*\n?", re.DOTALL)
    _CODE_FENCE_RE: Pattern = re.compile(r"```.*?```", re.DOTALL)
    _TILDE_FENCE_RE: Pattern = re.compile(r"~~~.*?~~~", re.DOTALL)
    _WS_HARD_RE: Pattern = re.compile(r"[ \t]+\n")
    _MANY_NL_RE: Pattern = re.compile(r"\n{3,}")
    _SOFT_HYPHEN_MIDWORD_RE: Pattern = re.compile(r"(?<!`)\b(\w+)-\n(\w+)\b(?!`)")
    _LIST_BULLET_PREFIX_RE: Pattern = re.compile(r"(?m)^\s*([-*]|\d+\.)\s+")
    _FIRST_H1_RE: Pattern = re.compile(r"^\s*#\s+.+\n+", re.MULTILINE)

    # Диапазоны Unicode для эмодзи и контрольных символов
    _EMOJI_RANGES: Pattern = re.compile(
        r"[\U0001F1E6-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]"
    )
    _CONTROL_CHARS: Pattern = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")
    _EMOJI_MODIFIERS: Pattern = re.compile(r"[\uFE0F\u200D\U0001F3FB-\U0001F3FF]")

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Инициализация препроцессора.

        Args:
            config: Конфигурация препроцессинга. Если None, используется конфиг по умолчанию.
        """
        self.config = config or PreprocessingConfig()
        logger.debug("TextPreprocessor initialized with config: %s", self.config)

    def _split_code_aware(self, text: str) -> List[Tuple[str, bool]]:
        """
        Разделяет текст на сегменты, различая код и обычный текст.

        Args:
            text: Исходный текст для разделения

        Returns:
            Список кортежей (segment, is_code), где is_code указывает, является ли сегмент кодом
        """
        if not text:
            return []

        segments: List[Tuple[str, bool]] = []
        last_position = 0

        # Обрабатываем блоки кода с ```
        for match in self._CODE_FENCE_RE.finditer(text):
            # Текст до блока кода
            if match.start() > last_position:
                segments.append((text[last_position:match.start()], False))

            # Сам блок кода
            segments.append((match.group(0), True))
            last_position = match.end()

        # Обрабатываем оставшийся текст и блоки с ~~~
        remaining_text = text[last_position:]
        last_position = 0
        temp_segments: List[Tuple[str, bool]] = []

        for match in self._TILDE_FENCE_RE.finditer(remaining_text):
            # Текст до блока кода
            if match.start() > last_position:
                temp_segments.append((remaining_text[last_position:match.start()], False))

            # Сам блок кода
            temp_segments.append((match.group(0), True))
            last_position = match.end()

        # Добавляем оставшийся хвост
        if last_position < len(remaining_text):
            temp_segments.append((remaining_text[last_position:], False))

        segments.extend(temp_segments)
        return segments

    def _remove_emojis_and_control_chars(self, text: str) -> str:
        """
        Удаляет эмодзи и контрольные символы из текста.

        Args:
            text: Текст для очистки

        Returns:
            Очищенный текст
        """
        if not text:
            return text

        if self.config.remove_emojis:
            text = self._EMOJI_RANGES.sub("", text)
            text = self._EMOJI_MODIFIERS.sub("", text)

        if self.config.remove_control_chars:
            text = self._CONTROL_CHARS.sub("", text)

        return text

    def _normalize_text_segment(self, text: str, is_code: bool) -> str:
        """
        Нормализует сегмент текста в зависимости от его типа.

        Args:
            text: Текст сегмента
            is_code: Является ли сегмент кодом

        Returns:
            Нормализованный текст сегмента
        """
        if is_code:
            return text  # Код не нормализуем

        normalized = text

        # Удаление эмодзи и контрольных символов
        normalized = self._remove_emojis_and_control_chars(normalized)

        # Нормализация пробелов и символов
        if self.config.normalize_whitespace:
            normalized = normalized.replace("\u00A0", " ")  # NBSP -> пробел
            normalized = self._WS_HARD_RE.sub("\n", normalized)
            normalized = self._MANY_NL_RE.sub("\n\n", normalized)

        # Склейка переносов слов
        if self.config.fix_word_hyphenation:
            normalized = self._SOFT_HYPHEN_MIDWORD_RE.sub(r"\1\2", normalized)

        # Нормализация маркеров списков
        if self.config.normalize_list_bullets:
            normalized = self._LIST_BULLET_PREFIX_RE.sub("\n• ", normalized)

        return normalized

    def preprocess(self, text: str) -> str:
        """
        Основной метод препроцессинга текста.

        Args:
            text: Исходный текст для обработки

        Returns:
            Обработанный и нормализованный текст

        Raises:
            ValueError: Если текст None
        """
        if text is None:
            raise ValueError("Input text cannot be None")

        if not text.strip():
            logger.debug("Received empty or whitespace-only text")
            return text

        original_length = len(text)
        logger.debug("Starting preprocessing of text (length: %d)", original_length)

        # Удаление лицензионного заголовка
        if self.config.remove_license_header:
            text = self._HEAD_LICENSE_RE.sub("", text, count=1)
            logger.debug("Removed license header if present")

        # Разделение на сегменты (код/текст)
        segments = self._split_code_aware(text)
        logger.debug("Split text into %d segments", len(segments))

        processed_segments: List[str] = []
        leading_h1_removed = False

        for segment, is_code in segments:
            if is_code:
                processed_segments.append(segment)
                continue

            # Удаление первого H1 заголовка (только один раз)
            if self.config.remove_leading_h1 and not leading_h1_removed:
                segment = self._FIRST_H1_RE.sub("", segment, count=1)
                leading_h1_removed = True
                logger.debug("Removed leading H1 header if present")

            # Нормализация текстового сегмента
            normalized_segment = self._normalize_text_segment(segment, is_code)
            processed_segments.append(normalized_segment)

        # Сборка результата
        result = "".join(processed_segments).strip()

        # Добавление завершающего переноса строки
        if self.config.ensure_trailing_newline and result and not result.endswith("\n"):
            result += "\n"

        final_length = len(result)
        logger.debug(
            "Preprocessing completed. Original length: %d, Final length: %d, Reduction: %.1f%%",
            original_length, final_length,
            ((original_length - final_length) / original_length * 100) if original_length else 0
        )

        return result


# Глобальный экземпляр препроцессора для обратной совместимости
_DEFAULT_PREPROCESSOR = TextPreprocessor()


def preprocess_text(text: str) -> str:
    """
    Функция для обратной совместимости.
    Выполняет препроцессинг текста с настройками по умолчанию.

    Args:
        text: Исходный текст для обработки

    Returns:
        Обработанный и нормализованный текст
    """
    return _DEFAULT_PREPROCESSOR.preprocess(text)


# Утилитарные функции
def create_preprocessor(config: PreprocessingConfig) -> TextPreprocessor:
    """
    Фабричная функция для создания препроцессора с заданной конфигурацией.

    Args:
        config: Конфигурация препроцессинга

    Returns:
        Настроенный экземпляр TextPreprocessor
    """
    return TextPreprocessor(config)


def get_default_preprocessor() -> TextPreprocessor:
    """
    Возвращает препроцессор с настройками по умолчанию.

    Returns:
        Стандартный экземпляр TextPreprocessor
    """
    return _DEFAULT_PREPROCESSOR
