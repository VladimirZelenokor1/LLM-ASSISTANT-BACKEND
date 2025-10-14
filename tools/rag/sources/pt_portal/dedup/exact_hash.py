# path: tools/rag/sources/pt_portal/dedup/exact_hash.py
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Константа для размера чанка при чтении файлов (1 МБ)
_CHUNK_SIZE = 1024 * 1024


def text_hash(text: str) -> str:
    """Вычисляет SHA256 хэш текстовой строки.

    Args:
        text: Входная текстовая строка для хэширования

    Returns:
        SHA256 хэш в виде hex-строки

    Example:
        >>> text_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    if not text:
        logger.warning("Computing hash for empty string")

    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def file_hash(file_path: str | Path) -> str:
    """Вычисляет SHA256 хэш файла, читая его чанками.

    Эффективно обрабатывает большие файлы, не загружая их полностью в память.

    Args:
        file_path: Путь к файлу для хэширования

    Returns:
        SHA256 хэш файла в виде hex-строки

    Raises:
        FileNotFoundError: Если файл не существует
        PermissionError: Если нет прав для чтения файла
        OSError: При других ошибках ввода-вывода

    Example:
        >>> file_hash("/path/to/file.pdf")
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    hasher = hashlib.sha256()

    try:
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(_CHUNK_SIZE), b""):
                hasher.update(chunk)

        logger.debug(f"Computed hash for file: {file_path}")
        return hasher.hexdigest()

    except PermissionError as error:
        logger.error(f"Permission denied reading file: {file_path}")
        raise PermissionError(f"Cannot read file: {file_path}") from error
    except OSError as error:
        logger.error(f"OS error reading file {file_path}: {error}")
        raise OSError(f"Error reading file: {file_path}") from error


class ExactHashDeduplicator:
    """Дубликатор для точного хэширования текстов и файлов."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def is_duplicate_text(self, text: str) -> bool:
        """Проверяет, является ли текст дубликатом.

        Args:
            text: Текст для проверки

        Returns:
            True если текст уже встречался, иначе False
        """
        hash_value = text_hash(text)
        is_duplicate = hash_value in self._seen_hashes

        if not is_duplicate:
            self._seen_hashes.add(hash_value)

        return is_duplicate

    def is_duplicate_file(self, file_path: str | Path) -> bool:
        """Проверяет, является ли файл дубликатом.

        Args:
            file_path: Путь к файлу для проверки

        Returns:
            True если файл уже встречался, иначе False
        """
        hash_value = file_hash(file_path)
        is_duplicate = hash_value in self._seen_hashes

        if not is_duplicate:
            self._seen_hashes.add(hash_value)

        return is_duplicate

    def add_text(self, text: str) -> bool:
        """Добавляет текст в коллекцию и возвращает статус дубликата.

        Args:
            text: Текст для добавления

        Returns:
            True если текст уже был в коллекции, иначе False
        """
        return self.is_duplicate_text(text)

    def add_file(self, file_path: str | Path) -> bool:
        """Добавляет файл в коллекцию и возвращает статус дубликата.

        Args:
            file_path: Путь к файлу для добавления

        Returns:
            True если файл уже был в коллекции, иначе False
        """
        return self.is_duplicate_file(file_path)

    def get_seen_count(self) -> int:
        """Возвращает количество уникальных хэшей в коллекции."""
        return len(self._seen_hashes)

    def clear(self) -> None:
        """Очищает коллекцию хэшей."""
        self._seen_hashes.clear()
        logger.info("Cleared all stored hashes")
