# path: tools/rag/sources/pt_portal/dedup/near_dup.py
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from rapidfuzz.fuzz import token_set_ratio

logger = logging.getLogger(__name__)

# Константы по умолчанию
_DEFAULT_THRESHOLD = 92
_DEFAULT_MIN_LENGTH = 600
_DEFAULT_TEXT_SAMPLE_LENGTH = 2000
_LENGTH_BUCKET_SIZE = 1000


def _len_bucket(text_length: int) -> str:
    """Вычисляет бакет длины текста.

    Args:
        text_length: Длина текста в символах

    Returns:
        Строка бакета в формате "начало-конец"

    Example:
        >>> _len_bucket(1250)
        "1000-2000"
    """
    bucket_start = (text_length // _LENGTH_BUCKET_SIZE) * _LENGTH_BUCKET_SIZE
    bucket_end = bucket_start + _LENGTH_BUCKET_SIZE
    return f"{bucket_start}-{bucket_end}"


def _section_from_meta(
        url: Optional[str],
        section: Optional[str],
        product_code: Optional[str]
) -> str:
    """Извлекает секцию из метаданных.

    Args:
        url: URL документа
        section: Секция документа
        product_code: Код продукта

    Returns:
        Нормализованное название секции
    """
    if section:
        return section.strip("/").split("/")[0]

    # Извлекаем секцию из URL для случаев отсутствия section
    if url:
        match = re.match(r"^\w+://([^/]+)/", url)
        if match:
            return match.group(1)

    return product_code or "root"


class NearDupChecker:
    """Детектор почти дубликатов на основе rapidfuzz token_set_ratio.

    Использует бакетизацию по: product_code + section + length_bucket.
    """

    def __init__(
            self,
            state_path: str,
            threshold: int = _DEFAULT_THRESHOLD,
            min_len: int = _DEFAULT_MIN_LENGTH
    ) -> None:
        """Инициализация детектора почти дубликатов.

        Args:
            state_path: Путь к файлу состояния
            threshold: Пороговое значение схожести (0-100)
            min_len: Минимальная длина текста для проверки
        """
        self.state_path = Path(state_path)
        self.threshold = threshold
        self.min_len = min_len
        self.reprs: Dict[str, Tuple[str, str]] = {}  # bucket -> (hash, text_sample)

        self._load_state()
        logger.info(
            f"NearDupChecker initialized: threshold={threshold}, "
            f"min_len={min_len}, loaded_reprs={len(self.reprs)}"
        )

    def _load_state(self) -> None:
        """Загружает состояние из файла."""
        if not self.state_path.exists():
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"State file does not exist, creating: {self.state_path}")
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            self.reprs = {key: tuple(value) for key, value in data.get("reprs", {}).items()}
            logger.debug(f"Loaded {len(self.reprs)} representations from {self.state_path}")

        except (json.JSONDecodeError, IOError) as error:
            logger.error(f"Failed to load state from {self.state_path}: {error}")
            self.reprs = {}

    def _save_state(self) -> None:
        """Сохраняет состояние в файл."""
        try:
            data = {"reprs": {key: list(value) for key, value in self.reprs.items()}}

            with open(self.state_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)

            logger.debug(f"Saved {len(self.reprs)} representations to {self.state_path}")

        except IOError as error:
            logger.error(f"Failed to save state to {self.state_path}: {error}")

    def _bucket_key(
            self,
            *,
            text: str,
            url: Optional[str],
            section: Optional[str],
            product_code: Optional[str]
    ) -> str:
        """Генерирует ключ бакета для текста.

        Args:
            text: Текст для бакетизации
            url: URL документа
            section: Секция документа
            product_code: Код продукта

        Returns:
            Ключ бакета в формате "product:section:length_bucket"
        """
        section_name = _section_from_meta(url, section, product_code)
        length_bucket = _len_bucket(len(text))
        product = product_code or "root"

        return f"{product}:{section_name}:{length_bucket}"

    def is_near_dup(
            self,
            *,
            text: str,
            h: str,
            url: Optional[str] = None,
            section: Optional[str] = None,
            product_code: Optional[str] = None
    ) -> bool:
        """Проверяет, является ли текст почти дубликатом.

        Args:
            text: Текст для проверки
            h: Хэш текста (для отслеживания)
            url: URL документа
            section: Секция документа
            product_code: Код продукта

        Returns:
            True если текст является почти дубликатом, иначе False
        """
        if not text or len(text) < self.min_len:
            logger.debug(f"Text too short for near-dup check: {len(text)} chars")
            return False

        bucket_key = self._bucket_key(
            text=text, url=url, section=section, product_code=product_code
        )

        representative = self.reprs.get(bucket_key)

        # Если в бакете нет представителя, устанавливаем текущий текст
        if not representative:
            self.reprs[bucket_key] = (h, text[:_DEFAULT_TEXT_SAMPLE_LENGTH])
            logger.debug(f"New bucket created: {bucket_key}")
            return False

        existing_hash, existing_text = representative

        # Сравниваем с существующим представителем
        similarity_score = token_set_ratio(text, existing_text)

        if similarity_score >= self.threshold:
            logger.debug(
                f"Near-duplicate found: score={similarity_score}, "
                f"bucket={bucket_key}, current_hash={h}, existing_hash={existing_hash}"
            )
            return True

        # Обновляем представителя бакета текущим текстом
        self.reprs[bucket_key] = (h, text[:_DEFAULT_TEXT_SAMPLE_LENGTH])
        logger.debug(f"Bucket updated: {bucket_key}, new_hash={h}")

        return False

    def flush(self) -> None:
        """Сохраняет состояние на диск."""
        self._save_state()
        logger.info(f"State flushed to {self.state_path}")

    def get_stats(self) -> Dict[str, any]:
        """Возвращает статистику детектора.

        Returns:
            Словарь со статистикой
        """
        buckets_by_product: Dict[str, int] = {}

        for bucket_key in self.reprs:
            product = bucket_key.split(":")[0]
            buckets_by_product[product] = buckets_by_product.get(product, 0) + 1

        return {
            "total_buckets": len(self.reprs),
            "buckets_by_product": buckets_by_product,
            "threshold": self.threshold,
            "min_length": self.min_len
        }

    def __enter__(self) -> NearDupChecker:
        """Поддержка контекстного менеджера."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Автоматическое сохранение при выходе из контекста."""
        self.flush()
