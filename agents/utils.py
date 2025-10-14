# agents/utils.py
from __future__ import annotations

"""
Утилиты для обработки текста, маршрутизации и системных операций.

Содержит вспомогательные функции, используемые классификатором, 
агентом и инструментами системы маршрутизации запросов.

Публичные функции:
- Текст: normalize_text, clamp01
- Маршрутизация: choose_route_by_priorities, combine_scores  
- Системные: now_ms, gen_trace_id, Timer
"""

import re
import time
import unicodedata
from contextlib import contextmanager
from secrets import token_hex
from time import perf_counter_ns
from typing import Dict, Iterable, List, Mapping, Optional, TypeVar, Union

__all__ = [
    "normalize_text",
    "clamp01",
    "choose_route_by_priorities",
    "combine_scores",
    "now_ms",
    "gen_trace_id",
    "Timer",
]


def normalize_text(text: str) -> str:
    """Нормализует текст для сопоставления по ключевым словам и паттернам.

    Выполняет последовательность преобразований:
    1. Приведение к строке (для нестроковых входов)
    2. NFC-нормализация Юникода (композиция символов)
    3. Приведение к нижнему регистру
    4. Замена последовательностей пробельных символов на одиночный пробел
    5. Обрезка пробелов по краям

    Args:
        text: Исходный текст для нормализации

    Returns:
        Нормализованный текст в нижнем регистре без лишних пробелов

    Examples:
        >>> normalize_text("  Привет,   МИР!  ")
        'привет, мир!'
        >>> normalize_text("Café & naïve")
        'café & naïve'
    """
    if not isinstance(text, str):
        text = str(text or "")

    # NFC нормализация для совместимости с предопределенными ключевыми словами
    normalized = unicodedata.normalize("NFC", text)

    # Приведение к нижнему регистру
    normalized = normalized.lower()

    # Замена всех последовательностей пробельных символов на одиночный пробел
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized.strip()


def clamp01(value: float) -> float:
    """Ограничивает значение числом в диапазоне [0.0, 1.0].

    Args:
        value: Исходное числовое значение

    Returns:
        Значение, ограниченное диапазоном [0.0, 1.0]

    Examples:
        >>> clamp01(1.5)
        1.0
        >>> clamp01(-0.5)
        0.0
        >>> clamp01(0.75)
        0.75
        >>> clamp01(float('nan'))
        0.0
    """
    if value != value:  # Проверка на NaN
        return 0.0
    return max(0.0, min(1.0, value))


def choose_route_by_priorities(
        candidates: Iterable[str],
        priorities: Iterable[str],
        allow_combine: bool = True,
) -> List[str]:
    """Выбирает маршрут обработки на основе кандидатов и приоритетов.

    Алгоритм выбора:
    1. Нормализует имена инструментов к нижнему регистру
    2. Фильтрует кандидатов, оставляя только присутствующие в приоритетах
    3. Сортирует по порядку приоритетов
    4. Возвращает один или несколько инструментов в зависимости от флага

    Args:
        candidates: Набор инструментов-кандидатов
        priorities: Приоритетный порядок инструментов
        allow_combine: Разрешить комбинирование нескольких инструментов

    Returns:
        Отсортированный список выбранных инструментов

    Raises:
        ValueError: Если candidates или priorities пусты после нормализации

    Examples:
        >>> choose_route_by_priorities(
        ...     ["web", "docs"], ["docs", "sql", "web"], True
        ... )
        ['docs', 'web']
        >>> choose_route_by_priorities(
        ...     ["web", "docs"], ["docs", "sql", "web"], False
        ... )
        ['docs']
    """
    # Нормализация и валидация входных данных
    candidate_set = {
        str(candidate).strip().lower()
        for candidate in candidates
        if candidate and str(candidate).strip()
    }

    priority_list = [
        str(priority).strip().lower()
        for priority in priorities
        if priority and str(priority).strip()
    ]

    if not candidate_set:
        raise ValueError("Candidates cannot be empty after normalization")

    if not priority_list:
        raise ValueError("Priorities cannot be empty after normalization")

    # Выбор инструментов в порядке приоритетов
    selected_route = [
        priority for priority in priority_list
        if priority in candidate_set
    ]

    if not selected_route:
        return []

    return selected_route if allow_combine else [selected_route[0]]


# Generic тип для словарей с оценками
ScoreDict = TypeVar("ScoreDict", bound=Dict[str, float])


def combine_scores(
        *score_dicts: Mapping[str, float],
        weights: Optional[Iterable[float]] = None,
) -> Dict[str, float]:
    """Объединяет несколько словарей оценок с учетом весов.

    Объединяет оценки из разных источников путем взвешенного суммирования.
    Ключи, отсутствующие в некоторых словарях, считаются нулевыми.

    Args:
        *score_dicts: Произвольное количество словарей с оценками
        weights: Веса для каждого словаря (равные по умолчанию)

    Returns:
        Словарь с объединенными взвешенными оценками

    Raises:
        ValueError: Если количество весов не совпадает с количеством словарей
        TypeError: Если оценки не могут быть преобразованы в float

    Examples:
        >>> combine_scores(
        ...     {"docs": 0.8, "web": 0.3},
        ...     {"docs": 0.6, "sql": 0.9},
        ...     weights=[1.0, 0.5]
        ... )
        {'docs': 1.1, 'web': 0.3, 'sql': 0.45}
    """
    if not score_dicts:
        return {}

    # Подготовка весов
    weight_list: List[float]
    if weights is None:
        weight_list = [1.0] * len(score_dicts)
    else:
        weight_list = list(weights)
        if len(weight_list) != len(score_dicts):
            raise ValueError(
                f"Number of weights ({len(weight_list)}) must match "
                f"number of score dictionaries ({len(score_dicts)})"
            )

    # Объединение оценок
    combined: Dict[str, float] = {}

    for score_dict, weight in zip(score_dicts, weight_list):
        for key, score in score_dict.items():
            try:
                numeric_score = float(score)
                combined[key] = combined.get(key, 0.0) + numeric_score * weight
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Invalid score value '{score}' for key '{key}': {e}"
                ) from e

    return combined


def now_ms() -> int:
    """Возвращает монотонное время в миллисекундах.

    Использует perf_counter_ns() для получения монотонного времени,
    не подверженного изменениям системных часов.

    Returns:
        Текущее монотонное время в миллисекундах

    Examples:
        >>> start = now_ms()
        >>> # ... выполнение операции ...
        >>> elapsed = now_ms() - start
    """
    return perf_counter_ns() // 1_000_000


def gen_trace_id(length: int = 8) -> str:
    """Генерирует уникальный идентификатор трассировки.

    Args:
        length: Длина идентификатора в байтах (по умолчанию 8)

    Returns:
        Строка с шестнадцатеричным идентификатором

    Examples:
        >>> gen_trace_id()
        'e3b0c44298fc1c14'
        >>> gen_trace_id(4)
        'a1b2c3d4'
    """
    if length <= 0:
        raise ValueError("Length must be positive integer")

    return token_hex(length)


class Timer:
    """Контекстный менеджер для измерения времени выполнения.

    Attributes:
        elapsed_ms: Время выполнения в миллисекундах
        elapsed_ns: Время выполнения в наносекундах

    Examples:
        >>> with Timer() as timer:
        ...     # выполнение операции
        ...     time.sleep(0.1)
        >>> print(f"Operation took {timer.elapsed_ms:.2f} ms")
    """

    def __init__(self) -> None:
        self._start_ns: Optional[int] = None
        self.elapsed_ns: Optional[int] = None

    def __enter__(self) -> Timer:
        self._start_ns = perf_counter_ns()
        return self

    def __exit__(self, *args: object) -> None:
        if self._start_ns is not None:
            self.elapsed_ns = perf_counter_ns() - self._start_ns

    @property
    def elapsed_ms(self) -> float:
        """Возвращает время выполнения в миллисекундах."""
        if self.elapsed_ns is None:
            return 0.0
        return self.elapsed_ns / 1_000_000

    @property
    def elapsed_seconds(self) -> float:
        """Возвращает время выполнения в секундах."""
        if self.elapsed_ns is None:
            return 0.0
        return self.elapsed_ns / 1_000_000_000


@contextmanager
def timing(description: str = "Operation") -> Iterable[float]:
    """Контекстный менеджер для измерения и логирования времени выполнения.

    Args:
        description: Описание измеряемой операции

    Yields:
        Время выполнения в секундах

    Examples:
        >>> with timing("Database query") as elapsed:
        ...     # выполнение запроса
        ...     time.sleep(0.1)
        >>> print(f"Query took {elapsed:.3f} seconds")
    """
    start = perf_counter_ns()
    try:
        yield 0.0
    finally:
        end = perf_counter_ns()
        elapsed_seconds = (end - start) / 1_000_000_000
        print(f"{description} completed in {elapsed_seconds:.3f}s")
