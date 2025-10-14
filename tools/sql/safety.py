from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Pattern


@dataclass
class GuardResult:
    """Результат проверки безопасности SQL-запроса."""
    ok: bool
    reason: str = ""


class SQLSafetyValidator:
    """Валидатор безопасности SQL-запросов."""

    # Запрещённые SQL ключевые слова и операции
    FORBIDDEN_PATTERNS: List[Pattern] = [
        re.compile(r"\binsert\b", re.IGNORECASE),
        re.compile(r"\bupdate\b", re.IGNORECASE),
        re.compile(r"\bdelete\b", re.IGNORECASE),
        re.compile(r"\balter\b", re.IGNORECASE),
        re.compile(r"\bdrop\b", re.IGNORECASE),
        re.compile(r"\bcreate\b", re.IGNORECASE),
        re.compile(r"\btruncate\b", re.IGNORECASE),
        re.compile(r"\bvacc?uum\b", re.IGNORECASE),
        re.compile(r"\bpragma\b", re.IGNORECASE),
        re.compile(r"\battach\b", re.IGNORECASE),
        re.compile(r"\bdetach\b", re.IGNORECASE),
        re.compile(r"\bgrant\b", re.IGNORECASE),
        re.compile(r"\brevoke\b", re.IGNORECASE),
        re.compile(r"\bset\b", re.IGNORECASE),
        re.compile(r"\breplace\b", re.IGNORECASE),
    ]

    @staticmethod
    def _strip_comments(sql: str) -> str:
        """
        Удаляет комментарии из SQL-запроса.

        Args:
            sql: Исходный SQL-запрос

        Returns:
            SQL-запрос без комментариев
        """
        # Удаляем блочные комментарии /* ... */
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        # Удаляем строчные комментарии --
        sql = re.sub(r"--[^\n]*", "", sql)
        return sql

    def normalize_sql(self, sql: str) -> str:
        """
        Нормализует SQL-запрос: удаляет комментарии и лишние пробелы.

        Args:
            sql: Исходный SQL-запрос

        Returns:
            Нормализованный SQL-запрос
        """
        if not sql:
            return ""

        sql = self._strip_comments(sql)
        sql = sql.strip().strip(";").strip()
        sql = re.sub(r"\s+", " ", sql)
        return sql

    def validate_select_only(self, sql: str, max_limit: int = 100) -> GuardResult:
        """
        Проверяет, что SQL-запрос является безопасным SELECT-запросом.

        Args:
            sql: SQL-запрос для проверки
            max_limit: Максимально допустимый LIMIT (не используется, оставлен для совместимости)

        Returns:
            GuardResult с результатом проверки
        """
        if not sql:
            return GuardResult(False, "Empty SQL query")

        raw_sql = sql
        normalized_sql = self.normalize_sql(raw_sql).lower()

        # Проверка на наличие точек с запятой
        if ";" in raw_sql:
            return GuardResult(False, "Only one statement is allowed (no semicolons).")

        # Проверка что запрос начинается с SELECT
        if not normalized_sql.startswith("select "):
            return GuardResult(False, "Only SELECT statements are allowed.")

        # Проверка запрещённых ключевых слов
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.search(normalized_sql):
                keyword = pattern.pattern.strip('\\b')
                return GuardResult(False, f"Forbidden keyword detected: {keyword}")

        return GuardResult(True, "")


# Функции для обратной совместимости
_validator = SQLSafetyValidator()

FORBIDDEN = [
    r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\balter\b", r"\bdrop\b",
    r"\bcreate\b", r"\btruncate\b", r"\bvacc?uum\b", r"\bpragma\b",
    r"\battach\b", r"\bdetach\b", r"\bgrant\b", r"\brevoke\b",
    r"\bset\b", r"\breplace\b"
]


def _strip_comments(sql: str) -> str:
    """Устаревшая функция для обратной совместимости."""
    return SQLSafetyValidator._strip_comments(sql)


def normalize_sql(sql: str) -> str:
    """Устаревшая функция для обратной совместимости."""
    return _validator.normalize_sql(sql)


def validate_select_only(sql: str, max_limit: int = 100) -> GuardResult:
    """Устаревшая функция для обратной совместимости."""
    return _validator.validate_select_only(sql, max_limit)
