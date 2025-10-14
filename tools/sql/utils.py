from __future__ import annotations
from typing import Any, Iterable, Sequence, Tuple, List, Dict, Union
from sqlalchemy import create_engine, event, Engine, CursorResult, Row
from sqlalchemy.engine import Connection


class DatabaseUtils:
    """Утилиты для работы с базой данных."""

    @staticmethod
    def get_engine(db_url: str, readonly: bool = True, echo: bool = False) -> Engine:
        """
        Создает и настраивает движок SQLAlchemy.

        Args:
            db_url: URL для подключения к базе данных
            readonly: Флаг режима только для чтения
            echo: Флаг вывода SQL-запросов в лог

        Returns:
            Настроенный движок SQLAlchemy
        """
        engine = create_engine(db_url, echo=echo, future=True)

        if readonly:
            @event.listens_for(engine, "connect")
            def set_readonly(dbapi_conn: Any, conn_record: Any) -> None:  # pragma: no cover
                """
                Устанавливает настройки только для чтения при подключении.

                Args:
                    dbapi_conn: DBAPI соединение
                    conn_record: Запись о соединении
                """
                cursor = dbapi_conn.cursor()
                try:
                    cursor.execute("SET default_transaction_read_only = on;")
                    cursor.execute("SET statement_timeout = 5000;")
                finally:
                    cursor.close()

        return engine

    @staticmethod
    def rows_to_dicts(
            rows: Union[Iterable[Row], Iterable[Tuple], CursorResult],
            columns: Sequence[str]
    ) -> List[Dict[str, Any]]:
        """
        Преобразует строки результата SQL в список словарей.

        Args:
            rows: Результат выполнения SQL-запроса
            columns: Список названий колонок

        Returns:
            Список словарей, где каждый словарь представляет строку

        Raises:
            ValueError: Если rows пустой или columns не соответствует rows
        """
        if not rows:
            return []

        data: List[Dict[str, Any]] = []

        for row in rows:
            if hasattr(row, "_mapping"):
                # SQLAlchemy Row объект (современный стиль)
                data.append(dict(row._mapping))
            elif hasattr(row, "_asdict"):
                # SQLAlchemy Row объект с методом _asdict
                data.append(row._asdict())
            elif isinstance(row, dict):
                # Уже словарь
                data.append(row)
            else:
                # Обычный tuple или список - создаем словарь по колонкам
                row_dict = {}
                for i, column in enumerate(columns):
                    if i < len(row):
                        row_dict[column] = row[i]
                    else:
                        raise ValueError(f"Column '{column}' index out of range for row {row}")
                data.append(row_dict)

        return data

    @staticmethod
    def format_answer(
            rows: Union[Iterable[Row], Iterable[Tuple], CursorResult],
            columns: Sequence[str]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Форматирует результат SQL-запроса в читаемый ответ и структурированные данные.

        Args:
            rows: Результат выполнения SQL-запроса
            columns: Список названий колонок

        Returns:
            Кортеж (текстовый ответ, список словарей с данными)
        """
        data = DatabaseUtils.rows_to_dicts(rows, columns)

        if not data:
            phrase = "Данные не найдены."
        elif len(data) == 1:
            if len(columns) == 1:
                # Одна колонка, одна строка - простой ответ
                value = next(iter(data[0].values()))
                phrase = f"Найдено: {value}"
            else:
                # Несколько колонок, одна строка
                phrase = "Найдена 1 запись."
        else:
            # Несколько строк
            phrase = f"Найдено записей: {len(data)}."

            # Добавляем информацию о колонках для большей ясности
            if columns and len(columns) <= 5:  # Не перегружаем для многих колонок
                phrase += f" Колонки: {', '.join(columns)}"

        return phrase, data


# Функции для обратной совместимости
def get_engine(db_url: str, readonly: bool = True, echo: bool = False) -> Engine:
    """Устаревшая функция для обратной совместимости."""
    return DatabaseUtils.get_engine(db_url, readonly, echo)


def rows_to_json(rows: Any, columns: Sequence[str]) -> List[Dict[str, Any]]:
    """Устаревшая функция для обратной совместимости."""
    return DatabaseUtils.rows_to_dicts(rows, columns)


def format_answer(rows: Any, columns: Sequence[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """Устаревшая функция для обратной совместимости."""
    return DatabaseUtils.format_answer(rows, columns)
