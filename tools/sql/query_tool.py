from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Dict
from sqlalchemy import text
import yaml

from .safety import validate_select_only
from .utils import get_engine, format_answer
from . import prompts

try:
    from core.llm_provider import get_llm

    LLM_AVAILABLE = True
except Exception:  # pragma: no cover
    LLM_AVAILABLE = False

logger = logging.getLogger("sql_qa")


@dataclass
class DatabaseConfig:
    """Конфигурация подключения к базе данных."""
    url: str
    readonly: bool = True
    echo: bool = False
    schema: str = "public"

    @classmethod
    def load_from_file_and_env(cls, path: str | None = None) -> DatabaseConfig:
        """Загружает конфигурацию из файла и переменных окружения."""
        path = path or os.getenv("DB_CONFIG_YAML", "configs/db.yaml")
        cfg = {}

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        db_config = cfg.setdefault("db", {})

        # Приоритет: переменные окружения > конфиг файл > значения по умолчанию
        url = os.getenv("DB_URL") or db_config.get("url",
                                                   "postgresql+psycopg://nlq_user:nlq_pass@localhost:5432/transformers")
        readonly = str(os.getenv("DB_READONLY", db_config.get("readonly", "true"))).lower() == "true"
        echo = bool(db_config.get("echo", False))
        schema = os.getenv("DB_SCHEMA", db_config.get("schema", "public"))

        return cls(url=url, readonly=readonly, echo=echo, schema=schema)


class QueryTool:
    """Основной класс для выполнения NLP-запросов к базе данных."""

    def __init__(self, config: DatabaseConfig | None = None):
        self.config = config or DatabaseConfig.load_from_file_and_env()

    def render_schema_for_prompt(self) -> str:
        """Генерирует описание схемы БД для промпта."""
        engine = get_engine(self.config.url, readonly=True)
        lines: List[str] = []

        with engine.connect() as conn:
            # Получаем список таблиц
            tables = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """), {"schema": self.config.schema}).fetchall()

            for (table_name,) in tables:
                # Получаем колонки для каждой таблицы
                columns = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :table_name
                    ORDER BY ordinal_position
                """), {"schema": self.config.schema, "table_name": table_name}).fetchall()

                column_defs = []
                for col_name, data_type, nullable, default in columns:
                    col_def = f"{col_name} {data_type}"
                    if default:
                        col_def += f" DEFAULT {default}"
                    if nullable == "NO":
                        col_def += " NOT NULL"
                    column_defs.append(col_def)

                lines.append(f"{table_name}({', '.join(column_defs)})")

        return "\n".join(lines)

    def _rule_based_nl2sql(self, question: str) -> str:
        """Rule-based преобразование естественного языка в SQL."""
        q = question.lower().strip()

        # Правила должны быть безопасны от SQL-инъекций
        if "тимлид" in q or "team lead" in q:
            return "SELECT t.name FROM public.team AS t WHERE t.role = 'maintainer' ORDER BY t.id ASC LIMIT 1"

        if ("сколько" in q and "core" in q) or "core-dev" in q:
            return "SELECT COUNT(*) AS count FROM public.team AS t WHERE t.role = 'core-dev'"

        if "какие модули у" in q and "maintainer" in q:
            # Извлекаем имя безопасным способом
            name_part = q.split("maintainer", 1)[1].strip().strip("?")
            # В реальном коде здесь должна быть валидация имени
            return ("SELECT m.name FROM public.modules AS m JOIN public.team AS t ON m.owner_id = t.id "
                    f"WHERE t.name = '{name_part}' AND t.role = 'maintainer' ORDER BY m.name ASC")

        if "токенизац" in q or "token" in q:
            return "SELECT m.name FROM public.modules AS m WHERE lower(m.name) LIKE '%token%' LIMIT 100"

        if "trainer" in q:
            return ("SELECT t.name, t.role FROM public.team AS t "
                    "JOIN public.modules AS m ON m.owner_id = t.id "
                    "WHERE m.name = 'transformers.trainer' LIMIT 100")

        return "SELECT name, role FROM public.team LIMIT 10"

    def _optimize_sql_for_relationships(self, sql: str, question: str) -> str:
        """Оптимизирует SQL для запросов о связях между таблицами."""
        q_lower = question.lower()
        sql_lower = sql.lower()

        relationship_keywords = ["модул", "владел", "отвеча", "связ", "отношен"]
        if any(keyword in q_lower for keyword in relationship_keywords):
            if "left join" in sql_lower:
                sql = sql.replace("LEFT JOIN", "INNER JOIN")
                logger.info("Optimized: Replaced LEFT JOIN with INNER JOIN for relationship query")

        return sql

    def nl_to_sql(self, question: str) -> str:
        """Преобразует естественный язык в SQL-запрос."""
        schema_text = self.render_schema_for_prompt()

        if LLM_AVAILABLE and get_llm is not None:
            try:
                provider = get_llm()
                messages = prompts.build_messages(question, schema_text)
                response = provider.generate(messages=messages, temperature=0)
                sql = response.text.strip()
                logger.info("LLM generated SQL: %s", sql)
            except Exception as e:
                logger.warning("LLM generation failed, fallback to rules: %s", e)
                sql = self._rule_based_nl2sql(question)
        else:
            sql = self._rule_based_nl2sql(question)

        return self._optimize_sql_for_relationships(sql, question)

    def execute_sql(self, sql: str) -> Tuple[Any, List[str]]:
        """Выполняет SQL-запрос и возвращает результаты."""
        engine = get_engine(self.config.url, readonly=self.config.readonly, echo=self.config.echo)

        with engine.connect() as conn:
            stmt = text(sql)
            result = conn.execute(stmt)
            rows = result.fetchall()
            columns = list(result.keys())

        return rows, columns

    def answer_from_db(self, question: str, dry_run: bool = False) -> QAResult:
        """Основной метод для получения ответа на вопрос."""
        sql = self.nl_to_sql(question=question)

        # Валидация безопасности SQL
        guard = validate_select_only(sql)
        if not guard.ok:
            raise ValueError(f"SQL validation failed: {guard.reason}")

        # Добавляем LIMIT если отсутствует
        if " limit " not in sql.lower():
            sql += " LIMIT 100"

        if dry_run:
            return QAResult(answer="dry-run: SQL generated", data=[], sql=sql, guarded=True)

        rows, columns = self.execute_sql(sql)
        phrase, data = format_answer(rows, columns)

        return QAResult(answer=phrase, data=data, sql=sql, guarded=True)


# Функции для обратной совместимости
def _load_db_config() -> dict:
    """Устаревшая функция для обратной совместимости."""
    config = DatabaseConfig.load_from_file_and_env()
    return {
        "db": {
            "url": config.url,
            "readonly": config.readonly,
            "echo": config.echo,
            "schema": config.schema
        }
    }


def render_schema_for_prompt(db_url: str, schema: str = "public") -> str:
    """Устаревшая функция для обратной совместимости."""
    tool = QueryTool(DatabaseConfig(url=db_url, schema=schema))
    return tool.render_schema_for_prompt()


def nl_to_sql(question: str, schema: str, llm: Optional[Any] = None) -> str:
    """Устаревшая функция для обратной совместимости."""
    tool = QueryTool(DatabaseConfig(url="", schema="public"))
    return tool.nl_to_sql(question)


def execute_sql(db_url: str, sql: str, echo: bool = False):
    """Устаревшая функция для обратной совместимости."""
    tool = QueryTool(DatabaseConfig(url=db_url, echo=echo))
    return tool.execute_sql(sql)


def answer_from_db(question: str, dry_run: bool = False) -> QAResult:
    """Устаревшая функция для обратной совместимости."""
    tool = QueryTool()
    return tool.answer_from_db(question, dry_run)


@dataclass
class QAResult:
    """Результат выполнения запроса."""
    answer: str
    data: List[Dict]
    sql: str
    guarded: bool
