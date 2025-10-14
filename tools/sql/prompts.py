from __future__ import annotations
from textwrap import dedent
from typing import List, Tuple, Dict, Any


class PromptBuilder:
    """Формирует промпты для генерации SQL-запросов на основе схемы БД."""

    SYSTEM_RULES = dedent("""
        You convert a user question about the product team and Transformers-like modules
        into a single PostgreSQL SELECT for the schema below.

        STRICT:
        - Output SQL only (no explanations, no comments).
        - One statement, no semicolons.
        - Only SELECT (no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/TRUNCATE/VACUUM/GRANT/REVOKE/SET/PRAGMA).
        - Prefer explicit columns (avoid SELECT *).
        - Add LIMIT 100 if result may be large.
        - Use INNER JOIN when showing relationships between tables (e.g., team and modules).
        - Use LEFT JOIN only when explicitly asked to include records that might not have matches.
        """).strip()

    FEW_SHOT_EXAMPLES: List[Tuple[str, str]] = [
        (
            "кто тимлид продукта?",
            "SELECT t.name FROM public.team AS t WHERE t.role = 'maintainer' ORDER BY t.id ASC LIMIT 1",
        ),
        (
            "сколько core-dev в команде?",
            "SELECT COUNT(*) AS count FROM public.team AS t WHERE t.role = 'core-dev'",
        ),
        (
            "какие модули у maintainer Ирина Петрова?",
            "SELECT m.name FROM public.modules AS m JOIN public.team AS t ON m.owner_id = t.id "
            "WHERE t.name = 'Ирина Петрова' AND t.role = 'maintainer' ORDER BY m.name ASC",
        ),
        (
            "кто отвечает за модуль transformers.trainer?",
            "SELECT t.name, t.role FROM public.team AS t JOIN public.modules AS m ON m.owner_id = t.id "
            "WHERE m.name = 'transformers.trainer'",
        ),
        (
            "сколько лет опыта у владельца vision-модуля?",
            "SELECT t.experience_years FROM public.team AS t JOIN public.modules AS m ON m.owner_id = t.id "
            "WHERE m.name = 'transformers.models.vit' LIMIT 1",
        ),
        (
            "покажи сотрудников и их модули, только у кого есть модули",
            "SELECT t.name, m.name FROM public.team AS t INNER JOIN public.modules AS m ON m.owner_id = t.id "
            "ORDER BY t.name ASC LIMIT 100",
        ),
        (
            "покажи сотрудников у которых есть модули",
            "SELECT t.name, m.name FROM public.team AS t INNER JOIN public.modules AS m ON m.owner_id = t.id "
            "ORDER BY t.name ASC LIMIT 100",
        ),
    ]

    @staticmethod
    def render_schema(schema: str) -> str:
        """Форматирует схему БД для вставки в промпт."""
        return "-- SCHEMA START\n" + schema.strip() + "\n-- SCHEMA END"

    def build_messages(self, question: str, schema: str) -> List[Dict[str, Any]]:
        """Собирает сообщения для LLM на основе схемы и примерами."""
        examples = "\n\n".join(f"Q: {q}\nSQL: {sql}" for q, sql in self.FEW_SHOT_EXAMPLES)
        system_content = (
            f"{self.SYSTEM_RULES}\n\n{self.render_schema(schema)}\n\n{examples}"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question.strip()},
        ]
