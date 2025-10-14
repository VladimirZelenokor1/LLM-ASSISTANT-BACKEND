from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sqlalchemy import create_engine, text, Engine

logger = logging.getLogger(__name__)


@dataclass
class SeedConfig:
    """Конфигурация для заполнения базы данных."""
    db_url: str = os.getenv("DB_URL_ADMIN", "postgresql+psycopg://nlq_user:nlq_pass@localhost:5432/transformers")
    create_tables: bool = True
    drop_existing: bool = False


@dataclass
class TeamMember:
    """Данные члена команды."""
    name: str
    role: str
    area: Optional[str]
    experience_years: int
    email: Optional[str]
    github: str
    location: Optional[str]


@dataclass
class Module:
    """Данные модуля."""
    name: str
    description: str
    github_owner: str


class DataSeeder:
    """Класс для заполнения базы данных тестовыми данными."""

    # DDL для создания таблиц
    DDL = """
    CREATE TABLE IF NOT EXISTS public.team (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        area TEXT,
        experience_years INTEGER NOT NULL DEFAULT 0,
        email TEXT,
        github TEXT,
        location TEXT
    );

    CREATE TABLE IF NOT EXISTS public.modules (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        owner_id INTEGER NOT NULL REFERENCES public.team(id)
    );

    CREATE INDEX IF NOT EXISTS idx_team_role ON public.team(role);
    CREATE INDEX IF NOT EXISTS idx_modules_owner ON public.modules(owner_id);
    """

    # Тестовые данные команды
    TEAM_DATA: List[TeamMember] = [
        TeamMember("Ирина Петрова", "maintainer", "NLP", 6, "irina@example.com", "irina-maint", "Moscow"),
        TeamMember("Алексей Смирнов", "core-dev", "NLP", 4, "alexey@example.com", "alex-core", "Saint Petersburg"),
        TeamMember("Мария Кузнецова", "core-dev", "NLP", 3, "maria@example.com", "maria-core", "Kazan"),
        TeamMember("Олег Иванов", "infra", "Training", 5, "oleg@example.com", "oleg-infra", "Moscow"),
        TeamMember("Дмитрий Орлов", "docs", "Docs", 2, "dmitry@example.com", "dmitry-docs", "Novosibirsk"),
        TeamMember("Светлана Сергеева", "nlp", "NLP", 5, "svetlana@example.com", "svet-nlp", "Moscow"),
        TeamMember("Павел Волков", "vision", "Vision", 4, "pavel@example.com", "pavel-vision", "Perm"),
        TeamMember("Анна Федорова", "core-dev", "Audio", 3, "anna@example.com", "anna-audio", "Samara"),
        TeamMember("Григорий Литвинов", "core-dev", "Utils", 2, "grigory@example.com", "grig-utils", "Tomsk"),
    ]

    # Тестовые данные модулей
    MODULES_DATA: List[Module] = [
        Module("transformers.pipelines", "Pipelines registry and helpers", "alex-core"),
        Module("transformers.trainer", "High-level training loop", "oleg-infra"),
        Module("transformers.tokenization", "Tokenizer base utilities", "svet-nlp"),
        Module("transformers.models.gpt2", "GPT-2 model package", "svet-nlp"),
        Module("transformers.models.bert", "BERT model package", "alex-core"),
        Module("transformers.models.vit", "Vision Transformer models", "pavel-vision"),
        Module("transformers.models.whisper", "Whisper ASR models", "anna-audio"),
        Module("transformers.utils", "Common helpers and adapters", "grig-utils"),
    ]

    def __init__(self, config: Optional[SeedConfig] = None):
        self.config = config or SeedConfig()
        self.engine: Optional[Engine] = None

    def _create_engine(self) -> Engine:
        """Создает и возвращает движок SQLAlchemy."""
        return create_engine(self.config.db_url, future=True)

    def _create_tables(self, connection) -> None:
        """Создает таблицы в базе данных."""
        logger.info("Creating database tables...")
        connection.exec_driver_sql(self.DDL)
        logger.info("Database tables created successfully")

    def _insert_team_data(self, connection) -> None:
        """Вставляет данные команды в базу данных."""
        logger.info("Inserting team data...")

        for member in self.TEAM_DATA:
            connection.execute(
                text("""
                    INSERT INTO public.team(name, role, area, experience_years, email, github, location)
                    VALUES (:name, :role, :area, :experience_years, :email, :github, :location)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "name": member.name,
                    "role": member.role,
                    "area": member.area,
                    "experience_years": member.experience_years,
                    "email": member.email,
                    "github": member.github,
                    "location": member.location,
                }
            )

        logger.info("Team data inserted successfully")

    def _insert_modules_data(self, connection) -> None:
        """Вставляет данные модулей в базу данных."""
        logger.info("Inserting modules data...")

        for module in self.MODULES_DATA:
            connection.execute(
                text("""
                    INSERT INTO public.modules(name, description, owner_id)
                    VALUES (:name, :description, (SELECT id FROM public.team WHERE github = :github LIMIT 1))
                    ON CONFLICT (name) DO NOTHING
                """),
                {
                    "name": module.name,
                    "description": module.description,
                    "github": module.github_owner,
                }
            )

        logger.info("Modules data inserted successfully")

    def _validate_data(self) -> None:
        """Проверяет корректность тестовых данных."""
        # Проверяем уникальность GitHub имен
        github_names = [member.github for member in self.TEAM_DATA]
        if len(github_names) != len(set(github_names)):
            raise ValueError("Duplicate GitHub names found in team data")

        # Проверяем уникальность имен модулей
        module_names = [module.name for module in self.MODULES_DATA]
        if len(module_names) != len(set(module_names)):
            raise ValueError("Duplicate module names found in modules data")

        # Проверяем, что все GitHub владельцев модулей существуют в команде
        team_githubs = set(github_names)
        for module in self.MODULES_DATA:
            if module.github_owner not in team_githubs:
                raise ValueError(f"Module owner '{module.github_owner}' not found in team data")

    def run(self) -> None:
        """
        Основной метод для заполнения базы данных тестовыми данными.

        Raises:
            Exception: Если произошла ошибка при работе с базой данных
        """
        logger.info("Starting database seeding...")

        try:
            self._validate_data()

            self.engine = self._create_engine()

            with self.engine.begin() as connection:
                if self.config.create_tables:
                    self._create_tables(connection)

                self._insert_team_data(connection)
                self._insert_modules_data(connection)

            logger.info("Database seeding completed successfully")

        except Exception as e:
            logger.error("Database seeding failed: %s", e)
            raise


# Функции для обратной совместимости
def run() -> None:
    """Устаревшая функция для обратной совместимости."""
    seeder = DataSeeder()
    seeder.run()


if __name__ == "__main__":
    # Настройка логирования для скрипта
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    run()
