from __future__ import annotations

import logging
from typing import Iterator, List, Optional

from sqlalchemy import (
    Integer,
    String,
    Text,
    ForeignKey,
    create_engine,
    CheckConstraint
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
    validates
)

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей SQLAlchemy."""

    def to_dict(self) -> dict:
        """Конвертирует объект модели в словарь.

        Returns:
            Словарь с атрибутами модели
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

    def __repr__(self) -> str:
        """Строковое представление объекта."""
        attributes = [
            f"{key}={value!r}"
            for key, value in self.to_dict().items()
        ]
        return f"{self.__class__.__name__}({', '.join(attributes)})"


class Team(Base):
    """Модель команды разработчиков.

    Attributes:
        id: Уникальный идентификатор члена команды
        name: Имя члена команды
        role: Роль в команде
        area: Область ответственности
        experience_years: Опыт работы в годах
        email: Контактный email
        github: GitHub профиль
        location: Местоположение
        modules: Список модулей, за которые отвечает член команды
    """

    __tablename__ = "team"
    __table_args__ = (
        CheckConstraint('experience_years >= 0', name='check_experience_non_negative'),
        {"schema": "public"}
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Уникальный идентификатор члена команды"
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Полное имя члена команды"
    )
    role: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Роль в команде (разработчик, менеджер, etc.)"
    )
    area: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Область ответственности или специализации"
    )
    experience_years: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Опыт работы в годах"
    )
    email: Mapped[Optional[str]] = mapped_column(
        String(150),
        comment="Контактный email адрес"
    )
    github: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="GitHub username"
    )
    location: Mapped[Optional[str]] = mapped_column(
        String(100),
        comment="Географическое местоположение"
    )

    # Отношения
    modules: Mapped[List[Module]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    @validates('name', 'role')
    def validate_required_fields(self, key: str, value: str) -> str:
        """Валидирует обязательные строковые поля.

        Args:
            key: Название поля
            value: Значение для валидации

        Returns:
            Проверенное значение

        Raises:
            ValueError: Если значение пустое или слишком длинное
        """
        if not value or not value.strip():
            raise ValueError(f"{key} cannot be empty")

        max_length = 100 if key == 'name' else 50
        if len(value) > max_length:
            raise ValueError(f"{key} cannot exceed {max_length} characters")

        return value.strip()

    @validates('email')
    def validate_email(self, key: str, value: Optional[str]) -> Optional[str]:
        """Валидирует формат email адреса.

        Args:
            key: Название поля
            value: Email для валидации

        Returns:
            Проверенный email или None

        Raises:
            ValueError: Если email имеет неверный формат
        """
        if value is None:
            return None

        if not value.strip():
            return None

        if '@' not in value or '.' not in value.split('@')[-1]:
            raise ValueError(f"Invalid email format: {value}")

        return value.strip().lower()

    @validates('experience_years')
    def validate_experience(self, key: str, value: int) -> int:
        """Валидирует опыт работы.

        Args:
            key: Название поля
            value: Количество лет опыта

        Returns:
            Проверенное значение опыта

        Raises:
            ValueError: Если опыт отрицательный
        """
        if value < 0:
            raise ValueError("Experience years cannot be negative")
        return value

    @property
    def module_count(self) -> int:
        """Возвращает количество модулей, за которые отвечает член команды.

        Returns:
            Количество модулей
        """
        return len(self.modules)

    def get_contact_info(self) -> dict[str, Optional[str]]:
        """Возвращает контактную информацию члена команды.

        Returns:
            Словарь с контактной информацией
        """
        return {
            'email': self.email,
            'github': self.github,
            'location': self.location
        }


class Module(Base):
    """Модель модуля системы.

    Attributes:
        id: Уникальный идентификатор модуля
        name: Название модуля
        description: Описание модуля
        owner_id: Идентификатор владельца модуля
        owner: Владелец модуля (член команды)
    """

    __tablename__ = "modules"
    __table_args__ = {"schema": "public"}

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Уникальный идентификатор модуля"
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        comment="Уникальное название модуля"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        comment="Подробное описание функциональности модуля"
    )
    owner_id: Mapped[int] = mapped_column(
        ForeignKey("public.team.id"),
        nullable=False,
        comment="Идентификатор владельца модуля"
    )

    # Отношения
    owner: Mapped[Team] = relationship(
        back_populates="modules",
        lazy="joined"
    )

    @validates('name')
    def validate_name(self, key: str, value: str) -> str:
        """Валидирует название модуля.

        Args:
            key: Название поля
            value: Название модуля

        Returns:
            Проверенное название модуля

        Raises:
            ValueError: Если название пустое или слишком длинное
        """
        if not value or not value.strip():
            raise ValueError("Module name cannot be empty")

        if len(value) > 100:
            raise ValueError("Module name cannot exceed 100 characters")

        return value.strip()

    @property
    def owner_name(self) -> str:
        """Возвращает имя владельца модуля.

        Returns:
            Имя владельца
        """
        return self.owner.name


def create_db_engine(db_url: str, echo: bool = False, **kwargs):
    """Создает движок SQLAlchemy для подключения к базе данных.

    Args:
        db_url: URL подключения к базе данных
        echo: Включить логирование SQL запросов
        **kwargs: Дополнительные параметры для create_engine

    Returns:
        Настроенный движок SQLAlchemy

    Raises:
        ValueError: Если db_url пустой
    """
    if not db_url:
        raise ValueError("Database URL cannot be empty")

    engine = create_engine(
        db_url,
        echo=echo,
        future=True,
        pool_pre_ping=True,  # Проверка соединения перед использованием
        pool_recycle=3600,  # Переподключение каждый час
        **kwargs
    )

    logger.info(f"Database engine created for URL: {db_url}")
    return engine


def get_session(db_url: str, echo: bool = False) -> Iterator[Session]:
    """Генератор сессий для работы с базой данных.

    Args:
        db_url: URL подключения к базе данных
        echo: Включить логирование SQL запросов

    Yields:
        Сессия SQLAlchemy

    Example:
        >>> with get_session("sqlite:///db.sqlite") as session:
        ...     teams = session.query(Team).all()
    """
    engine = create_db_engine(db_url, echo=echo)

    try:
        with Session(engine) as session:
            logger.debug("Database session created")
            yield session
            logger.debug("Database session committed and closed")
    except Exception as error:
        logger.error(f"Database session error: {error}")
        raise


def create_tables(engine, check_first: bool = True) -> None:
    """Создает таблицы в базе данных.

    Args:
        engine: Движок SQLAlchemy
        check_first: Проверить существование таблиц перед созданием
    """
    try:
        Base.metadata.create_all(engine, checkfirst=check_first)
        logger.info("Database tables created successfully")
    except Exception as error:
        logger.error(f"Failed to create database tables: {error}")
        raise


def drop_tables(engine, check_first: bool = True) -> None:
    """Удаляет таблицы из базы данных.

    Args:
        engine: Движок SQLAlchemy
        check_first: Проверить существование таблиц перед удалением
    """
    try:
        Base.metadata.drop_all(engine, checkfirst=check_first)
        logger.info("Database tables dropped successfully")
    except Exception as error:
        logger.error(f"Failed to drop database tables: {error}")
        raise


class DatabaseManager:
    """Менеджер для работы с базой данных."""

    def __init__(self, db_url: str, echo: bool = False):
        """Инициализация менеджера базы данных.

        Args:
            db_url: URL подключения к базе данных
            echo: Включить логирование SQL запросов
        """
        self.db_url = db_url
        self.echo = echo
        self.engine = create_db_engine(db_url, echo)

    def __enter__(self) -> DatabaseManager:
        """Поддержка контекстного менеджера."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Завершение работы контекстного менеджера."""
        self.engine.dispose()

    def get_session(self) -> Iterator[Session]:
        """Возвращает генератор сессий."""
        return get_session(self.db_url, self.echo)

    def health_check(self) -> bool:
        """Проверяет доступность базы данных.

        Returns:
            True если база данных доступна, иначе False
        """
        try:
            with self.engine.connect() as connection:
                connection.execute("SELECT 1")
            logger.debug("Database health check passed")
            return True
        except Exception as error:
            logger.error(f"Database health check failed: {error}")
            return False

    def get_table_stats(self) -> dict:
        """Возвращает статистику по таблицам.

        Returns:
            Словарь со статистикой таблиц
        """
        stats = {}
        try:
            with self.engine.connect() as connection:
                # Статистика таблицы team
                team_count = connection.execute(
                    "SELECT COUNT(*) FROM public.team"
                ).scalar()
                stats['team_count'] = team_count

                # Статистика таблицы modules
                module_count = connection.execute(
                    "SELECT COUNT(*) FROM public.modules"
                ).scalar()
                stats['module_count'] = module_count

            logger.debug(f"Table stats collected: {stats}")
        except Exception as error:
            logger.error(f"Failed to collect table stats: {error}")
            stats['error'] = str(error)

        return stats
