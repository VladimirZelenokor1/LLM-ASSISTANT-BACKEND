from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, ClassVar, Tuple, Callable

import yaml
from pydantic import BaseModel, Field, model_validator, field_validator

logger = logging.getLogger(__name__)


class PathsConfig(BaseModel):
    """Конфигурация путей."""
    input_dir: str = Field(..., description="Директория с исходными документами")
    preprocessed_dir: str = Field(..., description="Директория для препроцессированных документов")
    chunks_dir: str = Field(..., description="Директория для готовых чанков")

    @field_validator("*")
    @classmethod
    def _validate_paths(cls, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("Path cannot be empty")
        return v

    def ensure_directories(self) -> None:
        for path in (self.input_dir, self.preprocessed_dir, self.chunks_dir):
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", path)


class ChunkerConfig(BaseModel):
    """
    Поля и дефолты — как в СТАРОЙ версии. Проверки — только WARN.
    """
    strategy: Literal["rule_aware"] = Field(default="rule_aware", description="Стратегия чанкинга")
    unit: Literal["tokens", "chars"] = Field(default="tokens", description="Единица размера")

    max_tokens: int = 520
    overlap_sentences: int = 1
    sentence_max: int = 7
    min_chars: int = 180
    max_chunks_per_doc: int = 60

    # Бюджеты для кода/таблиц
    max_tokens_code: int = 300
    max_chars_code: int = 1600

    @model_validator(mode="after")
    def _soft_checks(self) -> "ChunkerConfig":
        try:
            est_chars = max(1, int(self.max_tokens)) * 4
            if int(self.max_chars_code) < est_chars:
                logger.warning(
                    "max_chars_code (%s) меньше ~оценки символов для max_tokens=%s (~%s). "
                    "Это может уменьшить размер чанков с кодом.",
                    self.max_chars_code, self.max_tokens, est_chars
                )
        except Exception:
            # никаких падений — только мягкие проверки
            pass
        return self


class ChunkingConfig(BaseModel):
    """
    Основная конфигурация процесса чанкинга.
    """
    paths: PathsConfig = Field(..., description="Конфигурация путей")
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig, description="Конфигурация чанкера")
    abbreviations: Dict[str, List[str]] = Field(default_factory=dict, description="Словарь аббревиатур")

    # ВАЖНО: объявляем как ClassVar, чтобы Pydantic не превращал это в ModelPrivateAttr/field
    ENV_SIMPLE_OVERRIDES: ClassVar[Dict[str, Tuple[str, str, Callable[[str], Any]]]] = {
        "CHUNK_MAX_TOKENS": ("chunker", "max_tokens", int),
        "CHUNK_MAX_TOKENS_CODE": ("chunker", "max_tokens_code", int),
        "CHUNK_MAX_CHARS_CODE": ("chunker", "max_chars_code", int),
        # Доп. (не ломает старую логику, можно не задавать)
        "CHUNK_OVERLAP_SENTENCES": ("chunker", "overlap_sentences", int),
        "CHUNK_MIN_CHARS": ("chunker", "min_chars", int),
        "CHUNK_STRATEGY": ("chunker", "strategy", str),
        "CHUNK_UNIT": ("chunker", "unit", str),
        "CHUNK_SENTENCE_MAX": ("chunker", "sentence_max", int),
        "CHUNK_MAX_CHUNKS_PER_DOC": ("chunker", "max_chunks_per_doc", int),
    }

    @classmethod
    def from_yaml(cls, path: str) -> "ChunkingConfig":
        """
        Загрузка YAML -> создание модели -> применение ENV (как в СТАРОЙ) -> возврат модели.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        logger.info("Loading configuration from: %s", path)

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Configuration file must contain a YAML dictionary")

        # 1) создаём модель (1:1 со старой семантикой)
        cfg = cls(**raw)

        # 2) применяем ENV-переопределения ПОСЛЕ создания модели (как в старой версии)
        for env_name, (section, field, caster) in cls.ENV_SIMPLE_OVERRIDES.items():
            val = os.getenv(env_name)
            if val is None:
                continue
            try:
                cast_val = caster(val)
                target = getattr(cfg, section, None)
                if target is not None and hasattr(target, field):
                    setattr(target, field, cast_val)
                    logger.info("Applied ENV override: %s=%s -> %s.%s", env_name, val, section, field)
            except Exception as e:
                logger.warning("Failed to apply ENV override %s=%s: %s", env_name, val, e)

        return cfg

    def validate_configuration(self) -> List[str]:
        """
        Доп. мягкие проверки (только предупреждения для пользователя).
        """
        warnings: List[str] = []

        if not Path(self.paths.input_dir).exists():
            warnings.append(f"Input directory does not exist: {self.paths.input_dir}")

        if self.chunker.min_chars >= self.chunker.max_chars_code:
            warnings.append(
                f"min_chars ({self.chunker.min_chars}) >= max_chars_code ({self.chunker.max_chars_code}); "
                f"рекомендуется увеличить max_chars_code или уменьшить min_chars."
            )

        return warnings

    def ensure_directories(self) -> None:
        self.paths.ensure_directories()

    # Оставляем .dict() для обратной совместимости со старым кодом
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths": self.paths.dict(),
            "chunker": self.chunker.dict(),
            "abbreviations": self.abbreviations,
        }

    def __str__(self) -> str:
        return (
            f"ChunkingConfig(paths={self.paths.dict()}, "
            f"chunker={self.chunker.dict()}, "
            f"abbreviations_keys={list(self.abbreviations.keys())})"
        )


# — Утилиты обратной совместимости —
def load_config(config_path: str) -> ChunkingConfig:
    return ChunkingConfig.from_yaml(config_path)


def create_default_config() -> ChunkingConfig:
    return ChunkingConfig(
        paths=PathsConfig(
            input_dir="./input",
            preprocessed_dir="./preprocessed",
            chunks_dir="./chunks",
        ),
        chunker=ChunkerConfig(),
        abbreviations={},
    )
