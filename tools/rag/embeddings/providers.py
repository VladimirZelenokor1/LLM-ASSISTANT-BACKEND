from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Type

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

EmbeddingArray = npt.NDArray[np.float32]


class ProviderType(str, Enum):
    E5 = "e5"
    OPENAI = "openai"


class EmbeddingTask(str, Enum):
    PASSAGE = "passage"
    QUERY = "query"


# =========================== Configs ===========================

@dataclass
class BaseEmbeddingConfig:
    model: str = ""  # конкретные дефолты проставим в наследниках, если пусто
    normalize: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    batch_size: int = 64  # как в старой реализации

    def validate(self) -> None:
        if not self.model:
            raise ValueError("Model name cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class E5Config(BaseEmbeddingConfig):
    device: str = "cpu"
    model_revision: Optional[str] = None
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model:
            self.model = "intfloat/e5-base-v2"  # дефолт как в старой версии
        self.validate()
        if not self.device:
            raise ValueError("Device cannot be empty")


@dataclass
class OpenAIConfig(BaseEmbeddingConfig):
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    MODEL_DIMENSIONS: Dict[str, int] = field(default_factory=lambda: {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    })

    def __post_init__(self) -> None:
        if not self.model:
            self.model = "text-embedding-3-small"
        self.validate()
        if self.model not in self.MODEL_DIMENSIONS:
            logger.warning(
                "Unknown OpenAI model: %s. Known: %s",
                self.model, list(self.MODEL_DIMENSIONS.keys())
            )


# =========================== Base Provider ===========================

class EmbeddingProvider(ABC):
    """
    Абстрактный провайдер.
    Старый контракт требует .name, .dim, embed_passages, embed_queries.
    """

    def __init__(self, config: BaseEmbeddingConfig):
        self._config = config
        self._initialize_provider()

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...

    # --- Back-compat alias ---
    @property
    def dim(self) -> int:
        """Совместимость со старой версией (alias на dimension)."""
        return self.dimension

    @property
    def config(self) -> BaseEmbeddingConfig:
        return self._config

    @abstractmethod
    def _initialize_provider(self) -> None:
        ...

    @abstractmethod
    def _embed_batch(self, texts: Sequence[str], task: EmbeddingTask) -> EmbeddingArray:
        ...

    def embed_passages(self, passages: Sequence[str]) -> EmbeddingArray:
        return self._embed_with_task(passages, EmbeddingTask.PASSAGE)

    def embed_queries(self, queries: Sequence[str]) -> EmbeddingArray:
        return self._embed_with_task(queries, EmbeddingTask.QUERY)

    def _embed_with_task(self, texts: Sequence[str], task: EmbeddingTask) -> EmbeddingArray:
        if not texts:
            # (0, dim) — как раньше
            return np.zeros((0, self.dim), dtype=np.float32)

        logger.debug(
            "Embedding %d texts with %s (model=%s, task=%s)",
            len(texts), self.name, self.config.model, task.value
        )

        start = time.time()
        all_embeddings: List[EmbeddingArray] = []
        try:
            bs = int(self.config.batch_size) or 64
            for i in range(0, len(texts), bs):
                batch = texts[i:i + bs]
                embs = self._embed_batch(batch, task)
                all_embeddings.append(embs)

            result = np.vstack(all_embeddings) if all_embeddings else np.zeros((0, self.dim), dtype=np.float32)
            elapsed = time.time() - start
            if elapsed > 0:
                logger.debug("Embedding completed: %d texts in %.2fs (%.1f texts/s)",
                             len(texts), elapsed, len(texts) / elapsed)
            return result
        except Exception as e:
            logger.error("Embedding failed for %d texts with %s: %s", len(texts), self.name, e)
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, dim={self.dim})"


class EmbeddingError(Exception):
    pass


# =========================== E5 Provider ===========================

class E5Provider(EmbeddingProvider):
    """
    intfloat/e5-* с префиксами 'query:' / 'passage:'.
    Поведение и интерфейс — как в старой версии.
    """

    def __init__(self, config: E5Config):
        self._model: Any = None
        self._dim: int = 0
        super().__init__(config)

    @property
    def name(self) -> str:
        return ProviderType.E5.value

    @property
    def dimension(self) -> int:
        return self._dim

    def _initialize_provider(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "sentence_transformers is required for E5. Install: pip install sentence-transformers"
            ) from e

        logger.info("Initializing E5 provider with model: %s, device: %s",
                    self.config.model, self.config.device)
        try:
            self._model = SentenceTransformer(
                self.config.model,
                device=self.config.device,
                revision=getattr(self.config, 'model_revision', None),
                trust_remote_code=getattr(self.config, 'trust_remote_code', False),
            )
            self._dim = int(self._model.get_sentence_embedding_dimension())
            logger.debug("E5 initialized: dim=%d, normalize=%s", self._dim, self.config.normalize)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize E5 model: {e}") from e

    def _embed_batch(self, texts: Sequence[str], task: EmbeddingTask) -> EmbeddingArray:
        # Префиксы, как раньше
        if task == EmbeddingTask.PASSAGE:
            prefixed = [f"passage: {t}" for t in texts]
        else:
            prefixed = [f"query: {t}" for t in texts]

        try:
            embs = self._model.encode(
                prefixed,
                batch_size=int(self.config.batch_size) or 64,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
                show_progress_bar=False,
            )
            return np.asarray(embs, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"E5 encode failed: {e}") from e


# =========================== OpenAI Provider ===========================

class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI embeddings (>=1.x). Возвращаем L2-нормализованные вектора,
    как делали вручную раньше.
    """

    def __init__(self, config: OpenAIConfig):
        self._client: Any = None
        self._model_dimension: int = 1536
        super().__init__(config)

    @property
    def name(self) -> str:
        return ProviderType.OPENAI.value

    @property
    def dimension(self) -> int:
        return self._model_dimension

    def _initialize_provider(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise EmbeddingError("openai package is required for OpenAI provider. Install: pip install openai") from e

        logger.info("Initializing OpenAI provider with model: %s", self.config.model)
        try:
            self._client = OpenAI(
                api_key=self.config.api_key,
                organization=getattr(self.config, 'organization', None),
                base_url=getattr(self.config, 'base_url', None),
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
            )
            dim = self.config.MODEL_DIMENSIONS.get(self.config.model)
            self._model_dimension = int(dim) if dim else 1536
            logger.debug("OpenAI initialized: dim=%d, normalize=%s", self._model_dimension, self.config.normalize)
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize OpenAI client: {e}") from e

    def _embed_batch(self, texts: Sequence[str], task: EmbeddingTask) -> EmbeddingArray:
        # E5-совместимые префиксы
        prefixed = [f"{'passage' if task == EmbeddingTask.PASSAGE else 'query'}: {t}" for t in texts]
        try:
            resp = self._client.embeddings.create(model=self.config.model, input=prefixed)
            arr = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            if self.config.normalize:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.maximum(norms, 1e-12)
            return arr
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise EmbeddingError(f"OpenAI API call failed: {e}") from e


# =========================== Factory ===========================

class EmbeddingProviderFactory:
    """Создание провайдеров по словарю конфигурации."""

    _CONFIG_BY_TYPE: Dict[ProviderType, Type[BaseEmbeddingConfig]] = {
        ProviderType.E5: E5Config,
        ProviderType.OPENAI: OpenAIConfig,
    }
    _CLASS_BY_TYPE: Dict[ProviderType, Type[EmbeddingProvider]] = {
        ProviderType.E5: E5Provider,
        ProviderType.OPENAI: OpenAIProvider,
    }

    @staticmethod
    def create_provider(provider_type: ProviderType, config_dict: Dict[str, Any]) -> EmbeddingProvider:
        if provider_type not in EmbeddingProviderFactory._CLASS_BY_TYPE:
            raise ValueError(f"Unknown provider type: {provider_type}")

        cfg_cls = EmbeddingProviderFactory._CONFIG_BY_TYPE[provider_type]
        prov_cls = EmbeddingProviderFactory._CLASS_BY_TYPE[provider_type]

        try:
            cfg = cfg_cls(**(config_dict or {}))
            prov = prov_cls(cfg)
            logger.info("Created %s provider: model=%s, dim=%d", provider_type.value, cfg.model, prov.dim)
            return prov
        except Exception as e:
            raise EmbeddingError(f"Failed to create {provider_type.value} provider: {e}") from e

    @staticmethod
    def create_provider_from_config(full_config: Dict[str, Any]) -> EmbeddingProvider:
        provider_cfg = full_config.get("provider", {}) or {}
        name = (provider_cfg.get("name") or "").lower()
        try:
            ptype = ProviderType(name)
        except ValueError as e:
            raise ValueError(f"Unsupported provider: {name}") from e
        specific = provider_cfg.get(name, {}) or {}
        return EmbeddingProviderFactory.create_provider(ptype, specific)


# =========================== Back-compat helpers ===========================

def create_e5_provider(config_dict: Dict[str, Any]) -> E5Provider:
    return EmbeddingProviderFactory.create_provider(ProviderType.E5, config_dict)  # type: ignore[return-value]


def create_openai_provider(config_dict: Dict[str, Any]) -> OpenAIProvider:
    return EmbeddingProviderFactory.create_provider(ProviderType.OPENAI, config_dict)  # type: ignore[return-value]
