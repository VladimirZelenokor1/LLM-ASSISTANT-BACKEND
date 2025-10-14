# path: core/settings.py
from __future__ import annotations
import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# --- загрузим .env один раз, ДО чтения YAML/Settings
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


# ---------- YAML-схемы (простые, без плейсхолдеров) ----------
class LLMCore(BaseModel):
    provider: str = "openrouter"
    model: str = "openai/gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: int = 512
    timeout_sec: int = 60
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = True
    max_retries: int = 3
    tool_choice: Optional[str] = "auto"
    response_format: str = "none"
    json_schema_path: Optional[str] = None
    extra_headers: Dict[str, str] = {}


class OpenAIBlock(BaseModel):
    api_key: Optional[str] = None


class OpenRouterBlock(BaseModel):
    api_key: Optional[str] = None
    base_url: Optional[str] = "https://openrouter.ai/api/v1"
    extra_headers: Dict[str, str] = {}


class OllamaBlock(BaseModel):
    host: Optional[str] = "http://localhost:11434"


class VLLMBlock(BaseModel):
    base_url: Optional[str] = "http://localhost:8000/v1"


class TransformersBlock(BaseModel):
    model: Optional[str] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "auto"


class LlamaCppBlock(BaseModel):
    model_path: Optional[str] = None
    n_ctx: int = 4096
    n_gpu_layers: int = -1


class _YamlConfig(BaseModel):
    llm: LLMCore = LLMCore()
    openai: OpenAIBlock = OpenAIBlock()
    openrouter: OpenRouterBlock = OpenRouterBlock()
    ollama: OllamaBlock = OllamaBlock()
    vllm: VLLMBlock = VLLMBlock()
    transformers: TransformersBlock = TransformersBlock()
    llama_cpp: LlamaCppBlock = LlamaCppBlock()


# ---------- Settings из ENV (.env) ----------
class LLMSettings(BaseSettings):
    # core
    provider: str = "openrouter"
    model: str = "openai/gpt-4o-mini"
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_output_tokens: int = 512
    timeout_sec: int = 60
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stream: bool = True
    max_retries: int = 3
    tool_choice: Optional[str] = "auto"
    response_format: str = "none"
    json_schema_path: Optional[str] = None
    extra_headers: Dict[str, str] = {}

    # provider-specific (ENV aliases)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    openrouter_referer: Optional[str] = Field(default=None, alias="OPENROUTER_REFERER")
    openrouter_app_title: Optional[str] = Field(default=None, alias="OPENROUTER_APP_TITLE")
    openrouter_base_url: Optional[str] = Field(default=None, alias="OPENROUTER_BASE_URL")

    ollama_host: Optional[str] = Field(default=None, alias="OLLAMA_HOST")
    vllm_base_url: Optional[str] = Field(default=None, alias="VLLM_BASE_URL")
    transformers_model: Optional[str] = Field(default=None, alias="TRANSFORMERS_MODEL")

    llama_cpp_model_path: Optional[str] = Field(default=None, alias="LLAMA_CPP_MODEL_PATH")
    llama_cpp_n_ctx: Optional[int] = Field(default=None, alias="LLAMA_CPP_N_CTX")
    llama_cpp_n_gpu_layers: Optional[int] = Field(default=None, alias="LLAMA_CPP_N_GPU_LAYERS")

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),  # дубль безопасности
        env_prefix="",
        extra="ignore",
        populate_by_name=True,
    )

    # helpers
    def get_api_key(self) -> Optional[str]:
        if self.provider == "openai":
            return self.openai_api_key
        if self.provider == "openrouter":
            return self.openrouter_api_key
        return self.openai_api_key

    def get_base_url(self) -> Optional[str]:
        if self.provider == "vllm":
            return self.vllm_base_url or self.base_url
        if self.provider == "openrouter":
            return self.openrouter_base_url or self.base_url
        if self.provider == "ollama":
            return self.ollama_host
        return self.base_url

    def get_json_schema(self) -> Optional[Dict[str, Any]]:
        if self.response_format != "json_schema" or not self.json_schema_path:
            return None
        p = Path(self.json_schema_path)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def get_openrouter_headers(self) -> dict[str, str]:
        """Заголовки для OpenRouter (без None/пустых значений)."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}" if self.openrouter_api_key else None,
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_referer or "https://github.com/your-org/llm-assistant",
            "X-Title": self.openrouter_app_title or "LLM Assistant",
        }
        return {k: v for k, v in headers.items() if isinstance(v, str) and v.strip()}


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """
    1) читаем .env (через BaseSettings)
    2) дополняем из YAML только пустые поля
    """
    env_settings = LLMSettings()  # уже подтянул .env

    cfg_path = Path(os.getenv("LLM_CONFIG_PATH", _PROJECT_ROOT / "configs/llm.yaml"))
    y = _YamlConfig(**_read_yaml(cfg_path)) if cfg_path.exists() else _YamlConfig()

    # копия для дозаполнения
    s = env_settings.model_copy(deep=True)

    # core-значения: заполняем если пусто/None
    for name in [
        "provider", "model", "base_url", "temperature", "max_output_tokens",
        "timeout_sec", "top_p", "frequency_penalty", "presence_penalty",
        "stream", "max_retries", "tool_choice", "response_format",
        "json_schema_path", "extra_headers"
    ]:
        if getattr(s, name) in (None, "", []):
            setattr(s, name, getattr(y.llm, name))

    # provider-specific — только если пусто
    if not s.openrouter_base_url and y.openrouter.base_url:
        s.openrouter_base_url = y.openrouter.base_url
    if not s.openrouter_api_key and y.openrouter.api_key:
        s.openrouter_api_key = y.openrouter.api_key  # опционально
    if y.openrouter.extra_headers and not s.extra_headers:
        s.extra_headers = y.openrouter.extra_headers

    if not s.openai_api_key and y.openai.api_key:
        s.openai_api_key = y.openai.api_key

    if not s.ollama_host and y.ollama.host:
        s.ollama_host = y.ollama.host

    if not s.vllm_base_url and y.vllm.base_url:
        s.vllm_base_url = y.vllm.base_url

    if not s.transformers_model and y.transformers.model:
        s.transformers_model = y.transformers.model

    if not s.llama_cpp_model_path and y.llama_cpp.model_path:
        s.llama_cpp_model_path = y.llama_cpp.model_path
    if not s.llama_cpp_n_ctx and y.llama_cpp.n_ctx:
        s.llama_cpp_n_ctx = y.llama_cpp.n_ctx
    if not s.llama_cpp_n_gpu_layers and y.llama_cpp.n_gpu_layers is not None:
        s.llama_cpp_n_gpu_layers = y.llama_cpp.n_gpu_layers

    return s
