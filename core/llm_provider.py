# core/llm_provider.py
"""
Единый интерфейс поставщика услуг LLM.

Особенности:
- Клиенты httpx, поддерживающие синхронизацию, с объединением подключений
- Потоковая передача с поддержкой инструментальных вызовов
- Нормализация ответа / завершения / использования у разных поставщиков
- Комплексная обработка ошибок и повторные попытки
- Обратно совместимый метод get_llm() (без аргументов, одноэлементный)
- Оболочка в стиле ответов OpenAI: client.responses.create(...)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .settings import LLMSettings

logger = logging.getLogger(__name__)


# =============================================================================
# МОДЕЛИ ДАННЫХ
# =============================================================================

class FinishReason(Enum):
    """Причины завершения генерации модели."""
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """Представление вызова инструмента (function call)."""
    id: Optional[str]
    type: str  # "function"
    function_name: Optional[str]
    arguments_json: str  # raw JSON string


@dataclass
class LLMResponse:
    """Нормализованный ответ от LLM провайдера."""
    text: str
    usage: Dict[str, int]
    finish_reason: Optional[FinishReason] = None
    model: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class LLMChunk:
    """Чанк потокового ответа от LLM."""
    delta_text: str = ""
    finish_reason: Optional[FinishReason] = None
    usage: Optional[Dict[str, int]] = None
    delta_tool_call: Optional[ToolCall] = None


# =============================================================================
# ИСКЛЮЧЕНИЯ
# =============================================================================

class LLMError(Exception):
    """Базовое исключение для ошибок LLM провайдера."""


class LLMTimeoutError(LLMError):
    """Таймаут запроса к LLM."""


class LLMRateLimitError(LLMError):
    """Превышение лимитов запросов."""


class LLMAuthorizationError(LLMError):
    """Ошибка авторизации."""


class LLMBadRequestError(LLMError):
    """Некорректный запрос."""


class LLMServerError(LLMError):
    """Ошибка сервера LLM."""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _map_finish_reason(reason: Optional[str]) -> FinishReason:
    """Преобразует строковую причину завершения в enum."""
    if not reason:
        return FinishReason.UNKNOWN
    try:
        return FinishReason(reason)
    except ValueError:
        logger.debug("Unknown finish reason: %s", reason)
        return FinishReason.UNKNOWN


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Нормализует сообщения чата к минимальной схеме."""
    normalized = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        # Нормализация ролей; "developer" маппим в "system" (как в старой версии)
        if role == "developer":
            role = "system"
        elif role not in {"system", "user", "assistant", "tool", "function"}:
            role = "user"

        normalized.append({
            "role": role,
            "content": content if content is not None else ""
        })
    return normalized


# =============================================================================
# ИНТЕРФЕЙС БАЗОВОГО ПРОВАЙДЕРА
# =============================================================================

class LLMProvider(ABC):
    """Абстрактный базовый класс для LLM провайдеров."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> LLMProvider:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Освобождает ресурсы клиента."""
        if self._client:
            try:
                self._client.close()
            except Exception as error:
                logger.warning("Error closing HTTP client: %s", error)

    @abstractmethod
    def generate(
            self,
            messages: List[Dict[str, Any]],
            *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            timeout: Optional[float] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
            response_format: Optional[Union[str, Dict[str, Any]]] = None,
            extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        ...

    @abstractmethod
    def generate_stream(
            self,
            messages: List[Dict[str, Any]],
            **kwargs: Any,
    ) -> Generator[LLMChunk, None, None]:
        ...


# =============================================================================
# OPENAI-COMPATIBLE ПРОВАЙДЕРЫ
# =============================================================================

class OpenAIProvider(LLMProvider):
    """Провайдер для OpenAI и совместимых API."""

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self._initialize_client()

    def _initialize_client(self) -> None:
        api_key = self.settings.get_api_key()
        if not api_key:
            raise LLMAuthorizationError("API key is required for OpenAI provider")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers = {k: v for k, v in headers.items() if v is not None}

        self._client = httpx.Client(
            base_url=self.settings.base_url or "https://api.openai.com/v1",
            timeout=self.settings.timeout_sec,
            headers=headers,
        )

    def _build_request_payload(
            self,
            messages: List[Dict[str, Any]],
            temperature: Optional[float],
            max_tokens: Optional[int],
            top_p: Optional[float],
            tools: Optional[List[Dict[str, Any]]],
            tool_choice: Optional[Union[str, Dict[str, Any]]],
            response_format: Optional[Union[str, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": _normalize_messages(messages),
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.settings.max_output_tokens,
            "top_p": top_p if top_p is not None else self.settings.top_p,
            "frequency_penalty": self.settings.frequency_penalty,
            "presence_penalty": self.settings.presence_penalty,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        return payload

    def _handle_http_error(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            if status_code in (401, 403):
                raise LLMAuthorizationError("Authentication failed") from error
            elif status_code == 408:
                raise LLMTimeoutError("Request timeout") from error
            elif status_code in (409, 422):
                raise LLMBadRequestError(f"Bad request: {error.response.text}") from error
            elif status_code == 429:
                raise LLMRateLimitError("Rate limit exceeded") from error
            elif 500 <= status_code < 600:
                raise LLMServerError(f"Server error {status_code}") from error
            else:
                raise LLMError(f"HTTP error {status_code}: {error.response.text}") from error

    @retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMServerError, LLMTimeoutError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(
            self,
            messages: List[Dict[str, Any]],
            *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            timeout: Optional[float] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
            response_format: Optional[Union[str, Dict[str, Any]]] = None,
            extra_headers: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:
        payload = self._build_request_payload(
            messages, temperature, max_tokens, top_p, tools, tool_choice, response_format
        )
        payload["stream"] = False

        headers = dict(self._client.headers)
        if extra_headers:
            headers.update(extra_headers)

        try:
            logger.debug("Sending request to OpenAI API with model: %s", self.settings.model)
            response = self._client.post(
                "/chat/completions",
                json=payload,
                timeout=timeout or self.settings.timeout_sec,
                headers=headers,
            )
            self._handle_http_error(response)
            data = response.json()
            return self._parse_response(data)
        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Request timeout") from error
        except Exception as error:
            if isinstance(error, LLMError):
                raise
            raise LLMError(f"OpenAI API error: {error}") from error

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        choice = data["choices"][0]
        message = choice["message"]

        tool_calls: Optional[List[ToolCall]] = None
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                tool_calls.append(
                    ToolCall(
                        id=tool_call.get("id"),
                        type=tool_call.get("type", "function"),
                        function_name=function.get("name"),
                        arguments_json=function.get("arguments", ""),
                    )
                )

        usage_data = data.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

        return LLMResponse(
            text=message.get("content", ""),
            usage=usage,
            finish_reason=_map_finish_reason(choice.get("finish_reason")),
            model=data.get("model"),
            tool_calls=tool_calls,
            raw=data,
        )

    def generate_stream(
            self,
            messages: List[Dict[str, Any]],
            **kwargs: Any,
    ) -> Generator[LLMChunk, None, None]:
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        top_p = kwargs.get("top_p")
        timeout = kwargs.get("timeout")
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        response_format = kwargs.get("response_format")
        extra_headers = kwargs.get("extra_headers")

        payload = self._build_request_payload(
            messages, temperature, max_tokens, top_p, tools, tool_choice, response_format
        )
        payload["stream"] = True

        headers = dict(self._client.headers)
        if extra_headers:
            headers.update(extra_headers)

        try:
            with self._client.stream(
                    "POST",
                    "/chat/completions",
                    json=payload,
                    timeout=timeout or self.settings.timeout_sec,
                    headers=headers,
            ) as response:
                self._handle_http_error(response)

                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data)
                            yield from self._parse_stream_chunk(chunk_data)
                        except json.JSONDecodeError:
                            logger.debug("Failed to parse JSON chunk: %s", data)
                            continue

        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Stream timeout") from error
        except Exception as error:
            if isinstance(error, LLMError):
                raise
            raise LLMError(f"Stream error: {error}") from error

    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Generator[LLMChunk, None, None]:
        choice = chunk_data.get("choices", [{}])[0]
        delta = choice.get("delta", {})

        if "content" in delta and delta["content"] is not None:
            yield LLMChunk(delta_text=delta["content"])

        if "tool_calls" in delta and delta["tool_calls"]:
            tool_call = delta["tool_calls"][0]
            function = tool_call.get("function", {})
            yield LLMChunk(
                delta_tool_call=ToolCall(
                    id=tool_call.get("id"),
                    type=tool_call.get("type", "function"),
                    function_name=function.get("name"),
                    arguments_json=function.get("arguments", ""),
                )
            )

        if choice.get("finish_reason"):
            yield LLMChunk(
                delta_text="",
                finish_reason=_map_finish_reason(choice.get("finish_reason")),
            )


class OpenRouterProvider(OpenAIProvider):
    """Провайдер для OpenRouter API (OpenAI-совместимый endpoint)."""

    def _initialize_client(self) -> None:
        if not self.settings.openrouter_api_key:
            raise LLMAuthorizationError("OpenRouter API key is required")

        base_url = self.settings.get_base_url() or "https://openrouter.ai/api/v1"
        openrouter_headers = self.settings.get_openrouter_headers()  # может вернуть {'HTTP-Referer': ..., 'X-Title': ...}

        headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Content-Type": "application/json",
            **(openrouter_headers or {}),
        }
        headers = {k: v for k, v in headers.items() if v is not None}

        self._client = httpx.Client(
            base_url=base_url,
            timeout=self.settings.timeout_sec,
            headers=headers,
        )


class VLLMProvider(OpenAIProvider):
    """Провайдер для vLLM и других OpenAI-совместимых серверов."""

    def _initialize_client(self) -> None:
        self._client = httpx.Client(
            base_url=self.settings.base_url or "http://localhost:8000/v1",
            timeout=self.settings.timeout_sec,
            headers={"Content-Type": "application/json"},
        )


# =============================================================================
# ПРОЧИЕ ПРОВАЙДЕРЫ (Ollama, Transformers, Llama.cpp)
# =============================================================================

class OllamaProvider(LLMProvider):
    """Провайдер для локального Ollama."""

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self._client = httpx.Client(
            base_url=settings.ollama_host or "http://localhost:11434",
            timeout=settings.timeout_sec,
        )

    @retry(
        retry=retry_if_exception_type((LLMRateLimitError, LLMServerError, LLMTimeoutError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def generate(
            self,
            messages: List[Dict[str, Any]],
            *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> LLMResponse:
        payload = {
            "model": self.settings.model,
            "messages": _normalize_messages(messages),
            "options": {
                "temperature": temperature or self.settings.temperature,
                "num_predict": max_tokens or self.settings.max_output_tokens,
                "top_p": top_p or self.settings.top_p,
            },
            "stream": False,
        }
        try:
            response = self._client.post("/api/chat", json=payload, timeout=timeout or self.settings.timeout_sec)
            response.raise_for_status()
            data = response.json()
            content = (data.get("message") or {}).get("content", "")
            return LLMResponse(
                text=content,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                finish_reason=FinishReason.STOP,
                model=self.settings.model,
                raw=data,
            )
        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Ollama timeout") from error
        except httpx.HTTPStatusError as error:
            raise LLMServerError(f"Ollama HTTP {error.response.status_code}") from error
        except Exception as error:
            raise LLMError(f"Ollama error: {error}") from error

    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Generator[LLMChunk, None, None]:
        payload = {
            "model": self.settings.model,
            "messages": _normalize_messages(messages),
            "options": {
                "temperature": kwargs.get("temperature", self.settings.temperature),
                "num_predict": kwargs.get("max_tokens", self.settings.max_output_tokens),
                "top_p": kwargs.get("top_p", self.settings.top_p),
            },
            "stream": True,
        }
        try:
            with self._client.stream(
                    "POST",
                    "/api/chat",
                    json=payload,
                    timeout=kwargs.get("timeout", self.settings.timeout_sec),
            ) as response:
                response.raise_for_status()
                for raw in response.iter_lines():
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if obj.get("done"):
                        fr = obj.get("done_reason")
                        yield LLMChunk(delta_text="", finish_reason=_map_finish_reason(fr))
                        break
                    msg = obj.get("message") or {}
                    if "content" in msg and msg["content"] is not None:
                        yield LLMChunk(delta_text=msg["content"])
        except httpx.TimeoutException as error:
            raise LLMTimeoutError("Ollama stream timeout") from error
        except Exception as error:
            raise LLMError(f"Ollama stream error: {error}") from error


class TransformersProvider(LLMProvider):
    """Провайдер для локальных transformers моделей."""

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self.pipeline = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import pipeline, AutoTokenizer
            import torch
            logger.info("Loading transformers model: %s", self.settings.transformers_model)
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.transformers_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            device_map = "auto"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.pipeline = pipeline(
                "text-generation",
                model=self.settings.transformers_model,
                tokenizer=self.tokenizer,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except ImportError as error:
            raise ImportError("Transformers not installed. pip install transformers torch") from error
        except Exception as error:
            raise LLMError(f"Failed to load transformers model: {error}") from error

    def generate(
            self,
            messages: List[Dict[str, Any]],
            *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> LLMResponse:
        prompt = self._format_messages(messages)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens or self.settings.max_output_tokens,
            temperature=temperature or self.settings.temperature,
            top_p=top_p or self.settings.top_p,
            do_sample=True,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = outputs[0]["generated_text"]
        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(generated_text))
        return LLMResponse(
            text=generated_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            finish_reason=FinishReason.STOP,
            model=self.settings.transformers_model,
        )

    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Generator[LLMChunk, None, None]:
        resp = self.generate(messages, **kwargs)
        for w in resp.text.split():
            yield LLMChunk(delta_text=w + " ")
        yield LLMChunk(delta_text="", finish_reason=resp.finish_reason, usage=resp.usage)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        for m in _normalize_messages(messages):
            r, c = m["role"], m["content"]
            if r == "system":
                parts.append(f"<|system|>\n{c}")
            elif r == "user":
                parts.append(f"<|user|>\n{c}")
            elif r == "assistant":
                parts.append(f"<|assistant|>\n{c}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)


class LlamaCppProvider(LLMProvider):
    """Провайдер для llama.cpp."""

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self._llm = None
        if not settings.llama_cpp_model_path:
            raise ValueError("LLAMA_CPP_MODEL_PATH is required for llama_cpp provider")

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.settings.llama_cpp_model_path,
                n_ctx=self.settings.llama_cpp_n_ctx or 4096,
                n_gpu_layers=self.settings.llama_cpp_n_gpu_layers or -1,
                verbose=False,
            )
        except ImportError as error:
            raise ImportError("llama-cpp-python not installed") from error
        except Exception as error:
            raise LLMError(f"Failed to load llama.cpp model: {error}") from error

    def generate(
            self,
            messages: List[Dict[str, Any]],
            *,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            timeout: Optional[float] = None,
            **kwargs: Any,
    ) -> LLMResponse:
        self._ensure_loaded()
        result = self._llm.create_chat_completion(
            messages=_normalize_messages(messages),
            temperature=temperature or self.settings.temperature,
            max_tokens=max_tokens or self.settings.max_output_tokens,
            top_p=top_p or self.settings.top_p,
            stream=False,
        )
        choice = result["choices"][0]
        message = choice["message"]
        usage_data = result.get("usage", {})
        usage = {
            "prompt_tokens": usage_data.get("prompt_tokens", 0),
            "completion_tokens": usage_data.get("completion_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }
        return LLMResponse(
            text=message.get("content", ""),
            usage=usage,
            finish_reason=_map_finish_reason(choice.get("finish_reason")),
            model=self.settings.model,
            raw=result,
        )

    def generate_stream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> Generator[LLMChunk, None, None]:
        self._ensure_loaded()
        stream = self._llm.create_chat_completion(
            messages=_normalize_messages(messages),
            temperature=kwargs.get("temperature", self.settings.temperature),
            max_tokens=kwargs.get("max_tokens", self.settings.max_output_tokens),
            top_p=kwargs.get("top_p", self.settings.top_p),
            stream=True,
        )
        for ch in stream:
            choice = ch.get("choices", [{}])[0]
            delta = choice.get("delta", {}) or {}
            if "content" in delta and delta["content"] is not None:
                yield LLMChunk(delta_text=delta["content"])
            if choice.get("finish_reason"):
                yield LLMChunk(delta_text="", finish_reason=_map_finish_reason(choice.get("finish_reason")))


# =============================================================================
# ФАБРИКА, ОТВЕТНАЯ ПРОКЛАДКА, ГЛОБАЛЬНЫЙ ДОСТУП
# =============================================================================

def _ensure_responses_shim(client: LLMProvider) -> None:
    """
    Навешивает адаптер .responses.create(...) (OpenAI Responses API look-alike)
    на любой провайдер, если у него нет такого атрибута.
    Возвращает объект с полями: output_text, text, content, model, raw, output.
    """
    if hasattr(client, "responses"):
        return

    class _Resp:
        def __init__(self, text: str, model: Optional[str], raw: Any):
            self.output_text = text
            self.text = text
            self.content = text
            self.model = model
            self.raw = raw or {"text": text}
            # Приближённый формат OpenAI Responses API
            self.output = [{"content": [{"type": "output_text", "text": text}]}]

    class _ResponsesProxy:
        def __init__(self, outer: LLMProvider):
            self._outer = outer

        def create(self, *, model=None, input=None, messages=None,
                   temperature=None, max_tokens=None, top_p=None, **kwargs):
            # Приводим к messages при необходимости
            if messages is None:
                if input is None:
                    raise ValueError("Either 'messages' or 'input' must be provided")
                if isinstance(input, str):
                    messages = [{"role": "user", "content": input}]
                elif isinstance(input, (list, tuple)):
                    parts = []
                    for p in input:
                        if isinstance(p, dict) and "text" in p:
                            parts.append(p["text"])
                        else:
                            parts.append(str(p))
                    messages = [{"role": "user", "content": "".join(parts)}]
                else:
                    messages = [{"role": "user", "content": str(input)}]

            resp = self._outer.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            return _Resp(text=resp.text, model=resp.model, raw=resp.raw)

    setattr(client, "responses", _ResponsesProxy(client))


def build_llm(settings: LLMSettings) -> LLMProvider:
    """Создает экземпляр LLM провайдера по настройкам."""
    provider_name = settings.provider.lower()
    providers = {
        "openai": OpenAIProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "vllm": VLLMProvider,
        "transformers": TransformersProvider,
        "llama_cpp": LlamaCppProvider,
    }
    if provider_name not in providers:
        available = list(providers.keys())
        raise ValueError(f"Unknown LLM provider: {provider_name}. Available: {available}")

    provider_class = providers[provider_name]
    instance = provider_class(settings)
    _ensure_responses_shim(instance)  # важно для кода, который использует .responses
    return instance


# Глобальный singleton (как в старой версии)
_llm_instance: Optional[LLMProvider] = None


def get_llm() -> LLMProvider:
    """Возвращает глобальный экземпляр LLM провайдера (singleton, без аргументов)."""
    global _llm_instance
    if _llm_instance is None:
        from .settings import get_llm_settings
        settings = get_llm_settings()
        _llm_instance = build_llm(settings)
        logger.info("Initialized LLM provider: %s", settings.provider)
    return _llm_instance


def reset_llm() -> None:
    """Сбрасывает глобальный экземпляр провайдера (для тестов/хотрелоада)."""
    global _llm_instance
    if _llm_instance is not None:
        try:
            _llm_instance.close()
        except Exception as error:
            logger.warning("Error closing LLM instance: %s", error)
        finally:
            _llm_instance = None
    logger.debug("LLM provider reset")
