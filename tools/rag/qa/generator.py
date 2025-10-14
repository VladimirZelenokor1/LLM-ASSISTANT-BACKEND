from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Protocol

from core.settings import get_llm_settings
from core.llm_provider import get_llm, LLMResponse, LLMChunk

logger = logging.getLogger(__name__)


class GenerationType(str, Enum):
    """Типы генерации ответов."""
    COMPLETE = "complete"
    STREAMING = "streaming"


class MessageRole(str, Enum):
    """Роли сообщений в диалоге."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Модель сообщения в диалоге."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует сообщение в словарь."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Citation:
    """Модель цитаты/источника."""
    id: str
    text: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует цитату в словарь."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata
        }


@dataclass
class GenerationConfig:
    """Конфигурация генерации."""
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Результат генерации ответа."""
    answer: str
    citations: List[Citation]
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует результат в словарь."""
        result = {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "model": self.model,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.usage:
            result["usage"] = self.usage
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        if self.error:
            result["error"] = self.error
            
        return result


@dataclass
class StreamingChunk:
    """Чанк потокового ответа."""
    delta: str
    finish_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует чанк в словарь."""
        result = {
            "type": "chunk",
            "delta": self.delta,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
            
        return result


@dataclass
class StreamingMetadata:
    """Метаданные потокового ответа."""
    model: str
    citations: List[Citation]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует метаданные в словарь."""
        return {
            "type": "metadata",
            "model": self.model,
            "citations": [citation.to_dict() for citation in self.citations],
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class StreamingError:
    """Ошибка потокового ответа."""
    error: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует ошибку в словарь."""
        return {
            "type": "error",
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }


class GenerationError(Exception):
    """Исключение для ошибок генерации."""
    pass


class PromptValidator:
    """Валидатор промптов и входных данных."""
    
    @staticmethod
    def validate_system_prompt(prompt: str) -> None:
        """Валидирует системный промпт."""
        if not prompt or not prompt.strip():
            raise ValueError("System prompt cannot be empty")
        
        if len(prompt) > 10000:
            raise ValueError("System prompt is too long (max 10000 characters)")
    
    @staticmethod
    def validate_user_prompt(prompt: str) -> None:
        """Валидирует пользовательский промпт."""
        if not prompt or not prompt.strip():
            raise ValueError("User prompt cannot be empty")
        
        if len(prompt) > 50000:
            raise ValueError("User prompt is too long (max 50000 characters)")
    
    @staticmethod
    def validate_citations(citations: List[Dict[str, Any]]) -> List[Citation]:
        """Валидирует и конвертирует цитаты."""
        validated_citations = []
        
        for i, citation_dict in enumerate(citations or []):
            try:
                citation_id = citation_dict.get("id", f"citation_{i}")
                citation_text = citation_dict.get("text", "")
                
                if not citation_text.strip():
                    logger.warning(f"Citation {citation_id} has empty text, skipping")
                    continue
                
                citation = Citation(
                    id=citation_id,
                    text=citation_text,
                    source=citation_dict.get("source"),
                    metadata=citation_dict.get("metadata", {})
                )
                validated_citations.append(citation)
                
            except Exception as e:
                logger.warning(f"Invalid citation at index {i}: {e}")
                continue
        
        return validated_citations
    
    @staticmethod
    def prepare_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Подготавливает сообщения для LLM."""
        PromptValidator.validate_system_prompt(system_prompt)
        PromptValidator.validate_user_prompt(user_prompt)
        
        return [
            {"role": MessageRole.SYSTEM.value, "content": system_prompt},
            {"role": MessageRole.USER.value, "content": user_prompt}
        ]


class LLMProvider(Protocol):
    """Protocol для LLM провайдеров."""
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse: ...
    
    def generate_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[LLMChunk, None, None]: ...


class AnswerGenerator:
    """
    Генератор ответов с использованием единого LLM провайдера.
    
    Обеспечивает:
    - Генерацию полных ответов с цитатами
    - Потоковую генерацию ответов
    - Валидацию входных данных
    - Обработку ошибок
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Инициализирует генератор ответов.
        
        Args:
            llm_provider: Провайдер LLM (если None, используется провайдер по умолчанию)
        """
        self.llm = llm_provider or get_llm()
        self.settings = get_llm_settings()
        self.validator = PromptValidator()
        
        logger.info(
            f"AnswerGenerator initialized with model: {self.settings.model}, "
            f"provider: {type(self.llm).__name__}"
        )
    
    def generate_answer(
        self,
        system_prompt: str,
        user_prompt: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Генерирует полный ответ с цитатами.
        
        Args:
            system_prompt: Системный промпт
            user_prompt: Пользовательский промпт
            citations: Список цитат/источников
            **kwargs: Дополнительные параметры генерации
            
        Returns:
            Словарь с ответом и метаданными
            
        Raises:
            GenerationError: При ошибках генерации
        """
        logger.info("Starting answer generation")
        
        try:
            # Валидируем и подготавливаем входные данные
            messages = self.validator.prepare_messages(system_prompt, user_prompt)
            validated_citations = self.validator.validate_citations(citations)
            
            # Получаем конфигурацию генерации
            generation_config = self._build_generation_config(**kwargs)
            
            # Выполняем генерацию
            response: LLMResponse = self.llm.generate(
                messages,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_tokens,
                top_p=generation_config.top_p,
                frequency_penalty=generation_config.frequency_penalty,
                presence_penalty=generation_config.presence_penalty,
                stop=generation_config.stop_sequences or None
            )
            
            # Создаем результат
            result = GenerationResult(
                answer=response.text,
                citations=validated_citations,
                model=response.model or self.settings.model,
                usage=response.usage,
                finish_reason=response.finish_reason.value if response.finish_reason else None
            )
            
            logger.info(
                f"Answer generation completed: {len(response.text)} characters, "
                f"model: {result.model}, finish_reason: {result.finish_reason}"
            )
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            
            error_result = GenerationResult(
                answer=f"Sorry, I couldn't generate an answer due to an error: {str(e)}",
                citations=[],
                model=self.settings.model,
                error=str(e)
            )
            
            return error_result.to_dict()
    
    def generate_answer_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Генерирует потоковый ответ с цитатами.
        
        Args:
            system_prompt: Системный промпт
            user_prompt: Пользовательский промпт
            citations: Список цитат/источников
            **kwargs: Дополнительные параметры генерации
            
        Yields:
            Словари с чанками ответа и метаданными
            
        Raises:
            GenerationError: При ошибках генерации
        """
        logger.info("Starting streaming answer generation")
        
        try:
            # Валидируем и подготавливаем входные данные
            messages = self.validator.prepare_messages(system_prompt, user_prompt)
            validated_citations = self.validator.validate_citations(citations)
            
            # Получаем конфигурацию генерации
            generation_config = self._build_generation_config(**kwargs)
            
            # Отправляем метаданные
            metadata = StreamingMetadata(
                model=self.settings.model,
                citations=validated_citations
            )
            yield metadata.to_dict()
            
            # Стримим ответ
            for chunk in self.llm.generate_stream(
                messages,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_tokens,
                top_p=generation_config.top_p,
                frequency_penalty=generation_config.frequency_penalty,
                presence_penalty=generation_config.presence_penalty,
                stop=generation_config.stop_sequences or None
            ):
                streaming_chunk = StreamingChunk(
                    delta=chunk.delta_text,
                    finish_reason=chunk.finish_reason.value if chunk.finish_reason else None
                )
                yield streaming_chunk.to_dict()
                
        except Exception as e:
            logger.error(f"Streaming answer generation failed: {e}", exc_info=True)
            
            error = StreamingError(error=str(e))
            yield error.to_dict()
    
    def _build_generation_config(self, **kwargs) -> GenerationConfig:
        """
        Создает конфигурацию генерации из аргументов и настроек.
        
        Args:
            **kwargs: Параметры переопределения
            
        Returns:
            Конфигурация генерации
        """
        return GenerationConfig(
            temperature=kwargs.get('temperature', self.settings.temperature),
            max_tokens=kwargs.get('max_tokens', self.settings.max_output_tokens),
            top_p=kwargs.get('top_p', self.settings.top_p),
            frequency_penalty=kwargs.get('frequency_penalty', 0.0),
            presence_penalty=kwargs.get('presence_penalty', 0.0),
            stop_sequences=kwargs.get('stop_sequences', [])
        )


class AnswerGeneratorFactory:
    """Фабрика для создания генераторов ответов."""
    
    @staticmethod
    def create_default_generator() -> AnswerGenerator:
        """
        Создает генератор ответов с настройками по умолчанию.
        
        Returns:
            Экземпляр AnswerGenerator
        """
        return AnswerGenerator()
    
    @staticmethod
    def create_generator_with_provider(llm_provider: LLMProvider) -> AnswerGenerator:
        """
        Создает генератор ответов с указанным провайдером.
        
        Args:
            llm_provider: Провайдер LLM
            
        Returns:
            Экземпляр AnswerGenerator
        """
        return AnswerGenerator(llm_provider=llm_provider)


# Функции для обратной совместимости
def create_answer_generator() -> AnswerGenerator:
    """
    Создает генератор ответов для обратной совместимости.
    
    Returns:
        Экземпляр AnswerGenerator
    """
    return AnswerGeneratorFactory.create_default_generator()


# Пример использования
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    
    # Создание генератора
    generator = create_answer_generator()
    
    # Пример системного и пользовательского промптов
    system_prompt = "You are a helpful assistant that provides accurate and concise answers."
    user_prompt = "Explain the concept of machine learning in simple terms."
    
    # Пример цитат
    citations = [
        {
            "id": "citation_1",
            "text": "Machine learning is a subset of artificial intelligence.",
            "source": "Introduction to ML",
            "metadata": {"page": 42}
        }
    ]
    
    try:
        # Генерация ответа
        result = generator.generate_answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            citations=citations,
            temperature=0.7
        )
        
        print("Generated Answer:")
        print(result["answer"])
        print(f"\nModel: {result['model']}")
        print(f"Citations: {len(result['citations'])}")
        
    except Exception as e:
        print(f"Error: {e}")