# tools/rag/qa/qa_tool.py
from __future__ import annotations

import logging
import os
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import yaml

from core.llm_provider import LLMResponse, get_llm
from core.settings import get_llm_settings
from ..embeddings.providers import E5Config, E5Provider
from ..stores.qdrant_reader import QdrantVectorReader
from .prompt import Evidence, PromptBuilder
from .retriever import Retriever

logger = logging.getLogger(__name__)


class DocumentQA:
    """Основной инструмент вопросов и ответов по документам."""

    def __init__(
            self,
            config_path: Optional[str] = None,
            config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Инициализация QA системы.

        Args:
            config_path: Путь к файлу конфигурации
            config_dict: Словарь конфигурации (альтернатива файлу)
        """
        self.config = self._load_config(config_path, config_dict)
        self._initialize_components()

    def _load_config(
            self,
            config_path: Optional[str],
            config_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Загружает конфигурацию из файла или словаря.

        Returns:
            Словарь с конфигурацией
        """
        if config_dict:
            return config_dict

        config_path = config_path or os.getenv('QA_CONFIG', 'configs/qa.yaml')
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as error:
            logger.error(f"Failed to load config from {config_path}: {error}")
            raise

    def _initialize_components(self) -> None:
        """Инициализирует все компоненты системы из конфигурации."""
        self.embedding_provider = self._create_embedding_provider()
        self.vector_reader = self._create_vector_reader()
        self.retriever = self._create_retriever()
        self.llm, self.llm_settings = self._create_llm_provider()
        self.prompt_builder = self._create_prompt_builder()

    def _create_embedding_provider(self) -> E5Provider:
        """Создает провайдер эмбеддингов."""
        embedding_config = self.config.get('embedding', {})

        if embedding_config.get('provider', 'e5') == 'e5':
            e5_config = E5Config(
                model=embedding_config.get('e5', {}).get('model', 'intfloat/e5-base-v2'),
                device=embedding_config.get('e5', {}).get('device', 'cpu')
            )
            return E5Provider(e5_config)

        logger.warning("Unsupported embedding provider, falling back to E5")
        return E5Provider(E5Config())

    def _create_vector_reader(self) -> QdrantVectorReader:
        """Создает векторный ридер для Qdrant."""
        retrieval_config = self.config['retrieval']

        if retrieval_config['store'] != 'qdrant':
            raise ValueError(f"Unsupported store: {retrieval_config['store']}")

        return QdrantVectorReader(
            url=os.getenv('QDRANT_URL', 'http://localhost:6333'),
            api_key=os.getenv('QDRANT_API_KEY'),
            parents_collection=retrieval_config['parents_collection'],
            chunks_collection=retrieval_config['chunks_collection'],
            vector_size=self.embedding_provider.dim
        )

    def _create_retriever(self) -> Retriever:
        """Создает ретривер для поиска документов."""
        retrieval_config = self.config['retrieval']

        return Retriever(
            vector_reader=self.vector_reader,
            embedding_provider=self.embedding_provider,
            top_k_parents=retrieval_config.get('top_k_parents', 3),
            top_k_children_per_parent=retrieval_config.get('top_k_children_per_parent', 4),
            min_score=retrieval_config.get('min_score', 0.3),
            alpha_parent_child=retrieval_config.get('alpha_parent_child', 0.7)
        )

    def _create_llm_provider(self) -> tuple[Any, Any]:
        """Создает провайдер LLM с настройками."""
        try:
            llm = get_llm()
            llm_settings = get_llm_settings()
            logger.info(f"Initialized LLM: {llm.name}, model: {llm_settings.model}")
            return llm, llm_settings
        except Exception as error:
            logger.error(f"LLM initialization failed: {error}")
            logger.warning("Using EchoProvider fallback")
            return self._create_echo_provider(), None

    def _create_echo_provider(self) -> Any:
        """Создает эхо-провайдер для fallback."""

        class EchoProvider:
            name = "echo"

            def generate(self, messages: list, **kwargs: Any) -> Any:
                user_message = messages[-1]['content'] if messages else "No messages"
                return type('LLMResponse', (), {
                    'text': f"Echo response (LLM not configured): {user_message[:100]}...",
                    'model': 'echo',
                    'usage': {},
                    'finish_reason': None
                })()

        return EchoProvider()

    def _create_prompt_builder(self) -> PromptBuilder:
        """Создает строитель промптов."""
        prompt_config = self.config.get('prompt', {})
        return PromptBuilder(
            citation_style=prompt_config.get('citation_style', 'number')
        )

    def answer_from_docs(
            self,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> Dict[str, Any]:
        """Генерирует ответ на вопрос на основе документов.

        Args:
            query: Вопрос пользователя
            filters: Фильтры для поиска документов
            **kwargs: Дополнительные параметры

        Returns:
            Словарь с ответом и метаданными
        """
        retrieval_config = {**self.config['retrieval'], **kwargs}
        prompt_config = {**self.config.get('prompt', {}), **kwargs}
        filters = filters or retrieval_config.get('filters', {})

        try:
            return self._generate_answer(query, filters, retrieval_config, prompt_config)
        except Exception as error:
            logger.exception("Error processing query in answer_from_docs: %s", error)
            return self._create_error_response(str(error))

    def _generate_answer(
            self,
            query: str,
            filters: Dict[str, Any],
            retrieval_config: Dict[str, Any],
            prompt_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Генерирует ответ на основе извлеченных документов."""
        # 1. Получаем релевантные документы
        retrieval_result = self.retriever.retrieve(query, filters)
        chunks = getattr(retrieval_result, "chunks", None)

        if chunks is None:
            chunks_list = []
        elif isinstance(chunks, np.ndarray):
            chunks_list = chunks.tolist()
        else:
            chunks_list = list(chunks)

        if len(chunks_list) == 0:
            return self._create_empty_response(query)

        # 2. Преобразуем в Evidence
        evidences = [
            Evidence(
                text=chunk.text,
                score=chunk.score,
                metadata=chunk.metadata
            )
            for chunk in chunks_list
        ]

        # 3. Строим промпты
        system_prompt = self.prompt_builder.build_system_prompt(
            prompt_config.get('system_policy')
        )
        # total_chunks может быть numpy-числом; делаем разумный fallback
        total_chunks = getattr(retrieval_result, "total_chunks", None)

        if total_chunks is None:
            total_chunks = len(evidences)
        else:
            try:
                total_chunks = int(total_chunks)
            except Exception:
                total_chunks = len(evidences)

        user_prompt = self.prompt_builder.build_user_prompt(
            query, evidences, total_chunks
        )

        # 4. Извлекаем цитаты
        citations = self.prompt_builder.extract_citations_from_evidences(evidences)

        # 5. Генерируем ответ
        response = self._generate_llm_response(
            system_prompt, user_prompt, prompt_config
        )

        return self._create_success_response(query, response, citations, evidences)

    def _generate_llm_response(
            self,
            system_prompt: str,
            user_prompt: str,
            prompt_config: Dict[str, Any]
    ) -> LLMResponse:
        """Генерирует ответ через LLM."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        temperature = prompt_config.get('temperature', getattr(self.llm_settings, 'temperature', 0.1))
        max_tokens = prompt_config.get('max_output_tokens', getattr(self.llm_settings, 'max_output_tokens', 1000))

        return self.llm.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def _create_empty_response(self, query: str) -> Dict[str, Any]:
        """Создает ответ при отсутствии релевантных документов."""
        return {
            "answer": "I couldn't find any relevant information in the documentation to answer your question.",
            "citations": [],
            "retrieved_chunks": 0,
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _create_success_response(
            self,
            query: str,
            response: LLMResponse,
            citations: list,
            evidences: list
    ) -> Dict[str, Any]:
        """Создает успешный ответ с результатами."""
        return {
            "answer": response.text,
            "citations": citations,
            "model": getattr(response, 'model', getattr(self.llm_settings, 'model', 'unknown')),
            "usage": getattr(response, 'usage', {}),
            "finish_reason": getattr(response, 'finish_reason', None),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieved_chunks": len(evidences),
            "query": query,
            "provider": self.llm.name
        }

    def _create_error_response(self, error: str) -> Dict[str, Any]:
        """Создает ответ с ошибкой."""
        return {
            "answer": f"Sorry, I encountered an error while processing your question: {error}",
            "citations": [],
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Глобальный инстанс для удобного использования
_global_qa: Optional[DocumentQA] = None


def answer_from_docs(query: str, **kwargs: Any) -> Dict[str, Any]:
    """Удобная функция для использования глобального инстанса QA."""
    global _global_qa
    if _global_qa is None:
        _global_qa = DocumentQA()
    return _global_qa.answer_from_docs(query, **kwargs)


def get_qa_instance() -> DocumentQA:
    """Возвращает глобальный инстанс QA."""
    global _global_qa
    if _global_qa is None:
        _global_qa = DocumentQA()
    return _global_qa
