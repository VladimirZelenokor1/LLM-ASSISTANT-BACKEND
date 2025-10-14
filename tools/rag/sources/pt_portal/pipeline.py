# path: tools/rag/sources/pt_portal/pipeline.py
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from .dedup.exact_hash import text_hash
from .dedup.near_dup import NearDupChecker
from .fetch.repo_loader import RepoLoader, RepoSpec
from .normalize.strategies import GenericNormalizer
from .parse.markdown_parser import MarkdownParser
from .storage.doc_store import DocStore
from .storage.schema import IngestionError, IngestionStats, PTDocument, PTMeta

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Загружает конфигурацию пайплайна из YAML файла.

    Args:
        config_path: Путь к файлу конфигурации

    Returns:
        Словарь с конфигурацией

    Raises:
        FileNotFoundError: Если файл не существует
        yaml.YAMLError: Если файл содержит невалидный YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        logger.info(f"Loaded configuration from {config_file}")
        return config or {}

    except yaml.YAMLError as error:
        logger.error(f"Invalid YAML in config file {config_file}: {error}")
        raise
    except Exception as error:
        logger.error(f"Failed to load config from {config_file}: {error}")
        raise


class PTIngestionPipeline:
    """Пайплайн ингрессии для RAG системы портала PT.

    Обрабатывает только репозитории с md/mdx файлами (например, Transformers).
    Поддерживает клонирование, парсинг, нормализацию и дедупликацию документов.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Инициализация пайплайна ингрессии.

        Args:
            config: Словарь с конфигурацией пайплайна

        Raises:
            KeyError: Если отсутствуют обязательные параметры конфигурации
        """
        self.config = config
        self._validate_config()

        # Инициализация компонентов
        self.doc_store = DocStore(config["storage"]["processed_dir"])
        self.near_dup_checker = NearDupChecker(
            state_path=config["storage"]["dedup_state"],
            threshold=int(config.get("dedup", {}).get("threshold", 92)),
            min_len=int(config.get("dedup", {}).get("min_len", 600)),
        )
        self.normalizer = GenericNormalizer()
        self.markdown_parser = MarkdownParser()

        logger.info("PTIngestionPipeline initialized successfully")

    def _validate_config(self) -> None:
        """Проверяет обязательные параметры конфигурации.

        Raises:
            KeyError: Если отсутствуют обязательные параметры
        """
        required_sections = ["storage"]
        for section in required_sections:
            if section not in self.config:
                raise KeyError(f"Missing required config section: {section}")

        storage_required = ["processed_dir", "raw_repos_dir", "dedup_state"]
        for key in storage_required:
            if key not in self.config["storage"]:
                raise KeyError(f"Missing required storage config: {key}")

    def _create_base_metadata(
            self,
            *,
            product: str,
            product_code: str,
            version: str,
            lang: str
    ) -> PTMeta:
        """Создает базовые метаданные для документа.

        Args:
            product: Название продукта
            product_code: Код продукта
            version: Версия документации
            lang: Язык документации

        Returns:
            Объект PTMeta с базовыми метаданными
        """
        return PTMeta(
            product=product,
            product_code=product_code,
            version=version,
            lang=lang,
        )

    def _save_document(self, doc: PTDocument, stats: IngestionStats) -> None:
        """Сохраняет документ с проверкой дедупликации.

        Args:
            doc: Документ для сохранения
            stats: Статистика для обновления
        """
        # Пропускаем пустые документы
        if not doc.text or not doc.text.strip():
            logger.debug("Skipping empty document")
            return

        # Вычисляем хэш текста
        text_hash_value = text_hash(doc.text)
        doc.meta.hash_text = text_hash_value

        # Проверяем на near-дубликаты
        if self.near_dup_checker.is_near_dup(
                text=doc.text,
                h=text_hash_value,
                url=doc.meta.canonical_url,
                section=doc.meta.section,
                product_code=doc.meta.product_code,
        ):
            stats.dedup_dropped_near += 1
            logger.debug(f"Near duplicate dropped: {doc.meta.repo_path}")
            return

        # Сохраняем документ
        try:
            self.doc_store.save(doc)
            stats.persisted_processed += 1
            stats.processed_ok += 1
            logger.debug(f"Document saved: {doc.meta.repo_path}")
        except Exception as error:
            stats.processed_err += 1
            stats.add_error(
                stage="save_document",
                message=f"Failed to save document: {error}",
                url=doc.meta.repo_path
            )

    def _process_repository_source(self, source_config: Dict[str, Any], stats: IngestionStats) -> None:
        """Обрабатывает один репозиторий-источник.

        Args:
            source_config: Конфигурация источника репозитория
            stats: Статистика для обновления
        """
        repo_url = source_config.get("repo", "unknown")
        product_code = source_config.get("product_code", "unknown")

        logger.info(f"Processing repository source: {product_code} ({repo_url})")

        try:
            # Создаем спецификацию репозитория
            spec = RepoSpec(
                remote=source_config["repo"],
                version=source_config.get("version", "main"),
                docs_path=source_config.get("docs_path", ""),
                include_glob=source_config.get("include_glob", ["**/*.md", "**/*.mdx"]),
                exclude_glob=source_config.get("exclude_glob", []),
                dest_root=self.config["storage"]["raw_repos_dir"],
            )

            # Загружаем репозиторий
            loader = RepoLoader(spec)
            repo_dir = loader.clone_or_pull()
            commit_hash = loader.current_commit(repo_dir)

            # Находим файлы для обработки
            files = loader.find_files(repo_dir)
            if not files:
                raise RuntimeError(
                    f"No files matched in repository. "
                    f"repo_dir={repo_dir}, docs_path={spec.docs_path}, "
                    f"include={spec.include_glob}, exclude={spec.exclude_glob}"
                )

            stats.repo_files_found += len(files)
            logger.info(f"Found {len(files)} files in repository")

            # Обрабатываем каждый файл
            progress_desc = f"Parsing {product_code}@{spec.version}"
            for file_path in tqdm(files, desc=progress_desc):
                self._process_single_file(
                    file_path=file_path,
                    repo_dir=repo_dir,
                    source_config=source_config,
                    commit_hash=commit_hash,
                    stats=stats
                )

        except Exception as error:
            stats.processed_err += 1
            stats.add_error(
                stage="repo_clone",
                message=str(error),
                url=repo_url
            )
            logger.error(f"Failed to process repository {repo_url}: {error}")

    def _process_single_file(
            self,
            file_path: Path,
            repo_dir: Path,
            source_config: Dict[str, Any],
            commit_hash: str,
            stats: IngestionStats
    ) -> None:
        """Обрабатывает один файл из репозитория.

        Args:
            file_path: Путь к файлу
            repo_dir: Корневая директория репозитория
            source_config: Конфигурация источника
            commit_hash: Хэш коммита репозитория
            stats: Статистика для обновления
        """
        try:
            # Читаем содержимое файла
            with open(file_path, "r", encoding="utf-8") as file:
                text_content = file.read()

            # Вычисляем относительный путь
            docs_path = source_config.get("docs_path", "")
            if docs_path:
                base_path = repo_dir / docs_path
            else:
                base_path = repo_dir

            relative_path = os.path.relpath(file_path, base_path).replace("\\", "/")

            # Создаем метаданные
            meta = self._create_base_metadata(
                product=source_config["product"],
                product_code=source_config["product_code"],
                version=source_config.get("version", "main"),
                lang=source_config.get("lang", "en"),
            )

            # Заполняем метаданные репозитория
            self._populate_repo_metadata(
                meta=meta,
                source_config=source_config,
                commit_hash=commit_hash,
                relative_path=relative_path,
                original_text=text_content
            )

            # Парсим документ
            doc = self.markdown_parser.parse_file(
                text=text_content,
                meta=meta,
                repo_path=relative_path
            )

            # Нормализуем документ
            try:
                doc = self.normalizer.normalize(doc)
            except Exception as error:
                stats.add_error(
                    stage="normalize",
                    message=str(error),
                    url=str(file_path)
                )
                logger.warning(f"Normalization failed for {file_path}: {error}")

            stats.repo_files_parsed += 1
            self._save_document(doc, stats)

        except Exception as error:
            stats.processed_err += 1
            stats.add_error(
                stage="repo_parse",
                message=str(error),
                url=str(file_path)
            )
            logger.error(f"Failed to process file {file_path}: {error}")

    def _populate_repo_metadata(
            self,
            meta: PTMeta,
            source_config: Dict[str, Any],
            commit_hash: str,
            relative_path: str,
            original_text: str
    ) -> None:
        """Заполняет метаданные репозитория.

        Args:
            meta: Объект метаданных для заполнения
            source_config: Конфигурация источника
            commit_hash: Хэш коммита
            relative_path: Относительный путь файла
            original_text: Исходный текст файла
        """
        meta.source_kind = "repo"
        meta.repo_remote = source_config["repo"]
        meta.repo_commit = commit_hash
        meta.repo_path = relative_path
        meta.url = f"repo://{source_config['product_code']}/{commit_hash}/{relative_path}"
        meta.canonical_url = meta.url
        meta.fetched_at = datetime.utcnow()
        meta.content_type = "md"
        meta.source_format = "markdown"
        meta.hash_raw = text_hash(original_text)

    def run(self) -> IngestionStats:
        """Запускает пайплайн ингрессии.

        Returns:
            Статистика выполнения пайплайна
        """
        logger.info("Starting PT ingestion pipeline")

        stats = IngestionStats()
        repo_sources: List[Dict[str, Any]] = self.config.get("repo_sources", []) or []
        stats.sources_total = len(repo_sources)

        if not repo_sources:
            logger.warning("No repository sources configured")
            return stats

        logger.info(f"Processing {len(repo_sources)} repository sources")

        # Обрабатываем каждый источник
        for source_config in repo_sources:
            self._process_repository_source(source_config, stats)

        # Сохраняем состояние дедупликации
        self.near_dup_checker.flush()

        # Логируем итоговую статистику
        logger.info(
            f"Pipeline completed: {stats.processed_ok} OK, "
            f"{stats.processed_err} errors, "
            f"{stats.dedup_dropped_near} near-duplicates dropped"
        )

        return stats

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Возвращает информацию о конфигурации пайплайна.

        Returns:
            Словарь с информацией о пайплайне
        """
        repo_sources = self.config.get("repo_sources", [])

        return {
            "pipeline_type": "repo_only",
            "repo_sources_count": len(repo_sources),
            "repo_products": [
                {
                    "product": source.get("product"),
                    "product_code": source.get("product_code"),
                    "version": source.get("version", "main"),
                    "lang": source.get("lang", "en")
                }
                for source in repo_sources
            ],
            "storage": {
                "processed_dir": self.config["storage"]["processed_dir"],
                "raw_repos_dir": self.config["storage"]["raw_repos_dir"],
            },
            "dedup": {
                "threshold": self.config.get("dedup", {}).get("threshold", 92),
                "min_len": self.config.get("dedup", {}).get("min_len", 600),
            }
        }
