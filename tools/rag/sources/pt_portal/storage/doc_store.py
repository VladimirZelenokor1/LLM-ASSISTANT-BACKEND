# path: tools/rag/sources/pt_portal/storage/doc_store.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from slugify import slugify

from .schema import PTDocument

logger = logging.getLogger(__name__)

# Константы для ограничений
_MAX_FILENAME_LENGTH = 120
_HASH_SUFFIX_LENGTH = 8
_MIN_TEXT_LENGTH = 10  # Минимальная длина текста для сохранения


class DocStore:
    """Хранилище для обработки и сохранения документов в формате JSONL."""

    def __init__(self, processed_dir: str | Path) -> None:
        """Инициализация хранилища документов.

        Args:
            processed_dir: Корневая директория для сохранения обработанных документов
        """
        self.processed_dir = Path(processed_dir)
        logger.info(f"Initialized DocStore with directory: {self.processed_dir}")

    def _get_document_directory(self, product_code: str, version: str, lang: str) -> Path:
        """Генерирует путь к директории для сохранения документа.

        Args:
            product_code: Код продукта
            version: Версия документации
            lang: Язык документации

        Returns:
            Путь к целевой директории
        """
        return self.processed_dir / product_code / version / lang

    def _generate_filename(self, doc: PTDocument) -> str:
        """Генерирует имя файла на основе метаданных документа.

        Args:
            doc: Документ для генерации имени файла

        Returns:
            Сгенерированное имя файла с расширением .jsonl
        """
        # Определяем базовое имя из различных источников
        base_name = (
                doc.meta.title
                or doc.meta.repo_path
                or doc.meta.canonical_url
                or "document"
        )

        # Нормализуем имя файла
        normalized_name = slugify(base_name)[:_MAX_FILENAME_LENGTH]
        if not normalized_name:
            normalized_name = "document"

        # Добавляем суффикс из хэша для избежания коллизий
        hash_suffix = ""
        if hasattr(doc.meta, "hash_text") and doc.meta.hash_text:
            hash_suffix = f"-{doc.meta.hash_text[:_HASH_SUFFIX_LENGTH]}"

        return f"{normalized_name}{hash_suffix}.jsonl"

    def _prepare_payload(self, doc: PTDocument) -> dict:
        """Подготавливает данные документа для сериализации в JSON.

        Args:
            doc: Документ для сериализации

        Returns:
            Словарь с данными документа

        Raises:
            ValueError: Если документ не может быть сериализован
        """
        try:
            # Пытаемся использовать model_dump для pydantic v2
            meta_data = doc.meta.model_dump(mode="json")
        except AttributeError:
            try:
                # Fallback для pydantic v1
                meta_data = doc.meta.dict()
            except AttributeError as error:
                logger.error(f"Failed to serialize document metadata: {error}")
                raise ValueError("Document metadata cannot be serialized") from error

        return {
            "text": doc.text,
            "meta": meta_data,
            "source_ref": doc.source_ref,
        }

    def save(self, doc: PTDocument) -> str:
        """Сохраняет документ в формате JSONL.

        Args:
            doc: Документ для сохранения

        Returns:
            Путь к сохраненному файлу или пустая строка если документ не сохранен

        Raises:
            ValueError: Если документ невалиден
        """
        if not doc or not isinstance(doc, PTDocument):
            raise ValueError("Invalid document: must be PTDocument instance")

        # Проверяем, что текст документа достаточно длинный для сохранения
        if not doc.text or not doc.text.strip() or len(doc.text.strip()) < _MIN_TEXT_LENGTH:
            logger.warning(
                f"Document text too short or empty ({len(doc.text or '')} chars), skipping save"
            )
            return ""

        try:
            # Создаем целевую директорию
            target_dir = self._get_document_directory(
                doc.meta.product_code,
                doc.meta.version,
                doc.meta.lang
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            # Генерируем имя файла
            filename = self._generate_filename(doc)
            file_path = target_dir / filename

            # Подготавливаем данные для сохранения
            payload = self._prepare_payload(doc)

            # Сохраняем документ
            with open(file_path, "w", encoding="utf-8") as file:
                json_line = json.dumps(payload, ensure_ascii=False)
                file.write(json_line + "\n")

            logger.debug(f"Document saved to: {file_path}")
            return str(file_path)

        except Exception as error:
            logger.error(f"Failed to save document: {error}")
            raise

    def save_batch(self, documents: List[PTDocument]) -> List[str]:
        """Сохраняет пакет документов.

        Args:
            documents: Список документов для сохранения

        Returns:
            Список путей к сохраненным файлам (пустые строки для пропущенных документов)
        """
        logger.info(f"Saving batch of {len(documents)} documents")

        saved_paths = []
        success_count = 0

        for i, doc in enumerate(documents):
            try:
                file_path = self.save(doc)
                saved_paths.append(file_path)
                if file_path:
                    success_count += 1
            except Exception as error:
                logger.error(f"Failed to save document {i}: {error}")
                saved_paths.append("")

        logger.info(f"Successfully saved {success_count}/{len(documents)} documents")
        return saved_paths

    def load(self, file_path: str | Path) -> Optional[PTDocument]:
        """Загружает документ из файла JSONL.

        Args:
            file_path: Путь к файлу JSONL

        Returns:
            Загруженный документ или None в случае ошибки
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {path}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as file:
                json_line = file.readline().strip()
                if not json_line:
                    logger.warning(f"Empty file: {path}")
                    return None

                data = json.loads(json_line)

                # Восстанавливаем документ из данных
                # Предполагается, что PTDocument имеет соответствующий конструктор
                # или методы для десериализации
                return PTDocument(
                    text=data.get("text", ""),
                    meta=data.get("meta", {}),
                    source_ref=data.get("source_ref", "")
                )

        except (json.JSONDecodeError, KeyError, Exception) as error:
            logger.error(f"Failed to load document from {path}: {error}")
            return None

    def load_all_from_directory(
            self,
            product_code: str,
            version: str,
            lang: str
    ) -> List[PTDocument]:
        """Загружает все документы из директории продукта/версии/языка.

        Args:
            product_code: Код продукта
            version: Версия документации
            lang: Язык документации

        Returns:
            Список загруженных документов
        """
        target_dir = self._get_document_directory(product_code, version, lang)

        if not target_dir.exists():
            logger.warning(f"Directory not found: {target_dir}")
            return []

        documents = []
        jsonl_files = list(target_dir.glob("*.jsonl"))

        logger.info(f"Loading {len(jsonl_files)} documents from {target_dir}")

        for file_path in jsonl_files:
            doc = self.load(file_path)
            if doc:
                documents.append(doc)

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    def get_stats(self) -> dict:
        """Возвращает статистику хранилища.

        Returns:
            Словарь со статистикой
        """
        if not self.processed_dir.exists():
            return {"total_documents": 0, "products": {}}

        stats = {
            "total_documents": 0,
            "products": {},
            "total_size_bytes": 0
        }

        for product_dir in self.processed_dir.iterdir():
            if not product_dir.is_dir():
                continue

            product_stats = {
                "versions": {},
                "total_documents": 0
            }

            for version_dir in product_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                version_stats = {
                    "languages": {},
                    "total_documents": 0
                }

                for lang_dir in version_dir.iterdir():
                    if not lang_dir.is_dir():
                        continue

                    jsonl_files = list(lang_dir.glob("*.jsonl"))
                    lang_count = len(jsonl_files)
                    version_stats["languages"][lang_dir.name] = lang_count
                    version_stats["total_documents"] += lang_count

                    # Подсчет общего размера
                    for file_path in jsonl_files:
                        stats["total_size_bytes"] += file_path.stat().st_size

                product_stats["versions"][version_dir.name] = version_stats
                product_stats["total_documents"] += version_stats["total_documents"]

            stats["products"][product_dir.name] = product_stats
            stats["total_documents"] += product_stats["total_documents"]

        return stats
