# path: tools/rag/sources/pt_portal/parse/markdown_parser.py
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

import yaml

from ..storage.schema import PTDocument, PTMeta

logger = logging.getLogger(__name__)

# Регулярные выражения для парсинга
_FRONT_MATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TABS_BLOCK_RE = re.compile(r"<Tabs[^>]*>(?P<inner>.*?)</Tabs>", re.DOTALL | re.IGNORECASE)
_TAB_ITEM_RE = re.compile(
    r"<TabItem[^>]*\bvalue=[\"'](?P<val>[^\"']+)[\"'][^>]*>(?P<body>.*?)</TabItem>",
    re.DOTALL | re.IGNORECASE
)
_WHITESPACE_BEFORE_NEWLINE_RE = re.compile(r"[ \t]+\n")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$", re.MULTILINE)

# Константы для обработки
_PREFERRED_TAB_VALUES = ["pip", "python", "pt", "bash", "conda", "source"]
_DOCS_PREFIXES = {"docs", "site", "content", "src", "source", "en"}
_INDEX_FILES = {"index", "readme", "_index"}
_COMPONENT_TAGS = {
    "tip": "**Tip:**",
    "note": "**Note:**",
    "warning": "**Warning:**"
}


def _strip_front_matter(text: str) -> Tuple[str, Optional[str]]:
    """Удаляет front matter из текста и возвращает его содержимое.

    Args:
        text: Исходный текст с front matter

    Returns:
        Кортеж (текст без front matter, сырой YAML front matter или None)
    """
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return text, None

    start, end = match.span()
    return text[end:], match.group(1)


def _pick_h1(text: str) -> Optional[str]:
    """Извлекает первый заголовок H1 из текста.

    Args:
        text: Текст для поиска заголовков

    Returns:
        Текст первого H1 заголовка или None
    """
    match = _H1_RE.search(text)
    if not match:
        return None

    # Очищаем заголовок от HTML-тегов и бэктиков
    title_text = match.group(1)
    title_text = _HTML_TAG_RE.sub("", title_text)
    title_text = title_text.replace("`", "").strip()

    return title_text or None


def _title_from_basename(repo_path: str) -> str:
    """Генерирует заголовок на основе имени файла.

    Args:
        repo_path: Путь к файлу в репозитории

    Returns:
        Сгенерированный заголовок
    """
    if not repo_path:
        return "Document"

    base_name = os.path.splitext(os.path.basename(repo_path))[0]

    # Для индексных файлов используем имя родительской директории
    if base_name.lower() in _INDEX_FILES:
        parent_dir = os.path.basename(os.path.dirname(repo_path))
        base_name = parent_dir or base_name

    # Нормализуем имя файла в читаемый заголовок
    title = base_name.replace("_", " ").replace("-", " ").strip()
    title = re.sub(r"\s+", " ", title)

    return title[:1].upper() + title[1:] if title else "Document"


def _section_from_repo_path(repo_path: str) -> str:
    """Извлекает секцию из пути репозитория.

    Args:
        repo_path: Путь к файлу в репозитории

    Returns:
        Нормализованный путь секции
    """
    if not repo_path:
        return "/"

    parts = repo_path.replace("\\", "/").split("/")

    # Убираем типичные префиксы doc-структур
    filtered_parts = [
        part for part in parts
        if part.lower() not in _DOCS_PREFIXES
    ]

    if not filtered_parts:
        return "/"

    # Берем 1-2 первые директории (без файла)
    dirs = filtered_parts[:-1]
    if not dirs:
        return "/"

    if len(dirs) == 1:
        return f"/{dirs[0].strip('/')}/"

    return f"/{dirs[0].strip('/')}/{dirs[1].strip('/')}/"


def _prefer_single_tab(markdown_text: str) -> str:
    """Заменяет блоки Tabs на содержимое наиболее предпочтительного TabItem.

    Args:
        markdown_text: Текст Markdown с блоками Tabs

    Returns:
        Текст с замененными блоками Tabs
    """

    def _pick_best_tab(match: re.Match) -> str:
        inner_content = match.group("inner")
        tab_items = list(_TAB_ITEM_RE.finditer(inner_content))

        if not tab_items:
            return inner_content

        # Выбираем наиболее предпочтительный tab по значению value
        best_tab = min(
            tab_items,
            key=lambda item: (
                _PREFERRED_TAB_VALUES.index(item.group("val"))
                if item.group("val") in _PREFERRED_TAB_VALUES
                else 999
            )
        )

        return best_tab.group("body").strip()

    return _TABS_BLOCK_RE.sub(_pick_best_tab, markdown_text)


def _strip_mdx_noise(text: str) -> str:
    """Удаляет служебные конструкции MDX и JSX из текста.

    Args:
        text: Исходный текст с MDX/JSX

    Returns:
        Очищенный текст
    """
    # Заменяем компоненты Tip, Note, Warning на текстовые аналоги
    for tag, replacement in _COMPONENT_TAGS.items():
        text = re.sub(
            f"<{tag}[^>]*>",
            f"\n{replacement} ",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(f"</{tag}>", "\n", text, flags=re.IGNORECASE)

    # Удаляем import/export statements
    text = re.sub(r"(?m)^\s*(import|export)\b.*?$", "", text)

    # Удаляем одиночные и парные JSX-теги
    text = re.sub(r"</?[A-Za-z][A-Za-z0-9_.-]*(\s+[^<>]*)?/?>", "", text)

    return text


def _normalize_text(text: str) -> str:
    """Нормализует текст, сохраняя кодовые блоки нетронутыми.

    Args:
        text: Исходный текст для нормализации

    Returns:
        Нормализованный текст
    """
    chunks = []
    last_end = 0

    # Обрабатываем текст вокруг кодовых блоков
    for match in _CODE_FENCE_RE.finditer(text):
        # Текст до кодового блока
        head = text[last_end:match.start()]
        head = _strip_mdx_noise(head)
        head = _WHITESPACE_BEFORE_NEWLINE_RE.sub("\n", head)
        head = _MULTIPLE_NEWLINES_RE.sub("\n\n", head)
        chunks.append(head)

        # Кодовый блок оставляем как есть
        chunks.append(match.group(0))
        last_end = match.end()

    # Обрабатываем хвост текста после последнего кодового блока
    tail = text[last_end:]
    tail = _strip_mdx_noise(tail)
    tail = _WHITESPACE_BEFORE_NEWLINE_RE.sub("\n", tail)
    tail = _MULTIPLE_NEWLINES_RE.sub("\n\n", tail)
    chunks.append(tail)

    return "".join(chunks).strip()


class MarkdownParser:
    """Парсер Markdown/MDX документов.

    Извлекает метаданные, обрабатывает front matter, нормализует текст
    и генерирует структурированные документы для дальнейшей обработки.
    """

    def parse_file(
            self,
            *,
            text: str,
            meta: PTMeta,
            repo_path: str
    ) -> PTDocument:
        """Парсит Markdown/MDX файл и возвращает структурированный документ.

        Args:
            text: Содержимое Markdown/MDX файла
            meta: Метаданные документа
            repo_path: Путь к файлу в репозитории

        Returns:
            Структурированный документ PTDocument

        Raises:
            ValueError: Если переданные аргументы невалидны
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        if not meta or not isinstance(meta, PTMeta):
            raise ValueError("Meta must be a valid PTMeta instance")

        logger.info(f"Parsing markdown file: {repo_path}")

        # Обрабатываем блоки Tabs
        processed_text = _prefer_single_tab(text)

        # Извлекаем front matter
        body, front_matter = _strip_front_matter(processed_text)

        # Определяем заголовок
        title = self._extract_title(body, front_matter, repo_path)

        # Определяем секцию
        section = meta.section or _section_from_repo_path(repo_path)

        # Нормализуем текст
        clean_text = _normalize_text(body)

        # Заполняем метаданные
        self._populate_metadata(
            meta, title, section, clean_text, front_matter
        )

        logger.info(
            f"Successfully parsed: {repo_path}, "
            f"title: {title}, section: {section}, "
            f"chars: {len(clean_text)}, words: {meta.content_words}"
        )

        return PTDocument(
            text=clean_text,
            meta=meta,
            source_ref=repo_path,
        )

    def _extract_title(
            self,
            body: str,
            front_matter: Optional[str],
            repo_path: str
    ) -> str:
        """Извлекает заголовок из различных источников.

        Args:
            body: Основной текст документа
            front_matter: Содержимое front matter
            repo_path: Путь к файлу в репозитории

        Returns:
            Извлеченный заголовок
        """
        # Пробуем извлечь H1 из тела документа
        title = _pick_h1(body)

        # Если H1 не найден, пробуем из front matter
        if not title and front_matter:
            title = self._extract_title_from_front_matter(front_matter)

        # Если все еще нет заголовка, генерируем из имени файла
        if not title:
            title = _title_from_basename(repo_path)

        return title

    def _extract_title_from_front_matter(self, front_matter: str) -> Optional[str]:
        """Извлекает заголовок из front matter YAML.

        Args:
            front_matter: Содержимое front matter в формате YAML

        Returns:
            Извлеченный заголовок или None
        """
        try:
            fm_obj = yaml.safe_load(front_matter) or {}

            # Пробуем разные поля для заголовка
            title_fields = ["title", "sidebar_label"]
            for field in title_fields:
                title = fm_obj.get(field)
                if isinstance(title, str) and title.strip():
                    # Очищаем заголовок от HTML и бэктиков
                    clean_title = _HTML_TAG_RE.sub("", title)
                    clean_title = clean_title.replace("`", "").strip()
                    if clean_title:
                        return clean_title

        except yaml.YAMLError as error:
            logger.warning(f"Failed to parse front matter YAML: {error}")
        except Exception as error:
            logger.warning(f"Unexpected error parsing front matter: {error}")

        return None

    def _populate_metadata(
            self,
            meta: PTMeta,
            title: str,
            section: str,
            clean_text: str,
            front_matter: Optional[str]
    ) -> None:
        """Заполняет метаданные документа.

        Args:
            meta: Объект метаданных для заполнения
            title: Заголовок документа
            section: Секция документа
            clean_text: Очищенный текст документа
            front_matter: Содержимое front matter
        """
        meta.title = title
        meta.section = section
        meta.content_length_chars = len(clean_text)
        meta.content_words = len(re.findall(r"\w+", clean_text, re.UNICODE))
        meta.has_code_blocks = bool(_CODE_FENCE_RE.search(clean_text))
        meta.has_tables = bool(
            "|" in clean_text and _TABLE_ROW_RE.search(clean_text)
        )

        # Сохраняем front matter в extra
        if front_matter:
            meta.extra = meta.extra or {}
            meta.extra["front_matter"] = front_matter