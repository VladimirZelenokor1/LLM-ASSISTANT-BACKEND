# agents/policies.py
from __future__ import annotations

"""
Политики классификации запросов для маршрутизации.

Содержит ключевые слова и регулярные выражения для первичной классификации
пользовательских запросов по трём направлениям: документация, SQL-база, веб-поиск.
"""

import re
from typing import List, Pattern

__all__ = [
    "DOCS_KEYWORDS",
    "SQL_KEYWORDS",
    "DOCS_PATTERNS",
    "WEB_KEYWORDS",
    "WEB_PATTERNS",
]

# =============================================================================
# КЛЮЧЕВЫЕ СЛОВА ДЛЯ ДОКУМЕНТАЦИИ TRANSFORMERS
# =============================================================================

DOCS_KEYWORDS: List[str] = [
    # Основные понятия и платформа
    "transformers", "huggingface", "hugging face", "pipeline", "tokenizer",
    "model", "bert", "gpt", "transformer", "attention", "fine-tuning",
    "fine-tune", "pre-training", "dataset", "datasets", "trainer", "training",
    "inference", "pipelines",

    # Архитектуры и компоненты API
    "auto", "config", "configuration", "tokenization", "vocab", "weights",
    "checkpoint", "hub", "repository", "repo", "github", "documentation",
    "example", "tutorial", "guide", "how to", "usage", "using", "apply",
    "application", "implement", "implementation", "class", "function", "method",
    "parameter", "argument", "return", "code", "source code", "api", "reference",

    # Конкретные модели и компоненты
    "whisper", "clip", "blip", "detr", "yolos", "perceiver", "vision", "text",
    "multimodal", "encoder", "decoder", "seq2seq", "encoder-decoder", "causal",
    "autoregressive", "masked", "language", "audio", "image",

    # Технические термины и инструменты
    "embedding", "layer", "head", "pooler", "activation", "normalization",
    "dropout", "optimizer", "scheduler", "loss", "metric", "evaluation",
    "validation", "callback", "utility", "processor", "feature extractor",

    # Файлы и структуры проекта
    "model card", "model_card", "readme", "license", "tokenizer.json",
    "config.json", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack",
    "safetensors", "generation", "utils", "processing",
]

# =============================================================================
# КЛЮЧЕВЫЕ СЛОВА ДЛЯ SQL (ОРГАНИЗАЦИОННЫЕ ДАННЫЕ)
# =============================================================================

SQL_KEYWORDS: List[str] = [
    # Вопросы о количестве и статистике
    "сколько", "count",

    # Команда и сотрудники
    "кто", "команда", "сотрудник", "тимлид", "разработчик", "опыт", "лет",
    "team", "employee", "staff", "member", "lead", "manager",

    # Организационная структура
    "модуль", "владелец", "отвечает", "ответственный", "роль", "должность",
    "стаж", "опыт работы", "состав", "команды", "участник", "член команды",

    # Контактная информация
    "контакт", "почта", "email", "город", "локация",

    # Роли и специализации
    "core-dev", "maintainer", "инфра", "docs", "vision", "nlp", "audio", "training",
]

# =============================================================================
# КЛЮЧЕВЫЕ СЛОВА ДЛЯ ВЕБ-ПОИСКА
# =============================================================================

WEB_KEYWORDS: List[str] = [
    # Поисковые системы и платформы
    "в интернете", "гугл", "google", "поиск", "twitter", "reddit", "hn",
    "stack overflow", "stackoverflow", "форум", "github issue",

    # Актуальность и новости
    "актуальные новости", "последн", "релиз сегодня", "новост", "пресс-релиз",

    # Внешние источники и проблемы
    "за пределами документации", "за пределами", "обсуждение", "issue", "bug",
    "problem", "error",
]

# =============================================================================
# РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ ДЛЯ ДОКУМЕНТАЦИИ
# =============================================================================

DOCS_PATTERNS: List[Pattern] = [
    # Сигнатуры безопасности (унаследованные)
    re.compile(r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE),
    re.compile(r"\b(XSpider|MaxPatrol)\b", re.IGNORECASE),

    # Архитектуры моделей
    re.compile(
        r"\b(BERT|GPT-2|GPT-3|RoBERTa|DistilBERT|Transformer|T5|BART|XLNet)\b",
        re.IGNORECASE,
    ),

    # Классы transformers API
    re.compile(
        r"\b(AutoModel|AutoTokenizer|AutoConfig|BertModel|BertTokenizer|GPT2Model|GPT2Tokenizer)\b",
        re.IGNORECASE,
    ),

    # Вызовы pipeline
    re.compile(r"\b(pipeline|Pipeline)\s+[\"']([^\"']+)[\"']", re.IGNORECASE),

    # Методы from_pretrained
    re.compile(r"from_pretrained\([\"']([^\"']+)[\"']\)", re.IGNORECASE),
    re.compile(r"\.from_pretrained\([\"']([^\"']+)[\"']\)", re.IGNORECASE),

    # Импорты и обращения к transformers
    re.compile(r"transformers\.\w+\.\w+", re.IGNORECASE),
]

# =============================================================================
# РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ ДЛЯ ВЕБ-ПОИСКА
# =============================================================================

WEB_PATTERNS: List[Pattern] = [
    # Указания на интернет-поиск
    re.compile(r"\b(в\s+интернете|в\s+сети)\b", re.IGNORECASE),

    # Упоминания релизов
    re.compile(r"\b(последн(ий|яя|ее)\s+релиз)\b", re.IGNORECASE),

    # Технические форумы и платформы
    re.compile(r"\b(stack\s+overflow|stackoverflow)\b", re.IGNORECASE),

    # GitHub issues
    re.compile(r"\b(github\s+issue|issue\s+#\d+)\b", re.IGNORECASE),
]
