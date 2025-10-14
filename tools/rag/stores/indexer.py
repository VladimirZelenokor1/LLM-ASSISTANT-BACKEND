# tools/rag/indexing/indexer.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.rag.embeddings.cli import cmd_embed, cmd_sanity
from tools.rag.embeddings.utils import (
    iter_chunk_files, iter_chunks_jsonl, iter_parents_jsonl
)

logger = logging.getLogger(__name__)


@dataclass
class IndexingConfig:
    embeddings_config_path: str
    chunks_root_dir: str
    chunks_glob: str = "*.jsonl"
    batch_size: int = 64
    recreate_collection: bool = False

    def __post_init__(self) -> None:
        cfg = Path(self.embeddings_config_path)
        if not cfg.exists():
            raise FileNotFoundError(f"Embeddings config not found: {cfg}")

        root = Path(self.chunks_root_dir)
        # Раньше не падали — создадим при необходимости
        if not root.exists():
            logger.warning("Chunks root dir is missing, creating: %s", root)
            root.mkdir(parents=True, exist_ok=True)


@dataclass
class IndexingStats:
    total_chunk_files: int = 0
    total_chunks: int = 0
    total_parents: int = 0
    indexed_chunks: int = 0
    indexed_parents: int = 0
    failed_chunks: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    errors: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


class RAGIndexer:
    def __init__(self, config: IndexingConfig):
        self.config = config
        self.stats = IndexingStats()
        logger.info("RAGIndexer initialized: %s", self.config)

    # ---- public API ----

    def index_all(self) -> bool:
        """Полная индексация — как раньше: вызываем cmd_embed()."""
        self._tick_start()
        try:
            class Args:
                def __init__(self, config_path: str):
                    self.config = config_path

            rc = cmd_embed(Args(self.config.embeddings_config_path))
            if rc == 0:
                self._collect_stats()
                self._tick_finish(success=True)
                logger.info("Indexing completed successfully")
                return True
            else:
                self._tick_finish(success=False)
                self.stats.errors.append(f"cmd_embed exited with code {rc}")
                logger.error("Indexing failed: cmd_embed -> %s", rc)
                return False
        except Exception as e:
            self._tick_finish(success=False)
            self.stats.errors.append(str(e))
            logger.exception("Indexing error")
            return False

    def index_incremental(self, document_ids: List[str]) -> bool:
        """Сохраняем старое поведение — полной переиндексацией."""
        logger.info("Incremental indexing (fallback to full). docs=%d", len(document_ids))
        return self.index_all()

    def verify_index(self) -> Dict[str, Any]:
        """Санити-проверка — как раньше: cmd_sanity()."""

        class Args:
            def __init__(self, config_path: str):
                self.config = config_path

        rc = cmd_sanity(Args(self.config.embeddings_config_path))
        status = "healthy" if rc == 0 else "issues_found"
        return {
            "sanity_check_result": rc,
            "collections_checked": ["main_chunks", "parent_chunks"],
            "status": status
        }

    def get_index_stats(self) -> Dict[str, Any]:
        if self.stats.total_chunk_files == 0:
            # обновим, если ещё не собирали
            self._collect_stats()
        return {
            "total_chunk_files": self.stats.total_chunk_files,
            "total_chunks": self.stats.total_chunks,
            "total_parents": self.stats.total_parents,
            "indexed_chunks": self.stats.indexed_chunks,
            "indexed_parents": self.stats.indexed_parents,
            "failed_chunks": self.stats.failed_chunks,
            "started_at": self.stats.started_at,
            "finished_at": self.stats.finished_at,
            "duration_seconds": self.stats.duration_seconds,
            "errors": self.stats.errors,
        }

    # ---- internals ----

    def _tick_start(self) -> None:
        self.stats.started_at = datetime.now().isoformat(timespec="seconds")

    def _tick_finish(self, success: bool) -> None:
        end = datetime.now()
        self.stats.finished_at = end.isoformat(timespec="seconds")
        if self.stats.started_at:
            try:
                start = datetime.fromisoformat(self.stats.started_at)
                self.stats.duration_seconds = (end - start).total_seconds()
            except Exception:
                pass

    def _collect_stats(self) -> None:
        """Считаем ровно как прежде — по файлам/строкам JSONL."""
        try:
            root = self.config.chunks_root_dir
            pattern = self.config.chunks_glob

            files = list(iter_chunk_files(root, pattern))
            self.stats.total_chunk_files = len(files)

            total_chunks = 0
            for fp in files:
                try:
                    total_chunks += sum(1 for _ in iter_chunks_jsonl(fp))
                except Exception as e:
                    logger.warning("Failed to read %s: %s", fp, e)
                    self.stats.failed_chunks += 1

            # parents
            parents_total = 0
            parents_dir = os.path.join(root, "_parents")
            if os.path.isdir(parents_dir):
                for name in sorted(os.listdir(parents_dir)):
                    if not name.endswith(".jsonl"):
                        continue
                    ppath = os.path.join(parents_dir, name)
                    try:
                        parents_total += sum(1 for _ in iter_parents_jsonl(ppath))
                    except Exception as e:
                        logger.warning("Failed to read parents %s: %s", ppath, e)

            self.stats.total_chunks = total_chunks
            self.stats.total_parents = parents_total
            # считаем, что всё, что посчитали, уехало (как раньше)
            self.stats.indexed_chunks = total_chunks
            self.stats.indexed_parents = parents_total

            logger.info("Stats: files=%d chunks=%d parents=%d",
                        self.stats.total_chunk_files, total_chunks, parents_total)
        except Exception as e:
            logger.warning("Could not collect stats: %s", e)
            self.stats.errors.append(f"stats: {e}")


def create_indexer(config_path: str, chunks_root: str = "data/chunks") -> RAGIndexer:
    """Совместимая фабрика, как раньше."""
    cfg = IndexingConfig(
        embeddings_config_path=config_path,
        chunks_root_dir=chunks_root,
    )
    return RAGIndexer(cfg)