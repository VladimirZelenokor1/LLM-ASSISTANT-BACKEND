from __future__ import annotations
import glob
import json
import os
import uuid
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


# ---------------- io ----------------

def iter_chunk_files(root: str, pattern: str) -> List[str]:
    """
    Возвращает отсортированный список путей к обычным .jsonl с чанками,
    исключая _parents (они читаются отдельно).
    """
    root = (root or "").rstrip("/")
    files = glob.glob(os.path.join(root, pattern), recursive=True)
    files = [f for f in files if "/_parents/" not in f and "\\_parents\\" not in f]
    files.sort()
    return files


@dataclass
class ChunkRow:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


def _uuid5_from(s: str) -> str:
    """Стабильный UUID из произвольной строки (для Qdrant и т.п.)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def _extract_chunk_id(meta: Dict[str, Any], text: str, fallback_prefix: str) -> str:
    """
    Достаём настоящий chunk_id:
      1) meta.extra.chunk_id
      2) meta.chunk_id
      3) fallback: <fallback_prefix>__seq<seq>__<sha8> или <fallback_prefix>__<sha8>
    """
    if not isinstance(meta, dict):
        meta = {}

    extra = meta.get("extra") or {}
    if isinstance(extra, dict):
        cid = extra.get("chunk_id")
        if cid:
            return str(cid)

    cid = meta.get("chunk_id")
    if cid:
        return str(cid)

    seq = meta.get("seq")
    digest = sha1((text or "").encode("utf-8", "ignore")).hexdigest()[:8]
    if seq is not None:
        return f"{fallback_prefix}__seq{seq}__{digest}"
    return f"{fallback_prefix}__{digest}"


def iter_chunks_jsonl(path: str) -> Iterable[ChunkRow]:
    """
    Читает обычные чанки (НЕ _parents). Гарантирует корректный chunk_id
    (берёт из meta.extra.chunk_id / meta.chunk_id) и дублирует его в meta.chunk_id.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            text = row.get("text", "")
            meta = row.get("meta", {}) or {}

            # стабильный префикс для fallback
            pref = (
                    meta.get("repo_path")
                    or meta.get("canonical_url")
                    or os.path.splitext(os.path.basename(path))[0]
            )

            chunk_id = _extract_chunk_id(meta, text, pref)

            # дублируем в meta.chunk_id (для удобного payload)
            if isinstance(meta, dict):
                meta["chunk_id"] = chunk_id

            yield ChunkRow(chunk_id=chunk_id, text=text, meta=meta)


# ---------------- parents ----------------

@dataclass
class ParentRow:
    parent_id: str
    text: str
    meta: Dict[str, Any]  # минимальные поля для payload (product/lang/h_path/child_ids/...)


def iter_parents_jsonl(path: str) -> Iterable[ParentRow]:
    """
    Читает parent-чанки с полными метаданными.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "parent":
                continue

            pid = row["parent_id"]
            text = row.get("text", "")
            meta = row.get("meta", {}) or {}

            # Обязательные поля для parent-чанков
            required = {
                "parent_id": pid,
                "document_id": row.get("document_id"),
                "h_path": row.get("h_path", []),
                "child_ids": row.get("child_ids", []),
                "kind": "parent",
                "type": "parent",
            }
            for k, v in required.items():
                meta.setdefault(k, v)

            # Переносим документные меты с верхнего уровня при их наличии
            for k in ("product", "product_code", "version", "lang", "title", "section"):
                if k in row and k not in meta:
                    meta[k] = row[k]

            yield ParentRow(parent_id=pid, text=text, meta=meta)


# ---------------- l2 ----------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-нормализация по строкам. Возвращает float32.
    """
    if x.size == 0:
        return x.astype(np.float32) if x.dtype != np.float32 else x
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def check_l2_unit(x: np.ndarray, tol: float = 1e-3) -> bool:
    """
    Проверяет, что L2-норма каждого вектора ~ 1.0 (+/- tol).
    """
    if x.size == 0:
        return True
    n = np.linalg.norm(x, axis=1)
    return np.all((np.abs(n - 1.0) <= tol))
