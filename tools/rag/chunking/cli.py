from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz.fuzz import token_set_ratio

from .export import JsonlWriter
from .models import ChunkingReport
from .policies import ChunkingConfig
from .preprocess import preprocess_text
from .splitters import RuleAwareSplitter, _tok_len

logger = logging.getLogger(__name__)


# ---------- helpers: совместимые с СТАРОЙ логикой ----------

def _stable_document_id(meta: Dict[str, Any], fallback: str) -> str:
    for k in ("doc_id", "repo_path", "canonical_url", "url"):
        v = meta.get(k)
        if v:
            v_str = str(v).replace('\\', '/').strip('/')
            if v_str:
                return v_str
    return str(fallback).replace('\\', '/').strip('/')


def _stable_chunk_id(document_id: str, seq: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{document_id}_{seq}_{h}"


def _dedup_chunks(chunks: List[Any]) -> List[Any]:
    seen = set()
    unique = []
    for c in chunks:
        text = c["text"] if isinstance(c, dict) else getattr(c, "text", "")
        h = sha1((text or "").encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(c)
    return unique


def _get_meta(c: Any, k: str, default=None):
    """
    Унифицированное извлечение meta-атрибутов:
    - для dict-чанка: c["meta"][k]
    - для объектного чанка: c.meta.k (dataclass/pydantic)
    - поддерживает оба варианта (старые 'h_path', 'extra', 'seq' и т.п.)
    """
    if isinstance(c, dict):
        meta = c.get("meta", {}) or {}
        return meta.get(k, default)

    meta = getattr(c, "meta", None)
    if meta is None:
        meta = getattr(c, "metadata", None)

    if meta is None:
        return default

    # прямой атрибут
    if hasattr(meta, k):
        return getattr(meta, k)

    # алиасы из более «новых» реализаций
    if k == "h_path" and hasattr(meta, "heading_path"):
        return getattr(meta, "heading_path")

    # попытка достать в extra
    extra = getattr(meta, "extra", None)
    if isinstance(extra, dict) and k in extra:
        return extra.get(k, default)

    return default


def _ensure_meta_dict(c: Any) -> Dict[str, Any]:
    if isinstance(c, dict):
        if "meta" not in c or c["meta"] is None:
            c["meta"] = {}
        return c["meta"]
    # объектный — возвращаем dict-представление только когда нужно обновлять массово
    m = getattr(c, "meta", None)
    if isinstance(m, dict):
        return m
    # если dataclass/pydantic — вернём временный dict-представитель
    view: Dict[str, Any] = {}
    for key in ("seq", "h_path", "title", "section", "lang"):
        v = getattr(m, key, None)
        if v is not None:
            view[key] = v
    ex = getattr(m, "extra", None)
    if isinstance(ex, dict):
        view["extra"] = ex
    return view


def _set_meta(c: Any, k: str, value):
    if isinstance(c, dict):
        meta = _ensure_meta_dict(c)
        meta[k] = value
        return
    # объектный чанк: старательно поставить на c.meta.k
    m = getattr(c, "meta", None)
    if m is None:
        setattr(c, "meta", {})
        m = c.meta
    try:
        setattr(m, k, value)
    except Exception:
        # если meta — dict
        try:
            if not isinstance(m, dict):
                # заменим на dict-мету, чтобы не падать
                setattr(c, "meta", {})
                m = c.meta
            m[k] = value
        except Exception:
            logger.debug("Failed to set meta %s on chunk", k, exc_info=True)


def _set_meta_extra(c: Any, key: str, value):
    """
    Безопасно пишет meta.extra[key] для:
      - чанка-словаря,
      - объектного чанка (датакласс/pydantic).
    """
    if isinstance(c, dict):
        meta = c.setdefault("meta", {})
        extra = meta.setdefault("extra", {})
        extra[key] = value
        return

    meta = getattr(c, "meta", None)
    if meta is None:
        # создадим словарь
        setattr(c, "meta", type("M", (), {})())
        meta = c.meta
    if getattr(meta, "extra", None) is None:
        try:
            setattr(meta, "extra", {})
        except Exception:
            # если мета не поддерживает setattr — заменим мету на dict
            setattr(c, "meta", {"extra": {}})
            meta = c.meta
    try:
        meta.extra[key] = value
    except Exception:
        # meta.extra может быть не dict — принудительно заменим
        setattr(meta, "extra", {})
        meta.extra[key] = value


def _group_key_for_parent(c: Any):
    h_path = _get_meta(c, "h_path") or []
    if isinstance(h_path, str):
        h_path = [h_path]
    if h_path:
        return tuple(h_path[:2])
    title = _get_meta(c, "title", None)
    return (title,) if title else ("root",)


def _near_dedup_neighbors(chs: List[Any], near_thr: int, contain_thr: float) -> List[Any]:
    if not chs:
        return chs

    def _sec_key(c):
        hp = _get_meta(c, "h_path") or []
        if isinstance(hp, str):
            hp = [hp]
        return tuple(hp[:2]) if hp else ("root",)

    out = []
    last_by_sec: Dict[Tuple[str, ...], Tuple[Any, int]] = {}

    for c in chs:
        key = _sec_key(c)
        text_c = c["text"] if isinstance(c, dict) else getattr(c, "text", "")
        if not text_c.strip():
            continue

        if key in last_by_sec:
            prev_c, prev_tok = last_by_sec[key]
            text_p = prev_c["text"] if isinstance(prev_c, dict) else getattr(prev_c, "text", "")
            if token_set_ratio(text_p, text_c) >= near_thr:
                continue
            cur_tok = _tok_len(text_c)
            shared_est = min(prev_tok, cur_tok)
            novel_ratio = max(0, cur_tok - shared_est) / max(1, cur_tok)
            if novel_ratio < contain_thr:
                continue

        cur_tok = _tok_len(text_c)
        last_by_sec[key] = (c, cur_tok)
        out.append(c)
    return out


def _build_parents(document_id: str, chunks: List[Any], target_tokens: int = 1200,
                   document_meta: Dict[str, Any] = None):
    """
    Создает родительские чанки с метаданными из исходного документа — строго как в СТАРОМ.
    """
    if document_meta is None:
        document_meta = {}

    parents: List[Dict[str, Any]] = []
    groups = defaultdict(list)
    for c in chunks:
        groups[_group_key_for_parent(c)].append(c)

    # Стабильные метаданные документа
    stable_meta = {
        "product": document_meta.get("product"),
        "product_code": document_meta.get("product_code"),
        "version": document_meta.get("version"),
        "lang": document_meta.get("lang"),
        "title": document_meta.get("title"),
        "section": document_meta.get("section"),
    }
    stable_meta = {k: v for k, v in stable_meta.items() if v is not None}

    for key, items in groups.items():
        acc = ""
        child_ids: List[str] = []

        def flush():
            nonlocal acc, child_ids
            if not acc.strip():
                return

            parent_id = f"{document_id}__parent__{'__'.join([str(x) for x in key if x])}"

            preview = acc.strip()[:1000]
            if len(acc.strip()) > 1000:
                preview += "..."

            parent_meta = {
                **stable_meta,
                "type": "parent",
                "parent_id": parent_id,
                "document_id": document_id,
                "h_path": list(key),
                "_preview": preview,
            }
            stats = {
                "child_count": len(child_ids),
                "total_length": len(acc.strip()),
                "token_count": _tok_len(acc.strip()),
            }
            parent_meta["stats"] = stats

            parents.append({
                "type": "parent",
                "parent_id": parent_id,
                "document_id": document_id,
                "h_path": list(key),
                "child_ids": child_ids[:],
                "text": acc.strip(),
                "meta": parent_meta,
            })
            acc = ""
            child_ids = []

        for c in items:
            cid = None
            extra = _get_meta(c, "extra", {}) or {}
            if isinstance(extra, dict):
                cid = extra.get("chunk_id")
            text = c["text"] if isinstance(c, dict) else getattr(c, "text", "")
            cand = (acc + "\n\n" + text).strip() if acc else text
            if _tok_len(cand) > target_tokens:
                flush()
                acc = text
                child_ids = [cid] if cid else []
            else:
                acc = cand
                if cid:
                    child_ids.append(cid)
        flush()

    return parents


def _assign_neighbors_by_section(chunks: List[Any]):
    groups = defaultdict(list)
    for c in chunks:
        groups[_group_key_for_parent(c)].append(c)

    for items in groups.values():
        n = len(items)
        for i, c in enumerate(items):
            prev_id = None
            next_id = None
            if i > 0:
                extra_prev = _get_meta(items[i - 1], "extra", {}) or {}
                prev_id = extra_prev.get("chunk_id") if isinstance(extra_prev, dict) else None
            if i + 1 < n:
                extra_next = _get_meta(items[i + 1], "extra", {}) or {}
                next_id = extra_next.get("chunk_id") if isinstance(extra_next, dict) else None
            if prev_id:
                _set_meta_extra(c, "prev_id", prev_id)
            if next_id:
                _set_meta_extra(c, "next_id", next_id)


def _ensure_dir_for_path(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ---------- processor (обёртка поверх старых хелперов) ----------

@dataclass
class ChunkProcessor:
    config: ChunkingConfig
    report: ChunkingReport = field(default_factory=ChunkingReport)
    total_sum_chars: int = 0
    total_chunks: int = 0

    NEAR_THR: int = 95
    CONTAIN_THR: float = 0.10
    PARENT_TARGET_TOKENS: int = 1200

    def run(self) -> None:
        os.makedirs(self.config.paths.preprocessed_dir, exist_ok=True)
        os.makedirs(self.config.paths.chunks_dir, exist_ok=True)

        files = sorted(str(p) for p in Path(self.config.paths.input_dir).rglob("*.jsonl"))
        if not files:
            logger.warning("No .jsonl files found under: %s", self.config.paths.input_dir)

        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    line = f.readline()
                if not line:
                    continue

                rec = json.loads(line)
                text = rec.get("text", "")
                meta = rec.get("meta", {}) or {}

                # 1) препроцесс
                text_p = preprocess_text(text)

                # 2) document_id
                raw_id = _stable_document_id(meta, os.path.splitext(os.path.basename(path))[0])
                document_id = str(raw_id).strip("/\\")

                # 3) сохраняем preprocessed
                pp_path = os.path.join(self.config.paths.preprocessed_dir, f"{document_id}.jsonl")
                _ensure_dir_for_path(pp_path)
                with open(pp_path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"document_id": document_id, "text": text_p, "meta": meta},
                                       ensure_ascii=False) + "\n")

                # 4) чанкинг
                splitter = RuleAwareSplitter(self.config, meta)
                chunks = splitter.split(text_p)

                # 5) дедуп
                chunks = _dedup_chunks(chunks)
                chunks = _near_dedup_neighbors(chunks, near_thr=self.NEAR_THR, contain_thr=self.CONTAIN_THR)

                # 6) seq и chunk_id
                for i, c in enumerate(chunks, 1):
                    _set_meta(c, "seq", i)

                for c in chunks:
                    text_c = c["text"] if isinstance(c, dict) else getattr(c, "text", "")
                    seq = int(_get_meta(c, "seq", 0) or 0)
                    chunk_id = _stable_chunk_id(document_id, seq, text_c or "")
                    _set_meta_extra(c, "chunk_id", chunk_id)

                # 6.1) вливаем документные поля в чанки
                doc_keys = [
                    "product", "product_code", "version", "lang",
                    "title", "section", "repo_path", "url", "canonical_url"
                ]
                doc_meta = {k: meta.get(k) for k in doc_keys if meta.get(k) is not None}
                doc_meta["document_id"] = document_id

                for c in chunks:
                    if isinstance(c, dict):
                        m = c.setdefault("meta", {})
                        m.update(doc_meta)
                    else:
                        current_meta = getattr(c, "meta", {})
                        if hasattr(current_meta, 'extra'):
                            if not hasattr(current_meta.extra, '__setitem__'):
                                current_meta.extra = {}
                            current_meta.extra.update(doc_meta)
                        elif isinstance(current_meta, dict):
                            current_meta.setdefault('extra', {}).update(doc_meta)
                        else:
                            try:
                                if not hasattr(current_meta, 'extra'):
                                    setattr(current_meta, 'extra', {})
                                getattr(current_meta, 'extra').update(doc_meta)
                            except Exception:
                                logger.debug("Failed to merge document meta into chunk", exc_info=True)

                # 7) prev/next
                _assign_neighbors_by_section(chunks)

                # 8) parent-чанки
                parents = _build_parents(document_id, chunks, target_tokens=self.PARENT_TARGET_TOKENS,
                                         document_meta=meta)
                if parents:
                    parents_dir = os.path.join(self.config.paths.chunks_dir, "_parents")
                    ppath = os.path.join(parents_dir, f"{document_id}.jsonl")
                    _ensure_dir_for_path(ppath)
                    with open(ppath, "w", encoding="utf-8") as f:
                        for p in parents:
                            f.write(json.dumps(p, ensure_ascii=False) + "\n")

                # 9) запись чанков
                ch_path = os.path.join(self.config.paths.chunks_dir, f"{document_id}.jsonl")
                _ensure_dir_for_path(ch_path)
                rep = JsonlWriter(ch_path).write(chunks)

                # 10) метрики
                self.total_sum_chars += sum(len(c["text"] if isinstance(c, dict) else getattr(c, "text", ""))
                                            for c in chunks)
                self.total_chunks += len(chunks)
                self.report.chunks_total += rep.chunks_total
                self.report.chunks_with_code += rep.chunks_with_code
                self.report.chunks_with_table += rep.chunks_with_table

            except Exception as e:
                print(f"[error] {path}: {e.__class__.__name__}: {e}")
                logger.exception("Error processing file %s", path)
                continue

        self.report.avg_len_chars = (self.total_sum_chars / self.total_chunks) if self.total_chunks else 0.0

        # Итог — печать в точности как в СТАРОЙ версии:
        print("Done.")
        print(f"- Docs: {len(files)}")
        print(f"- Chunks total: {self.report.chunks_total}")
        print(f"- With code/table: {self.report.chunks_with_code}/{self.report.chunks_with_table}")
        print(f"- Avg len (chars): {self.report.avg_len_chars:.1f}")


# ---------- CLI ----------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    ap = argparse.ArgumentParser(description="Document chunking processor")
    ap.add_argument("--config", required=True, help="Path to configuration file")
    ap.add_argument("--input-dir")
    ap.add_argument("--preprocessed-dir")
    ap.add_argument("--chunks-dir")
    args = ap.parse_args()

    cfg = ChunkingConfig.from_yaml(args.config)
    if args.input_dir:
        cfg.paths.input_dir = args.input_dir
    if args.preprocessed_dir:
        cfg.paths.preprocessed_dir = args.preprocessed_dir
    if args.chunks_dir:
        cfg.paths.chunks_dir = args.chunks_dir

    ChunkProcessor(cfg).run()


if __name__ == "__main__":
    main()
