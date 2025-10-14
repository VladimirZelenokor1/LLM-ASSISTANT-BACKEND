from __future__ import annotations

import argparse
import os
import uuid
from typing import Any, Dict, List

import yaml

from .providers import (
    E5Config, E5Provider,
    OpenAIConfig, OpenAIProvider,
    EmbeddingProvider,
)
from .utils import (
    iter_chunk_files, iter_chunks_jsonl, iter_parents_jsonl,
    l2_normalize, check_l2_unit, ChunkRow,
)
from ..stores.qdrant_store import QdrantVectorStore

import logging

logger = logging.getLogger(__name__)


# ---------------- helpers ----------------

def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _make_provider(cfg: Dict[str, Any]) -> EmbeddingProvider:
    """
    Совместимо со старым embeddings.yaml:
    provider:
      name: e5|openai
      e5: {...} / openai: {...}
    """
    name = cfg["provider"]["name"].lower()
    if name == "e5":
        return E5Provider(E5Config(**(cfg["provider"].get("e5") or {})))
    if name == "openai":
        return OpenAIProvider(OpenAIConfig(**(cfg["provider"].get("openai") or {})))
    raise ValueError(f"Unknown provider: {name}")


def _make_qdrant(sc: Dict[str, Any], dim: int) -> QdrantVectorStore:
    """
    Создаёт коллекцию (если нужно) и индексы payload, как раньше.
    """
    store = QdrantVectorStore(
        url=sc["url"],
        api_key=sc.get("api_key") or None,
        collection=sc["collection"],
        vector_size=dim,
        distance=sc.get("distance", "Cosine"),  # строка уйдет в нормализатор в qdrant_store
    )
    store.ensure_collection(on_missing=sc.get("on_missing", "create"))
    store.create_payload_index(sc.get("payload_index", []))
    return store


def _uuidize(ids: List[str]) -> List[str]:
    """UUIDv5 для Qdrant id (строки)."""
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, s)) for s in ids]


# ---------------- commands ----------------

def cmd_embed(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)
    prov = _make_provider(cfg)

    st = cfg["stores"]

    # Не завязываемся на 'primary', создаём, если блок есть в конфиге
    qdrant_main = _make_qdrant(st["qdrant_main"], prov.dim) if "qdrant_main" in st else None
    qdrant_par = _make_qdrant(st["qdrant_parents"], prov.dim) if "qdrant_parents" in st else None

    if qdrant_main is None:
        logger.warning("[embed] qdrant_main store is not configured — upsert will be skipped")
    if qdrant_par is None:
        logger.info("[embed] qdrant_parents store is not configured — parent upsert will be skipped")

    root = cfg["chunks"]["root_dir"]
    glob_pat = cfg["chunks"]["glob"]
    bs = int(cfg["chunks"]["batch_size"])

    # ---- MAIN CHUNKS ----
    files = iter_chunk_files(root, glob_pat)
    tot = emb = ups = 0
    for fp in files:
        rows = list(iter_chunks_jsonl(fp))
        if not rows:
            continue

        for i in range(0, len(rows), bs):
            batch: List[ChunkRow] = rows[i: i + bs]
            texts = [r.text for r in batch]
            ids = _uuidize([r.chunk_id for r in batch])

            v = prov.embed_passages(texts)
            v = l2_normalize(v)
            assert v.shape[1] == prov.dim and check_l2_unit(v)

            payloads: List[Dict[str, Any]] = []
            for r in batch:
                meta = dict(r.meta)
                # превью и гарантия chunk_id в payload
                meta["_preview"] = (r.text or "")[:1000]
                meta.setdefault("chunk_id", r.chunk_id)
                payloads.append(meta)

            if qdrant_main:
                qdrant_main.upsert(ids, v.tolist(), payloads)
                ups += len(batch)

            emb += len(batch)
            tot += len(batch)
        print(f"[embed/main] file={os.path.basename(fp)} rows={len(rows)}")

    if "qdrant_main" in st:
        print(f"[embed/main] total={tot} embedded={emb} upserted={ups} collection={st['qdrant_main']['collection']}")
    else:
        print(f"[embed/main] total={tot} embedded={emb} upserted={ups} (no qdrant_main configured)")

    # ---- PARENT CHUNKS ----
    par_dir = os.path.join(root, "_parents")
    if os.path.isdir(par_dir):
        pfiles = sorted([os.path.join(par_dir, f) for f in os.listdir(par_dir) if f.endswith(".jsonl")])
        ptot = pemb = pups = 0
        for fp in pfiles:
            rows = list(iter_parents_jsonl(fp))
            if not rows:
                continue

            # отфильтруем очень короткие parents
            rows = [r for r in rows if len((r.text or "").strip()) >= 80]

            for i in range(0, len(rows), bs):
                batch = rows[i: i + bs]
                texts = [r.text for r in batch]
                ids = _uuidize([r.parent_id for r in batch])

                v = prov.embed_passages(texts)
                v = l2_normalize(v)
                assert v.shape[1] == prov.dim and check_l2_unit(v)

                payloads = []
                for r in batch:
                    m = dict(r.meta)
                    m["_preview"] = (r.text or "")[:1000]
                    m.setdefault("parent_id", r.parent_id)
                    m.setdefault("kind", "parent")
                    payloads.append(m)

                if qdrant_par:
                    qdrant_par.upsert(ids, v.tolist(), payloads)
                    pups += len(batch)

                pemb += len(batch)
                ptot += len(batch)
            print(f"[embed/parents] file={os.path.basename(fp)} rows={len(rows)}")

        if "qdrant_parents" in st:
            print(f"[embed/parents] total={ptot} embedded={pemb} upserted={pups} "
                  f"collection={st['qdrant_parents']['collection']}")
        else:
            print(f"[embed/parents] total={ptot} embedded={pemb} upserted={pups} (no qdrant_parents configured)")

    return 0


def cmd_sanity(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)
    prov = _make_provider(cfg)

    # main + parents
    qm_cfg = cfg["stores"]["qdrant_main"]
    qdr_main = QdrantVectorStore(
        url=qm_cfg["url"],
        api_key=qm_cfg.get("api_key") or None,
        collection=qm_cfg["collection"],
        vector_size=prov.dim,
        distance=qm_cfg.get("distance", "Cosine"),
    )

    qp = None
    if "qdrant_parents" in cfg["stores"]:
        qp_cfg = cfg["stores"]["qdrant_parents"]
        qp = QdrantVectorStore(
            url=qp_cfg["url"],
            api_key=qp_cfg.get("api_key") or None,
            collection=qp_cfg["collection"],
            vector_size=prov.dim,
            distance=qp_cfg.get("distance", "Cosine"),
        )

    k = int(cfg.get("sanity", {}).get("k", 5))
    queries = cfg.get("sanity", {}).get("queries", ["transformers pipeline", "quantization"])

    base_filter = {"product_code": "transformers", "lang": "en"}

    for q in queries:
        print(f"\n[Q] {q}")
        qv = prov.embed_queries([q])
        qv = l2_normalize(qv)

        # --- main ---
        hits = qdr_main.query(qv[0].tolist(), top_k=k, filters=base_filter)
        if not hits:
            print("  (no hits in main collection)")
        else:
            for r in hits:
                title = (r.payload or {}).get("title") or (r.payload or {}).get("section") or "-"
                cid = (r.payload or {}).get("chunk_id") or r.id
                preview = (r.payload or {}).get("_preview", "")[:200]
                print(f"  • main  score={r.score:.4f}  title={title}  id={cid}")
                if preview:
                    print(f"     preview: {preview}...")

        # --- parents ---
        if qp is not None:
            parent_filter = {"product_code": "transformers", "kind": "parent"}
            phits = qp.query(qv[0].tolist(), top_k=min(3, k), filters=parent_filter)
            if not phits:
                print("  (no hits in parents)")
            else:
                for r in phits:
                    hpath = (r.payload or {}).get("h_path") or "-"
                    pid = (r.payload or {}).get("parent_id") or r.id
                    child_count = len((r.payload or {}).get("child_ids", []))
                    preview = (r.payload or {}).get("_preview", "")[:200]
                    print(f"  • parent score={r.score:.4f}  h_path={hpath}  children={child_count}  id={pid}")
                    if preview:
                        print(f"     preview: {preview}...")

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """
    Управление индексером (как в старой версии).
    """
    from tools.rag.stores.indexer import create_indexer  # важно: тот же импорт, что и раньше

    indexer = create_indexer(args.config)

    if args.index_action == "full":
        success = indexer.index_all()
        if success:
            stats = indexer.get_index_stats()
            print(f"Indexing completed. Stats: {stats}")
            return 0
        else:
            print("Indexing failed")
            return 1

    elif args.index_action == "verify":
        result = indexer.verify_index()
        print(f"Verification: {result}")
        return 0

    elif args.index_action == "stats":
        stats = indexer.get_index_stats()
        print(f"Statistics: {stats}")
        return 0


# ---------------- main ----------------

def main() -> int:
    p = argparse.ArgumentParser(prog="rag-embed", description="Embeddings → Qdrant")
    p.add_argument("--config", default="configs/embeddings.yaml", help="path to embeddings.yaml")

    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="Embed chunks and upsert")
    e.set_defaults(func=cmd_embed)

    s = sub.add_parser("sanity", help="Quick retrieval sanity check")
    s.set_defaults(func=cmd_sanity)

    i = sub.add_parser("index", help="Index management operations")
    i.add_argument("index_action", choices=["full", "verify", "stats"], help="Indexing action to perform")
    i.set_defaults(func=cmd_index)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
