# tools/rag/stores/qdrant_reader.py
from __future__ import annotations

import logging
import numpy as np
from typing import Any, Dict, List, Optional

from qdrant_client.models import Filter, FieldCondition, MatchAny

from .base import ChildHit, ParentHit, VectorReader
from .qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def _normalize_query_vector(vec):
    """
    Делает из эмбеддинга корректный список float для Qdrant.
    Возвращает None, если эмбеддинг пустой.
    """
    if vec is None:
        return None

    if isinstance(vec, np.ndarray):
        if vec.size == 0:
            return None

        return vec.astype(float).ravel().tolist()

    try:
        seq = list(vec)
    except TypeError:
        seq = [vec]

    if len(seq) == 0:
        return None

    return [float(x) for x in seq]


class QdrantVectorReader(VectorReader):
    """
    Ридер поверх двух коллекций Qdrant: родителей и дочерних чанков.
    Совместим с прежним поведением: COSINE, payload['_preview'], payload['extra'], и фильтрацией по chunk_id.
    """

    def __init__(
            self,
            url: str,
            api_key: Optional[str] = None,
            parents_collection: str = "transformers_parents",
            chunks_collection: str = "transformers_chunks",
            vector_size: int = 768,
    ) -> None:
        if not url:
            raise ValueError("Qdrant URL cannot be empty")
        if vector_size <= 0:
            raise ValueError("Vector size must be positive")

        self.parents_collection = parents_collection
        self.chunks_collection = chunks_collection

        # Инициализируем сторы (distance допускает любые регистры — см. qdrant_store)
        self.parents_client = QdrantVectorStore(
            url=url, api_key=api_key, collection=parents_collection, vector_size=vector_size, distance="COSINE"
        )
        self.chunks_client = QdrantVectorStore(
            url=url, api_key=api_key, collection=chunks_collection, vector_size=vector_size, distance="COSINE"
        )

        logger.info(
            "QdrantVectorReader initialized: parents=%s, chunks=%s, vector_size=%d",
            parents_collection, chunks_collection, vector_size
        )

    # ---------- helpers ----------

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        """Основной текст берем из '_preview', затем пытаемся из fallback-полей."""
        if not payload:
            return ""
        for field in ("_preview", "text", "content", "chunk_text", "body", "description", "summary", "title"):
            val = payload.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def _extract_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Возвращаем «чистые» метаданные: без векторов и без служебных '_' ключей, но с мерджем extra."""
        if not payload:
            return {}
        meta: Dict[str, Any] = {}
        for k, v in payload.items():
            if k in {"vector", "embedding", "_preview"}:
                continue
            if k.startswith("_"):
                continue
            meta[k] = v
        extra = payload.get("extra")
        if isinstance(extra, dict):
            # плоско вливаем extra, как было раньше
            for k, v in extra.items():
                meta[k] = v
        return meta

    def _child_ids_filter(self, child_ids: List[str]) -> Optional[Filter]:
        """Строим фильтр для выборки по chunk_id из списка child_ids."""
        if not child_ids:
            return None
        return Filter(should=[FieldCondition(key="chunk_id", match=MatchAny(any=child_ids))])

    # ---------- API ----------

    def search_parents(
            self,
            query_embedding: List[float],
            top_k: int,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[ParentHit]:
        query_vector = _normalize_query_vector(query_embedding)
        if not query_vector:
            raise ValueError("Query embedding cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            results = self.parents_client.query(query_vec=query_vector, top_k=top_k, filters=filters)
        except Exception as e:
            logger.error("Error searching parents: %s", e)
            return []

        hits: List[ParentHit] = []
        for r in results:
            payload = r.payload or {}
            text = self._extract_text(payload)
            meta = self._extract_metadata(payload)
            try:
                hits.append(ParentHit(id=str(r.id), text=text, score=float(r.score), meta=meta))
            except Exception as ve:
                # На случай, если score неожиданно вне [0,1] для другой метрики — мягко приводим.
                logger.debug("ParentHit validation warning: %s (id=%s, score=%s)", ve, r.id, r.score)
                hits.append(ParentHit(id=str(r.id), text=text, score=max(0.0, min(1.0, float(r.score))), meta=meta))

        logger.info("Found %d parent hits", len(hits))
        return hits

    def search_children(
            self,
            parent_hits: List[ParentHit],
            query_embedding: List[float],
            top_k_each: int,
    ) -> Dict[str, List[ChildHit]]:
        query_vector = _normalize_query_vector(query_embedding)
        if not parent_hits:
            raise ValueError("Parent hits cannot be empty")
        if top_k_each <= 0:
            raise ValueError("top_k_each must be positive")

        out: Dict[str, List[ChildHit]] = {}
        for parent in parent_hits:
            child_ids = parent.meta.get("child_ids") if isinstance(parent.meta, dict) else None
            if not isinstance(child_ids, list) or not child_ids:
                logger.debug("Parent %s has no child_ids", parent.id)
                out[parent.id] = []
                continue

            flt = self._child_ids_filter([str(c) for c in child_ids if c])
            if flt is None:
                out[parent.id] = []
                continue

            try:
                results = self.chunks_client.query(query_vec=query_vector, top_k=top_k_each, filters=flt)
            except Exception as e:
                logger.error("Error searching children for parent %s: %s", parent.id, e)
                out[parent.id] = []
                continue

            childs: List[ChildHit] = []
            for r in results:
                payload = r.payload or {}
                text = self._extract_text(payload)
                meta = self._extract_metadata(payload)
                try:
                    childs.append(
                        ChildHit(id=str(r.id), text=text, score=float(r.score), meta=meta, parent_id=parent.id))
                except Exception as ve:
                    logger.debug("ChildHit validation warning: %s (id=%s, score=%s)", ve, r.id, r.score)
                    childs.append(
                        ChildHit(
                            id=str(r.id),
                            text=text,
                            score=max(0.0, min(1.0, float(r.score))),
                            meta=meta,
                            parent_id=parent.id,
                        )
                    )
            out[parent.id] = childs

        return out

    # ---- Back-compat helper (если где-то до сих пор зовут старую сигнатуру) ----
    def search_children_by_ids(
            self,
            parent_ids: List[str],
            query_embedding: List[float],
            top_k_each: int,
    ) -> Dict[str, List[ChildHit]]:
        """Обратная совместимость: принимает parent_ids, делает поиск родителей и затем детей."""
        if not parent_ids:
            return {}
        # Быстрый поиск родителей по их точкам не делаем — у нас нет прямого индекса по parent_id в payload.
        # Поэтому берём top_k побольше и фильтруем по совпадению parent_id в payload/ID.
        parents = self.search_parents(query_embedding, top_k=max(len(parent_ids) * 3, 10))
        parents = [p for p in parents if p.meta.get("parent_id") in parent_ids or p.id in parent_ids]
        return self.search_children(parents, query_embedding, top_k_each)

    def get_collection_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "parents_collection": self.parents_collection,
            "chunks_collection": self.chunks_collection,
            "parents_count": 0,
            "chunks_count": 0,
            "status": "unknown",
        }
        try:
            pcount = self.parents_client.get_points_count()
            ccount = self.chunks_client.get_points_count()
            stats["parents_count"] = pcount
            stats["chunks_count"] = ccount
            stats["status"] = "healthy"
        except Exception as e:
            logger.error("Error getting collection stats: %s", e)
            stats["status"] = "error"
            stats["error"] = str(e)
        return stats

    def get_search_stats(self) -> Dict[str, Any]:
        return {
            "reader_type": self.__class__.__name__,
            "vector_size": self.parents_client.vector_size,
            "distance_metric": str(self.parents_client.distance),
            "collections": self.get_collection_stats(),
        }

    def health_check(self) -> Dict[str, Any]:
        info = {"status": "unknown", "components": {}}
        try:
            conn = self.parents_client.get_collection_info()
            info["components"]["parents"] = {
                "exists": conn is not None,
                "points": self.parents_client.get_points_count(),
            }
            conn2 = self.chunks_client.get_collection_info()
            info["components"]["chunks"] = {
                "exists": conn2 is not None,
                "points": self.chunks_client.get_points_count(),
            }
            info["status"] = "healthy"
        except Exception as e:
            info["status"] = "unhealthy"
            info["error"] = str(e)
        return info

    def __enter__(self) -> "QdrantVectorReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Явно закрывать клиентов не обязательно (закрываются в store), но и не вредно:
        try:
            self.parents_client.client.close()
        except Exception:
            pass
        try:
            self.chunks_client.client.close()
        except Exception:
            pass
