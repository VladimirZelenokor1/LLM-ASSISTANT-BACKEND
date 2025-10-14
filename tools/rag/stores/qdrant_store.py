# tools/rag/stores/qdrant_store.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, CollectionInfo, UpdateResult, ScoredPoint, PayloadSchemaType
)

logger = logging.getLogger(__name__)


def _normalize_distance(distance: Any) -> Distance:
    """
    Принимает строку в любом регистре ("Cosine", "COSINE", "cosine", "cos")
    и возвращает Distance.*. Поддерживает синонимы.
    """
    if isinstance(distance, Distance):
        return distance

    s = str(distance).strip()
    # 1) точные значения из официальной спеки
    direct = {
        "Cosine": Distance.COSINE,
        "Euclid": Distance.EUCLID,
        "Dot": Distance.DOT,
        "COSINE": Distance.COSINE,
        "EUCLID": Distance.EUCLID,
        "DOT": Distance.DOT,
    }
    if s in direct:
        return direct[s]

    # 2) гибкие алиасы (без учёта регистра, пробелов и знаков)
    key = s.lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "cosine": Distance.COSINE,
        "cos": Distance.COSINE,
        "angular": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "l2": Distance.EUCLID,
        "dot": Distance.DOT,
        "dotproduct": Distance.DOT,
    }
    if key in aliases:
        return aliases[key]

    raise ValueError(f"Unsupported distance metric: {distance}")


class QdrantVectorStore:
    """Клиент для работы с векторным хранилищем Qdrant."""

    def __init__(
            self,
            url: str,
            api_key: Optional[str],
            collection: str,
            vector_size: int,
            distance: Union[str, Distance] = "COSINE",  # совместимо со старой реализацией
    ) -> None:
        if not url:
            raise ValueError("Qdrant URL cannot be empty")
        if not collection:
            raise ValueError("Collection name cannot be empty")
        if vector_size <= 0:
            raise ValueError("Vector size must be positive")

        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection
        self.vector_size = vector_size
        self.distance = _normalize_distance(distance)

    def ensure_collection(self, on_missing: str = "create") -> None:
        """Ensure collection exists (create if missing and on_missing='create')."""
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection in names:
            return
        if on_missing != "create":
            raise RuntimeError(f"Collection {self.collection} missing")

        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
        )
        logger.info(f"Collection {self.collection} created (size={self.vector_size}, distance={self.distance})")

    def create_payload_index(self, fields: Iterable[Any]) -> None:
        """
        Создает payload-индексы.

        Поддерживает два формата:
          - список строк: ["product_code", "lang", ...] → KEYWORD
          - список словарей: [{"name":"title","schema":"keyword"}, {"name":"h_path","schema":"keyword"}]
        """
        if not fields:
            logger.info("No payload indexes requested")
            return

        def _schema_from(value: Any):
            # поддержка и строк, и словарей
            if isinstance(value, str):
                return value, PayloadSchemaType.KEYWORD  # по умолчанию
            if isinstance(value, dict):
                name = value.get("name")
                schema = str(value.get("schema", "keyword")).lower()
                mapping = {
                    "keyword": PayloadSchemaType.KEYWORD,
                    "integer": PayloadSchemaType.INTEGER,
                    "float": PayloadSchemaType.FLOAT,
                    "bool": PayloadSchemaType.BOOL,
                    "datetime": PayloadSchemaType.DATETIME,
                    "geo": PayloadSchemaType.GEO,
                }
                return name, mapping.get(schema, PayloadSchemaType.KEYWORD)
            return None, None

        created = 0
        for f in fields:
            name, schema = _schema_from(f)
            if not name or not schema:
                logger.warning(f"Skip invalid index spec: {f!r}")
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=name,
                    field_schema=schema,
                    wait=True,
                )
                created += 1
                logger.debug(f"Created payload index: {name} ({schema})")
            except Exception as e:
                logger.warning(f"Failed to create index for field {name}: {e}")

        logger.info(f"Created {created} payload index(es)")

    def upsert(
            self,
            ids: List[Union[str, int]],
            vectors: List[List[float]],
            payloads: List[Dict[str, Any]],
    ) -> UpdateResult:
        """Вставьте точки для сбора (дождитесь завершения)."""
        if not all((ids, vectors, payloads)):
            raise ValueError("ids, vectors and payloads cannot be empty")
        if len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors and payloads must have the same length")

        points = [
            PointStruct(id=str(ids[i]), vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        result = self.client.upsert(self.collection, points=points, wait=True)
        logger.info(f"Upserted {len(points)} points into '{self.collection}'")
        return result

    def _build_filter(self, filters: Optional[Union[Dict[str, Any], Filter]]) -> Optional[Filter]:
        """Принимаем как фильтры dict (старого образца), так и готовые объекты фильтра."""
        if filters is None:
            return None
        if isinstance(filters, Filter):
            return filters
        if isinstance(filters, dict):
            conds = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
            return Filter(must=conds) if conds else None
        raise ValueError(f"Unsupported filter type: {type(filters)}")

    def query(
            self,
            query_vec: List[float],
            top_k: int = 5,
            filters: Optional[Union[Dict[str, Any], Filter]] = None,
    ) -> List[ScoredPoint]:
        """Запрашивайте похожие векторы с помощью дополнительных фильтров."""
        if not query_vec:
            raise ValueError("Query vector cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Не жёстко валидируем длину: Qdrant сам вернёт ошибку, если не совпадёт
        flt = self._build_filter(filters)
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vec,
            limit=top_k,
            query_filter=flt,
        )

    # Доп. утилиты (не критичны для пайплайна, но полезны)

    def get_collection_info(self) -> Optional[CollectionInfo]:
        try:
            for c in self.client.get_collections().collections:
                if c.name == self.collection:
                    return c
            return None
        except Exception as e:
            logger.error(f"Failed to get collection info for {self.collection}: {e}")
            return None

    def get_points_count(self) -> int:
        try:
            res = self.client.count(collection_name=self.collection)
            return getattr(res, "count", 0)
        except Exception as e:
            logger.error(f"Failed to count points for {self.collection}: {e}")
            return 0

    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(collection_name=self.collection)
            logger.info(f"Collection {self.collection} deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {self.collection}: {e}")
            return False

    def __enter__(self) -> "QdrantVectorStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.client.close()
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")
