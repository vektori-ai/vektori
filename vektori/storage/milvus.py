"""Milvus storage backend — dedicated vector database with partition-key isolation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Collection name suffixes
_FACTS = "facts"
_SENTENCES = "sentences"
_EPISODES = "episodes"

# Typed rows in dynamic schema
_FACT_RECORD = "fact"
_SENTENCE_RECORD = "sentence"
_EPISODE_RECORD = "episode"
_SESSION_RECORD = "session"


def _col(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv_to_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _ids_to_csv(values: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ",".join(ordered)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class MilvusBackend(StorageBackend):
    """
    Milvus storage backend using pymilvus high-level client.

    Uses exactly three collections:
        - {prefix}_facts
        - {prefix}_sentences
        - {prefix}_episodes

    Partition key:
        user_id (all collections)

    Sessions are stored as typed records (`record_type = "session"`) in the
    sentences collection to keep the 3-collection constraint.
    """

    def __init__(
        self,
        url: str = "http://localhost:19530",
        token: str | None = None,
        prefix: str = "vektori",
        embedding_dim: int = 1024,
    ) -> None:
        self.url = url
        self.token = token
        self.prefix = prefix
        self.embedding_dim = embedding_dim

        self._client: Any | None = None
        self._datatype: Any | None = None
        self._json_is_native = False
        self._sentence_embedder: Any | None = None

    # ── Collection name helpers ────────────────────────────────────────────────

    @property
    def _facts_col(self) -> str:
        return _col(self.prefix, _FACTS)

    @property
    def _sentences_col(self) -> str:
        return _col(self.prefix, _SENTENCES)

    @property
    def _episodes_col(self) -> str:
        return _col(self.prefix, _EPISODES)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        try:
            from pymilvus import AsyncMilvusClient, DataType, MilvusClient
        except ImportError as e:
            raise ImportError("pymilvus required: pip install 'vektori[milvus]'") from e

        self._datatype = DataType
        self._json_is_native = hasattr(DataType, "JSON")
        client_kwargs: dict[str, Any] = {"uri": self.url}
        if self.token:
            client_kwargs["token"] = self.token

        # Prefer asyncio-native client; gracefully fallback to sync client.
        try:
            self._client = AsyncMilvusClient(**client_kwargs)
        except Exception:
            self._client = MilvusClient(**client_kwargs)

        await self._ensure_collection(self._facts_col, self._fact_schema())
        await self._ensure_collection(self._sentences_col, self._sentence_schema())
        await self._ensure_collection(self._episodes_col, self._episode_schema())

        for col in (self._facts_col, self._sentences_col, self._episodes_col):
            try:
                await self._call("load_collection", collection_name=col)
            except Exception:
                # Some Milvus setups auto-load; safe to continue.
                pass

        logger.info("Milvus backend initialized at %s (prefix=%s)", self.url, self.prefix)

    async def close(self) -> None:
        if not self._client:
            return
        close_fn = getattr(self._client, "close", None)
        if close_fn is None:
            self._client = None
            return

        try:
            if inspect.iscoroutinefunction(close_fn):
                await close_fn()
            else:
                await asyncio.to_thread(close_fn)
        except Exception:
            pass
        finally:
            self._client = None

    async def _ensure_collection(
        self,
        name: str,
        field_specs: list[dict[str, Any]],
    ) -> None:
        if await self._has_collection(name):
            return

        schema = None
        create_schema_fn = getattr(self._client, "create_schema", None)
        if create_schema_fn is not None:
            schema = create_schema_fn(auto_id=False, enable_dynamic_field=True)
            for spec in field_specs:
                schema.add_field(**spec)

        index_params = None
        prepare_index_fn = getattr(self._client, "prepare_index_params", None)
        if prepare_index_fn is not None:
            index_params = prepare_index_fn()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="AUTOINDEX",
            )

        if schema is not None:
            kwargs: dict[str, Any] = {
                "collection_name": name,
                "schema": schema,
            }
            if index_params is not None:
                kwargs["index_params"] = index_params
            await self._call("create_collection", **kwargs)
            return

        # Fallback for very old client APIs.
        await self._call(
            "create_collection",
            collection_name=name,
            dimension=self.embedding_dim,
            metric_type="COSINE",
            auto_id=False,
            id_type="string",
        )

    async def _has_collection(self, name: str) -> bool:
        for kwargs in ({"collection_name": name}, {"name": name}):
            try:
                exists = await self._call("has_collection", **kwargs)
                return bool(exists)
            except Exception:
                continue

        try:
            names = await self._call("list_collections")
            return name in (names or [])
        except Exception:
            return False

    async def _call(self, method: str, **kwargs: Any) -> Any:
        if not self._client:
            raise RuntimeError("Milvus client is not initialized")
        fn = getattr(self._client, method)
        if inspect.iscoroutinefunction(fn):
            try:
                return await fn(**kwargs)
            except RuntimeError:
                pass  # loop mismatch — fall through to sync fallback
            except Exception as exc:
                # pymilvus decorators wrap RuntimeError in MilvusException
                # before our handler sees it; detect the wrapped loop error.
                if "attached to a different loop" not in str(exc):
                    raise
            # pytest-asyncio (or other multi-loop runtimes) may reuse a backend
            # instance across event loops, which breaks grpc.aio internals.
            logger.warning("AsyncMilvusClient loop mismatch detected; falling back to sync client")
            from pymilvus import MilvusClient

            fallback_kwargs: dict[str, Any] = {"uri": self.url}
            if self.token:
                fallback_kwargs["token"] = self.token
            self._client = MilvusClient(**fallback_kwargs)
            fn = getattr(self._client, method)
            return await asyncio.to_thread(fn, **kwargs)
        return await asyncio.to_thread(fn, **kwargs)

    def _json_value(self, metadata: dict[str, Any] | None) -> Any:
        payload = metadata or {}
        if self._json_is_native:
            return payload
        return json.dumps(payload)

    def set_sentence_embedder(self, embedder: Any) -> None:
        """Inject the embedder used by the ingestion/retrieval pipeline."""
        self._sentence_embedder = embedder

    async def _embed_quotes(self, quotes: list[str]) -> list[list[float]]:
        """Best-effort quote embedding using the configured pipeline embedder."""
        if not quotes:
            return []
        if self._sentence_embedder is None:
            logger.debug("MilvusBackend.find_sentences_by_similarity: no embedder configured")
            return []

        embed_batch = getattr(self._sentence_embedder, "embed_batch", None)
        if embed_batch is not None:
            if inspect.iscoroutinefunction(embed_batch):
                vectors = await embed_batch(quotes)
            else:
                vectors = await asyncio.to_thread(embed_batch, quotes)
            return [list(v) for v in vectors if isinstance(v, list)]

        embed_one = getattr(self._sentence_embedder, "embed", None)
        if embed_one is None:
            logger.warning(
                "MilvusBackend.find_sentences_by_similarity: configured embedder has no embed/embed_batch"
            )
            return []

        vectors: list[list[float]] = []
        for quote in quotes:
            if inspect.iscoroutinefunction(embed_one):
                vector = await embed_one(quote)
            else:
                vector = await asyncio.to_thread(embed_one, quote)
            if isinstance(vector, list):
                vectors.append(vector)
        return vectors

    async def _query(
        self,
        collection_name: str,
        filter_expr: str | None = None,
        output_fields: list[str] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "limit": limit,
            "offset": offset,
        }
        if output_fields is not None:
            kwargs["output_fields"] = output_fields

        if filter_expr:
            type_errors: list[TypeError] = []
            for key in ("filter", "expr"):
                try:
                    rows = await self._call("query", **{**kwargs, key: filter_expr})
                    return self._normalize_query_rows(rows)
                except TypeError as e:
                    type_errors.append(e)
                    continue
            raise RuntimeError(
                "Milvus client does not support filtered query parameters ('filter'/'expr')"
            ) from type_errors[-1]

        rows = await self._call("query", **kwargs)
        return self._normalize_query_rows(rows)

    async def _query_all(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str],
        batch_size: int = 200,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            batch = await self._query(
                collection_name=collection_name,
                filter_expr=filter_expr,
                output_fields=output_fields,
                limit=batch_size,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        return rows

    async def _search(
        self,
        collection_name: str,
        embedding: list[float],
        filter_expr: str,
        limit: int,
        output_fields: list[str],
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "data": [embedding],
            "limit": limit,
            "output_fields": output_fields,
            "search_params": {"metric_type": "COSINE", "params": {}},
        }

        for key in ("filter", "expr"):
            try:
                raw = await self._call("search", **{**kwargs, key: filter_expr})
                return self._normalize_search_rows(raw, embedding)
            except TypeError as e:
                last_type_error = e
                continue

        raise RuntimeError(
            "Milvus client does not support filtered search parameters ('filter'/'expr')"
        ) from last_type_error

    def _normalize_query_rows(self, rows: Any) -> list[dict[str, Any]]:
        if rows is None:
            return []
        if isinstance(rows, dict):
            data = rows.get("data") or []
            return [dict(r) for r in data if isinstance(r, dict)]
        if isinstance(rows, list):
            normalized: list[dict[str, Any]] = []
            for row in rows:
                if isinstance(row, dict):
                    normalized.append(dict(row))
            return normalized
        return []

    def _normalize_search_rows(
        self, raw: Any, query_embedding: list[float]
    ) -> list[dict[str, Any]]:
        if raw is None:
            return []

        data = raw
        if isinstance(raw, dict):
            data = raw.get("data") or raw.get("result") or []

        if isinstance(data, list) and data and isinstance(data[0], list):
            hits = data[0]
        elif isinstance(data, list):
            hits = data
        else:
            hits = []

        rows: list[dict[str, Any]] = []

        for hit in hits:
            if isinstance(hit, dict):
                entity = hit.get("entity") or hit.get("fields") or {}
                if not entity:
                    entity = {
                        k: v
                        for k, v in hit.items()
                        if k not in {"distance", "score", "id", "pk", "primary_key"}
                    }
                hid = hit.get("id") or hit.get("pk") or hit.get("primary_key") or entity.get("id")
                score = hit.get("score")
                distance_raw = hit.get("distance")
            else:
                entity = getattr(hit, "entity", {}) or {}
                hid = getattr(hit, "id", None)
                score = getattr(hit, "score", None)
                distance_raw = getattr(hit, "distance", None)

            row = dict(entity)
            if hid is not None:
                row["id"] = str(hid)

            emb = row.get("embedding")
            if isinstance(emb, list) and emb:
                distance = 1.0 - _cosine_similarity(query_embedding, emb)
            elif distance_raw is not None:
                distance = float(distance_raw)
            elif score is not None:
                distance = 1.0 - float(score)
            else:
                distance = 1.0

            row["distance"] = max(0.0, float(distance))
            rows.append(row)

        rows.sort(key=lambda r: r.get("distance", 1.0))
        return rows

    async def _upsert_rows(self, collection_name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            await self._call("upsert", collection_name=collection_name, data=rows)
            await self._flush_collection(collection_name)
            return
        except (NotImplementedError, AttributeError):
            logger.debug("Upsert not supported; falling back to delete+insert")
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "unsupported" not in exc_msg and "not implemented" not in exc_msg:
                raise
            logger.debug("Upsert not supported (%s); falling back to delete+insert", exc)

        # Fallback for clients without upsert support.
        ids = [str(r["id"]) for r in rows if r.get("id")]
        if ids:
            ors = " or ".join([f'id == "{_escape(i)}"' for i in ids])
            try:
                await self._delete(collection_name, ors)
            except RuntimeError as e:
                logger.warning(
                    "Milvus delete fallback unsupported before insert; continuing "
                    "with insert (collection=%s, filter=%s): %s",
                    collection_name,
                    ors,
                    e,
                )
            except Exception as e:
                logger.error(
                    "Milvus delete fallback failed; aborting insert to avoid "
                    "duplicate records (collection=%s, filter=%s, ids=%s): %s",
                    collection_name,
                    ors,
                    ids,
                    e,
                )
                raise
        await self._call("insert", collection_name=collection_name, data=rows)
        await self._flush_collection(collection_name)

    async def _flush_collection(self, collection_name: str) -> None:
        """Best-effort flush so writes are query-visible immediately."""
        for kwargs in (
            {"collection_name": collection_name},
            {"collection_names": [collection_name]},
        ):
            try:
                await self._call("flush", **kwargs)
                return
            except TypeError:
                continue
            except Exception:
                return

    async def _delete(self, collection_name: str, filter_expr: str) -> None:
        last_type_error: TypeError | None = None
        for key in ("filter", "expr"):
            try:
                await self._call("delete", collection_name=collection_name, **{key: filter_expr})
                await self._flush_collection(collection_name)
                return
            except TypeError as e:
                last_type_error = e
                continue

        raise RuntimeError(
            "Milvus client does not support filtered delete parameters ('filter'/'expr')"
        ) from last_type_error

    async def _get_one(
        self,
        collection_name: str,
        filter_expr: str,
        output_fields: list[str],
    ) -> dict[str, Any] | None:
        rows = await self._query(
            collection_name=collection_name,
            filter_expr=filter_expr,
            output_fields=output_fields,
            limit=1,
            offset=0,
        )
        return rows[0] if rows else None

    def _fact_output_fields(self) -> list[str]:
        return [
            "id",
            "text",
            "embedding",
            "user_id",
            "agent_id",
            "session_id",
            "subject",
            "confidence",
            "mentions",
            "is_active",
            "superseded_by",
            "event_time",
            "created_at",
            "metadata",
            "source_sentence_ids",
            "record_type",
        ]

    def _sentence_output_fields(self) -> list[str]:
        return [
            "id",
            "text",
            "embedding",
            "user_id",
            "agent_id",
            "session_id",
            "turn_number",
            "sentence_index",
            "role",
            "content_hash",
            "mentions",
            "is_active",
            "created_at",
            "next_sentence_id",
            "metadata",
            "started_at",
            "ended_at",
            "record_type",
        ]

    def _episode_output_fields(self) -> list[str]:
        return [
            "id",
            "text",
            "embedding",
            "user_id",
            "agent_id",
            "session_id",
            "is_active",
            "created_at",
            "metadata",
            "fact_ids",
            "record_type",
        ]

    # ── Schema builders ────────────────────────────────────────────────────────

    def _metadata_field(self) -> dict[str, Any]:
        if self._json_is_native:
            return {"field_name": "metadata", "datatype": self._datatype.JSON}
        return {
            "field_name": "metadata",
            "datatype": self._datatype.VARCHAR,
            "max_length": 8192,
        }

    def _fact_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "field_name": "id",
                "datatype": self._datatype.VARCHAR,
                "is_primary": True,
                "max_length": 64,
            },
            {"field_name": "text", "datatype": self._datatype.VARCHAR, "max_length": 8192},
            {
                "field_name": "embedding",
                "datatype": self._datatype.FLOAT_VECTOR,
                "dim": self.embedding_dim,
            },
            {
                "field_name": "user_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 256,
                "is_partition_key": True,
            },
            {"field_name": "agent_id", "datatype": self._datatype.VARCHAR, "max_length": 256},
            {
                "field_name": "session_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 256,
            },
            {"field_name": "subject", "datatype": self._datatype.VARCHAR, "max_length": 256},
            {"field_name": "confidence", "datatype": self._datatype.FLOAT},
            {"field_name": "mentions", "datatype": self._datatype.INT64},
            {"field_name": "is_active", "datatype": self._datatype.BOOL},
            {
                "field_name": "superseded_by",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "event_time",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "created_at",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "source_sentence_ids",
                "datatype": self._datatype.VARCHAR,
                "max_length": 8192,
            },
            {
                "field_name": "record_type",
                "datatype": self._datatype.VARCHAR,
                "max_length": 32,
            },
            self._metadata_field(),
        ]

    def _sentence_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "field_name": "id",
                "datatype": self._datatype.VARCHAR,
                "is_primary": True,
                "max_length": 64,
            },
            {"field_name": "text", "datatype": self._datatype.VARCHAR, "max_length": 8192},
            {
                "field_name": "embedding",
                "datatype": self._datatype.FLOAT_VECTOR,
                "dim": self.embedding_dim,
            },
            {
                "field_name": "user_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 256,
                "is_partition_key": True,
            },
            {"field_name": "agent_id", "datatype": self._datatype.VARCHAR, "max_length": 256},
            {
                "field_name": "session_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 256,
            },
            {"field_name": "turn_number", "datatype": self._datatype.INT64},
            {"field_name": "sentence_index", "datatype": self._datatype.INT64},
            {"field_name": "role", "datatype": self._datatype.VARCHAR, "max_length": 32},
            {
                "field_name": "content_hash",
                "datatype": self._datatype.VARCHAR,
                "max_length": 128,
            },
            {"field_name": "mentions", "datatype": self._datatype.INT64},
            {"field_name": "is_active", "datatype": self._datatype.BOOL},
            {
                "field_name": "created_at",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "next_sentence_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "record_type",
                "datatype": self._datatype.VARCHAR,
                "max_length": 32,
            },
            {
                "field_name": "started_at",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "ended_at",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            self._metadata_field(),
        ]

    def _episode_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "field_name": "id",
                "datatype": self._datatype.VARCHAR,
                "is_primary": True,
                "max_length": 64,
            },
            {"field_name": "text", "datatype": self._datatype.VARCHAR, "max_length": 8192},
            {
                "field_name": "embedding",
                "datatype": self._datatype.FLOAT_VECTOR,
                "dim": self.embedding_dim,
            },
            {
                "field_name": "user_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 256,
                "is_partition_key": True,
            },
            {"field_name": "agent_id", "datatype": self._datatype.VARCHAR, "max_length": 256},
            {
                "field_name": "session_id",
                "datatype": self._datatype.VARCHAR,
                "max_length": 256,
            },
            {"field_name": "is_active", "datatype": self._datatype.BOOL},
            {
                "field_name": "created_at",
                "datatype": self._datatype.VARCHAR,
                "max_length": 64,
            },
            {
                "field_name": "fact_ids",
                "datatype": self._datatype.VARCHAR,
                "max_length": 8192,
            },
            {
                "field_name": "record_type",
                "datatype": self._datatype.VARCHAR,
                "max_length": 32,
            },
            self._metadata_field(),
        ]

    # ── Sentences ──────────────────────────────────────────────────────────────

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        from vektori.ingestion.hasher import generate_content_hash

        if not sentences:
            return 0

        if not embeddings or len(sentences) != len(embeddings):
            raise ValueError(
                f"sentences/embeddings length mismatch: "
                f"{len(sentences)} sentences vs {len(embeddings)} embeddings"
            )

        rows: list[dict[str, Any]] = []

        for sent, emb in zip(sentences, embeddings):
            if self.embedding_dim and len(emb) != self.embedding_dim:
                raise ValueError(
                    f"Embedding has {len(emb)} dimensions, expected {self.embedding_dim}"
                )

            content_hash = generate_content_hash(
                sent["session_id"],
                f"{sent['turn_number']}_{sent['sentence_index']}",
                sent["text"],
            )

            existing = await self._get_one(
                self._sentences_col,
                (
                    f'record_type == "{_SENTENCE_RECORD}" and '
                    f'content_hash == "{_escape(content_hash)}"'
                ),
                self._sentence_output_fields(),
            )

            if existing:
                sid = existing.get("id")
                mentions = int(existing.get("mentions") or 1) + 1
            else:
                sid = sent["id"]
                mentions = 1

            rows.append(
                {
                    "id": str(sid),
                    "text": sent["text"],
                    "embedding": emb,
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "session_id": sent["session_id"],
                    "turn_number": int(sent["turn_number"]),
                    "sentence_index": int(sent["sentence_index"]),
                    "role": sent.get("role", "user"),
                    "content_hash": content_hash,
                    "mentions": mentions,
                    "is_active": True,
                    "created_at": existing.get("created_at") if existing else _utcnow_iso(),
                    "next_sentence_id": existing.get("next_sentence_id", "") if existing else "",
                    "record_type": _SENTENCE_RECORD,
                    "metadata": (
                        existing.get("metadata", self._json_value({}))
                        if existing
                        else self._json_value({})
                    ),
                    "started_at": existing.get("started_at", "") if existing else "",
                    "ended_at": existing.get("ended_at", "") if existing else "",
                }
            )

        await self._upsert_rows(self._sentences_col, rows)
        return len(rows)

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        parts = [
            f'record_type == "{_SENTENCE_RECORD}"',
            f'user_id == "{_escape(user_id)}"',
            "is_active == true",
        ]
        if agent_id is not None:
            parts.append(f'agent_id == "{_escape(agent_id)}"')

        hits = await self._search(
            collection_name=self._sentences_col,
            embedding=embedding,
            filter_expr=" and ".join(parts),
            limit=limit,
            output_fields=self._sentence_output_fields(),
        )
        return [self._to_sentence(row) for row in hits]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        if not quotes:
            return []

        cleaned_quotes = [q.strip() for q in quotes if isinstance(q, str) and q.strip()]
        if not cleaned_quotes:
            return []

        try:
            quote_embeddings = await self._embed_quotes(cleaned_quotes)
        except Exception as e:
            logger.warning("Quote embedding failed in find_sentences_by_similarity: %s", e)
            return []

        if not quote_embeddings:
            return []

        filter_expr = (
            f'record_type == "{_SENTENCE_RECORD}" and '
            f'session_id == "{_escape(session_id)}" and '
            "is_active == true"
        )

        seen: set[str] = set()
        ordered_ids: list[str] = []

        for emb in quote_embeddings:
            try:
                hits = await self._search(
                    collection_name=self._sentences_col,
                    embedding=emb,
                    filter_expr=filter_expr,
                    limit=max(10, len(cleaned_quotes) * 5),
                    output_fields=self._sentence_output_fields(),
                )
            except Exception as e:
                logger.warning("Sentence similarity search failed for quote embedding: %s", e)
                continue

            for row in hits:
                similarity = 1.0 - float(row.get("distance", 1.0))
                if similarity < threshold:
                    continue

                sentence_id = str(row.get("id") or "")
                sentence_text = str(row.get("text") or row.get("sentence") or "")
                dedup_key = sentence_id or sentence_text
                if not dedup_key or dedup_key in seen:
                    continue

                seen.add(dedup_key)
                ordered_ids.append(sentence_id or sentence_text)

        return ordered_ids

    async def search_sentences_in_session(
        self,
        embedding: list[float],
        session_id: str,
        limit: int = 3,
        threshold: float = 0.75,
    ) -> list[str]:
        filter_expr = (
            f'record_type == "{_SENTENCE_RECORD}" and '
            f'session_id == "{_escape(session_id)}" and '
            "is_active == true"
        )
        hits = await self._search(
            collection_name=self._sentences_col,
            embedding=embedding,
            filter_expr=filter_expr,
            limit=max(limit * 4, limit),
            output_fields=self._sentence_output_fields(),
        )
        ids: list[str] = []
        for row in hits:
            similarity = 1.0 - float(row.get("distance", 1.0))
            if similarity >= threshold:
                rid = str(row.get("id"))
                if rid and rid not in ids:
                    ids.append(rid)
            if len(ids) >= limit:
                break
        return ids

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        rows = await self._query_all(
            collection_name=self._sentences_col,
            filter_expr=(
                f'record_type == "{_SENTENCE_RECORD}" and session_id == "{_escape(session_id)}"'
            ),
            output_fields=self._sentence_output_fields(),
            batch_size=200,
        )

        needle = quote.lower()
        for row in rows:
            if needle in str(row.get("text") or "").lower():
                return self._to_sentence(row)
        return None

    # ── Facts ──────────────────────────────────────────────────────────────────

    async def insert_fact(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_time: datetime | None = None,
    ) -> str:
        if self.embedding_dim and len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding has {len(embedding)} dimensions, expected {self.embedding_dim}"
            )

        fact_id = str(uuid.uuid4())
        row = {
            "id": fact_id,
            "text": text,
            "embedding": embedding,
            "user_id": user_id,
            "agent_id": agent_id or "",
            "session_id": session_id or "",
            "subject": subject or "",
            "confidence": float(confidence),
            "mentions": 1,
            "is_active": True,
            "superseded_by": superseded_by_target or "",
            "event_time": event_time.isoformat() if event_time else "",
            "created_at": _utcnow_iso(),
            "source_sentence_ids": "",
            "record_type": _FACT_RECORD,
            "metadata": self._json_value(metadata),
        }
        await self._upsert_rows(self._facts_col, [row])
        return fact_id

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        limit: int = 10,
        active_only: bool = True,
        before_date: datetime | None = None,
        after_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        parts = [
            f'record_type == "{_FACT_RECORD}"',
            f'user_id == "{_escape(user_id)}"',
        ]
        if active_only:
            parts.append("is_active == true")
        if agent_id is not None:
            parts.append(f'agent_id == "{_escape(agent_id)}"')
        if session_id is not None:
            parts.append(f'session_id == "{_escape(session_id)}"')
        if subject is not None:
            parts.append(f'subject == "{_escape(subject)}"')
        if before_date is not None:
            parts.append(f'event_time <= "{_escape(before_date.isoformat())}"')
        if after_date is not None:
            parts.append(f'event_time >= "{_escape(after_date.isoformat())}"')

        hits = await self._search(
            collection_name=self._facts_col,
            embedding=embedding,
            filter_expr=" and ".join(parts),
            limit=limit,
            output_fields=self._fact_output_fields(),
        )
        return [self._to_fact(row) for row in hits]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        parts = [
            f'record_type == "{_FACT_RECORD}"',
            f'user_id == "{_escape(user_id)}"',
            "is_active == true",
        ]
        if agent_id is not None:
            parts.append(f'agent_id == "{_escape(agent_id)}"')

        rows = await self._query_all(
            collection_name=self._facts_col,
            filter_expr=" and ".join(parts),
            output_fields=self._fact_output_fields(),
            batch_size=max(200, limit + offset),
        )
        normalized = [self._to_fact(row) for row in rows]
        normalized.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return normalized[offset : offset + limit]

    async def deactivate_fact(
        self,
        fact_id: str,
        superseded_by: str | None = None,
    ) -> None:
        fact = await self._get_one(
            self._facts_col,
            f'id == "{_escape(fact_id)}" and record_type == "{_FACT_RECORD}"',
            self._fact_output_fields(),
        )
        if not fact:
            return
        fact["is_active"] = False
        if superseded_by is not None:
            fact["superseded_by"] = superseded_by
        await self._upsert_rows(self._facts_col, [fact])

    async def increment_fact_mentions(self, fact_id: str) -> None:
        fact = await self._get_one(
            self._facts_col,
            f'id == "{_escape(fact_id)}" and record_type == "{_FACT_RECORD}"',
            self._fact_output_fields(),
        )
        if not fact:
            return
        fact["mentions"] = int(fact.get("mentions") or 1) + 1
        await self._upsert_rows(self._facts_col, [fact])

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        parts = [
            f'record_type == "{_FACT_RECORD}"',
            f'user_id == "{_escape(user_id)}"',
            "is_active == true",
            f'text == "{_escape(text)}"',
        ]
        if agent_id is not None:
            parts.append(f'agent_id == "{_escape(agent_id)}"')

        row = await self._get_one(
            self._facts_col,
            " and ".join(parts),
            self._fact_output_fields(),
        )
        return self._to_fact(row) if row else None

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        chain: list[dict[str, Any]] = []
        visited: set[str] = set()
        current = fact_id

        while current and current not in visited and len(chain) < 50:
            visited.add(current)
            row = await self._get_one(
                self._facts_col,
                f'id == "{_escape(current)}" and record_type == "{_FACT_RECORD}"',
                self._fact_output_fields(),
            )
            if not row:
                break
            fact = self._to_fact(row)
            chain.append(fact)
            current = str(row.get("superseded_by") or "")

        return chain

    # ── Edges ──────────────────────────────────────────────────────────────────

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        if not edges:
            return 0

        for edge in edges:
            if edge.get("edge_type") != "NEXT":
                continue
            source_id = str(edge["source_id"])
            target_id = str(edge["target_id"])
            row = await self._get_one(
                self._sentences_col,
                (f'id == "{_escape(source_id)}" and record_type == "{_SENTENCE_RECORD}"'),
                self._sentence_output_fields(),
            )
            if not row:
                continue
            row["next_sentence_id"] = target_id
            await self._upsert_rows(self._sentences_col, [row])

        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []

        seeds: list[dict[str, Any]] = []
        for sid in sentence_ids:
            row = await self._get_one(
                self._sentences_col,
                f'id == "{_escape(sid)}" and record_type == "{_SENTENCE_RECORD}"',
                self._sentence_output_fields(),
            )
            if row:
                seeds.append(row)

        expanded: dict[str, dict[str, Any]] = {}

        for seed in seeds:
            session_id = str(seed.get("session_id") or "")
            sentence_index = int(seed.get("sentence_index") or 0)

            rows = await self._query_all(
                collection_name=self._sentences_col,
                filter_expr=(
                    f'record_type == "{_SENTENCE_RECORD}" and '
                    f'session_id == "{_escape(session_id)}" and '
                    f"sentence_index >= {sentence_index - window} and "
                    f"sentence_index <= {sentence_index + window} and "
                    "is_active == true"
                ),
                output_fields=self._sentence_output_fields(),
                batch_size=200,
            )
            for row in rows:
                rid = str(row.get("id"))
                expanded[rid] = self._to_sentence(row)

        ordered = list(expanded.values())
        ordered.sort(
            key=lambda s: (
                s.get("session_id") or "",
                s.get("turn_number") or 0,
                s.get("sentence_index") or 0,
            )
        )
        return ordered

    # ── Join-table equivalents ────────────────────────────────────────────────

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        await self.insert_fact_sources([(fact_id, sentence_id)])

    async def insert_fact_sources(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return

        by_fact: dict[str, list[str]] = {}
        for fid, sid in pairs:
            by_fact.setdefault(fid, []).append(sid)

        max_retries = 5
        for fid, sentence_ids in by_fact.items():
            for attempt in range(max_retries):
                row = await self._get_one(
                    self._facts_col,
                    f'id == "{_escape(fid)}" and record_type == "{_FACT_RECORD}"',
                    self._fact_output_fields(),
                )
                if not row:
                    break
                previous_csv = row.get("source_sentence_ids", "")
                merged = _ids_to_csv(_csv_to_ids(previous_csv) + sentence_ids)
                if merged == previous_csv:
                    break  # nothing to update
                # CAS check: verify the row still has the value we read
                cas_check = await self._get_one(
                    self._facts_col,
                    (
                        f'id == "{_escape(fid)}" and record_type == "{_FACT_RECORD}" and '
                        f'source_sentence_ids == "{_escape(str(previous_csv))}"'
                    ),
                    ["id"],
                )
                if cas_check:
                    row["source_sentence_ids"] = merged
                    await self._upsert_rows(self._facts_col, [row])
                    break
                # Row was modified concurrently; retry
                if attempt == max_retries - 1:
                    logger.warning(
                        "CAS retry exhausted for fact %s; writing last merged value", fid
                    )
                    row["source_sentence_ids"] = merged
                    await self._upsert_rows(self._facts_col, [row])

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []

        seen: set[str] = set()
        result: list[str] = []

        for fid in fact_ids:
            row = await self._get_one(
                self._facts_col,
                f'id == "{_escape(fid)}" and record_type == "{_FACT_RECORD}"',
                ["source_sentence_ids", "id"],
            )
            if not row:
                continue
            for sid in _csv_to_ids(row.get("source_sentence_ids")):
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)

        return result

    async def get_sentences_by_ids(self, sentence_ids: list[str]) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []

        rows: list[dict[str, Any]] = []
        for sid in sentence_ids:
            row = await self._get_one(
                self._sentences_col,
                (
                    f'id == "{_escape(sid)}" and '
                    f'record_type == "{_SENTENCE_RECORD}" and '
                    "is_active == true"
                ),
                self._sentence_output_fields(),
            )
            if row:
                rows.append(self._to_sentence(row))

        rows.sort(
            key=lambda s: (
                s.get("session_id") or "",
                s.get("turn_number") or 0,
                s.get("sentence_index") or 0,
            )
        )
        return rows

    # ── Episodes ──────────────────────────────────────────────────────────────

    async def insert_episode(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        if self.embedding_dim and len(embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding has {len(embedding)} dimensions, expected {self.embedding_dim}"
            )

        agent_scope = agent_id or ""
        episode_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{agent_scope}::{text}"))
        existing = await self._get_one(
            self._episodes_col,
            (
                f'id == "{_escape(episode_id)}" and '
                f'record_type == "{_EPISODE_RECORD}" and '
                f'agent_id == "{_escape(agent_scope)}"'
            ),
            self._episode_output_fields(),
        )
        if existing:
            return episode_id

        row = {
            "id": episode_id,
            "text": text,
            "embedding": embedding,
            "user_id": user_id,
            "agent_id": agent_id or "",
            "session_id": session_id or "",
            "is_active": True,
            "created_at": _utcnow_iso(),
            "fact_ids": "",
            "record_type": _EPISODE_RECORD,
            "metadata": self._json_value({}),
        }
        await self._upsert_rows(self._episodes_col, [row])
        return episode_id

    async def insert_episode_fact(self, episode_id: str, fact_id: str) -> None:
        max_retries = 5
        for attempt in range(max_retries):
            row = await self._get_one(
                self._episodes_col,
                f'id == "{_escape(episode_id)}" and record_type == "{_EPISODE_RECORD}"',
                self._episode_output_fields(),
            )
            if not row:
                return
            previous_csv = row.get("fact_ids", "")
            merged = _ids_to_csv(_csv_to_ids(previous_csv) + [fact_id])
            if merged == previous_csv:
                return  # nothing to update
            # CAS check: verify the row still has the value we read
            cas_check = await self._get_one(
                self._episodes_col,
                (
                    f'id == "{_escape(episode_id)}" and record_type == "{_EPISODE_RECORD}" and '
                    f'fact_ids == "{_escape(str(previous_csv))}"'
                ),
                ["id"],
            )
            if cas_check:
                row["fact_ids"] = merged
                await self._upsert_rows(self._episodes_col, [row])
                return
            # Row was modified concurrently; retry
            if attempt == max_retries - 1:
                logger.warning(
                    "CAS retry exhausted for episode %s; writing last merged value",
                    episode_id,
                )
                row["fact_ids"] = merged
                await self._upsert_rows(self._episodes_col, [row])

    async def get_episodes_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []

        fact_set = set(fact_ids)
        rows = await self._query_all(
            collection_name=self._episodes_col,
            filter_expr=f'record_type == "{_EPISODE_RECORD}" and is_active == true',
            output_fields=self._episode_output_fields(),
            batch_size=200,
        )

        episodes: list[dict[str, Any]] = []
        seen: set[str] = set()

        for row in rows:
            linked = set(_csv_to_ids(row.get("fact_ids")))
            if linked.intersection(fact_set):
                eid = str(row.get("id"))
                if eid in seen:
                    continue
                seen.add(eid)
                episodes.append(self._to_episode(row))

        return episodes

    async def search_episodes(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        parts = [
            f'record_type == "{_EPISODE_RECORD}"',
            f'user_id == "{_escape(user_id)}"',
            "is_active == true",
        ]
        if agent_id is not None:
            parts.append(f'agent_id == "{_escape(agent_id)}"')

        hits = await self._search(
            collection_name=self._episodes_col,
            embedding=embedding,
            filter_expr=" and ".join(parts),
            limit=limit,
            output_fields=self._episode_output_fields(),
        )

        return [self._to_episode(row) for row in hits]

    # ── Sessions (stored in sentences collection as typed records) ────────────

    def _session_marker_id(self, session_id: str, user_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_OID, f"session::{user_id}::{session_id}"))

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        marker_id = self._session_marker_id(session_id, user_id)
        existing = await self._get_one(
            self._sentences_col,
            (
                f'id == "{_escape(marker_id)}" and record_type == "{_SESSION_RECORD}" and '
                f'user_id == "{_escape(user_id)}"'
            ),
            self._sentence_output_fields(),
        )

        started_at_value = (
            started_at.isoformat()
            if started_at is not None
            else (existing.get("started_at") if existing else _utcnow_iso())
        )

        if existing and metadata is None:
            metadata_value = existing.get("metadata")
            if metadata_value is None:
                metadata_value = self._json_value({})
        else:
            metadata_value = self._json_value(metadata)

        row = {
            "id": marker_id,
            "text": "",
            "embedding": [0.0] * self.embedding_dim,
            "user_id": user_id,
            "agent_id": agent_id or "",
            "session_id": session_id,
            "turn_number": -1,
            "sentence_index": -1,
            "role": "session",
            "content_hash": marker_id,
            "mentions": 1,
            "is_active": True,
            "created_at": existing.get("created_at") if existing else _utcnow_iso(),
            "next_sentence_id": "",
            "record_type": _SESSION_RECORD,
            "metadata": metadata_value,
            "started_at": started_at_value,
            "ended_at": existing.get("ended_at", "") if existing else "",
        }

        await self._upsert_rows(self._sentences_col, [row])

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        marker = await self._get_one(
            self._sentences_col,
            (
                f'record_type == "{_SESSION_RECORD}" and '
                f'session_id == "{_escape(session_id)}" and '
                f'user_id == "{_escape(user_id)}"'
            ),
            self._sentence_output_fields(),
        )
        if not marker:
            return None

        sentence_rows = await self._query_all(
            collection_name=self._sentences_col,
            filter_expr=(
                f'record_type == "{_SENTENCE_RECORD}" and '
                f'session_id == "{_escape(session_id)}" and '
                f'user_id == "{_escape(user_id)}" and '
                "is_active == true"
            ),
            output_fields=self._sentence_output_fields(),
            batch_size=200,
        )

        sentences = [self._to_sentence(r) for r in sentence_rows]
        sentences.sort(key=lambda s: (s.get("turn_number") or 0, s.get("sentence_index") or 0))

        return {
            "id": session_id,
            "user_id": marker.get("user_id"),
            "agent_id": marker.get("agent_id") or None,
            "started_at": marker.get("started_at") or None,
            "ended_at": marker.get("ended_at") or None,
            "metadata": _parse_metadata(marker.get("metadata")),
            "sentences": sentences,
        }

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        parts = [
            f'record_type == "{_SESSION_RECORD}"',
            f'user_id == "{_escape(user_id)}"',
        ]
        if agent_id is not None:
            parts.append(f'agent_id == "{_escape(agent_id)}"')

        rows = await self._query_all(
            collection_name=self._sentences_col,
            filter_expr=" and ".join(parts),
            output_fields=["id"],
            batch_size=200,
        )
        return len(rows)

    # ── GDPR ───────────────────────────────────────────────────────────────────

    async def delete_user(self, user_id: str) -> int:
        total = 0
        filter_expr = f'user_id == "{_escape(user_id)}"'

        for collection_name in (self._facts_col, self._sentences_col, self._episodes_col):
            rows = await self._query_all(
                collection_name=collection_name,
                filter_expr=filter_expr,
                output_fields=["id"],
                batch_size=500,
            )
            total += len(rows)
            if rows:
                await self._delete(collection_name, filter_expr)

        logger.info("Deleted %d Milvus rows for user %s", total, user_id)
        return total

    # ── Row converters ─────────────────────────────────────────────────────────

    def _to_sentence(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id")),
            "text": row.get("text"),
            "session_id": row.get("session_id"),
            "turn_number": int(row.get("turn_number") or 0),
            "sentence_index": int(row.get("sentence_index") or 0),
            "role": row.get("role") or "user",
            "mentions": int(row.get("mentions") or 1),
            "is_active": bool(row.get("is_active", True)),
            "created_at": row.get("created_at"),
            "distance": float(row.get("distance", 1.0)),
        }

    def _to_fact(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id")),
            "text": row.get("text"),
            "confidence": float(row.get("confidence") or 1.0),
            "mentions": int(row.get("mentions") or 1),
            "session_id": row.get("session_id") or None,
            "subject": row.get("subject") or None,
            "is_active": bool(row.get("is_active", True)),
            "superseded_by": row.get("superseded_by") or None,
            "metadata": _parse_metadata(row.get("metadata")),
            "event_time": row.get("event_time") or None,
            "created_at": row.get("created_at") or None,
            "distance": float(row.get("distance", 1.0)),
        }

    def _to_episode(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(row.get("id")),
            "text": row.get("text"),
            "session_id": row.get("session_id") or None,
            "created_at": row.get("created_at") or None,
            "distance": float(row.get("distance", 1.0)),
        }
