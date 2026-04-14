"""Vektori CLI — add, search, and manage memory from the terminal."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

import typer

app = typer.Typer(help="Vektori memory engine — self-hosted, zero-config.", no_args_is_help=True)

_CONFIG_PATH = Path.home() / ".vektori" / "config.json"

_FALLBACK_EXTRACTION = "openai:gpt-4o-mini"
_FALLBACK_EMBEDDING = "sentence-transformers:all-MiniLM-L6-v2"
_FALLBACK_CHAT = "openai:gpt-4o-mini"

_EXTRACTION_HELP = (
    "LLM for fact extraction. e.g. 'litellm:groq/llama-3.3-70b-versatile', "
    "'litellm:ollama/llama3' [env: VEKTORI_EXTRACTION_MODEL]"
)
_EMBEDDING_HELP = (
    "Embedding model. e.g. 'sentence-transformers:all-MiniLM-L6-v2' (local), "
    "'openai:text-embedding-3-small' [env: VEKTORI_EMBEDDING_MODEL]"
)
_BACKEND_HELP = (
    "Storage backend: sqlite (default), postgres, memory, neo4j, qdrant, milvus. "
    "[env: VEKTORI_STORAGE_BACKEND]"
)
_DATABASE_URL_HELP = (
    "Connection URL for the backend. "
    "Postgres: postgresql://user:pw@host/db  "
    "Neo4j: bolt://host:7687  "
    "Qdrant: http://host:6333  "
    "Milvus: http://host:19530 or https://<cluster-endpoint>  "
    "[env: VEKTORI_DATABASE_URL]"
)
_QDRANT_API_KEY_HELP = "Qdrant Cloud API key. [env: QDRANT_API_KEY]"
_CHAT_HELP = (
    "Chat model for the native harness. e.g. 'openai:gpt-4o-mini', "
    "'litellm:groq/llama-3.3-70b-versatile' [env: VEKTORI_CHAT_MODEL]"
)


# ---------------------------------------------------------------------------
# Persistent config helpers
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        try:
            return json.loads(_CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_config(data: dict) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(data, indent=2))


def _default_extraction() -> str:
    cfg = _load_config()
    return (
        os.environ.get("VEKTORI_EXTRACTION_MODEL")
        or cfg.get("extraction_model")
        or _FALLBACK_EXTRACTION
    )


def _default_embedding() -> str:
    cfg = _load_config()
    return (
        os.environ.get("VEKTORI_EMBEDDING_MODEL")
        or cfg.get("embedding_model")
        or _FALLBACK_EMBEDDING
    )


def _default_chat_model() -> str:
    cfg = _load_config()
    return os.environ.get("VEKTORI_CHAT_MODEL") or cfg.get("chat_model") or _FALLBACK_CHAT


def _warn_openai(model: str, var: str) -> None:
    if model.startswith("openai:") and not os.environ.get("OPENAI_API_KEY"):
        typer.echo(
            f"[warn] {var}={model!r} requires OPENAI_API_KEY.\n"
            "       Alternatives:\n"
            "         litellm:groq/llama-3.3-70b-versatile   (GROQ_API_KEY)\n"
            "         litellm:ollama/llama3                   (local, no key)\n"
            "         sentence-transformers:all-MiniLM-L6-v2  (local, embeddings only)\n",
            err=True,
        )


def _silence_litellm() -> None:
    """Suppress LiteLLM's verbose stdout/stderr output."""
    import logging

    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("litellm").setLevel(logging.CRITICAL)
    # Suppress asyncio SSL transport errors on event loop close (cosmetic noise)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    try:
        import litellm

        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    except Exception:
        pass


def _default_storage_backend() -> str:
    cfg = _load_config()
    return os.environ.get("VEKTORI_STORAGE_BACKEND") or cfg.get("storage_backend") or "sqlite"


def _default_database_url() -> str | None:
    cfg = _load_config()
    return os.environ.get("VEKTORI_DATABASE_URL") or cfg.get("database_url")


def _client(
    extraction_model: str,
    embedding_model: str,
    sync_extraction: bool = True,
    storage_backend: str | None = None,
    database_url: str | None = None,
    qdrant_api_key: str | None = None,
):
    from vektori.client import Vektori
    from vektori.config import VektoriConfig

    _silence_litellm()
    cfg = VektoriConfig(
        extraction_model=extraction_model,
        embedding_model=embedding_model,
        async_extraction=not sync_extraction,
        enable_retrieval_gate=False,
        storage_backend=storage_backend or _default_storage_backend(),
        database_url=database_url or _default_database_url(),
        qdrant_api_key=qdrant_api_key or os.environ.get("QDRANT_API_KEY"),
        milvus_token=os.environ.get("MILVUS_TOKEN") or os.environ.get("MILVUS_API_KEY"),
    )
    return Vektori(config=cfg)


def _out(data: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@app.command()
def config(
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", help=_EMBEDDING_HELP
    ),
    chat_model: str | None = typer.Option(None, "--chat-model", help=_CHAT_HELP),
    storage_backend: str | None = typer.Option(None, "--storage-backend", help=_BACKEND_HELP),
    database_url: str | None = typer.Option(None, "--database-url", help=_DATABASE_URL_HELP),
    show: bool = typer.Option(False, "--show", help="Print current config."),
    reset: bool = typer.Option(False, "--reset", help="Reset to defaults."),
) -> None:
    """Set default models and storage so you don't have to pass flags every time.

    \b
        Examples:
            vektori config --extraction-model "litellm:groq/llama-3.3-70b-versatile"
            vektori config --embedding-model "sentence-transformers:all-MiniLM-L6-v2"
            vektori config --storage-backend qdrant --database-url http://localhost:6333
            vektori config --storage-backend neo4j --database-url "bolt://localhost:7687"
            vektori config --storage-backend milvus --database-url http://localhost:19530
            vektori config --show
    """
    if reset:
        if _CONFIG_PATH.exists():
            _CONFIG_PATH.unlink()
        typer.echo("Config reset to defaults.")
        return

    cfg = _load_config()

    if extraction_model:
        cfg["extraction_model"] = extraction_model
    if embedding_model:
        cfg["embedding_model"] = embedding_model
    if chat_model:
        cfg["chat_model"] = chat_model
    if storage_backend:
        cfg["storage_backend"] = storage_backend
    if database_url:
        cfg["database_url"] = database_url

    if extraction_model or embedding_model or chat_model or storage_backend or database_url:
        _save_config(cfg)
        typer.echo(f"Saved to {_CONFIG_PATH}")

    if show or not (
        extraction_model or embedding_model or chat_model or storage_backend or database_url or reset
    ):
        typer.echo(f"Config file     : {_CONFIG_PATH}")
        typer.echo(f"extraction      : {_default_extraction()}")
        typer.echo(f"embedding       : {_default_embedding()}")
        typer.echo(f"chat            : {_default_chat_model()}")
        typer.echo(f"storage_backend : {_default_storage_backend()}")
        typer.echo(f"database_url    : {_default_database_url() or '(default)'}")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
) -> None:
    """Initialise storage backend (SQLite by default)."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    _warn_openai(em, "--extraction-model")
    _warn_openai(eb, "--embedding-model")

    async def _run() -> None:
        v = _client(
            em,
            eb,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        await v._ensure_initialized()
        await v.close()

    asyncio.run(_run())
    typer.echo(f"Vektori initialised (extraction={em}, embedding={eb})")


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@app.command()
def add(
    text: str = typer.Argument(..., help="Text to store as a memory."),
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    session_id: str | None = typer.Option(
        None, "--session-id", "-s", help="Session ID (auto-generated if omitted)."
    ),
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    no_extraction: bool = typer.Option(
        False,
        "--no-extraction",
        help="Skip LLM fact extraction. Stores sentence only — useful for testing without an API key.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add a memory. Extraction runs synchronously so facts are ready immediately."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    if not no_extraction:
        _warn_openai(em, "--extraction-model")
    _warn_openai(eb, "--embedding-model")
    sid = session_id or str(uuid.uuid4())
    messages = [{"role": "user", "content": text}]

    async def _run() -> dict:
        v = _client(
            em,
            eb,
            sync_extraction=not no_extraction,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            result = await v.add(messages, session_id=sid, user_id=user_id)
        finally:
            await v.close()
        return result

    try:
        result = asyncio.run(_run())
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    if as_json:
        _out(result, True)
    else:
        typer.echo(
            f"Stored — {result['sentences_stored']} sentence(s), extraction: {result['extraction']}"
        )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Max results to return."),
    depth: str = typer.Option("l1", "--depth", "-d", help="Search depth: l0, l1, or l2."),
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    expand: bool = typer.Option(
        False, "--expand", help="Use LLM to generate query variants before searching."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Search memories with a natural language query."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    _warn_openai(eb, "--embedding-model")

    async def _run() -> dict:
        v = _client(
            em,
            eb,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            result = await v.search(query, user_id=user_id, top_k=top_k, depth=depth, expand=expand)
        finally:
            await v.close()
        return result

    try:
        result = asyncio.run(_run())
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    if as_json:
        _out(result, True)
        return

    facts = result.get("facts", [])
    sentences = result.get("sentences", [])

    if facts:
        for i, fact in enumerate(facts, 1):
            score = fact.get("score")
            score_str = f"  [{score:.3f}]" if score is not None else ""
            typer.echo(f"{i}.{score_str} {fact['text']}")
    elif sentences:
        typer.echo(
            "(showing raw sentences — run with an extraction model to get structured facts)\n"
        )
        for i, sent in enumerate(sentences, 1):
            dist = sent.get("distance")
            score_str = f"  [{1 - dist:.3f}]" if dist is not None else ""
            typer.echo(f"{i}.{score_str} {sent['text']}")
    else:
        typer.echo("No memories found.")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_memories(
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all active memories (extracted facts) for a user."""
    eb = embedding_model or _default_embedding()

    async def _run() -> list:
        v = _client(
            _default_extraction(),
            eb,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            facts = await v.get_facts(user_id=user_id)
        finally:
            await v.close()
        return facts

    facts = asyncio.run(_run())
    if as_json:
        _out(facts, True)
        return

    if not facts:
        typer.echo("No memories found.")
        return
    for i, fact in enumerate(facts, 1):
        created = fact.get("created_at", "")[:10]
        prefix = f"[{created}] " if created else ""
        typer.echo(f"{i}. {prefix}{fact['text']}")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@app.command()
def delete(
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Delete all memories for a user."""
    if not yes:
        typer.confirm(f"Delete ALL memories for user '{user_id}'?", abort=True)

    async def _run() -> int:
        v = _client(
            _default_extraction(),
            _default_embedding(),
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            n = await v.delete_user(user_id=user_id)
        finally:
            await v.close()
        return n

    n = asyncio.run(_run())
    if as_json:
        _out({"deleted": n, "user_id": user_id}, True)
    else:
        typer.echo(f"Deleted {n} record(s) for user '{user_id}'.")


# ---------------------------------------------------------------------------
# remember (alias for add)
# ---------------------------------------------------------------------------


@app.command()
def remember(
    text: str = typer.Argument(..., help="Text to store as a memory."),
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    session_id: str | None = typer.Option(
        None, "--session-id", "-s", help="Session ID (auto-generated if omitted)."
    ),
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Remember something. Alias for `add` — reads naturally in agent prompts."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    _warn_openai(em, "--extraction-model")
    sid = session_id or str(uuid.uuid4())
    messages = [{"role": "user", "content": text}]

    async def _run() -> dict:
        v = _client(
            em,
            eb,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            result = await v.add(messages, session_id=sid, user_id=user_id)
        finally:
            await v.close()
        return result

    try:
        result = asyncio.run(_run())
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    if as_json:
        _out(result, True)
    else:
        typer.echo(
            f"Stored — {result['sentences_stored']} sentence(s), extraction: {result['extraction']}"
        )


# ---------------------------------------------------------------------------
# recall (alias for search)
# ---------------------------------------------------------------------------


@app.command()
def recall(
    query: str = typer.Argument(..., help="What to recall."),
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Max results to return."),
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Recall memories. Alias for `search` — reads naturally in agent prompts."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()

    async def _run() -> dict:
        v = _client(
            em,
            eb,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            result = await v.search(query, user_id=user_id, top_k=top_k, depth="l1")
        finally:
            await v.close()
        return result

    try:
        result = asyncio.run(_run())
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    facts = result.get("facts", [])
    sentences = result.get("sentences", [])

    if as_json:
        # Richer output for agents — include session_id, created_at, source
        agent_facts = [
            {
                "text": f["text"],
                "score": f.get("score"),
                "session_id": f.get("session_id"),
                "created_at": f.get("created_at"),
                "source": f.get("source"),
            }
            for f in facts
        ]
        _out({"facts": agent_facts, "memory_found": len(agent_facts) > 0}, True)
        return

    if facts:
        for i, fact in enumerate(facts, 1):
            score = fact.get("score")
            score_str = f"  [{score:.3f}]" if score is not None else ""
            typer.echo(f"{i}.{score_str} {fact['text']}")
    elif sentences:
        for i, sent in enumerate(sentences, 1):
            dist = sent.get("distance")
            score_str = f"  [{1 - dist:.3f}]" if dist is not None else ""
            typer.echo(f"{i}.{score_str} {sent['text']}")
    else:
        typer.echo("No memories found.")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@app.command()
def stats(
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Show memory stats for a user."""

    async def _run() -> dict:
        v = _client(
            _default_extraction(),
            _default_embedding(),
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        await v._ensure_initialized()
        try:
            facts = await v.get_facts(user_id=user_id)
            session_count = await v.db.count_sessions(user_id=user_id)
        finally:
            await v.close()

        dates = [f.get("created_at", "") for f in facts if f.get("created_at")]
        oldest = min(dates)[:10] if dates else "—"
        newest = max(dates)[:10] if dates else "—"

        return {
            "user_id": user_id,
            "facts": len(facts),
            "sessions": session_count,
            "oldest": oldest,
            "newest": newest,
        }

    try:
        data = asyncio.run(_run())
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    if as_json:
        _out(data, True)
    else:
        typer.echo(
            f"facts: {data['facts']}  sessions: {data['sessions']}  "
            f"oldest: {data['oldest']}  newest: {data['newest']}"
        )


# ---------------------------------------------------------------------------
# inject  — detect and ingest Claude Code / Codex sessions into memory
# ---------------------------------------------------------------------------


def _slug_to_path(slug: str) -> str:
    """Convert Claude Code project slug back to a readable path."""
    return slug.replace("-", "/", 1).replace("-", " ").strip()


def _discover_claude_sessions() -> list[dict]:
    """Find all Claude Code session JSONL files."""
    base = Path.home() / ".claude" / "projects"
    if not base.exists():
        return []
    sessions = []
    for project_dir in sorted(base.iterdir()):
        if not project_dir.is_dir():
            continue
        project_slug = project_dir.name
        for jsonl in sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            sessions.append({
                "source": "claude-code",
                "project": project_slug,
                "session_id": jsonl.stem,
                "path": jsonl,
                "mtime": jsonl.stat().st_mtime,
            })
    return sessions


def _discover_codex_sessions() -> list[dict]:
    """Find all Codex session JSONL files."""
    base = Path.home() / ".codex" / "sessions"
    if not base.exists():
        return []
    sessions = []
    for jsonl in sorted(base.rglob("rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        sessions.append({
            "source": "codex",
            "project": None,  # resolved from session_meta
            "session_id": jsonl.stem,
            "path": jsonl,
            "mtime": jsonl.stat().st_mtime,
        })
    return sessions


def _parse_claude_session(path: Path) -> tuple[str | None, list[dict]]:
    """Parse a Claude Code JSONL transcript into messages. Returns (project_cwd, messages)."""
    messages = []
    cwd = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                # Extract cwd from summary if available
                if obj.get("type") == "summary" and not cwd:
                    cwd = obj.get("cwd")
                msg_type = obj.get("type", "")
                msg = obj.get("message", {})
                role = msg.get("role", "")
                if msg_type not in ("user", "assistant") or role not in ("user", "assistant"):
                    continue
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block["text"])
                    text = " ".join(parts).strip()
                elif isinstance(content, str):
                    text = content.strip()
                else:
                    continue
                if text:
                    messages.append({"role": role, "content": text})
    except Exception:
        pass
    return cwd, messages


def _parse_codex_session(path: Path) -> tuple[str | None, list[dict]]:
    """Parse a Codex rollout JSONL into messages. Returns (cwd, messages)."""
    messages = []
    cwd = None
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                obj_type = obj.get("type", "")
                payload = obj.get("payload", {})
                # Extract cwd from session_meta
                if obj_type == "session_meta" and not cwd:
                    cwd = payload.get("cwd")
                    continue
                if obj_type != "response_item":
                    continue
                role = payload.get("role", "")
                if role not in ("user", "assistant"):
                    continue
                content = payload.get("content", [])
                parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    # user: input_text, assistant: output_text
                    if btype in ("input_text", "output_text"):
                        text = block.get("text", "").strip()
                        # skip env context blocks injected by codex
                        if text and not text.startswith("<environment_context>") and not text.startswith("<permissions"):
                            parts.append(text)
                text = " ".join(parts).strip()
                if text:
                    messages.append({"role": role, "content": text})
    except Exception:
        pass
    return cwd, messages


agent_app = typer.Typer(help="Native conversational harness commands.")
app.add_typer(agent_app, name="agent")


@agent_app.command("chat")
def agent_chat(
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    agent_id: str = typer.Option("default-agent", "--agent-id", "-a", help="Agent ID."),
    context_path: str | None = typer.Option(
        None,
        "--context-path",
        help="Optional path to agents.md or vektori.yaml for agent instructions.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID to reuse for the native harness.",
    ),
    chat_model: str | None = typer.Option(
        None,
        "--chat-model",
        envvar="VEKTORI_CHAT_MODEL",
        help=_CHAT_HELP,
    ),
    extraction_model: str | None = typer.Option(
        None,
        "--extraction-model",
        "-m",
        envvar="VEKTORI_EXTRACTION_MODEL",
        help=_EXTRACTION_HELP,
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        "-e",
        envvar="VEKTORI_EMBEDDING_MODEL",
        help=_EMBEDDING_HELP,
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    profile_store_path: str | None = typer.Option(
        None,
        "--profile-store-path",
        help="Optional SQLite path for durable profile patches.",
    ),
) -> None:
    """Start an interactive chat loop using `VektoriAgent`."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    cm = chat_model or _default_chat_model()
    _warn_openai(em, "--extraction-model")
    _warn_openai(eb, "--embedding-model")
    _warn_openai(cm, "--chat-model")

    async def _run() -> None:
        from vektori.agent import AgentConfig, VektoriAgent
        from vektori.models.factory import create_chat_model

        memory = _client(
            em,
            eb,
            sync_extraction=True,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        agent = VektoriAgent(
            memory=memory,
            model=create_chat_model(cm),
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            config=AgentConfig(
                background_add=True,
                profile_store_path=profile_store_path,
            ),
            context_path=context_path,
        )

        typer.echo("Native agent chat (type 'quit' to exit)\n")
        try:
            while True:
                user_input = typer.prompt("You").strip()
                if user_input.lower() in {"quit", "exit"}:
                    break
                result = await agent.chat(user_input)
                typer.echo(f"Assistant: {result.content}\n")
        finally:
            await agent.close()
            await memory.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        typer.echo("\nExiting.")
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)


@app.command()
def inject(
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    source: str = typer.Option(
        "auto",
        "--source",
        "-s",
        help="Session source: auto (detect both), claude-code, codex.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        "-p",
        help="Filter by project path substring (e.g. 'oss-vektori').",
    ),
    since: int | None = typer.Option(
        None,
        "--since",
        help="Only ingest sessions modified in the last N days.",
    ),
    session_id: str | None = typer.Option(
        None,
        "--session",
        help="Ingest a specific session ID only.",
    ),
    list_only: bool = typer.Option(False, "--list", help="List detected sessions without ingesting."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
) -> None:
    """Detect Claude Code and Codex sessions and ingest them into memory.

    \b
        Examples:
            # list what vektori found
            vektori inject -u dev --list

            # ingest all sessions from the last 7 days
            vektori inject -u dev --since 7 --yes

            # ingest only sessions from a specific project
            vektori inject -u dev --project oss-vektori --yes

            # ingest one session by ID
            vektori inject -u dev --session d06bb817 --yes
    """
    import time

    # Discover sessions
    all_sessions: list[dict] = []
    if source in ("auto", "claude-code"):
        all_sessions.extend(_discover_claude_sessions())
    if source in ("auto", "codex"):
        all_sessions.extend(_discover_codex_sessions())

    # Filters
    if project:
        all_sessions = [s for s in all_sessions if project.lower() in str(s["path"]).lower()]
    if since is not None:
        cutoff = time.time() - since * 86400
        all_sessions = [s for s in all_sessions if s["mtime"] >= cutoff]
    if session_id:
        all_sessions = [s for s in all_sessions if session_id in s["session_id"]]

    if not all_sessions:
        typer.echo("No sessions found matching the given filters.")
        return

    # Show what was found
    typer.echo(f"Found {len(all_sessions)} session(s):\n")
    for i, s in enumerate(all_sessions, 1):
        import datetime
        ts = datetime.datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M")
        project_hint = s.get("project") or "unknown"
        if len(project_hint) > 40:
            project_hint = "..." + project_hint[-37:]
        typer.echo(f"  {i:>3}. [{s['source']:11}] {ts}  {project_hint}  {s['session_id'][:8]}")

    if list_only:
        return

    typer.echo("")
    if not yes:
        typer.confirm(f"Ingest {len(all_sessions)} session(s) into vektori?", abort=True)

    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    _warn_openai(em, "--extraction-model")

    async def _run() -> dict:
        v = _client(em, eb, storage_backend=storage_backend, database_url=database_url)
        ok = 0
        skipped = 0
        try:
            for s in all_sessions:
                if s["source"] == "claude-code":
                    cwd, messages = _parse_claude_session(s["path"])
                else:
                    cwd, messages = _parse_codex_session(s["path"])

                # Skip near-empty sessions
                if len(messages) < 2:
                    skipped += 1
                    continue

                sid = f"{s['source']}:{s['session_id']}"
                meta = {"source": s["source"], "project": s.get("project") or cwd or ""}
                await v.add(messages, session_id=sid, user_id=user_id, metadata=meta)
                ok += 1
                typer.echo(f"  ✓ {s['source']} {s['session_id'][:8]}  ({len(messages)} turns)")
        finally:
            await v.close()
        return {"ingested": ok, "skipped": skipped}

    try:
        result = asyncio.run(_run())
    except Exception as e:
        typer.echo(f"[error] {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nDone — {result['ingested']} session(s) ingested, {result['skipped']} skipped (too short).")


# ---------------------------------------------------------------------------
# context  — dump memory as an injectable context block
# ---------------------------------------------------------------------------


@app.command()
def context(
    user_id: str = typer.Option(..., "--user-id", "-u", envvar="VEKTORI_USER_ID", help="User ID."),
    query: str | None = typer.Argument(
        None,
        help="Optional query — returns relevant memories. Omit to dump all recent facts.",
    ),
    depth: str = typer.Option("l1", "--depth", "-d", help="Search depth: l0, l1, or l2."),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Max facts to include."),
    extraction_model: str | None = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: str | None = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    storage_backend: str | None = typer.Option(
        None, "--storage-backend", envvar="VEKTORI_STORAGE_BACKEND", help=_BACKEND_HELP
    ),
    database_url: str | None = typer.Option(
        None, "--database-url", envvar="VEKTORI_DATABASE_URL", help=_DATABASE_URL_HELP
    ),
    qdrant_api_key: str | None = typer.Option(
        None, "--qdrant-api-key", envvar="QDRANT_API_KEY", help=_QDRANT_API_KEY_HELP
    ),
    hook: bool = typer.Option(
        False,
        "--hook",
        help="Output Claude Code hook JSON (additionalContext injection) instead of plain text.",
    ),
) -> None:
    """Dump memory as a formatted context block — pipe into any AI or use as a Claude Code hook.

    \b
        Examples:
            # paste into any AI session:
            vektori context -u alice

            # query-scoped context:
            vektori context "auth bug" -u alice --depth l2

            # Claude Code hook mode (outputs hookSpecificOutput JSON):
            vektori context -u alice --hook
    """
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()

    async def _run() -> dict:
        v = _client(
            em,
            eb,
            storage_backend=storage_backend,
            database_url=database_url,
            qdrant_api_key=qdrant_api_key,
        )
        try:
            if query:
                result = await v.search(query, user_id=user_id, top_k=top_k, depth=depth)
                facts = result.get("facts", [])
                sentences = result.get("sentences", [])
                episodes = result.get("episodes", [])
            else:
                facts = await v.get_facts(user_id=user_id)
                facts = sorted(facts, key=lambda f: f.get("created_at", ""), reverse=True)[:top_k]
                sentences = []
                episodes = []
        finally:
            await v.close()
        return {"facts": facts, "sentences": sentences, "episodes": episodes}

    try:
        data = asyncio.run(_run())
    except Exception as e:
        if hook:
            typer.echo(json.dumps({"continue": True}))
        else:
            typer.echo(f"[error] {e}", err=True)
            raise typer.Exit(1)
        return

    facts = data["facts"]
    sentences = data["sentences"]
    episodes = data["episodes"]

    if not facts and not sentences:
        if hook:
            typer.echo(json.dumps({"continue": True}))
        else:
            typer.echo("No memories found.")
        return

    lines: list[str] = ["[vektori memory]"]

    if facts:
        lines.append("facts:")
        for f in facts:
            score = f.get("score")
            prefix = f"  [{score:.2f}] " if score is not None else "  "
            lines.append(f"{prefix}{f['text']}")

    if episodes:
        lines.append("episodes:")
        for ep in episodes[:3]:
            lines.append(f"  {ep.get('text', '')}")

    if sentences:
        lines.append("context:")
        cur_session = None
        for s in sentences:
            ssid = s.get("session_id", "")
            if ssid != cur_session:
                cur_session = ssid
                lines.append(f"  [{ssid[:8]}]")
            lines.append(f"    {s['text']}")

    block = "\n".join(lines)

    if hook:
        typer.echo(json.dumps({"continue": True, "hookSpecificOutput": {"additionalContext": block}}))
    else:
        typer.echo(block)


def main() -> None:
    app()
