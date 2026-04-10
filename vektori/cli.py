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

_EXTRACTION_HELP = (
    "LLM for fact extraction. e.g. 'litellm:groq/llama-3.3-70b-versatile', "
    "'litellm:ollama/llama3' [env: VEKTORI_EXTRACTION_MODEL]"
)
_EMBEDDING_HELP = (
    "Embedding model. e.g. 'sentence-transformers:all-MiniLM-L6-v2' (local), "
    "'openai:text-embedding-3-small' [env: VEKTORI_EMBEDDING_MODEL]"
)
_BACKEND_HELP = (
    "Storage backend: sqlite (default), postgres, memory, neo4j, qdrant. "
    "[env: VEKTORI_STORAGE_BACKEND]"
)
_DATABASE_URL_HELP = (
    "Connection URL for the backend. "
    "Postgres: postgresql://user:pw@host/db  "
    "Neo4j: bolt://host:7687  "
    "Qdrant: http://host:6333  "
    "[env: VEKTORI_DATABASE_URL]"
)
_QDRANT_API_KEY_HELP = "Qdrant Cloud API key. [env: QDRANT_API_KEY]"


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
    if storage_backend:
        cfg["storage_backend"] = storage_backend
    if database_url:
        cfg["database_url"] = database_url

    if extraction_model or embedding_model or storage_backend or database_url:
        _save_config(cfg)
        typer.echo(f"Saved to {_CONFIG_PATH}")

    if show or not (
        extraction_model or embedding_model or storage_backend or database_url or reset
    ):
        typer.echo(f"Config file     : {_CONFIG_PATH}")
        typer.echo(f"extraction      : {_default_extraction()}")
        typer.echo(f"embedding       : {_default_embedding()}")
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


def main() -> None:
    app()
