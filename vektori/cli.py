"""Vektori CLI — add, search, and manage memory from the terminal."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

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


def _client(extraction_model: str, embedding_model: str, sync_extraction: bool = True):
    from vektori.client import Vektori
    from vektori.config import VektoriConfig

    _silence_litellm()
    cfg = VektoriConfig(
        extraction_model=extraction_model,
        embedding_model=embedding_model,
        async_extraction=not sync_extraction,
        enable_retrieval_gate=False,
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
    extraction_model: Optional[str] = typer.Option(
        None, "--extraction-model", "-m", help=_EXTRACTION_HELP
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", "-e", help=_EMBEDDING_HELP
    ),
    show: bool = typer.Option(False, "--show", help="Print current config."),
    reset: bool = typer.Option(False, "--reset", help="Reset to defaults."),
) -> None:
    """Set default models so you don't have to pass --extraction-model / --embedding-model every time.

    \b
    Examples:
      vektori config --extraction-model "litellm:groq/llama-3.3-70b-versatile"
      vektori config --embedding-model "sentence-transformers:all-MiniLM-L6-v2"
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

    if extraction_model or embedding_model:
        _save_config(cfg)
        typer.echo(f"Saved to {_CONFIG_PATH}")

    if show or not (extraction_model or embedding_model or reset):
        typer.echo(f"Config file : {_CONFIG_PATH}")
        typer.echo(f"extraction  : {_default_extraction()}")
        typer.echo(f"embedding   : {_default_embedding()}")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    extraction_model: Optional[str] = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
) -> None:
    """Initialise local SQLite storage."""
    em = extraction_model or _default_extraction()
    eb = embedding_model or _default_embedding()
    _warn_openai(em, "--extraction-model")
    _warn_openai(eb, "--embedding-model")

    async def _run() -> None:
        v = _client(em, eb)
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
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID."),
    session_id: Optional[str] = typer.Option(
        None, "--session-id", "-s", help="Session ID (auto-generated if omitted)."
    ),
    extraction_model: Optional[str] = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    no_extraction: bool = typer.Option(
        False, "--no-extraction",
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
        v = _client(em, eb, sync_extraction=not no_extraction)
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
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Max results to return."),
    depth: str = typer.Option("l1", "--depth", "-d", help="Search depth: l0, l1, or l2."),
    extraction_model: Optional[str] = typer.Option(
        None, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL", help=_EXTRACTION_HELP
    ),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
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
        v = _client(em, eb)
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
        typer.echo("(showing raw sentences — run with an extraction model to get structured facts)\n")
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
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID."),
    embedding_model: Optional[str] = typer.Option(
        None, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL", help=_EMBEDDING_HELP
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all active memories (extracted facts) for a user."""
    eb = embedding_model or _default_embedding()

    async def _run() -> list:
        v = _client(_default_extraction(), eb)
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
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Delete all memories for a user."""
    if not yes:
        typer.confirm(f"Delete ALL memories for user '{user_id}'?", abort=True)

    async def _run() -> int:
        v = _client(_default_extraction(), _default_embedding())
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


def main() -> None:
    app()
