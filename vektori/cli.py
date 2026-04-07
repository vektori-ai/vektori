"""Vektori CLI — add, search, and manage memory from the terminal."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Optional

import typer

app = typer.Typer(help="Vektori memory engine — self-hosted, zero-config.", no_args_is_help=True)

# Model option defaults — override via env vars or CLI flags.
#
# Extraction model: "provider:model"
#   openai:gpt-4o-mini                                        (OPENAI_API_KEY)
#   litellm:groq/llama-3.3-70b-versatile                     (GROQ_API_KEY)
#   litellm:together_ai/meta-llama/Llama-3-70b-chat-hf       (TOGETHERAI_API_KEY)
#   litellm:ollama/llama3                                     (local, no key)
#
# Embedding model: "provider:model"
#   openai:text-embedding-3-small                             (OPENAI_API_KEY)
#   sentence-transformers:all-MiniLM-L6-v2                   (local, no key)
#   litellm:together_ai/togethercomputer/m2-bert-80M-8k-retrieval
#   litellm:ollama/nomic-embed-text                           (local, no key)
_DEFAULT_EXTRACTION = "openai:gpt-4o-mini"
_DEFAULT_EMBEDDING = "sentence-transformers:all-MiniLM-L6-v2"

_EXTRACTION_HELP = (
    "LLM for fact extraction. e.g. 'litellm:groq/llama-3.3-70b-versatile', "
    "'litellm:ollama/llama3' [env: VEKTORI_EXTRACTION_MODEL]"
)
_EMBEDDING_HELP = (
    "Embedding model. e.g. 'sentence-transformers:all-MiniLM-L6-v2' (local), "
    "'openai:text-embedding-3-small' [env: VEKTORI_EMBEDDING_MODEL]"
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
    try:
        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
    except Exception:
        pass


def _client(extraction_model: str, embedding_model: str, sync_extraction: bool = True):
    """
    Build a Vektori client.
    sync_extraction=True: extraction runs inline during add() — CLI default so
    facts are immediately visible after the command returns.
    """
    from vektori.client import Vektori

    from vektori.config import VektoriConfig

    _silence_litellm()
    cfg = VektoriConfig(
        extraction_model=extraction_model,
        embedding_model=embedding_model,
        async_extraction=not sync_extraction,
        # CLI search is always intentional — disable the conversational retrieval gate
        enable_retrieval_gate=False,
    )
    return Vektori(config=cfg)


def _out(data: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init(
    extraction_model: str = typer.Option(
        _DEFAULT_EXTRACTION, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL",
        help=_EXTRACTION_HELP,
    ),
    embedding_model: str = typer.Option(
        _DEFAULT_EMBEDDING, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL",
        help=_EMBEDDING_HELP,
    ),
) -> None:
    """Initialise local SQLite storage."""
    _warn_openai(extraction_model, "--extraction-model")
    _warn_openai(embedding_model, "--embedding-model")

    async def _run() -> None:
        v = _client(extraction_model, embedding_model)
        await v._ensure_initialized()
        await v.close()

    asyncio.run(_run())
    typer.echo(f"Vektori initialised (extraction={extraction_model}, embedding={embedding_model})")


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
    extraction_model: str = typer.Option(
        _DEFAULT_EXTRACTION, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL",
        help=_EXTRACTION_HELP,
    ),
    embedding_model: str = typer.Option(
        _DEFAULT_EMBEDDING, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL",
        help=_EMBEDDING_HELP,
    ),
    no_extraction: bool = typer.Option(
        False, "--no-extraction",
        help="Skip LLM fact extraction. Stores sentence only — useful for testing without an API key.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add a memory. Extraction runs synchronously so facts are ready immediately."""
    if not no_extraction:
        _warn_openai(extraction_model, "--extraction-model")
    _warn_openai(embedding_model, "--embedding-model")
    sid = session_id or str(uuid.uuid4())
    messages = [{"role": "user", "content": text}]

    async def _run() -> dict:
        v = _client(extraction_model, embedding_model, sync_extraction=not no_extraction)
        try:
            result = await v.add(messages, session_id=sid, user_id=user_id)
        finally:
            await v.close()
        return result

    try:
        result = asyncio.run(_run())
    except Exception as e:
        # Surface extraction errors clearly rather than a raw traceback
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
    extraction_model: str = typer.Option(
        _DEFAULT_EXTRACTION, "--extraction-model", "-m", envvar="VEKTORI_EXTRACTION_MODEL",
        help=_EXTRACTION_HELP,
    ),
    embedding_model: str = typer.Option(
        _DEFAULT_EMBEDDING, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL",
        help=_EMBEDDING_HELP,
    ),
    expand: bool = typer.Option(
        False, "--expand", help="Use LLM to generate query variants before searching."
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Search memories with a natural language query."""
    _warn_openai(embedding_model, "--embedding-model")

    async def _run() -> dict:
        v = _client(extraction_model, embedding_model)
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
        # No facts extracted yet — show raw sentence matches
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
    embedding_model: str = typer.Option(
        _DEFAULT_EMBEDDING, "--embedding-model", "-e", envvar="VEKTORI_EMBEDDING_MODEL",
        help=_EMBEDDING_HELP,
    ),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all active memories (extracted facts) for a user."""

    async def _run() -> list:
        v = _client(_DEFAULT_EXTRACTION, embedding_model)
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
        v = _client(_DEFAULT_EXTRACTION, _DEFAULT_EMBEDDING)
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
