"""Vektori CLI — add, search, and manage memory from the terminal."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Optional

import typer

app = typer.Typer(help="Vektori memory engine — self-hosted, zero-config.", no_args_is_help=True)


def _client():
    from vektori.client import Vektori

    return Vektori()


def _out(data: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Initialise local SQLite storage."""

    async def _run() -> None:
        v = _client()
        await v._ensure_initialized()
        await v.close()

    asyncio.run(_run())
    typer.echo("Vektori initialised. SQLite database ready.")


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
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Add a memory."""
    sid = session_id or str(uuid.uuid4())
    messages = [{"role": "user", "content": text}]

    async def _run() -> dict:
        v = _client()
        result = await v.add(messages, session_id=sid, user_id=user_id)
        await v.close()
        return result

    result = asyncio.run(_run())
    if as_json:
        _out(result, True)
    else:
        typer.echo(f"Stored — {result['sentences_stored']} sentence(s), extraction: {result['extraction']}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Max results to return."),
    depth: str = typer.Option("l1", "--depth", "-d", help="Search depth: l0, l1, or l2."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Search memories with a natural language query."""

    async def _run() -> dict:
        v = _client()
        result = await v.search(query, user_id=user_id, top_k=top_k, depth=depth)
        await v.close()
        return result

    result = asyncio.run(_run())
    if as_json:
        _out(result, True)
        return

    facts = result.get("facts", [])
    if not facts:
        typer.echo("No memories found.")
        return
    for i, fact in enumerate(facts, 1):
        score = fact.get("score")
        score_str = f"  [{score:.3f}]" if score is not None else ""
        typer.echo(f"{i}.{score_str} {fact['text']}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_memories(
    user_id: str = typer.Option(..., "--user-id", "-u", help="User ID."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all active memories for a user."""

    async def _run() -> list:
        v = _client()
        facts = await v.get_facts(user_id=user_id)
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
        v = _client()
        n = await v.delete_user(user_id=user_id)
        await v.close()
        return n

    n = asyncio.run(_run())
    if as_json:
        _out({"deleted": n, "user_id": user_id}, True)
    else:
        typer.echo(f"Deleted {n} record(s) for user '{user_id}'.")


def main() -> None:
    app()
