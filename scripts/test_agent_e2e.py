#!/usr/bin/env python3
"""
End-to-end integration test for VektoriAgent.

Tests all phases live against real providers:
  Phase 1 — basic chat + memory retrieval
  Phase 3 — profile patch learning and persistence
  Phase 4 — tool calling (search_memory, update_profile)
  Phase 5 — window snapshot save/resume

Run:
    OPENAI_API_KEY=sk-... python scripts/test_agent_e2e.py

Optional env vars:
    VEKTORI_MODEL          chat model string, default "openai:gpt-4o-mini"
    VEKTORI_EMBED_MODEL    embedding model string, default "openai:text-embedding-3-small"
    VEKTORI_EXTRACT_MODEL  extraction model string, default "openai:gpt-4o-mini"
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"

PASSED = f"{_GREEN}PASS{_RESET}"
FAILED = f"{_RED}FAIL{_RESET}"
SKIP = f"{_YELLOW}SKIP{_RESET}"


def _check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        print(f"  {PASSED}  {label}")
    else:
        print(f"  {FAILED}  {label}" + (f"\n         {detail}" if detail else ""))
    return condition


async def phase1_basic_chat(model_str: str, embed_str: str, extract_str: str) -> bool:
    """Phase 1 — chat() retrieves memory and generates a response."""
    print("\n[Phase 1] Basic chat + memory retrieval")
    from vektori import Vektori, VektoriAgent
    from vektori.agent import AgentConfig
    from vektori.models.factory import create_chat_model

    ok = True
    async with Vektori(
        embedding_model=embed_str,
        extraction_model=extract_str,
        async_extraction=False,
    ) as memory:
        # Seed some facts
        await memory.add(
            [
                {"role": "user", "content": "I prefer short answers."},
                {"role": "assistant", "content": "Got it, I'll be concise."},
            ],
            session_id="seed-1",
            user_id="e2e-user",
        )

        model = create_chat_model(model_str)
        config = AgentConfig(retrieve_on_every_turn=True, background_add=False)
        agent = VektoriAgent(
            memory=memory,
            model=model,
            user_id="e2e-user",
            config=config,
        )

        result = await agent.chat("What do I prefer?")
        ok &= _check(bool(result.content), "chat() returns non-empty content")
        ok &= _check(len(result.messages) >= 2, "window accumulates messages")
        ok &= _check("prompt_debug" in result.__dataclass_fields__, "prompt_debug present in result")

        result2 = await agent.chat("hi")
        ok &= _check(result2.retrieval_debug["reason"] in (
            "retrieval_gate_skipped", "retrieve_on_every_turn"
        ), "retrieval_debug.reason populated")

        await agent.close()
    return ok


async def phase3_profile_patches(model_str: str, embed_str: str, extract_str: str) -> bool:
    """Phase 3 — explicit profile patch learning + SQLite persistence."""
    print("\n[Phase 3] Profile patch learning")
    import tempfile
    from vektori import Vektori, VektoriAgent
    from vektori.agent import AgentConfig
    from vektori.models.factory import create_chat_model

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        profile_db = f"{tmp}/profile.db"
        async with Vektori(embedding_model=embed_str, extraction_model=extract_str) as memory:
            model = create_chat_model(model_str)
            config = AgentConfig(
                enable_profile_learning=True,
                background_add=False,
                profile_store_path=profile_db,
            )
            agent = VektoriAgent(memory=memory, model=model, user_id="e2e-user", config=config)

            result = await agent.chat("Call me Laxman.")
            ok &= _check(
                any(p.key == "preferences.name" for p in result.profile_updates),
                "name patch learned from 'Call me Laxman.'",
            )

            result2 = await agent.chat("Keep your answers short.")
            ok &= _check(
                any(p.key == "response_style.verbosity" for p in result2.profile_updates),
                "verbosity patch learned from 'Keep your answers short.'",
            )

            # Check the patch is injected in the next turn's system prompt
            result3 = await agent.chat("What do you know about me?")
            joined = " ".join(
                m["content"] for m in model.complete.call_args[0][0] if m["role"] == "system"
            )
            ok &= _check(
                "preferences.name" in joined or "Laxman" in joined,
                "profile patch appears in system prompt",
            )
            await agent.close()
    return ok


async def phase4_tool_calling(model_str: str, embed_str: str, extract_str: str) -> bool:
    """Phase 4 — tool calling loop (search_memory + update_profile)."""
    print("\n[Phase 4] Tool calling loop")
    from vektori import Vektori, VektoriAgent
    from vektori.agent import AgentConfig
    from vektori.models.factory import create_chat_model

    ok = True
    async with Vektori(
        embedding_model=embed_str,
        extraction_model=extract_str,
        async_extraction=False,
    ) as memory:
        await memory.add(
            [{"role": "user", "content": "I'm building a Rust async web service."}],
            session_id="seed-p4",
            user_id="e2e-tool-user",
        )

        model = create_chat_model(model_str)
        config = AgentConfig(
            enable_tool_calling=True,
            max_tool_round_trips=3,
            background_add=False,
        )
        agent = VektoriAgent(
            memory=memory,
            model=model,
            user_id="e2e-tool-user",
            config=config,
        )

        result = await agent.chat("What project am I working on?")
        ok &= _check(bool(result.content), "tool-calling chat returns content")
        # Tool calls may or may not fire depending on model discretion,
        # but at minimum it shouldn't crash
        ok &= _check(isinstance(result.tool_calls, list), "tool_calls is a list")
        if result.tool_calls:
            ok &= _check(
                result.tool_calls[0]["name"] in {"search_memory", "get_profile", "update_profile"},
                f"tool call name is valid: {result.tool_calls[0]['name']}",
            )
        else:
            print(f"  {SKIP}  model chose not to use tools this turn (acceptable)")

        await agent.close()
    return ok


async def phase5_window_persistence(model_str: str, embed_str: str, extract_str: str) -> bool:
    """Phase 5 — window snapshot save and resume."""
    print("\n[Phase 5] Window snapshot persistence")
    import tempfile
    from vektori import Vektori, VektoriAgent
    from vektori.agent import AgentConfig
    from vektori.models.factory import create_chat_model

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        windows_db = f"{tmp}/windows.db"
        async with Vektori(embedding_model=embed_str, extraction_model=extract_str) as memory:
            model = create_chat_model(model_str)
            config = AgentConfig(
                background_add=False,
                window_store_path=windows_db,
            )

            # Session A — have a conversation then save
            agent_a = VektoriAgent(
                memory=memory, model=model, user_id="e2e-window-user",
                session_id="sess-window-test", config=config,
            )
            await agent_a.chat("My favourite language is Elixir.")
            await agent_a.chat("I'm working on a distributed tracing system.")
            await agent_a.save_window()
            n_messages = len(agent_a.window.snapshot().recent_messages)
            await agent_a.close()

            ok &= _check(n_messages == 4, f"window has 4 messages before save ({n_messages} found)")

            # Session B — resume from saved snapshot
            agent_b = VektoriAgent(
                memory=memory, model=model, user_id="e2e-window-user",
                session_id="sess-window-test", config=config,
            )
            restored = await agent_b.resume_window()
            ok &= _check(restored is True, "resume_window() returns True")
            restored_msgs = agent_b.window.snapshot().recent_messages
            ok &= _check(
                len(restored_msgs) == n_messages,
                f"restored window has same message count ({len(restored_msgs)} vs {n_messages})",
            )
            ok &= _check(
                any("Elixir" in m["content"] for m in restored_msgs),
                "Elixir message present in restored window",
            )
            await agent_b.close()
    return ok


async def main() -> None:
    model_str = os.environ.get("VEKTORI_MODEL", "openai:gpt-4o-mini")
    embed_str = os.environ.get("VEKTORI_EMBED_MODEL", "openai:text-embedding-3-small")
    extract_str = os.environ.get("VEKTORI_EXTRACT_MODEL", "openai:gpt-4o-mini")

    if not os.environ.get("OPENAI_API_KEY"):
        print(f"{_RED}ERROR{_RESET}: OPENAI_API_KEY not set. Export it and re-run.")
        sys.exit(1)

    print(f"VektoriAgent e2e integration test")
    print(f"  model={model_str}  embed={embed_str}  extract={extract_str}\n")

    results: list[bool] = []
    for test in [phase1_basic_chat, phase3_profile_patches, phase4_tool_calling, phase5_window_persistence]:
        try:
            ok = await test(model_str, embed_str, extract_str)
        except Exception as e:
            print(f"  {FAILED}  unexpected exception: {e}")
            import traceback; traceback.print_exc()
            ok = False
        results.append(ok)

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} phases passed")
    if passed < total:
        print(f"{_RED}Some phases FAILED — see above.{_RESET}")
        sys.exit(1)
    else:
        print(f"{_GREEN}All phases PASSED.{_RESET}")


if __name__ == "__main__":
    asyncio.run(main())
