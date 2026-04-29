# Agent Harness Checklist

This tracks the current implementation status against `docs/AGENT_HARNESS_SPEC.md`.

## Implemented

- `VektoriAgent`, `AgentConfig`, and `AgentTurnResult`
- Separate `ChatModelProvider` / `ChatCompletionResult`
- `AgentContextLoader` for `agents.md` and `vektori.yaml`
- `MessageWindow` with local compaction
- `ProfilePatch` lane with in-memory and SQLite-backed stores
- Explicit profile learning (name, verbosity, units patterns)
- Deterministic prompt assembly with token budgeting and trimming
- Retrieval decision diagnostics on each turn
- Native CLI chat entrypoint via `vektori agent chat`
- **Phase 4 — Tool calling**: `vektori/tools/memory.py` with `search_memory`, `get_profile`, `update_profile` schemas + multi-round-trip tool loop in `VektoriAgent`
- **Phase 5 — Window persistence**: `SQLiteWindowStore` with `save_window()` / `resume_window()` on agent; sessions resumable across process restarts
- E2E integration test script: `scripts/test_agent_e2e.py`

## Partial

- Profile learning rules are explicit but still heuristic-driven (regex patterns only)
- Theory-of-Mind isolation exists in profile storage, not yet a broader runtime concept
- Context parsing is forgiving; not fully normalized for complex YAML
- Cold-path pipeline is `asyncio.create_task`, not a dedicated worker queue

## Missing / Next

- Anthropic `AnthropicChatModel` and Ollama registration in `CHAT_REGISTRY`
- Richer prompt budgeting with exact tokenization (tiktoken / provider tokenizers)
- Tool calling for Anthropic API (uses different tool_use schema)
- Compaction summaries optionally persisted as tagged episodic memory (Phase 5b)
- Cross-session planning / multi-agent shared windows
