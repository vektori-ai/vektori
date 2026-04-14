# Agent Harness Checklist

This tracks the current implementation status against `docs/AGENT_HARNESS_SPEC.md`.

## Implemented

- `VektoriAgent`, `AgentConfig`, and `AgentTurnResult`
- Separate `ChatModelProvider` / `ChatCompletionResult`
- `AgentContextLoader` for `agents.md` and `vektori.yaml`
- `MessageWindow` with local compaction
- `ProfilePatch` lane with in-memory and SQLite-backed stores
- Deterministic prompt assembly
- Prompt budgeting and trimming with diagnostics
- Retrieval decision diagnostics on each turn
- Native CLI chat entrypoint via `vektori agent chat`
- Example harness demos and focused tests

## Partial

- Profile learning rules are explicit but still heuristic-driven
- Theory-of-Mind isolation exists in profile storage and observer/observed patch scoping, but is not yet a broader runtime concept
- Context parsing is useful but still forgiving/basic rather than fully normalized
- Cold-path work is present but not yet a dedicated harness background pipeline

## Missing / Next

- Tool calling and `vektori/tools/memory.py`
- More chat providers beyond current OpenAI/LiteLLM path
- Richer prompt budgeting policy and exact tokenization
- More detailed turn observability beyond current prompt/retrieval diagnostics
- Persistent summary artifacts / extended window persistence
- Deeper context and profile lifecycle tests
