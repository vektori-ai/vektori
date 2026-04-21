# Vektori Agent Harness Specification
## Native Conversational Runtime for Memory-Centric Agents

**Authors:** Codex proposal for Vektori
**Status:** Design spec
**Last Updated:** April 14, 2026

---

## 1. Goal

Vektori already provides the core memory engine:
- storage backends
- ingestion and extraction pipelines
- retrieval across L0/L1/L2
- model providers for embeddings and extraction

What it does **not** provide today is a native conversational harness. Developers still have to:
- decide when to search memory
- convert `v.search()` output into prompt text
- manage raw chat history
- compact context when token limits are hit
- persist stable user preferences manually

This spec adds that missing orchestration layer.

**Target outcome:** a developer can build a stateful agent with Vektori in a few lines, without LangChain/LangGraph/CrewAI, while keeping Vektori's existing memory engine and storage model intact.

---

## 2. Design Principles

### 2.1 Keep the Existing Boundary

`Vektori` remains the memory engine. It should not become the chat runtime itself.

- `vektori.client.Vektori` stays focused on:
  - `add()`
  - `search()`
  - `get_facts()`
  - storage lifecycle
- New orchestration lives above it.

This avoids collapsing ingestion, retrieval, prompt assembly, and model execution into one class.

### 2.2 Separate Three Kinds of Memory

The harness must treat these as distinct:

1. **Core Memory**
   - static persona/instructions
   - stable user preferences
   - agent policy
   - loaded every turn

2. **Recall Memory**
   - retrieved facts/episodes/sentences
   - dynamic per-turn memory
   - only included when relevant

3. **Working Memory**
   - recent raw messages
   - rolling summaries of older turns
   - private short-term state of the current conversation

If these are collapsed into one prompt blob, the agent becomes noisy and hard to control.

### 2.3 Structured Self-Modification, Not Freeform Prompt Rewriting

The agent should be able to learn durable preferences, but it should **not** directly rewrite `agents.md` on every turn.

Instead:
- base instructions come from file
- stable learned preferences are stored as structured patches
- the runtime assembles an effective system prompt from:
  - base instructions
  - active patches
  - optional runtime overrides

This gives durable adaptation without uncontrolled prompt drift.

### 2.4 Additive Rollout

The first release should work without:
- tool calling
- cross-session planning logic
- provider-specific advanced APIs

Phase 1 should already solve the main problem:
`chat(user_message)` automatically retrieves memory, injects context, manages the short-term window, calls the model, and stores the exchange.

### 2.5 Hot Path vs. Cold Path (Zero Latency Tax)
For real-time and voice agents, generation of the response cannot wait for slow memory extraction. The harness splits execution into two paths:
- **Hot Path (Synchronous):** Retrieve existing memory, assemble prompt, generate fast response (streamable).
- **Cold Path (Asynchronous/Background):** After the turn/call, queue background tasks to do heavy processing (L0 fact extraction, L1 insight generation, profile patch learning, and window compaction).

### 2.6 Theory of Mind (ToM) Isolation
Memory and profile patches are recorded not as universal truths, but as subjective observations: `(Observer, Observed)`. This allows Agent A to construct a completely different mental model of a user than Agent B, preventing context bleeding.

---

## 3. Proposed Modules

### 3.1 New Public Modules

Add the following modules:

```text
vektori/
  agent.py
  context.py
  prompts.py
  memory/
    __init__.py
    window.py
    profile.py
  tools/
    __init__.py
    memory.py
```

### 3.2 Responsibilities

#### `vektori/agent.py`

Defines the public runtime API:
- `VektoriAgent`
- `AgentConfig`
- `AgentTurnResult`

Owns:
- turn loop
- retrieval orchestration
- context budgeting
- model invocation
- async persistence back into `Vektori`

#### `vektori/context.py`

Defines static context loading:
- `AgentContextLoader`
- parsers for `agents.md` and `vektori.yaml`

Owns:
- base system instructions
- static context sections
- policy flags
- optional defaults for response style and retrieval behavior

#### `vektori/memory/window.py`

Defines the short-term message window:
- raw turn buffer
- token accounting
- rolling summary
- compaction policy

#### `vektori/memory/profile.py`

Defines durable profile/preference state:
- stable user preferences
- learned instruction patches
- provenance and confidence

#### `vektori/prompts.py`

Defines deterministic prompt assembly:
- system prompt composition
- recall memory rendering
- summary blocks
- model-facing message construction

#### `vektori/tools/memory.py`

Defines optional tool schemas and tool handlers:
- `search_memory`
- `remember_preference`
- `get_profile`
- `update_profile`

Tool use is optional in phase 1 and enabled later.

---

## 4. Public API

### 4.1 `VektoriAgent`

Proposed constructor:

```python
class VektoriAgent:
    def __init__(
        self,
        memory: Vektori,
        model: ChatModelProvider,
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        config: AgentConfig | None = None,
        context_path: str | None = None,
    ) -> None: ...
```

Proposed runtime methods:

```python
async def chat(
    self,
    user_message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentTurnResult: ...

async def add_messages(
    self,
    messages: list[dict[str, str]],
    *,
    metadata: dict[str, Any] | None = None,
) -> None: ...

def reset_window(self) -> None: ...

async def close(self) -> None: ...
```

### 4.2 `AgentTurnResult`

Do not return only a string. The harness should expose what happened.

```python
@dataclass
class AgentTurnResult:
    content: str
    messages: list[dict[str, str]]
    memories_used: dict[str, list[dict[str, Any]]]
    summary_updated: bool
    profile_updates: list[ProfilePatch]
    tool_calls: list[ToolCallRecord]
    usage: dict[str, int] | None = None
```

This makes the runtime inspectable and easier to debug.

### 4.3 `AgentConfig`

```python
@dataclass
class AgentConfig:
    max_context_tokens: int = 12000
    reserve_response_tokens: int = 1500
    retrieval_depth: str = "l1"
    retrieval_top_k: int = 8
    retrieve_on_every_turn: bool = False
    enable_retrieval_gate: bool = True
    enable_profile_learning: bool = True
    enable_tool_calling: bool = False
    max_tool_round_trips: int = 3
    compaction_trigger_ratio: float = 0.8
    keep_last_n_turns: int = 6
    summary_max_tokens: int = 400
    persist_assistant_messages: bool = True
    background_add: bool = True
```

Phase 1 should support only a small, deterministic subset.

---

## 5. Model Interface Changes

## 5.1 Current Constraint

Today `LLMProvider` only supports:

```python
async def generate(prompt: str, max_tokens: int | None = None) -> str
```

That is enough for extraction but insufficient for:
- multi-message chat
- system prompts
- tool calling
- summarization with role-aware message history

### 5.2 Proposed Split

Do **not** overload extraction behavior into the same method forever.

Introduce a chat-oriented interface:

```python
class ChatModelProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResult: ...
```

And keep the extraction path:

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int | None = None) -> str: ...
```

### 5.3 Compatibility Strategy

Two safe options:

1. Add a new `ChatModelProvider` interface alongside `LLMProvider`.
2. Make provider implementations support both interfaces where possible.

Recommended approach:
- preserve `LLMProvider` for extraction
- add `ChatModelProvider` for the native harness
- let OpenAI / Anthropic / Gemini / LiteLLM implementations implement both

This avoids breaking the current extraction pipeline.

### 5.4 `ChatCompletionResult`

```python
@dataclass
class ChatCompletionResult:
    content: str | None
    tool_calls: list[ToolCall]
    raw_response: Any | None = None
    usage: dict[str, int] | None = None
```

This result shape supports both simple text-only turns and tool-driven loops.

---

## 6. Context File Standard

## 6.1 Supported Files

Support either:
- `agents.md`
- `vektori.yaml`

The loader should search in this order:
1. explicit `context_path`
2. current working directory
3. repo root

### 6.2 `agents.md`

`agents.md` is the human-friendly format.

Suggested sections:

```md
# Agent

## Persona
You are a concise technical assistant.

## Instructions
- Prefer direct answers.
- Ask clarifying questions only when necessary.

## Domain Context
- This project uses Vektori as the memory engine.

## Response Style
- Keep answers short unless asked for depth.

## Memory Policy
- Persist explicit user preferences.
- Do not store secrets unless explicitly requested.
```

The parser should be forgiving:
- unrecognized headings become generic context sections
- missing sections are allowed

### 6.3 `vektori.yaml`

`vektori.yaml` is the machine-friendly format.

Proposed shape:

```yaml
agent:
  name: support-agent
  persona: >
    You are a concise technical support assistant.
  instructions:
    - Prefer direct answers.
    - Use memory only when it materially improves the answer.
  response_style:
    verbosity: short
  memory:
    persist_preferences: true
    retrieve_default_depth: l1
    retrieve_top_k: 8
```

### 6.4 Loader Output

Both file types should normalize into:

```python
@dataclass
class LoadedAgentContext:
    persona: str
    instructions: list[str]
    response_style: dict[str, Any]
    memory_policy: dict[str, Any]
    extra_sections: dict[str, str]
```

---

## 7. Short-Term Memory Window

## 7.1 Problem

The native harness must own short-term state locally instead of depending on provider-specific thread APIs.

### 7.2 `MessageWindow`

```python
@dataclass
class WindowState:
    recent_messages: list[dict[str, str]]
    rolling_summary: str
    estimated_tokens: int
    compaction_count: int
```

API:

```python
class MessageWindow:
    def add(self, role: str, content: str) -> None: ...
    def snapshot(self) -> WindowState: ...
    def estimated_tokens(self) -> int: ...
    async def compact(self, summarizer: ChatModelProvider) -> bool: ...
    def reset(self) -> None: ...
```

### 7.3 Compaction Strategy

When token usage exceeds:

`max_context_tokens * compaction_trigger_ratio`

the window should:

1. keep the last `N` raw turns unchanged
2. summarize older messages into a short state block
3. replace those older raw messages with the new summary
4. mark that a compaction event occurred

### 7.4 Summary Format

Summary should be structured, not freeform prose:

```text
Conversation Summary
- Active goals:
- Confirmed preferences:
- Open questions:
- Constraints:
- Recent commitments:
```

This makes the summary more reusable than a narrative paragraph.

### 7.5 Interaction with Long-Term Memory

Compaction summary should remain local in phase 1.

Optional phase 2 behavior:
- store summary-derived artifacts in Vektori as a tagged episode or summary record
- mark them with metadata such as:
  - `source = "window_compaction"`
  - `session_id`
  - `compaction_index`

Do not mix summary-derived memory with direct fact extraction unless tagged explicitly.

---

## 8. Durable User Profile and Instruction Patches

## 8.1 Why This Is Separate

A stable preference like "keep answers short" is neither:
- ordinary retrieved recall memory
- nor just a recent message

It is a durable behavioral instruction.

### 8.2 `ProfilePatch`

```python
@dataclass
class ProfilePatch:
    key: str
    value: Any
    reason: str
    source: str
    observer_id: str  # Theory of Mind: Who learned this
    observed_id: str  # Theory of Mind: Who this is about
    confidence: float
    created_at: datetime
    last_confirmed_at: datetime | None = None
    active: bool = True
```

Examples:
- `response_style.verbosity = "short"`
- `preferences.communication_channel = "email"`
- `preferences.units = "metric"`

### 8.3 Persistence Options

Recommended phase 1 implementation:
- store profile patches in a small local table or backend abstraction

Avoid phase 1 shortcuts like:
- storing them only as generic facts
- rewriting `agents.md`

Reason:
- profile patches are policy inputs, not just retrievable content
- they need activation/deactivation semantics

### 8.4 Profile Learning Rule

Only persist durable patches when the user signal is explicit enough.

Good candidates:
- "Keep your answers short."
- "Remember that I prefer metric units."
- "Call me Laxman."

Bad candidates:
- one-off situational requests
- weak inference from tone
- transient plans

### 8.5 Effective System Prompt Assembly

At runtime:

```text
effective_system_prompt =
  base persona/instructions
  + active profile patches
  + optional runtime style overrides
```

The harness should never mutate the base file itself unless the developer explicitly asks for it.

---

## 9. Turn Execution Flow

## 9.1 Phase 1 Flow

For a call to `await agent.chat(user_message)` the execution is split to support real-time/voice workloads:

**Hot Path (Immediate Response):**
1. Add user message to `MessageWindow`.
2. Decide whether retrieval is warranted.
3. If yes, fast-fetch pre-computed memory via `memory.search(...)`.
4. Load base context from `AgentContextLoader` and active Profile Patches.
5. Assemble model messages.
6. Call chat model.
7. Append assistant response to window and return `AgentTurnResult` immediately.

**Cold Path (Background Queue / Post-Turn):**
1. Enqueue conversation history for background processing.
2. Background worker extracts L0 Facts and infers L1 Insights via `memory.add(...)`.
3. Background worker evaluates Theory of Mind updates for the observer-observed pair.
4. Background worker learns and persists new `ProfilePatch` objects.
5. Background worker compacts `MessageWindow` if over token budget.

### 9.2 Retrieval Decision

Reuse the repo's retrieval-gate philosophy:
- not every turn needs memory lookup
- personal, referential, or recall-like turns should trigger retrieval
- local follow-up turns with enough recent context can skip retrieval

Recommended default:
- `enable_retrieval_gate=True`
- `retrieve_on_every_turn=False`

### 9.3 Retrieval Depth Default

Default to `l1`:
- facts
- episodes/insights
- no full L2 sentence expansion unless requested

Why:
- `l0` is often too sparse for conversational grounding
- `l2` is too expensive for every turn

### 9.4 Persistence Timing

`memory.add()` should happen after the assistant responds.

Default behavior:
- queue in background when possible
- allow synchronous mode for tests

This preserves the repo's current async-ingestion design.

---

## 10. Prompt Assembly

## 10.1 Assembly Inputs

The prompt builder should receive:
- loaded agent context
- active profile patches
- retrieved recall memory
- rolling summary
- recent raw messages

### 10.2 Assembly Structure

Recommended message layout:

1. system message
2. optional memory/context message
3. optional summary message
4. recent user/assistant turns

The system message should remain concise and policy-focused.
Retrieved memory should not be mixed blindly into the system prompt if it is large.

### 10.3 Memory Rendering Format

Use labeled sections:

```text
Retrieved Memory

Facts
- User prefers short answers.
- User works on the Vektori codebase.

Episodes
- The user tends to want implementation details after agreeing on direction.
```

This makes the memory block inspectable and easier to debug.

### 10.4 Priority Order

When token pressure exists, trim in this order:

1. low-value retrieved sentences
2. lower-ranked episodes
3. lower-ranked facts
4. old recent messages already represented in summary

Do not trim:
- system instructions
- active high-priority profile patches
- latest user turn

---

## 11. Tool Calling

## 11.1 Scope

Tool calling is a later enhancement, not a prerequisite for the native harness.

### 11.2 Initial Tool Set

Proposed schemas:
- `search_memory`
- `get_profile`
- `update_profile`

Avoid exposing raw low-level storage tools in phase 1.

### 11.3 Tool Definitions

Example:

```json
{
  "name": "search_memory",
  "description": "Search Vektori long-term memory for relevant facts and episodes.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "depth": {"type": "string", "enum": ["l0", "l1", "l2"]},
      "top_k": {"type": "integer", "minimum": 1, "maximum": 20}
    },
    "required": ["query"]
  }
}
```

### 11.4 Tool Loop

If tool calling is enabled:

1. send tools with the model request
2. execute returned tool calls
3. append tool results as messages
4. repeat up to `max_tool_round_trips`
5. require final assistant text

### 11.5 Why Tools Are Not Enough Alone

Even with tool calling, the harness should still support automatic retrieval:
- many turns can be handled better by deterministic memory injection
- tool choice should be reserved for ambiguous or multi-step cases

So the design should support both:
- proactive retrieval by the harness
- explicit memory tools for the model

---

## 12. Storage and Data Model Impact

## 12.1 What Can Reuse Existing Vektori Storage

Can reuse immediately:
- `add()`
- `search()`
- existing sentence/fact/episode graph

### 12.2 New Storage Need: Profile Patches

The harness needs a persistence mechanism for:
- durable user preferences
- instruction patches
- deactivation/history

Recommended options:

1. add a small `profiles` or `instruction_patches` table to SQL backends and equivalent support in memory backend
2. add a generic metadata key-value table

Preferred approach:
- explicit `instruction_patches` abstraction

Reason:
- cleaner semantics
- easier deactivation and provenance
- avoids overloading facts for policy

### 12.3 No Schema Change Needed for Phase 1 Windowing

Short-term window state can remain in-memory inside the harness instance in phase 1.

Later, if resumable sessions are needed, window snapshots can be persisted.

---

## 13. Example Usage

## 13.1 Minimal Native Agent

```python
import asyncio
from vektori import Vektori
from vektori.agent import VektoriAgent
from vektori.models.factory import create_chat_model


async def main():
    memory = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )
    model = create_chat_model("openai:gpt-4o-mini")

    agent = VektoriAgent(
        memory=memory,
        model=model,
        user_id="demo-user",
        context_path="agents.md",
    )

    result = await agent.chat("What do you remember about how I like answers?")
    print(result.content)

    await agent.close()


asyncio.run(main())
```

This is the target product experience.

### 13.2 Phase 1 Demo File

Add:

```text
examples/vektori_agent_demo.py
```

That example should demonstrate:
- one constructor
- one `chat()` loop
- no manual `search()` to prompt glue

---

## 14. Implementation Plan

## 14.1 Phase 1: Native Harness

Deliver:
- `vektori/agent.py` (Hot path / response orchestration)
- `vektori/prompts.py`
- `vektori/memory/window.py`
- basic `ChatModelProvider`
- Background Worker Task/Queue primitive for cold-path async extractions
- `examples/vektori_agent_demo.py`

Behavior:
- automatic hot-path retrieval
- automatic prompt assembly
- background execution of `memory.add()` and compaction
- local short-term buffer
- no tool calling yet

### 14.2 Phase 2: Context File Loader

Deliver:
- `vektori/context.py`
- `agents.md` / `vektori.yaml` parsing
- normalized loaded context object

### 14.3 Phase 3: Profile Patches

Deliver:
- `vektori/memory/profile.py`
- patch persistence abstraction
- learned durable preferences in effective system prompt

### 14.4 Phase 4: Tool Calling

Deliver:
- `vektori/tools/memory.py`
- tool schemas
- model/tool loop support

### 14.5 Phase 5: Extended Window Persistence

Optional:
- resumable window state
- compaction summaries stored as tagged episodic memory

---

## 15. Non-Goals for Initial Version

The initial harness should **not** attempt to solve all of the following:

- multi-agent shared windows
- provider-specific thread APIs
- autonomous long-horizon planning
- graph-walk planning across many retrieval hops
- uncontrolled prompt rewriting
- fully automatic profile learning from weak signals

These add complexity before the core harness is stable.

---

## 16. Key Design Decisions

### 16.1 `Vektori` Stays a Memory Engine

The native harness is additive, not a rewrite of `client.py`.

### 16.2 Short-Term Memory Is Local and Deterministic

Context compaction belongs to the harness, not to the storage layer.

### 16.3 Durable Preferences Need a Separate Lane

Stable user instructions should not be modeled only as generic searchable facts.

### 16.4 Retrieval Remains Facts-First

The harness should keep using Vektori's current L0/L1/L2 retrieval strategy.
No new retrieval paradigm is required for phase 1.

### 16.5 Tool Calling Is Optional, Not Foundational

The first useful native harness does not depend on tool calling.

---

## 17. Open Questions

These should be resolved before implementation begins:

1. Should `episodes` be renamed to `insights` at the API level, or should the harness preserve the current naming and only document the alias?
2. Should profile patches live in the same storage backends as facts, or in a thin local sidecar store first?
3. Should `MessageWindow` token counting be heuristic-only in phase 1, or should providers optionally expose a tokenizer/count method?
4. Should automatic retrieval happen before every turn when the context file declares `memory.always_retrieve=true`?
5. Should compaction summaries ever be fed back into `memory.add()` automatically, or only remain local until phase 2?

---

## 18. Success Criteria

The native harness is successful when:

1. a developer can build a stateful agent with Vektori in under 10 lines of setup
2. no manual `search()` to prompt mapping is required
3. the agent can hold a long conversation without exceeding model context
4. stable user preferences can influence future turns without editing prompt files manually
5. the design remains backend-agnostic and does not break the current memory engine

---

## 19. Summary

Vektori already has the memory primitives. The missing layer is the runtime that turns memory into agent behavior.

The correct architecture is:

- keep `Vektori` as the memory engine
- add `VektoriAgent` as the native chat harness
- add a chat-capable model interface
- separate core memory, recall memory, and working memory
- implement structured profile patches instead of freeform prompt rewriting

That produces a memory-native agent runtime without discarding the current retrieval and ingestion architecture.