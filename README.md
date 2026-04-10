<p align="center">
  <img src="assets/logo/memory-stack-logo-transparent.svg" width="96" height="96" alt="Vektori logo" />
</p>

<h1 align="center">Vektori</h1>

<p align="center"><strong>Memory that remembers the story, not just the facts.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0" /></a>
  <a href="https://pypi.org/project/vektori/"><img src="https://img.shields.io/pypi/v/vektori" alt="PyPI" /></a>
  <a href="https://pypi.org/project/vektori/"><img src="https://img.shields.io/pypi/dm/vektori?color=blue" alt="PyPI Downloads" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+" /></a>
</p>

---

Most memory systems compress conversations into entity-relationship triples. You get the answer, but you lose the texture, the reasoning, the trajectory. Vektori uses a **three-layer sentence graph** so agents don't just recall preferences, they understand how things got there.

```
FACT LAYER (L0)      <- vector search surface. Short, crisp statements.
        |
EPISODE LAYER (L1)   <- patterns auto-discovered via graph traversal.
        |
SENTENCE LAYER (L2)  <- raw conversation. Sequential NEXT edges. The full story.
```

Search hits Facts, graph discovers Episodes, traces back to source Sentences. SQLite by default — swap to Postgres, Neo4j, or Qdrant when you're ready to scale.

---

## Benchmarks

| Benchmark | Score | Depth | Models |
|-----------|-------|-------|--------|
| LongMemEval-S | **73%** | L1 | BGE-M3 + Gemini Flash |

Still improving. Run your own in [`/benchmarks`](benchmarks/).

---

## Install

```bash
pip install vektori                      # SQLite + Postgres
pip install 'vektori[neo4j]'             # + Neo4j support
pip install 'vektori[qdrant]'            # + Qdrant support
pip install 'vektori[neo4j,qdrant]'      # all backends
```

No Docker, no external services. SQLite by default.

---

## 30-Second Quickstart

```python
import asyncio
from vektori import Vektori

async def main():
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )

    await v.add(
        messages=[
            {"role": "user", "content": "I only use WhatsApp, please don't email me."},
            {"role": "assistant", "content": "Got it, WhatsApp only."},
            {"role": "user", "content": "My outstanding amount is ₹45,000 and I can pay by Friday."},
        ],
        session_id="call-001",
        user_id="user-123",
    )

    results = await v.search(
        query="How does this user prefer to communicate?",
        user_id="user-123",
        depth="l1",  # facts + episodes
    )

    for fact in results["facts"]:
        print(f"[{fact['score']:.2f}] {fact['text']}")
    for episode in results["episodes"]:
        print(f"episode: {episode['text']}")

    await v.close()

asyncio.run(main())
```

**Output:**
```
[0.94] User prefers WhatsApp communication
[0.81] Outstanding balance of ₹45,000, payment expected Friday
episode: User consistently avoids email — route all comms to WhatsApp
```

---

## Retrieval Depths

Pick how deep you want to go.

| Depth | Returns | ~Tokens | When to use |
|-------|---------|---------|-------------|
| `l0`  | Facts only | 50-200 | Fast lookup, agent planning, tool calls |
| `l1`  | Facts + Episodes | 200-500 | **Default.** Full answer with context |
| `l2`  | Facts + Episodes + raw Sentences | 1000-3000 | Trajectory analysis, full story replay |

```python
# Just the facts
results = await v.search(query, user_id, depth="l0")

# Facts + episodes (recommended)
results = await v.search(query, user_id, depth="l1")

# Everything, with surrounding conversation context
results = await v.search(query, user_id, depth="l2", context_window=3)
```

---

## Build an Agent with Memory

Three lines to wire memory into any agent loop:

```python
import asyncio
from openai import AsyncOpenAI
from vektori import Vektori

client = AsyncOpenAI()

async def chat(user_id: str):
    v = Vektori(
        embedding_model="openai:text-embedding-3-small",
        extraction_model="openai:gpt-4o-mini",
    )
    session_id = f"session-{user_id}-001"
    history = []

    print("Chat with memory (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # 1. Pull relevant memory
        mem = await v.search(query=user_input, user_id=user_id, depth="l1")
        facts = "\n".join(f"- {f['text']}" for f in mem.get("facts", []))
        episodes = "\n".join(f"- {ep['text']}" for ep in mem.get("episodes", []))

        # 2. Inject into system prompt
        system = "You are a helpful assistant with memory.\n"
        if facts:    system += f"\nKnown facts:\n{facts}"
        if episodes: system += f"\nBehavioral episodes:\n{episodes}"

        # 3. Get response
        history.append({"role": "user", "content": user_input})
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, *history],
        )
        reply = resp.choices[0].message.content
        history.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")

        # 4. Store exchange
        await v.add(
            messages=[{"role": "user", "content": user_input},
                      {"role": "assistant", "content": reply}],
            session_id=session_id,
            user_id=user_id,
        )

    await v.close()

asyncio.run(chat("demo-user"))
```

More examples in [`/examples`](examples/):
- [`quickstart.py`](examples/quickstart.py) — fully local, zero API keys (Ollama)
- [`openai_agent.py`](examples/openai_agent.py) — OpenAI agent loop

---

## Storage Backends

```python
# SQLite (default) — zero config, starts instantly
v = Vektori()

# PostgreSQL + pgvector — production scale
v = Vektori(database_url="postgresql://localhost:5432/vektori")

# Neo4j — native graph traversal for Episode layer
v = Vektori(
    storage_backend="neo4j",
    database_url="bolt://localhost:7687",
    embedding_dimension=1024,   # must match your embedding model
)

# Qdrant — dedicated vector DB, cloud-ready
v = Vektori(
    storage_backend="qdrant",
    database_url="http://localhost:6333",
    embedding_dimension=1024,
)

# Qdrant Cloud
v = Vektori(
    storage_backend="qdrant",
    database_url="https://your-cluster.qdrant.io",
    qdrant_api_key="your-api-key",
    embedding_dimension=1024,
)

# In-memory — tests / CI
v = Vektori(storage_backend="memory")
```

**All backends via Docker:**
```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
docker compose up -d                 # starts Postgres, Neo4j, and Qdrant

# Postgres
DATABASE_URL=postgresql://vektori:vektori@localhost:5432/vektori python examples/quickstart_postgres.py

# Neo4j
VEKTORI_STORAGE_BACKEND=neo4j VEKTORI_DATABASE_URL=bolt://localhost:7687 vektori add "I prefer dark mode" --user-id u1

# Qdrant
VEKTORI_STORAGE_BACKEND=qdrant VEKTORI_DATABASE_URL=http://localhost:6333 vektori add "I prefer dark mode" --user-id u1
```

**CLI storage flags:**
```bash
vektori config --storage-backend qdrant --database-url http://localhost:6333
vektori add "my note" --user-id u1
vektori search "preferences" --user-id u1
```

---

## Model Support

Bring whatever model stack you have. Works with 10 providers out of the box.

```python
# OpenAI
v = Vektori(
    embedding_model="openai:text-embedding-3-small",
    extraction_model="openai:gpt-4o-mini",
)

# Azure OpenAI
# Ensure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set
# Note: The string after "azure:" must match your specific Azure deployment names
v = Vektori(
    embedding_model="azure:my-embedding-deployment",
    extraction_model="azure:my-gpt-4o-deployment",
)

# GitHub Models (Copilot)
# Requires GITHUB_TOKEN. You can get one by running `./scripts/get_github_token.sh`
v = Vektori(
    embedding_model="github:text-embedding-3-small",
    extraction_model="github:gpt-4o",
)

# Anthropic
v = Vektori(
    embedding_model="anthropic:voyage-3",
    extraction_model="anthropic:claude-haiku-4-5-20251001",
)

# Fully local, no API keys, no internet
v = Vektori(
    embedding_model="ollama:nomic-embed-text",
    extraction_model="ollama:llama3",
)

# Sentence Transformers (local, no Ollama required)
v = Vektori(embedding_model="sentence-transformers:all-MiniLM-L6-v2")

# BGE-M3 — multilingual, 1024-dim, best local embeddings we've found
v = Vektori(embedding_model="bge:BAAI/bge-m3")

# LiteLLM — 100+ providers through one interface
v = Vektori(extraction_model="litellm:groq/llama3-8b-8192")
```

---

## Why Not Mem0 / Zep?

| | Mem0 / Zep | **Vektori** |
|---|---|---|
| Memory model | Entity-relation triples | Three-layer sentence graph |
| What you get | The answer | The answer + reasoning + story |
| Patterns beyond facts | Manual graph queries | Auto-discovered (Episode layer) |
| Default backend | Requires external DB | SQLite, zero config |
| Fully local / offline | No | Yes (Ollama, BGE-M3, SentenceTransformers) |
| License | Partial OSS | Apache 2.0 |

Mem0 and Zep are solid tools. But they compress conversations into triples, so you get the *what* but not the *why* or how it changed over time. That matters when you're building agents that need to reason about a person's trajectory, not just their current state.

---

## Contributing

Issues, PRs, and ideas welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
pip install -e ".[dev]"
pytest tests/unit/
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
