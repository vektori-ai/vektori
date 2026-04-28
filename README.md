<div align="center">

<img src="assets/logo/memory-stack-logo-transparent.svg" width="96" height="96" alt="Vektori logo" />

<h1>Vektori</h1>

<p><strong>Memory that remembers the story, not just the facts.</strong></p>

<a href="https://github.com/vektori-ai/vektori">GitHub</a> · <a href="https://github.com/vektori-ai/vektori/issues">Issues</a> · <a href="./docs">Docs</a>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/vektori)](https://pypi.org/project/vektori/)
[![Downloads](https://img.shields.io/pypi/dm/vektori?color=blue)](https://pypi.org/project/vektori/)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/vektori-ai/vektori?style=flat&color=ffcb47&labelColor=black)](https://github.com/vektori-ai/vektori)
[![Issues](https://img.shields.io/github/issues/vektori-ai/vektori?labelColor=black&style=flat&color=ff80eb)](https://github.com/vektori-ai/vektori/issues)
[![Contributors](https://img.shields.io/github/contributors/vektori-ai/vektori?color=c4f042&labelColor=black&style=flat)](https://github.com/vektori-ai/vektori/graphs/contributors)
[![Last Commit](https://img.shields.io/github/last-commit/vektori-ai/vektori?color=c4f042&labelColor=black)](https://github.com/vektori-ai/vektori/commits/main)

👋 Questions, ideas, bugs → [GitHub Issues](https://github.com/vektori-ai/vektori/issues) · [Discussions](https://github.com/vektori-ai/vektori/discussions)

If Vektori has been useful, a ⭐ goes a long way.

</div>

---

## Why Vektori

Building agents that actually remember people is harder than it looks:

- **Facts aren't enough.** Knowing a user prefers WhatsApp is different from knowing they've asked three times and are getting frustrated. Most systems give you the what, not the why or how it changed.
- **Patterns stay invisible.** Spotting that someone's tone has been shifting across sessions requires more than point-in-time retrieval — you need to see the trajectory.
- **Context overhead explodes.** Stuffing raw conversation history into every prompt doesn't scale. You need structure, not just storage.

Vektori solves this with a **three-layer sentence graph**. Agents don't just recall preferences — they understand how things got there.

```
FACT LAYER (L0)      <- vector search surface. Short, crisp statements.
        |
EPISODE LAYER (L1)   <- patterns auto-discovered via graph traversal.
        |
SENTENCE LAYER (L2)  <- raw conversation. Sequential NEXT edges. The full story.
```

<p align="center">
  <img src="assets/screenshots/layers.jpeg" alt="Three-layer memory graph: Facts → Episodes → Sentences" width="680" />
</p>

Search hits Facts, graph discovers Episodes, traces back to source Sentences. SQLite by default — swap to Postgres, Neo4j, Qdrant, or Milvus when you're ready to scale.

---

## Benchmarks

Tested on long-horizon memory benchmarks — hundreds of turns, real user details buried deep in history.

| Benchmark | Vektori | Mem0 | Zep | Supermemory | Letta |
|-----------|---------|------|-----|-------------|-------|
| LoCoMo | **66%** | 66% | 58%† | ~70% | ~83% |
| LongMemEval-S | **73%** | — | 64% | 85% | — |

†Zep's self-reported score is 75%; independently re-evaluated at 58%. Scores across systems are not always directly comparable — model choice (GPT-4o vs GPT-4.1-mini vs local) significantly affects results. 

We used gemini-2.5-flash-lite because of token cost, better models imporve accuracy a lot. Benchmarks at L1 level

On LoCoMo and longmemEval, **the retrieved context contains the answer in 95% of questions** — the gap to 66% is a synthesis problem, not a retrieval one. Actively working on closing it, exploring RL.

Still improving — PRs and evals welcome. Run your own: [`/benchmarks`](benchmarks/)

---

## Install

```bash
pip install vektori                      # SQLite + Postgres
pip install 'vektori[neo4j]'             # + Neo4j support
pip install 'vektori[qdrant]'            # + Qdrant support
pip install 'vektori[milvus]'            # + Milvus support
pip install 'vektori[neo4j,qdrant,milvus]'  # all backends
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
| `l1`  | Facts + Episodes + source Sentences | 300-800 | **Default.** Full answer with context |
| `l2`  | Facts + Episodes + Sentences + ±N context window | 1000-3000 | Trajectory analysis, full story replay |

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

## Memory Explorer (local debug UI)

Browse stored graph data for a user (facts, episodes, sentences, links, and profile summary):

```bash
python -m tools.memory_explorer.server
```

Then open `http://127.0.0.1:8765`.

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

# Milvus — high-scale vector store with partition-key isolation
v = Vektori(
    storage_backend="milvus",
    database_url="http://localhost:19530",
    embedding_dimension=1024,
)

# Milvus / Zilliz Cloud
v = Vektori(
    storage_backend="milvus",
    database_url="https://your-cluster-endpoint",
    milvus_token="your-api-key-or-token",
    embedding_dimension=1024,
)

# In-memory — tests / CI
v = Vektori(storage_backend="memory")
```

**All backends via Docker:**
```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
docker compose up -d                 # starts Postgres, Neo4j, Qdrant, and Milvus

# Postgres
DATABASE_URL=postgresql://vektori:vektori@localhost:5432/vektori python examples/quickstart_postgres.py

# Neo4j
VEKTORI_STORAGE_BACKEND=neo4j VEKTORI_DATABASE_URL=bolt://localhost:7687 vektori add "I prefer dark mode" --user-id u1

# Qdrant
VEKTORI_STORAGE_BACKEND=qdrant VEKTORI_DATABASE_URL=http://localhost:6333 vektori add "I prefer dark mode" --user-id u1

# Milvus
VEKTORI_STORAGE_BACKEND=milvus VEKTORI_DATABASE_URL=http://localhost:19530 vektori add "I prefer dark mode" --user-id u1

# Milvus Cloud
MILVUS_TOKEN=your-api-key VEKTORI_STORAGE_BACKEND=milvus VEKTORI_DATABASE_URL=https://your-cluster-endpoint vektori add "I prefer dark mode" --user-id u1
```

**CLI storage flags:**
```bash
vektori config --storage-backend qdrant --database-url http://localhost:6333
vektori config --storage-backend milvus --database-url http://localhost:19530
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

## NVIDIA NIM - GPU-optimized models via [NVIDIA NIM](https://build.nvidia.com).
```python
# NVIDIA embedding models (Matryoshka: 384-2048 dimensions)
v = Vektori(
    embedding_model="nvidia:llama-nemotron-embed-1b-v2",
    embedding_dimension=1024,  # Optional: 384, 512, 768, 1024, or 2048
)

# NVIDIA LLM models (nvidia/ prefix auto-added)
v = Vektori(extraction_model="nvidia:llama-3.3-nemotron-super-49b-v1")

# Third-party models hosted on NVIDIA NIM (use full path)
v = Vektori(extraction_model="nvidia:z-ai/glm5")

```
---

## Contributing

Vektori is early and there's a lot of ground to cover. If you're building agents that need memory, your real-world feedback is the most valuable thing you can contribute.

- Found a bug or an edge case? [Open an issue](https://github.com/vektori-ai/vektori/issues)
- Have an idea or want to discuss direction? [Start a discussion](https://github.com/vektori-ai/vektori/discussions)
- Want to contribute code? See [CONTRIBUTING.md](CONTRIBUTING.md)

```bash
git clone https://github.com/vektori-ai/vektori
cd vektori
pip install -e ".[dev]"
pytest tests/unit/
```

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vektori-ai/vektori&type=timeline)](https://www.star-history.com/#vektori-ai/vektori&type=timeline)

---

## License

Apache 2.0. See [LICENSE](LICENSE).
