# Vektori — Open Source Memory Engine
## Technical Specification & Implementation Plan

**Authors:** Laxman & Manit
**Version:** 0.1.0
**Last Updated:** March 23, 2026

---

## 1. What We're Building

An open-source, self-hostable memory engine for AI agents that preserves full conversational context through a sentence graph architecture. Unlike knowledge graphs (Mem0, Zep) that compress conversations into entity-relationship triples, Vektori preserves the actual conversational flow — what was said, in what order, what happened next — and uses that structure for cross-session context retrieval.

**Core principle:** Knowledge graphs give agents the answer. Sentence graphs give agents the story.

### 1.1 Open-Core Strategy

**Open source (this repo):**
- Core memory engine (sentence graph construction, storage, retrieval)
- Fact/insight extraction pipeline
- Quality filtering
- Multi-model support (OpenAI, Anthropic, Ollama)
- Pluggable storage backends: SQLite (zero-config default) + PostgreSQL + pgvector (production)
- Python SDK (`pip install vektori`)

**Commercial (later, not in this repo):**
- Hosted cloud API
- Trajectory library (v2)
- Causal edge extraction (v2)
- Multi-tenant isolation
- Analytics dashboard
- Enterprise auth, SOC 2, HIPAA

---

## 2. Architecture Overview

### 2.1 Three-Layer Graph Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ╔═══════════════════════════════════════════════════════════════╗   │
│  ║                    FACT LAYER (L0 — top)                     ║   │
│  ║         Primary search surface. Crisp, query-aligned.        ║   │
│  ║                                                               ║   │
│  ║  [Borrower prefers    [Agent offered     [Outstanding amount  ║   │
│  ║   WhatsApp]            email]             is ₹45,000]        ║   │
│  ║      │                   │                    │               ║   │
│  ╚══════╪═══════════════════╪════════════════════╪═══════════════╝   │
│         │                   │                    │                   │
│    fact_sources         fact_sources         fact_sources            │
│         │                   │                    │                   │
│  ┌──────▼───────────────────▼────────────────────▼───────────────┐  │
│  │                  SENTENCE LAYER (L2 — bottom)                 │  │
│  │          Raw conversation, sequential flow                    │  │
│  │                                                               │  │
│  │  Session A (Call 2):                                          │  │
│  │  [s1]──NEXT──[s2]──NEXT──[s3]──NEXT──[s4]──NEXT──[s5]       │  │
│  │                                                               │  │
│  │  Session B (Call 5):                                          │  │
│  │  [s6]──NEXT──[s7]──NEXT──[s8]──NEXT──[s9]──NEXT──[s10]      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│         ▲                   ▲                    ▲                   │
│    insight_sources     insight_sources      insight_sources          │
│         │                   │                    │                   │
│  ╔══════╪═══════════════════╪════════════════════╪═══════════════╗   │
│  ║      │          INSIGHT LAYER (L1 — middle)   │               ║   │
│  ║      │   Inferred patterns. Cross-session.    │               ║   │
│  ║      │   Discovered via graph, NOT search.    │               ║   │
│  ║                                                               ║   │
│  ║  [Offering email after WhatsApp    [Empathy before payment   ║   │
│  ║   preference → disconnection]       discussion → better      ║   │
│  ║        │          │                  outcomes]                ║   │
│  ║   insight_facts  insight_facts          │                    ║   │
│  ║        │          │                insight_facts              ║   │
│  ║        ▼          ▼                     ▼                    ║   │
│  ║   [Borrower    [Agent               [Agent                   ║   │
│  ║    prefers      offered              acknowledged             ║   │
│  ║    WhatsApp]    email]               difficulty]              ║   │
│  ╚═══════════════════════════════════════════════════════════════╝   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

RETRIEVAL FLOW:
  1. Vector search on FACTS (crisp cosine match)
  2. From facts → trace to INSIGHTS via insight_facts (graph discovery)
  3. From facts → trace to SENTENCES via fact_sources
  4. From sentences → expand session context via NEXT edges
```

**Fact layer (L0, top) — primary search surface.** LLM-extracted explicit statements: "Borrower prefers WhatsApp." "Outstanding amount is ₹45,000." Short, crisp, high cosine similarity with direct queries. This is where vector search lands. Facts are subject to conflict resolution — new information supersedes old facts.

**Insight layer (L1, middle) — discovered via graph traversal, NOT vector search.** LLM-inferred patterns derived from multiple facts across sessions: "Offering email after borrower states WhatsApp preference leads to disconnection." Insights are longer, multi-concept statements that embed poorly for point queries but are extremely valuable as contextual enrichment. They're discovered by following `insight_facts` edges from matched facts — they come along for the ride, not as search targets.

**Sentence layer (L2, bottom) — raw conversation, sequential flow.** Real sentences from conversations. Connected sequentially within a session via `NEXT` edges. Immutable records of what was actually said. Expanded from matched facts via `fact_sources` edges + session context via `NEXT` edges.

### 2.2 Why Facts on Top (The Cosine Similarity Argument)

Cosine similarity works best on short, focused text. Facts are exactly that — "Borrower prefers WhatsApp" is 4 words and semantically laser-aligned with a query like "how does the borrower communicate?"

Insights are paragraph-length reasoning: "Offering email after borrower states WhatsApp preference correlates with call disconnection in 4/6 observed cases." This embeds as an average of multiple concepts (email, WhatsApp, preference, disconnection, statistics). It'll have decent similarity to MANY queries but won't be the TOP match for any specific one. Insights are diluted in embedding space.

So we search the crispest layer and discover the rest through graph traversal.

### 2.3 Why No Pre-Computed KNN Edges

The current system computes SIMILAR_TO edges via KNN (cosine ≥ 0.80) between sentences. This creates problems:

1. **Quadratic growth** — edge table grows with N² sentences
2. **Recomputation cost** — new sentences require KNN recalculation
3. **Noisy bridges** — even at 0.80 threshold, many connections are meaningless
4. **PPR drift** — required suppressing SIMILAR_TO weight to 0.25 and lowering damping to 0.5 to compensate

The insight layer replaces KNN edges as the cross-session bridge. An insight like "offering email after WhatsApp preference leads to disconnection" is derived from facts across call 2, call 5, and call 8. The insight IS the cross-session learning — no computed similarity needed.

pgvector handles semantic similarity at query time (on facts). The graph handles structural relationships (derivation, sequence, insight-to-fact) that embeddings can't capture.

### 2.4 Why Not a Knowledge Graph

A knowledge graph (Mem0-style) stores:
```
[Borrower] --prefers--> [WhatsApp]
```

Our three-layer graph stores:
```
Fact:     "Borrower prefers WhatsApp"
  ↓ fact_sources
Sentence: "I only use WhatsApp, please don't send me emails"
  → NEXT → "The agent offered to send details via email"
  → NEXT → "The borrower disconnected"
  ↑ insight_sources
Insight:  "Offering email after WhatsApp preference → disconnection"
```

Knowledge graphs give agents the answer. We give agents the answer (facts), the reasoning (insights), AND the story (sentences). Three dimensions of memory that no other system provides.

---

## 3. Database Schema

### 3.1 PostgreSQL + pgvector

Single database. No Neo4j, no Qdrant, no external dependencies. Developers run `docker compose up` and everything works.

```sql
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- SENTENCES: The bottom layer. Real things that were said.
-- ============================================================
CREATE TABLE sentences (
    id UUID PRIMARY KEY,
    -- Content
    text TEXT NOT NULL,
    embedding vector(1536),  -- configurable dimension

    -- Ownership & scoping
    user_id TEXT NOT NULL,
    agent_id TEXT,

    -- Session context
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,      -- position in conversation
    sentence_index INTEGER NOT NULL,   -- position within a turn
    role TEXT NOT NULL DEFAULT 'user',  -- 'user' or 'assistant'

    -- Deduplication & weighting
    content_hash TEXT NOT NULL,         -- SHA-256(session_id + sentence_index + text)
    mentions INTEGER DEFAULT 1,         -- incremented on re-encounter, powers IDF

    -- State
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Deterministic ID: same content in same position = same ID = no duplicates
-- On conflict, increment mentions counter (IDF weighting)
CREATE UNIQUE INDEX idx_sentences_content_hash ON sentences (content_hash);

-- Vector index for semantic search (IVFFlat for < 1M rows, switch to HNSW later)
CREATE INDEX idx_sentences_embedding ON sentences
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Scoping indexes
CREATE INDEX idx_sentences_user ON sentences (user_id);
CREATE INDEX idx_sentences_session ON sentences (session_id);
CREATE INDEX idx_sentences_user_agent ON sentences (user_id, agent_id);


-- ============================================================
-- FACTS: The top layer. LLM-extracted explicit knowledge.
-- PRIMARY SEARCH SURFACE — vector search lands here.
-- ============================================================
CREATE TABLE facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- Content
    text TEXT NOT NULL,                 -- the extracted fact as a clear statement
    embedding vector(1536),

    -- Ownership
    user_id TEXT NOT NULL,
    agent_id TEXT,

    -- State management
    is_active BOOLEAN DEFAULT true,
    superseded_by UUID REFERENCES facts(id),  -- points to newer contradicting fact
    confidence FLOAT DEFAULT 1.0,             -- extraction confidence

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_facts_embedding ON facts
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_facts_user ON facts (user_id);
CREATE INDEX idx_facts_user_agent ON facts (user_id, agent_id);
CREATE INDEX idx_facts_active ON facts (user_id, is_active) WHERE is_active = true;


-- ============================================================
-- INSIGHTS: The middle layer. LLM-inferred cross-session patterns.
-- NOT a vector search target. Discovered via graph traversal
-- from matched facts through insight_facts join table.
-- ============================================================
CREATE TABLE insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    -- Content
    text TEXT NOT NULL,                 -- the inferred insight as a clear statement
    embedding vector(1536),            -- stored for potential future use, NOT searched

    -- Ownership
    user_id TEXT NOT NULL,
    agent_id TEXT,

    -- Quality
    confidence FLOAT DEFAULT 1.0,      -- higher bar than facts (inferred, not stated)
    is_active BOOLEAN DEFAULT true,

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_insights_user ON insights (user_id);
CREATE INDEX idx_insights_user_agent ON insights (user_id, agent_id);
-- NOTE: No vector index on insights — they're discovered via graph, not search


-- ============================================================
-- EDGES: Typed relationships between sentences.
-- ============================================================
CREATE TABLE sentence_edges (
    source_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,            -- 'next' (sequential), 'contradiction'
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE INDEX idx_sentence_edges_source ON sentence_edges (source_id, edge_type);
CREATE INDEX idx_sentence_edges_target ON sentence_edges (target_id, edge_type);


-- ============================================================
-- FACT_SOURCES: Links facts (L0) to source sentences (L2).
-- Vertical bridge: top → bottom.
-- ============================================================
CREATE TABLE fact_sources (
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, sentence_id)
);

CREATE INDEX idx_fact_sources_fact ON fact_sources (fact_id);
CREATE INDEX idx_fact_sources_sentence ON fact_sources (sentence_id);


-- ============================================================
-- INSIGHT_SOURCES: Links insights (L1) to source sentences (L2).
-- Insights derive from raw conversation just like facts.
-- ============================================================
CREATE TABLE insight_sources (
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    PRIMARY KEY (insight_id, sentence_id)
);

CREATE INDEX idx_insight_sources_insight ON insight_sources (insight_id);
CREATE INDEX idx_insight_sources_sentence ON insight_sources (sentence_id);


-- ============================================================
-- INSIGHT_FACTS: Links insights (L1) to related facts (L0).
-- THIS IS THE KEY BRIDGE. When vector search finds a fact,
-- we discover insights by traversing this join table.
-- Insights → Facts is how the middle layer connects to the
-- search surface.
-- ============================================================
CREATE TABLE insight_facts (
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (insight_id, fact_id)
);

CREATE INDEX idx_insight_facts_insight ON insight_facts (insight_id);
CREATE INDEX idx_insight_facts_fact ON insight_facts (fact_id);


-- ============================================================
-- SESSIONS: Metadata about conversation sessions.
-- ============================================================
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);
```

### 3.2 Conflict Resolution

When new information contradicts existing facts:

**During fact extraction, the LLM is prompted to check for contradictions against existing active facts for that user/agent.** If a contradiction is found:

1. The old fact is marked `is_active = false`
2. The old fact's `superseded_by` field points to the new fact
3. The new fact is inserted as `is_active = true`
4. Both facts retain their `fact_sources` links to original sentences

**The old fact is never deleted.** It stays in the database with its sentence connections intact. This preserves history — you can always ask "what did the borrower say before changing their preference?" Active retrieval only returns `is_active = true` facts, but temporal/historical queries can access superseded facts.

**Example flow:**

```
Session 1: User says "I prefer email for communication"
  → Fact extracted: "User prefers email" (is_active=true)

Session 5: User says "Actually, only contact me on WhatsApp"
  → LLM detects contradiction with "User prefers email"
  → Old fact: is_active=false, superseded_by=new_fact_id
  → New fact: "User prefers WhatsApp" (is_active=true)

Retrieval query: "How does user prefer to communicate?"
  → Returns: "User prefers WhatsApp" (active fact)
  → Source sentences available for context

Historical query: "What communication preferences has this user expressed?"
  → Returns both facts with timestamps, showing the change
```

**For sentence-level contradictions:** Sentences themselves are never marked inactive — they're records of what was actually said. Only facts (the interpreted layer) get superseded. The sentence "I prefer email" is still a true record of what was said in session 1, even though the preference has changed.

**For insight staleness:** When a fact gets superseded, insights linked to that fact may become stale. However, insights are NOT automatically deactivated — they might still be valid if they're linked to multiple facts and only one was superseded. Instead, during the next extraction cycle, the LLM receives updated active facts and can generate revised insights. Old insights linked only to superseded facts naturally lose relevance because they won't be discovered through active fact retrieval (the `insight_facts` bridge only fires when a matched fact is active). This is self-cleaning — stale insights fade from retrieval without explicit garbage collection.

---

## 4. Ingestion Pipeline

### 4.1 Overview

```
User calls v.add(messages, session_id, ...)
                │
                ▼
   ┌──── SYNCHRONOUS (fast, user waits) ────┐
   │                                         │
   │  1. Sentence splitting                  │
   │  2. Quality filtering                   │
   │  3. Deterministic ID generation         │
   │  4. Compute embeddings (batch)          │
   │  5. Store sentences + NEXT edges        │
   │  6. Return success                      │
   │                                         │
   └─────────────────────────────────────────┘
                │
                ▼
   ┌──── ASYNCHRONOUS (background) ──────────┐
   │                                         │
   │  7. Extract facts/insights via LLM      │
   │  8. Check for contradictions            │
   │  9. Compute fact embeddings             │
   │  10. Store facts + fact_source edges    │
   │  11. Handle conflict resolution         │
   │                                         │
   └─────────────────────────────────────────┘
```

The user gets a response immediately after step 6. Fact extraction happens in the background. This is critical for latency — LLM calls for extraction can take 2-5 seconds. Don't make the user wait.

### 4.2 Step 1: Sentence Splitting

**Current system:** Modal-hosted NLTK/SaT model (external API call).
**New system:** Local Python, no network calls.

```python
import spacy

nlp = spacy.load("en_core_web_sm")  # lightweight, no GPU needed

def split_sentences(text: str) -> list[str]:
    """Split text into sentences, merge short fragments."""
    doc = nlp(text)
    raw_sentences = [sent.text.strip() for sent in doc.sents]
    return merge_short_sentences(raw_sentences)

def merge_short_sentences(sentences: list[str]) -> list[str]:
    """Merge fragments that start with conjunctions/prepositions back into parent."""
    MERGE_STARTERS = {'and', 'but', 'or', 'nor', 'yet', 'so', 'for',
                      'on', 'in', 'at', 'by', 'to', 'from', 'with',
                      'which', 'who', 'that', 'where', 'when'}
    merged = []
    for sent in sentences:
        first_word = sent.split()[0].lower().rstrip(',') if sent.split() else ''
        if merged and (first_word in MERGE_STARTERS or len(sent.split()) < 4):
            merged[-1] = merged[-1].rstrip('.') + ' ' + sent
        else:
            merged.append(sent)
    return merged
```

**Fallback if spacy is too heavy:** Use `nltk.sent_tokenize` with the same merge logic. Even lighter dependency.

### 4.3 Step 2: Quality Filtering

Port the existing 10-layer gauntlet to Python. This is one of our best features — keep it all.

```python
import re
from dataclasses import dataclass

@dataclass
class QualityConfig:
    min_chars: int = 15
    min_words: int = 5
    min_content_density: float = 0.15  # ratio of content words to total
    max_pronoun_ratio: float = 0.40

STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
             'would', 'could', 'should', 'may', 'might', 'shall', 'can',
             'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
             'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
             'my', 'your', 'his', 'its', 'our', 'their', 'what', 'which',
             'who', 'whom', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
             'by', 'from', 'and', 'but', 'or', 'not', 'no', 'so', 'if'}

PRONOUNS = {'this', 'that', 'it', 'they', 'them', 'these', 'those', 'he',
            'she', 'we', 'him', 'her', 'us'}

# Patterns to reject
JUNK_PATTERNS = [
    r'^(ok|okay|sure|yes|no|yeah|yep|nope|hmm|hm|ah|oh|uh|um|lol|haha|thanks|thank you|got it|right|cool|nice|great|fine|alright)\s*[.!?]*$',
    r'^(hey|hi|hello|bye|goodbye|cheers|ciao)\s*[.!?]*$',
]

CODE_PATTERNS = [
    r'[{}\[\]<>].*[{}\[\]<>]',         # brackets suggesting code
    r'^(import |from |def |class |const |let |var |function )',
    r'[a-zA-Z0-9+/]{40,}',             # base64 / credentials
    r'^(/|\\|[A-Z]:\\)',                # file paths
    r'https?://\S{50,}',               # long URLs
]

META_PATTERNS = [
    r'^(just for context|for reference|fyi|note:|update:)',
    r':$',                               # headers ending in colon
    r'^(explain|tell me about|describe|show me|help me with)\s',
]

def is_quality_sentence(text: str, config: QualityConfig = QualityConfig()) -> bool:
    """10-layer quality gauntlet. Returns True if sentence passes all checks."""
    text_clean = text.strip()
    words = text_clean.lower().split()

    # 1. Length check
    if len(text_clean) < config.min_chars or len(words) < config.min_words:
        return False

    # 2. Junk / filler / acknowledgment
    text_lower = text_clean.lower().strip('.,!? ')
    for pattern in JUNK_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    # 3. Code / credentials / file paths
    for pattern in CODE_PATTERNS:
        if re.search(pattern, text_clean):
            return False

    # 4. Meta-text / vague commands
    for pattern in META_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    # 5. Pronoun-heavy fragments
    pronoun_count = sum(1 for w in words if w in PRONOUNS)
    if len(words) > 0 and pronoun_count / len(words) > config.max_pronoun_ratio:
        return False

    # 6. Information density (content words vs stopwords)
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if len(words) > 0 and len(content_words) / len(words) < config.min_content_density:
        return False

    return True
```

**Important:** Make the quality filter configurable. Developers should be able to disable it or adjust thresholds. Some use cases want to store everything.

### 4.4 Step 3: Deterministic IDs

Keep the current approach. It's genuinely good.

```python
import hashlib
import uuid

def generate_sentence_id(session_id: str, sentence_index: int, text: str) -> str:
    """Deterministic UUID from content. Same content = same ID = no duplicates."""
    raw = f"{session_id}:{sentence_index}:{text}"
    hash_bytes = hashlib.sha256(raw.encode()).digest()
    return str(uuid.UUID(bytes=hash_bytes[:16]))
```

On upsert, if the sentence already exists, increment `mentions` counter (for IDF weighting).

### 4.5 Step 4-5: Embedding + Storage

```python
async def ingest_messages(
    self,
    messages: list[dict],  # [{"role": "user", "content": "..."}, ...]
    session_id: str,
    user_id: str,
    agent_id: str = None
) -> dict:
    """Synchronous ingestion: split, filter, embed, store. Returns immediately."""

    # 1. Split all messages into sentences
    all_sentences = []
    for turn_num, msg in enumerate(messages):
        raw_sents = split_sentences(msg["content"])
        for idx, text in enumerate(raw_sents):
            if msg["role"] == "user" and is_quality_sentence(text):
                # Only user sentences go into the sentence graph
                # (assistant messages used in fact extraction but not stored as nodes)
                all_sentences.append({
                    "text": text,
                    "session_id": session_id,
                    "turn_number": turn_num,
                    "sentence_index": idx,
                    "role": msg["role"],
                    "id": generate_sentence_id(session_id, f"{turn_num}_{idx}", text)
                })

    if not all_sentences:
        return {"status": "ok", "sentences_stored": 0}

    # 2. Batch embed
    texts = [s["text"] for s in all_sentences]
    embeddings = await self.embedder.embed_batch(texts)

    # 3. Upsert sentences (ON CONFLICT increment mentions)
    await self.db.upsert_sentences(all_sentences, embeddings, user_id, agent_id)

    # 4. Create NEXT edges (sequential within session)
    edges = []
    for i in range(len(all_sentences) - 1):
        edges.append({
            "source_id": all_sentences[i]["id"],
            "target_id": all_sentences[i + 1]["id"],
            "edge_type": "next",
            "weight": 1.0
        })
    await self.db.insert_edges(edges)

    # 5. Trigger async fact + insight extraction (non-blocking)
    self._schedule_extraction(messages, session_id, user_id, agent_id)

    return {
        "status": "ok",
        "sentences_stored": len(all_sentences),
        "extraction": "processing"  # facts + insights extracted async
    }
```

### 4.6 Step 7-11: Async Fact & Insight Extraction + Conflict Resolution

```python
async def _extract_facts_and_insights(
    self,
    messages: list[dict],
    session_id: str,
    user_id: str,
    agent_id: str = None
):
    """Background job: extract facts AND insights from conversation."""

    # Get existing active facts for contradiction checking
    existing_facts = await self.db.get_active_facts(user_id, agent_id, limit=50)
    existing_facts_text = "\n".join([f"- {f['text']}" for f in existing_facts])

    # Build the full conversation text
    conversation = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" for msg in messages
    ])

    prompt = f"""Analyze this conversation and extract two types of information:
FACTS and INSIGHTS.

CONVERSATION:
{conversation}

EXISTING KNOWN FACTS ABOUT THIS USER:
{existing_facts_text if existing_facts_text else "None yet."}

Return JSON with two arrays:

{{
  "facts": [
    {{
      "text": "clear, standalone factual statement",
      "confidence": 0.0-1.0,
      "source_quotes": ["exact quote from conversation this fact comes from"],
      "contradicts": "text of existing fact this contradicts, or null"
    }}
  ],
  "insights": [
    {{
      "text": "inferred pattern or observation, actionable for the agent",
      "confidence": 0.0-1.0,
      "derived_from_facts": ["fact text this insight is based on"],
      "source_quotes": ["exact quotes supporting this inference"]
    }}
  ]
}}

FACT rules:
- Explicit, verifiable information stated in the conversation
- Short, crisp statements (under 20 words ideally)
- One fact per statement — don't combine multiple pieces of info
- Check EVERY new fact against existing facts for contradictions

INSIGHT rules:
- Inferred patterns NOT explicitly stated in the conversation
- Must be actionable — "borrower gets defensive when X" is good,
  "conversation happened" is worthless
- Can reference multiple facts and span cross-session patterns
- Higher confidence bar — only extract insights you're reasonably sure about
- Examples: behavioral patterns, communication preferences, what approaches
  work/don't work, emotional triggers, timing patterns

Return ONLY the JSON, nothing else."""

    response = await self.llm.generate(prompt)
    extracted = parse_json_response(response)

    # ── Process FACTS ──
    inserted_fact_ids = {}  # text → id mapping for insight linking
    for fact_data in extracted.get("facts", []):
        fact_embedding = await self.embedder.embed(fact_data["text"])

        # Handle contradiction
        supersedes_id = None
        if fact_data.get("contradicts"):
            old_fact = await self.db.find_fact_by_text(
                user_id, fact_data["contradicts"], agent_id
            )
            if old_fact:
                supersedes_id = old_fact["id"]
                await self.db.deactivate_fact(old_fact["id"])

        fact_id = await self.db.insert_fact(
            text=fact_data["text"],
            embedding=fact_embedding,
            user_id=user_id,
            agent_id=agent_id,
            confidence=fact_data.get("confidence", 1.0),
            superseded_by_target=supersedes_id
        )
        inserted_fact_ids[fact_data["text"]] = fact_id

        # Link fact → source sentences
        if fact_data.get("source_quotes"):
            source_sents = await self.db.find_sentences_by_similarity(
                fact_data["source_quotes"], session_id, threshold=0.75
            )
            for sent_id in source_sents:
                await self.db.insert_fact_source(fact_id, sent_id)

    # ── Process INSIGHTS ──
    for insight_data in extracted.get("insights", []):
        insight_embedding = await self.embedder.embed(insight_data["text"])

        insight_id = await self.db.insert_insight(
            text=insight_data["text"],
            embedding=insight_embedding,
            user_id=user_id,
            agent_id=agent_id,
            confidence=insight_data.get("confidence", 1.0),
        )

        # Link insight → related facts (the key bridge)
        if insight_data.get("derived_from_facts"):
            for fact_text in insight_data["derived_from_facts"]:
                # Check if this fact was just inserted in this batch
                if fact_text in inserted_fact_ids:
                    await self.db.insert_insight_fact(
                        insight_id, inserted_fact_ids[fact_text]
                    )
                else:
                    # Search existing facts for a match
                    existing = await self.db.find_fact_by_text(
                        user_id, fact_text, agent_id
                    )
                    if existing:
                        await self.db.insert_insight_fact(
                            insight_id, existing["id"]
                        )

        # Link insight → source sentences
        if insight_data.get("source_quotes"):
            source_sents = await self.db.find_sentences_by_similarity(
                insight_data["source_quotes"], session_id, threshold=0.75
            )
            for sent_id in source_sents:
                await self.db.insert_insight_source(insight_id, sent_id)
```

---

## 5. Retrieval Pipeline

### 5.1 Tiered Retrieval Depth (L0 / L1 / L2)

Retrieval maps directly to the three-layer architecture. Each depth level adds one more layer:

| Depth | Layers Returned | What Agent Gets | ~Tokens | Use Case |
|-------|----------------|-----------------|---------|----------|
| **L0** | Facts only | Crisp answers: "Borrower prefers WhatsApp" | ~50-200 | Quick lookup, agent planning |
| **L1** | Facts + insights | Answers + patterns: "...and offering email after this leads to disconnection" | ~200-500 | Default. Answer + actionable context |
| **L2** | Facts + insights + sentences + session context | Full story: the actual conversation moments and surrounding flow | ~1000-3000 | Deep investigation, trajectory analysis |

```python
# L0: Just facts — search surface only (cheapest)
results = await v.search(query, user_id, depth="l0")

# L1: Facts + insights discovered from those facts (default)
results = await v.search(query, user_id, depth="l1")

# L2: Everything — facts + insights + source sentences + session context
results = await v.search(query, user_id, depth="l2", context_window=5)
```

### 5.2 Pipeline Overview

```
User calls v.search(query, user_id, ...)
                │
                ▼
   Step 1: Vector search over FACTS (L0 — top layer)
   → Top 5-10 most relevant active facts
   → Crisp cosine match on short, query-aligned text
   → (if depth="l0", return here)
                │
                ▼
   Step 2: Discover INSIGHTS linked to matched facts
   → Traverse insight_facts join table
   → Find insights that reference any of the matched facts
   → NOT vector search — graph discovery only
   → (if depth="l1", return facts + insights here)
                │
                ▼
   Step 3: Trace facts DOWN to source sentences
   → Traverse fact_sources join table
   → Get the actual conversation moments facts came from
                │
                ▼
   Step 4: Session expansion via NEXT edges
   → For each source sentence, grab ±N surrounding sentences
   → Reconstructs the conversational context window
                │
                ▼
   Step 5: Score, rank, return all three layers
   → Facts (primary, scored by cosine) + Insights (enrichment)
     + Sentences with session context (evidence)
```

### 5.3 Implementation

```python
async def search(
    self,
    query: str,
    user_id: str,
    agent_id: str = None,
    depth: str = "l1",           # "l0", "l1", or "l2"
    top_k: int = 10,
    context_window: int = 3,     # ±N sentences (only used at L2)
    include_superseded: bool = False  # for historical queries
) -> list[dict]:
    """
    Retrieve relevant memories for a query.

    depth="l0": facts only (cheapest)
    depth="l1": facts + insights (default)
    depth="l2": facts + insights + sentences + session context (richest)
    """
    query_embedding = await self.embedder.embed(query)

    # ── Step 1: Vector search over FACTS ──
    # This is the entry point. Facts are short + crisp = best cosine match.
    seed_facts = await self.db.search_facts(
        embedding=query_embedding,
        user_id=user_id,
        agent_id=agent_id,
        limit=top_k,
        active_only=not include_superseded
    )

    if not seed_facts:
        return []

    scored_facts = self._score_and_rank(seed_facts, query_embedding)

    if depth == "l0":
        return {"facts": scored_facts[:top_k]}

    # ── Step 2: Discover INSIGHTS linked to matched facts ──
    # Graph traversal through insight_facts, NOT vector search.
    seed_fact_ids = [f["id"] for f in scored_facts[:top_k]]
    related_insights = await self.db.get_insights_from_facts(
        fact_ids=seed_fact_ids,
        active_only=True
    )

    if depth == "l1":
        return {
            "facts": scored_facts[:top_k],
            "insights": related_insights
        }

    # ── Step 3: Trace facts down to source sentences ──
    source_sentence_ids = await self.db.get_source_sentences(seed_fact_ids)

    # ── Step 4: Session expansion via NEXT edges ──
    expanded_sentences = await self.db.expand_session_context(
        sentence_ids=source_sentence_ids,
        window=context_window
    )

    # ── L2: Return all three layers ──
    return {
        "facts": scored_facts[:top_k],
        "insights": related_insights,
        "sentences": expanded_sentences
    }
```

### 5.3 Scoring & Ranking

```python
def _score_and_rank(
    self,
    facts: list[dict],
    query_embedding,
    temporal_decay_rate: float = 0.001  # per day
) -> list[dict]:
    """
    Score facts by combining vector similarity, specificity, and recency.
    """
    now = datetime.utcnow()
    scored = []

    for fact in facts:
        # Base score: cosine similarity to query
        similarity = 1 - fact["distance"]  # pgvector returns distance

        # Confidence: LLM extraction confidence
        confidence = fact.get("confidence", 1.0)

        # Temporal decay: newer facts score higher
        age_days = (now - fact["created_at"]).total_seconds() / 86400
        recency = math.exp(-temporal_decay_rate * age_days)

        # Combined score
        fact["score"] = similarity * confidence * recency
        scored.append(fact)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
```

### 5.4 The Key SQL Query (L2 — All Three Layers in One Round Trip)

For maximum performance, the full L2 retrieval can run as a single SQL query:

```sql
-- Single query: facts → insights → source sentences → session expansion
WITH
-- Step 1: Seed facts via vector similarity (L0)
seed_facts AS (
    SELECT id, text, confidence, created_at,
           embedding <=> $1::vector AS distance
    FROM facts
    WHERE user_id = $2
      AND ($3::text IS NULL OR agent_id = $3)
      AND is_active = true
    ORDER BY embedding <=> $1::vector
    LIMIT $4
),

-- Step 2: Insights linked to matched facts (L1)
related_insights AS (
    SELECT DISTINCT i.id, i.text, i.confidence, i.created_at
    FROM insights i
    JOIN insight_facts inf ON i.id = inf.insight_id
    WHERE inf.fact_id IN (SELECT id FROM seed_facts)
      AND i.is_active = true
),

-- Step 3: Source sentences for matched facts (L2)
source_sentences AS (
    SELECT DISTINCT s.id, s.text, s.session_id, s.turn_number,
           s.sentence_index, s.created_at
    FROM sentences s
    JOIN fact_sources fs ON s.id = fs.sentence_id
    WHERE fs.fact_id IN (SELECT id FROM seed_facts)
),

-- Step 4: Session expansion (±N surrounding sentences)
expanded_sentences AS (
    SELECT s2.id, s2.text, s2.session_id, s2.turn_number,
           s2.sentence_index, s2.created_at
    FROM source_sentences src
    JOIN sentences s2 ON s2.session_id = src.session_id
    WHERE s2.sentence_index BETWEEN src.sentence_index - $5
                                AND src.sentence_index + $5
)

-- Return all three layers
SELECT 'fact' AS layer, id, text, confidence, created_at, distance
    FROM seed_facts
UNION ALL
SELECT 'insight' AS layer, id, text, confidence, created_at, NULL
    FROM related_insights
UNION ALL
SELECT 'sentence' AS layer, id, text, NULL, created_at, NULL
    FROM expanded_sentences;
```

One database, one round trip, all three layers. The insight discovery happens through a simple JOIN on `insight_facts` — no vector search on insights, no separate graph database, no KNN computation.

---

## 6. Conflict Resolution — Full Design

### 6.1 Types of Conflicts

**Direct contradiction:** "I prefer email" vs "Only contact me on WhatsApp"
→ Old fact gets `is_active=false`, `superseded_by` → new fact

**Temporal update:** "I live in Mumbai" vs "I moved to Bangalore"
→ Same as contradiction. Old fact superseded. Both retain sentence links for history.

**Partial update:** "I work at Google" vs "I got promoted to Senior Engineer at Google"
→ Not a contradiction — additive. Both facts stay active. The new fact adds detail.

**Ambiguous/conflicting signals:** "I like coffee" in one session, "I prefer tea" in another
→ LLM extraction should flag uncertainty with lower confidence score. Both facts can coexist if they have different session contexts. If the LLM determines it's a genuine preference change, treat as contradiction.

### 6.2 Contradiction Detection Prompt

The fact extraction prompt (Section 4.6) already includes existing facts and asks the LLM to check for contradictions. The key is providing enough existing facts as context without overwhelming the prompt.

**Strategy:** Retrieve the 50 most relevant existing facts for the user (via vector similarity to the new conversation), not ALL facts. This keeps the prompt manageable and focused on likely contradictions.

### 6.3 Conflict Resolution in the API

```python
# Users can query conflict history
async def get_fact_history(self, user_id: str, fact_id: str) -> list[dict]:
    """Get the full chain of superseded facts leading to the current one."""
    return await self.db.get_supersession_chain(fact_id)

# Example return:
# [
#   {"text": "User prefers email", "created_at": "2026-01-15", "is_active": false,
#    "superseded_by": "fact-456"},
#   {"text": "User prefers WhatsApp", "created_at": "2026-03-20", "is_active": true}
# ]
```

---

## 7. Developer API Design

### 7.1 Core Interface

```python
from vektori import Vektori

# Initialize — uses SQLite by default, zero config
v = Vektori(
    embedding_model="openai:text-embedding-3-small",     # or "ollama:...", "anthropic:..."
    extraction_model="openai:gpt-4o-mini",               # for fact + insight extraction
)

# ── Store memories ──
await v.add(
    messages=[
        {"role": "user", "content": "I only use WhatsApp, please don't email me"},
        {"role": "assistant", "content": "Understood, I'll contact you on WhatsApp."}
    ],
    session_id="call-005",
    user_id="borrower-123",
    agent_id="debt-collector-v2"  # optional
)

# ── Search: L0 (facts only — cheapest) ──
results = await v.search(
    query="How does this borrower prefer to communicate?",
    user_id="borrower-123",
    depth="l0"
)
# Returns:
# {
#   "facts": [
#     {"text": "Borrower prefers WhatsApp", "score": 0.94, "confidence": 0.95}
#   ]
# }

# ── Search: L1 (facts + insights — default) ──
results = await v.search(
    query="How should I approach this borrower?",
    user_id="borrower-123",
    depth="l1"
)
# Returns:
# {
#   "facts": [
#     {"text": "Borrower prefers WhatsApp", "score": 0.94},
#     {"text": "Agent offered email in call 2", "score": 0.82}
#   ],
#   "insights": [
#     {"text": "Offering email after borrower states WhatsApp preference correlates with disconnection",
#      "confidence": 0.85}
#   ]
# }

# ── Search: L2 (full story — richest) ──
results = await v.search(
    query="What happened with communication in past calls?",
    user_id="borrower-123",
    depth="l2",
    context_window=3
)
# Returns:
# {
#   "facts": [...],
#   "insights": [...],
#   "sentences": [
#     {"text": "I only use WhatsApp, please don't email me",
#      "session_id": "call-005", "turn_number": 3},
#     {"text": "The agent offered to send details via email",
#      "session_id": "call-002", "turn_number": 12},
#     ...surrounding session context via NEXT edges...
#   ]
# }

# ── Get session history ──
session = await v.get_session("call-005", user_id="borrower-123")

# ── Get all active facts for a user ──
facts = await v.get_facts(user_id="borrower-123")

# ── Get all insights for a user ──
insights = await v.get_insights(user_id="borrower-123")

# ── Get fact change history ──
history = await v.get_fact_history(user_id="borrower-123", fact_id="fact-uuid")

# ── Delete user data (GDPR) ──
await v.delete_user("borrower-123")
```

### 7.2 Configuration

```python
from vektori import Vektori, QualityConfig

v = Vektori(
    # Database
    database_url="postgresql://localhost:5432/vektori",

    # Embedding provider
    embedding_model="openai:text-embedding-3-small",
    embedding_dimension=1536,

    # LLM for fact + insight extraction
    extraction_model="openai:gpt-4o-mini",

    # Quality filtering
    quality_config=QualityConfig(
        enabled=True,           # set False to store everything
        min_chars=15,
        min_words=5,
        min_content_density=0.15,
        max_pronoun_ratio=0.40
    ),

    # Retrieval
    default_top_k=10,
    context_window=3,           # ±N sentences around matches
    temporal_decay_rate=0.001,  # per day

    # Async processing
    async_extraction=True,      # False = block until facts extracted (slower but simpler)
)
```

---

## 8. Multi-Model Support

### 8.1 Embedding Providers

```python
class EmbeddingProvider:
    """Abstract base. Implementations for each provider."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIEmbedder(EmbeddingProvider):
    """Uses text-embedding-3-small by default. Dimension: 1536."""

class AnthropicEmbedder(EmbeddingProvider):
    """Via Voyage AI embeddings."""

class OllamaEmbedder(EmbeddingProvider):
    """Local embeddings via Ollama. No API key needed."""
    # Default model: nomic-embed-text (768 dim)
    # Developers can use any Ollama-compatible embedding model

class SentenceTransformerEmbedder(EmbeddingProvider):
    """Direct HuggingFace sentence-transformers. Fully local, no Ollama needed."""
```

### 8.2 LLM Providers (for extraction)

```python
class LLMProvider:
    """Abstract base for fact extraction LLM."""

    async def generate(self, prompt: str) -> str: ...

class OpenAILLM(LLMProvider): ...    # gpt-4o-mini default
class AnthropicLLM(LLMProvider): ... # claude-3-haiku default
class OllamaLLM(LLMProvider): ...    # llama3 default
```

**Important for OSS adoption:** Support Ollama out of the box. Developers who don't want to pay for API keys should be able to run Vektori completely locally with Ollama for both embeddings and extraction. This is a huge advantage over Mem0 which requires OpenAI by default.

### 8.3 Model Factory (Config-Driven Instantiation)

Model providers are resolved automatically from a string identifier. Users never import provider classes directly.

```python
# vektori/models/factory.py

EMBEDDING_REGISTRY = {
    "openai": OpenAIEmbedder,
    "anthropic": AnthropicEmbedder,   # via Voyage AI
    "ollama": OllamaEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
}

LLM_REGISTRY = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "ollama": OllamaLLM,
}

def create_embedder(model_string: str, **kwargs) -> EmbeddingProvider:
    """
    Factory: resolve 'openai:text-embedding-3-small' into an OpenAIEmbedder instance.
    Format: 'provider:model_name' or just 'provider' for default model.
    """
    provider, _, model_name = model_string.partition(":")
    if provider not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding provider: {provider}. "
                         f"Available: {list(EMBEDDING_REGISTRY.keys())}")
    cls = EMBEDDING_REGISTRY[provider]
    return cls(model=model_name or None, **kwargs)

def create_llm(model_string: str, **kwargs) -> LLMProvider:
    """Same pattern for LLM providers."""
    provider, _, model_name = model_string.partition(":")
    if provider not in LLM_REGISTRY:
        raise ValueError(f"Unknown LLM provider: {provider}. "
                         f"Available: {list(LLM_REGISTRY.keys())}")
    cls = LLM_REGISTRY[provider]
    return cls(model=model_name or None, **kwargs)
```

Users interact with a simple string:
```python
v = Vektori(embedding_model="ollama:nomic-embed-text", extraction_model="ollama:llama3")
```

The factory resolves the right class, passes the right config. Adding a new provider = one class + one registry entry. No user-facing API change.

---

## 9. Storage Backend Abstraction

### 9.1 Design Principle

Inspired by OpenViking's backend-agnostic storage pattern: one interface, multiple backends, swapped via config. The same API works whether you're running locally on SQLite or in production on Postgres.

### 9.2 Backend Interface

```python
# vektori/storage/base.py
from abc import ABC, abstractmethod

class StorageBackend(ABC):
    """Abstract storage interface. All backends implement this."""

    # ── Sentences ──
    @abstractmethod
    async def upsert_sentences(self, sentences: list[dict], embeddings: list, 
                                user_id: str, agent_id: str = None) -> int: ...

    @abstractmethod
    async def search_sentences(self, embedding, user_id: str, agent_id: str = None,
                                limit: int = 10) -> list[dict]: ...

    # ── Facts ──
    @abstractmethod
    async def insert_fact(self, **kwargs) -> str: ...

    @abstractmethod
    async def search_facts(self, embedding, user_id: str, agent_id: str = None,
                            limit: int = 10, active_only: bool = True) -> list[dict]: ...

    @abstractmethod
    async def deactivate_fact(self, fact_id: str, superseded_by: str = None) -> None: ...

    # ── Insights ──
    @abstractmethod
    async def insert_insight(self, **kwargs) -> str: ...

    @abstractmethod
    async def get_insights_from_facts(self, fact_ids: list[str],
                                       active_only: bool = True) -> list[dict]: ...

    # ── Edges ──
    @abstractmethod
    async def insert_edges(self, edges: list[dict]) -> int: ...

    @abstractmethod
    async def expand_session_context(self, sentence_ids: list[str], 
                                      window: int = 3) -> list[dict]: ...

    # ── Links ──
    @abstractmethod
    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None: ...

    @abstractmethod
    async def insert_insight_fact(self, insight_id: str, fact_id: str) -> None: ...

    @abstractmethod
    async def insert_insight_source(self, insight_id: str, sentence_id: str) -> None: ...

    @abstractmethod
    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]: ...

    @abstractmethod
    async def get_facts_from_sentences(self, sentence_ids: list[str],
                                        exclude_ids: list[str] = None) -> list[dict]: ...

    # ── Lifecycle ──
    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def delete_user(self, user_id: str) -> int: ...
```

### 9.3 Backend Implementations

| Backend | Use Case | Dependencies | Vector Search |
|---------|----------|-------------|---------------|
| `sqlite` | Zero-config dev, quick prototyping | None (built-in) | sqlite-vec or brute-force cosine |
| `postgres` | Production, scale, concurrent access | PostgreSQL + pgvector | IVFFlat / HNSW index |
| `memory` | Unit tests, CI | None | Brute-force cosine |

### 9.4 Resolution

```python
STORAGE_REGISTRY = {
    "sqlite": SQLiteBackend,
    "postgres": PostgresBackend,
    "memory": MemoryBackend,
}

# Default: SQLite, zero config, no Docker needed
v = Vektori()  # uses ~/.vektori/vektori.db automatically

# Production: Postgres
v = Vektori(database_url="postgresql://localhost:5432/vektori")

# Testing: In-memory
v = Vektori(storage_backend="memory")
```

### 9.5 SQLite Backend Notes

SQLite + sqlite-vec gives you vector search without any external database. The developer experience becomes:

```bash
pip install vektori
python -c "from vektori import Vektori; v = Vektori(); print('ready')"
```

No Docker. No Postgres. No setup. This is the single biggest adoption lever — the easier the first 5 minutes, the more people try it. Postgres is recommended for production (concurrent writes, better indexing) but SQLite gets people started.

For the SQLite backend, vector search uses either sqlite-vec (if available) or brute-force cosine similarity over numpy arrays. Brute-force is fine for <10K sentences. At larger scale, the developer upgrades to Postgres.

---

## 10. Docker Setup (Optional — For Postgres Users)

Most developers won't need Docker. SQLite is the default and requires no setup. Docker is for developers who want Postgres for production-grade performance.

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: vektori
      POSTGRES_USER: vektori
      POSTGRES_PASSWORD: vektori
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  pgdata:
```

**Zero-config path (most developers):**
```bash
pip install vektori
python examples/quickstart.py    # uses SQLite, just works
```

**Postgres path (production):**
```bash
git clone https://github.com/vektori-ai/vektori.git
cd vektori
docker compose up -d             # postgres with pgvector
pip install vektori
python examples/quickstart_postgres.py
```

No Neo4j. No Qdrant. No Redis. No Modal. SQLite by default, Postgres when you're ready.

---

## 11. Repo Structure

```
vektori/
├── vektori/                      # Core Python package
│   ├── __init__.py               # exports Vektori class
│   ├── client.py                 # Main Vektori class (add, search, get_session, etc.)
│   ├── config.py                 # Configuration dataclasses
│   │
│   ├── ingestion/
│   │   ├── splitter.py           # Sentence splitting + merging
│   │   ├── filter.py             # Quality filtering gauntlet
│   │   ├── hasher.py             # Deterministic ID generation
│   │   └── extractor.py          # LLM fact extraction + conflict detection
│   │
│   ├── retrieval/
│   │   ├── search.py             # Main search pipeline (L0/L1/L2)
│   │   ├── scoring.py            # Scoring + temporal decay + ranking
│   │   └── expansion.py          # Session context expansion
│   │
│   ├── storage/
│   │   ├── base.py               # Abstract StorageBackend interface
│   │   ├── sqlite.py             # SQLite + sqlite-vec (zero-config default)
│   │   ├── postgres.py           # PostgreSQL + pgvector (production)
│   │   ├── memory.py             # In-memory (unit tests)
│   │   ├── schema.sql            # Postgres table definitions
│   │   └── migrations/           # Schema version management
│   │
│   ├── models/
│   │   ├── base.py               # Abstract embedding + LLM provider interfaces
│   │   ├── factory.py            # Config-driven provider resolution
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── ollama.py
│   │   └── sentence_transformers.py
│   │
│   └── utils/
│       ├── async_worker.py       # Background fact extraction worker
│       └── logging.py
│
├── examples/
│   ├── quickstart.py             # 10-line getting started (SQLite, zero config)
│   ├── quickstart_postgres.py    # Same but with Postgres backend
│   ├── crewai_integration.py     # Memory for CrewAI agents
│   ├── langgraph_integration.py  # Memory for LangGraph
│   ├── openai_agent.py           # Memory for basic OpenAI agent
│   └── ollama_local.py           # Fully local setup, no API keys
│
├── benchmarks/
│   ├── locomo/                   # LoCoMo benchmark runner
│   ├── longmemeval/              # LongMemEval-S benchmark runner
│   └── run_benchmarks.py
│
├── tests/
│   ├── unit/
│   │   ├── test_splitter.py
│   │   ├── test_filter.py
│   │   ├── test_hasher.py
│   │   ├── test_scoring.py
│   │   └── test_factory.py
│   ├── integration/
│   │   ├── test_ingestion.py
│   │   ├── test_retrieval.py
│   │   ├── test_conflicts.py
│   │   └── test_tiered.py
│   └── fixtures/
│       ├── sample_conversations.json
│       └── expected_facts.json
│
├── scripts/
│   ├── init.sql                  # Postgres database initialization
│   └── migrate.py                # Schema migrations
│
├── docker-compose.yml            # Optional: Postgres for production users
├── pyproject.toml                # Package config + Ruff + mypy + pytest
├── .pre-commit-config.yaml       # Ruff hooks
├── README.md
├── CONTRIBUTING.md
├── LICENSE                       # MIT
└── .github/
    └── workflows/
        └── ci.yml                # Lint + test on every PR
```

---

## 12. Code Quality & Development Practices

### 11.1 Tooling (Set Up Before Writing Any Code)

| Tool | Purpose | Why |
|------|---------|-----|
| **Ruff** | Linting + formatting + import sorting | Fast (Rust-based), replaces black + isort + flake8 in one tool |
| **mypy** | Type checking | Catches bugs early, signals code quality to contributors |
| **pre-commit** | Git hooks that run checks before every commit | Prevents bad code from entering the repo |
| **pytest** | Testing framework | Standard, good async support with pytest-asyncio |

### 11.2 pyproject.toml Config

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]  # errors, pyflakes, isort, naming, warnings, pyupgrade

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false     # start lenient, tighten over time

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 11.3 Pre-commit Config

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Set up on day 1:
```bash
pip install pre-commit
pre-commit install
```

### 11.4 Style Guidelines

- **Line width:** 100 characters
- **Indentation:** 4 spaces
- **Strings:** Double quotes
- **Type hints:** Required for all public API methods, encouraged elsewhere
- **Docstrings:** Required for all public classes and methods. Keep them 1-3 lines. Use Google style.
- **Naming:** snake_case for functions/variables, PascalCase for classes. No abbreviations in public API.

### 11.5 Testing Strategy

```
tests/
├── unit/                    # No external dependencies, fast
│   ├── test_splitter.py     # Sentence splitting + merging
│   ├── test_filter.py       # Quality gauntlet
│   ├── test_hasher.py       # Deterministic ID generation
│   ├── test_scoring.py      # Score calculation + temporal decay
│   └── test_factory.py      # Model + storage factory resolution
│
├── integration/             # Requires database (use memory backend in CI)
│   ├── test_ingestion.py    # Full add() pipeline
│   ├── test_retrieval.py    # Full search() pipeline
│   ├── test_conflicts.py    # Contradiction detection + resolution
│   └── test_tiered.py       # L0/L1/L2 retrieval depth
│
└── fixtures/
    ├── sample_conversations.json
    └── expected_facts.json
```

**CI rule:** All tests must pass on every PR. Unit tests use memory backend (no Docker). Integration tests use memory backend in CI, Postgres locally.

### 11.6 Commit Convention

Use conventional commits for clean changelog generation:

```
feat: add SQLite storage backend
fix: handle empty conversation in sentence splitter
docs: add CrewAI integration example
refactor: extract model factory from client.py
test: add conflict resolution integration tests
```

### 11.7 Versioning

Use semantic versioning with `setuptools-scm` to auto-generate version from git tags:

```bash
git tag v0.1.0
# pyproject.toml picks it up automatically
```

---

## 13. Build Timeline

### Phase 1: Core Engine (Weeks 1-2)
**Goal:** `pip install vektori` → add memories → search memories. Working end-to-end.

- [ ] **Day 1: Repo scaffolding + tooling (before any feature code)**
  - [ ] pyproject.toml with Ruff, mypy, pytest config
  - [ ] .pre-commit-config.yaml + install hooks
  - [ ] .github/workflows/ci.yml (run tests + lint on every PR)
  - [ ] LICENSE (MIT), CONTRIBUTING.md skeleton
- [ ] Storage backend abstraction (base.py interface)
- [ ] SQLite backend (zero-config default)
- [ ] PostgreSQL + pgvector backend (production)
- [ ] Memory backend (unit tests)
- [ ] Model factory (config-driven provider resolution)
- [ ] Docker compose (one-command Postgres for users who want it)
- [ ] Sentence splitter (spacy + merge logic)
- [ ] Quality filter (port 10-layer gauntlet)
- [ ] Deterministic ID generation
- [ ] OpenAI embedding provider
- [ ] Basic `add()` — split → filter → embed → store sentences + NEXT edges
- [ ] Basic `search()` — embed query → vector search → return results (L0)
- [ ] `quickstart.py` example working
- [ ] Unit tests for splitter, filter, hasher, factory

### Phase 2: Fact Layer + Extraction (Weeks 3-4)
**Goal:** LLM extracts facts from conversations. Facts become the primary retrieval unit.

- [ ] LLM provider abstraction + OpenAI implementation
- [ ] Fact extraction prompt + JSON parsing
- [ ] Async extraction worker (non-blocking ingestion)
- [ ] Conflict detection + resolution (superseded_by chain)
- [ ] Fact → sentence linking (fact_sources table)
- [ ] Tiered retrieval: L0 (facts only), L1 (facts + insights via graph), L2 (facts + insights + sentences + session context)
- [ ] Updated `search()`: vector search over facts → trace to sentences → expand context
- [ ] `get_facts()`, `get_fact_history()`, `get_session()` APIs
- [ ] Anthropic + Ollama model providers
- [ ] `ollama_local.py` example (zero API keys)
- [ ] Integration tests for full ingestion + retrieval + conflict resolution

### Phase 3: Polish + Integrations (Weeks 5-6)
**Goal:** Developer-ready. Good README. Framework integrations. Benchmarks.

- [ ] Sentence-transformer embedding provider (fully local, no Ollama needed)
- [ ] CrewAI integration example
- [ ] LangGraph integration example
- [ ] LoCoMo benchmark runner + results
- [ ] LongMemEval-S benchmark runner + results
- [ ] README with architecture diagram, quickstart, API docs
- [ ] CONTRIBUTING.md
- [ ] CI pipeline (GitHub Actions — tests on PR)
- [ ] `delete_user()` for GDPR compliance

### Phase 4: Launch (Weeks 7-8)
**Goal:** Ship it. Get eyes on it.

- [ ] Write launch blog post (the Riverline trajectory analysis angle)
- [ ] Prepare HN "Show HN" post
- [ ] Reddit posts (r/MachineLearning, r/LangChain, r/LocalLLaMA)
- [ ] Dev.to article
- [ ] Twitter/X launch thread
- [ ] Product Hunt listing (optional)
- [ ] MagicBall presentation (ping Sid)

---

## 14. Future: Phase 2 Features (Post-Launch)

These are NOT in v1. Design the schema to support them but don't build them yet.

### 14.1 Entity Co-Reference Edges (v2)
Extract entities during fact extraction. Link sentences that share entities across sessions. This becomes the smart cross-session bridge replacing KNN.

### 14.2 Causal Edges (v2)
Extract causal relationships: "agent did X → resulted in Y". This enables trajectory matching — the Riverline demo.

### 14.3 Trajectory Library (v2)
Match current conversation patterns to past trajectories. "This call is following a pattern where 4/6 similar calls disconnected. Here's what the 2 that recovered did differently."

### 14.4 Personalized PageRank (Optional v2)
Bring back PPR over the enriched graph (entity edges + causal edges). With typed, sparse edges instead of dense KNN, PPR will actually work as intended — propagating relevance through meaningful connections.

---

## 15. What We're NOT Building (Scope Control)

- ❌ UI/Dashboard (not for v1)
- ❌ Multi-tenant auth (commercial feature)
- ❌ Real-time streaming ingestion
- ❌ MCP server (v2, but design API to be MCP-compatible)
- ❌ REST API server (SDK-first, server can come later)
- ❌ Agent framework (we're memory infra, not an agent runtime)

---

## 16. Open Questions to Resolve During Build

1. **Embedding dimension:** Default to 1536 (OpenAI) but need to handle mixed dimensions if users switch providers. Pad? Separate indexes?

2. **Async extraction implementation:** Python `asyncio` tasks? Celery? Simple threading? For v1, probably just `asyncio.create_task` — keep it simple.

3. **Schema migrations:** How to handle schema changes between versions. Use Alembic? Or simple numbered SQL scripts?

4. **Rate limiting for LLM extraction:** If a user dumps 1000 messages at once, we don't want to fire 1000 LLM calls. Batch sessions? Queue?

5. **Embedding caching:** Cache embeddings to avoid recomputing for the same text? Probably yes — the deterministic ID gives us a natural cache key.

6. **Testing strategy:** Unit tests for splitter/filter, integration tests with real Postgres, mock LLM calls for extraction tests. Need test fixtures.

---

*Let's build this thing. — L & M*


also bro some things opus pointed out when i gave it ur plan
Validation: What Needs Attention
1. The fact extraction prompt reliability is your biggest risk.
The entire system depends on the LLM extracting good facts, linking them to correct source quotes, and detecting contradictions reliably. In Section 4.6, you're asking the LLM to do a LOT in one prompt: extract facts with confidence scores, extract insights with confidence scores, find source quotes for both, AND check contradictions against 50 existing facts. This is a complex structured-output task.
The risk: LLMs are inconsistent at multi-task structured output. Some runs will miss contradictions. Some will extract garbage facts. Some will hallucinate source quotes that don't match any sentence. The find_sentences_by_similarity(source_quotes, threshold=0.75) on line 698-699 partially mitigates this (fuzzy matching instead of exact), but you should expect ~10-20% of fact-to-sentence links to be wrong or missing in practice.
Recommendation: Split the extraction into two LLM calls — one for facts, one for insights. Simpler prompts get more reliable outputs. Also consider a validation step where you verify that source_quotes actually exist in the conversation (string matching before falling back to embedding similarity).

2. The sentence splitter choice (spaCy) adds dependency weight.
You're going from NLTK (lightweight) to spaCy (en_core_web_sm is ~12MB download, spaCy itself is ~50MB). For a library that sells "zero config, pip install, just works," that's meaningful. The merge logic on lines 419-431 is good but spaCy might be overkill for just sentence splitting.
Recommendation: Default to a lighter splitter. Options: NLTK punkt (you already know it works), or PySBD (rule-based, ~1MB, specifically designed for sentence boundary detection), or SaT/wtpsplit (neural, ~15MB, best on messy text). Reserve spaCy as an optional backend if the user already has it installed for other reasons.
3. The sentence_index in session expansion has a subtle bug.
In the L2 SQL query (line 946-952), session expansion joins on sentence_index BETWEEN src.sentence_index - $5 AND src.sentence_index + $5. But sentence_index is the index within a turn, not within a session. Two different turns could have sentence_index = 0. You'd need to join on both session_id AND order by (turn_number, sentence_index) to get correct sequential context.
The NEXT edges actually solve this correctly (line 579-588 creates sequential edges between consecutive sentences). For session expansion, traversing NEXT edges N hops in both directions would be more reliable than the index arithmetic.
4. The MIT license in the spec (line 1471) contradicts your OSS launch plan.
Your launch plan says Apache 2.0. The spec says MIT. Both are permissive, but Apache 2.0 is better for enterprise adoption (explicit patent grant) and matches every competitor (Mem0, OpenViking, Graphiti). Change this to Apache 2.0.
5. Insight generation timing is underspecified.
Facts are extracted per-session (when add() is called). But insights are supposed to be cross-session patterns. When do insights get generated? On every add() call? That means the LLM only sees the current conversation + existing facts — it can't see patterns across sessions it hasn't been shown.
For insights to be truly cross-session ("Offering email after WhatsApp preference → disconnection"), the LLM needs to see facts from multiple sessions. The current prompt (line 621-666) only sends the current conversation + existing facts text. It doesn't send the actual cross-session pattern data.
Recommendation: Either generate insights on a separate schedule (after N sessions, or periodically) with a prompt that explicitly shows facts from multiple sessions, or pass recent session summaries into the extraction prompt.
6. The find_fact_by_text for contradiction linking is fragile.
On line 679-682, you look up the contradicted fact by matching the text the LLM said it contradicts. But the LLM might paraphrase: the existing fact is "User prefers email" and the LLM says contradicts: "User likes email". Text matching would miss this. Use embedding similarity to find the contradicted fact, not text matching.
7. SQLite + vector search needs more detail.
You mention sqlite-vec or brute-force, but sqlite-vec is relatively new and might not support all the operations your StorageBackend interface requires (like filtered vector search with user_id AND agent_id AND is_active). Brute-force cosine works for <10K sentences but you should be explicit about this limitation in docs, and test the SQLite backend thoroughly since it's the one most users will hit first.


8. No rate limiting for LLM extraction. (idk about this)
Section 16 flags this as an open question. The answer: batch sessions. If someone calls add() 100 times in a minute, don't fire 100 LLM calls. Queue extraction requests and batch them (e.g., extract facts from up to 5 sessions in one LLM call, or debounce with a 5-second window).

BTW it also said
The single Postgres query for L2 retrieval (line 911-963) is elegant. One round trip, all three layers. CTEs chained together. No application-level graph traversal, no multiple DB calls. This is production-quality SQL.