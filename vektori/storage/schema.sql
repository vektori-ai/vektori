-- Vektori PostgreSQL schema
-- Run via: docker compose up -d (uses this file via docker-entrypoint-initdb.d)
-- Or: psql $DATABASE_URL -f scripts/init.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for find_sentences_by_similarity

-- ============================================================
-- SENTENCES: The bottom layer (L2). Raw conversation, sequential flow.
-- Immutable records of what was actually said.
-- ============================================================
CREATE TABLE IF NOT EXISTS sentences (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1536),          -- configurable dimension

    user_id TEXT NOT NULL,
    agent_id TEXT,

    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,      -- position in conversation
    sentence_index INTEGER NOT NULL,   -- position within a turn
    role TEXT NOT NULL DEFAULT 'user', -- 'user' or 'assistant'

    -- Deduplication + IDF weighting
    content_hash TEXT NOT NULL,        -- SHA-256(session_id:sentence_index:text)
    mentions INTEGER DEFAULT 1,        -- incremented on re-encounter

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Deterministic: same content in same position = same ID = no duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_sentences_content_hash ON sentences (content_hash);

-- Vector index for sentence search (IVFFlat for <1M rows, switch to HNSW later)
CREATE INDEX IF NOT EXISTS idx_sentences_embedding ON sentences
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_sentences_user ON sentences (user_id);
CREATE INDEX IF NOT EXISTS idx_sentences_session ON sentences (session_id);
CREATE INDEX IF NOT EXISTS idx_sentences_user_agent ON sentences (user_id, agent_id);


-- ============================================================
-- FACTS: The top layer (L0). LLM-extracted explicit knowledge.
-- PRIMARY SEARCH SURFACE — vector search lands here.
-- ============================================================
CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    embedding vector(1536),

    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,                           -- which session produced this fact
    subject TEXT,                              -- entity this fact is about (pre-filter discriminator)

    is_active BOOLEAN DEFAULT true,
    superseded_by UUID REFERENCES facts(id),  -- conflict resolution chain
    confidence FLOAT DEFAULT 1.0,
    mentions INTEGER DEFAULT 1,               -- incremented on cross-session semantic dedup

    -- Temporal anchor: when the conversation happened (session started_at),
    -- not when extraction ran. Used for time-aware retrieval filtering.
    event_time TIMESTAMPTZ,

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_facts_event_time ON facts (user_id, event_time) WHERE event_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_facts_embedding ON facts
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts (user_id);
CREATE INDEX IF NOT EXISTS idx_facts_user_agent ON facts (user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts (user_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (user_id, subject) WHERE subject IS NOT NULL;


-- ============================================================
-- EDGES: Typed relationships between sentences.
-- 'next': sequential flow within a session
-- 'contradiction': conflicting statements (optional, future use)
-- ============================================================
CREATE TABLE IF NOT EXISTS sentence_edges (
    source_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_sentence_edges_source ON sentence_edges (source_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_sentence_edges_target ON sentence_edges (target_id, edge_type);


-- ============================================================
-- FACT_SOURCES: Links facts (L0) to source sentences (L2).
-- Vertical bridge: fact → the actual words it was extracted from.
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_sources_fact ON fact_sources (fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_sources_sentence ON fact_sources (sentence_id);


-- ============================================================
-- SESSIONS: Metadata about conversation sessions.
-- ============================================================
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);


-- ============================================================
-- ============================================================
-- SYNTHESES: The middle layer (L1). LLM-generated episodic memory narratives.
-- Discovered via graph traversal from matched facts, also directly vector-searched.
-- ============================================================
CREATE TABLE IF NOT EXISTS syntheses (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1536),               -- for direct vector search at retrieval
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,                       -- session this synthesis came from
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_syntheses_user ON syntheses (user_id);
CREATE INDEX IF NOT EXISTS idx_syntheses_embedding ON syntheses
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Dedup: same synthesis text for same user and agent scope is idempotent
CREATE UNIQUE INDEX IF NOT EXISTS idx_syntheses_user_agent_text
    ON syntheses (user_id, COALESCE(agent_id, ''), text);


-- ============================================================
-- EPISODE_FACTS: Links syntheses (L1) to the facts (L0) they were derived from.
-- Graph edge: traversed after L0 vector search to surface syntheses.
-- ============================================================
CREATE TABLE IF NOT EXISTS synthesis_facts (
    synthesis_id UUID NOT NULL REFERENCES syntheses(id) ON DELETE CASCADE,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (synthesis_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_synthesis_facts_synthesis ON synthesis_facts (synthesis_id);
CREATE INDEX IF NOT EXISTS idx_synthesis_facts_fact ON synthesis_facts (fact_id);

-- EPISODES: The middle layer (L1). LLM-generated episodic memory narratives.
-- Discovered via graph traversal from matched facts, also directly vector-searched.
-- ============================================================
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1536),               -- for direct vector search at retrieval
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT,                       -- session this episode came from
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes (user_id);
CREATE INDEX IF NOT EXISTS idx_episodes_embedding ON episodes
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- Dedup: same episode text for same user is idempotent
CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_user_text ON episodes (user_id, text);


-- ============================================================
-- EPISODE_FACTS: Links episodes (L1) to the facts (L0) they were derived from.
-- Graph edge: traversed after L0 vector search to surface episodes.
-- ============================================================
CREATE TABLE IF NOT EXISTS episode_facts (
    episode_id UUID NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (episode_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_episode_facts_episode ON episode_facts (episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_facts_fact ON episode_facts (fact_id);


-- ============================================================
-- FACT_EDGES: Similarity edges between facts for PPR graph traversal.
-- Built at write time when a new fact has cosine sim > 0.75 with existing facts.
-- Enables multi-hop retrieval: query → matched facts → related facts → episodes.
-- ============================================================
CREATE TABLE IF NOT EXISTS fact_edges (
    source_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (source_id, target_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_edges_user ON fact_edges (user_id);
