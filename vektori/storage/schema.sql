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

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_facts_embedding ON facts
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts (user_id);
CREATE INDEX IF NOT EXISTS idx_facts_user_agent ON facts (user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_facts_active ON facts (user_id, is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (user_id, subject) WHERE subject IS NOT NULL;


-- ============================================================
-- INSIGHTS: The middle layer (L1). LLM-inferred cross-session patterns.
-- NOT a vector search target. Discovered via graph traversal from facts.
-- ============================================================
CREATE TABLE IF NOT EXISTS insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    embedding vector(1536),    -- stored but NOT indexed — not searched directly

    user_id TEXT NOT NULL,
    agent_id TEXT,

    confidence FLOAT DEFAULT 1.0,
    is_active BOOLEAN DEFAULT true,

    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- NOTE: No vector index on insights — they're discovered via graph, not search.
CREATE INDEX IF NOT EXISTS idx_insights_user ON insights (user_id);
CREATE INDEX IF NOT EXISTS idx_insights_user_agent ON insights (user_id, agent_id);


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
-- INSIGHT_SOURCES: Links insights (L1) to source sentences (L2).
-- ============================================================
CREATE TABLE IF NOT EXISTS insight_sources (
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    PRIMARY KEY (insight_id, sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_insight_sources_insight ON insight_sources (insight_id);
CREATE INDEX IF NOT EXISTS idx_insight_sources_sentence ON insight_sources (sentence_id);


-- ============================================================
-- INSIGHT_FACTS: Links insights (L1) to related facts (L0).
-- THE KEY BRIDGE.
-- Vector search finds facts → JOIN here → discovers insights.
-- This is how the insight layer is found without vector search.
-- ============================================================
CREATE TABLE IF NOT EXISTS insight_facts (
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (insight_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_insight_facts_insight ON insight_facts (insight_id);
CREATE INDEX IF NOT EXISTS idx_insight_facts_fact ON insight_facts (fact_id);


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
