-- Vektori PostgreSQL initialization script
-- Used by Docker (docker-entrypoint-initdb.d) and manual setup
-- Idempotent — safe to run multiple times (IF NOT EXISTS everywhere)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS sentences (
    id UUID PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1536),
    user_id TEXT NOT NULL,
    agent_id TEXT,
    session_id TEXT NOT NULL,
    turn_number INTEGER NOT NULL,
    sentence_index INTEGER NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    content_hash TEXT NOT NULL,
    mentions INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sentences_content_hash ON sentences (content_hash);
CREATE INDEX IF NOT EXISTS idx_sentences_embedding ON sentences
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_sentences_user ON sentences (user_id);
CREATE INDEX IF NOT EXISTS idx_sentences_session ON sentences (session_id);
CREATE INDEX IF NOT EXISTS idx_sentences_user_agent ON sentences (user_id, agent_id);

CREATE TABLE IF NOT EXISTS facts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    embedding vector(1536),
    user_id TEXT NOT NULL,
    agent_id TEXT,
    is_active BOOLEAN DEFAULT true,
    superseded_by UUID REFERENCES facts(id),
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

CREATE TABLE IF NOT EXISTS insights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    embedding vector(1536),
    user_id TEXT NOT NULL,
    agent_id TEXT,
    confidence FLOAT DEFAULT 1.0,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_insights_user ON insights (user_id);
CREATE INDEX IF NOT EXISTS idx_insights_user_agent ON insights (user_id, agent_id);

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

CREATE TABLE IF NOT EXISTS fact_sources (
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_sources_fact ON fact_sources (fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_sources_sentence ON fact_sources (sentence_id);

CREATE TABLE IF NOT EXISTS insight_sources (
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    sentence_id UUID NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    PRIMARY KEY (insight_id, sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_insight_sources_insight ON insight_sources (insight_id);
CREATE INDEX IF NOT EXISTS idx_insight_sources_sentence ON insight_sources (sentence_id);

CREATE TABLE IF NOT EXISTS insight_facts (
    insight_id UUID NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
    fact_id UUID NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    PRIMARY KEY (insight_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_insight_facts_insight ON insight_facts (insight_id);
CREATE INDEX IF NOT EXISTS idx_insight_facts_fact ON insight_facts (fact_id);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_id TEXT,
    started_at TIMESTAMPTZ DEFAULT now(),
    ended_at TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);
