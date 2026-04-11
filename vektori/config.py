"""Configuration dataclasses for Vektori."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityConfig:
    """Configuration for the sentence quality filter."""

    enabled: bool = True
    min_chars: int = 10
    min_words: int = 3
    min_content_density: float = 0.15
    max_pronoun_ratio: float = 0.40


@dataclass
class VektoriConfig:
    """Full configuration for a Vektori instance."""

    # Storage
    database_url: str | None = None  # None = SQLite default (~/.vektori/vektori.db)
    storage_backend: str = "sqlite"  # "sqlite", "postgres", "memory", "neo4j", "qdrant"
    qdrant_api_key: str | None = None  # Qdrant Cloud API key (not needed for local)

    # Embedding provider — format: "provider:model_name"
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dimension: int = 1536

    # LLM provider for fact + episode extraction
    extraction_model: str = "openai:gpt-4o-mini"

    # Quality filtering
    quality_config: QualityConfig = field(default_factory=QualityConfig)

    # Retrieval defaults
    default_top_k: int = 15
    context_window: int = 3  # ±N sentences around matched sentence (L2)
    temporal_decay_rate: float = 0.001  # per day

    # Extraction limits (per extraction batch)
    max_facts: int = 15  # max facts the LLM may return per session (prompt-level)

    # Token-threshold batching — fire extraction once buffered input exceeds this
    token_batch_threshold: int = 800  # ~800 tokens ≈ 3-4 turns before extraction fires

    # Hard token limits for extraction LLM calls (API-level, not prompt hints)
    max_extraction_input_tokens: int = 4000  # truncate conversation before sending to LLM
    max_extraction_output_tokens: int = (
        8192  # headroom for dense sessions that ignore max_facts limit
    )

    # Retrieval gate — cheap heuristic, no LLM, runs before any DB query
    enable_retrieval_gate: bool = True

    # Query expansion — LLM-generated paraphrase variants (expand=True in search())
    expansion_queries: int = 2  # variants to generate (total searches = this + 1 original)

    # Min score floor — facts below this are dropped from results (0.0 = disabled)
    min_retrieval_score: float = 0.3

    # Episode generation
    max_episodes: int = 3  # max episodic narratives generated per session
    max_episode_input_tokens: int = 1500
    max_episode_output_tokens: int = 1024

    # Processing
    async_extraction: bool = True  # False = block until facts extracted (slower but simpler)
