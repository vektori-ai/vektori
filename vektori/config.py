"""Configuration dataclasses for Vektori."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QualityConfig:
    """Configuration for the sentence quality filter."""

    enabled: bool = True
    min_chars: int = 15
    min_words: int = 5
    min_content_density: float = 0.15
    max_pronoun_ratio: float = 0.40


@dataclass
class VektoriConfig:
    """Full configuration for a Vektori instance."""

    # Storage
    database_url: str | None = None          # None = SQLite default (~/.vektori/vektori.db)
    storage_backend: str = "sqlite"          # "sqlite", "postgres", "memory"

    # Embedding provider — format: "provider:model_name"
    embedding_model: str = "openai:text-embedding-3-small"
    embedding_dimension: int = 1536

    # LLM provider for fact + insight extraction
    extraction_model: str = "openai:gpt-4o-mini"

    # Quality filtering
    quality_config: QualityConfig = field(default_factory=QualityConfig)

    # Retrieval defaults
    default_top_k: int = 10
    context_window: int = 3             # ±N sentences around matched sentence (L2)
    temporal_decay_rate: float = 0.001  # per day

    # Processing
    async_extraction: bool = True       # False = block until facts extracted (slower but simpler)
