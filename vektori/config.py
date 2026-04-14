"""Configuration dataclasses for Vektori."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExtractionConfig:
    """
    Controls how Vektori's LLM extracts facts (L0) and episodes (L1).

    There are three levels of customisation — use whichever fits your needs:

    **Level 1 — Agent-type preset (zero effort)**
    Set ``agent_type`` to one of the built-in values and Vektori automatically
    biases extraction toward what matters for that agent.

    Supported values:
      ``"general"``   — default, balanced extraction (no bias)
      ``"presales"``  — pain points, budget signals, decision makers, objections,
                        competitors, product interests
      ``"sales"``     — deal stage, pricing, close dates, stakeholders, blockers,
                        next steps, legal concerns
      ``"account_management"`` — renewals, expansions, health signals,
                        escalations, stakeholder changes
      ``"support"``   — issue descriptions, error messages, resolution steps,
                        satisfaction signals, escalation triggers
      ``"onboarding"`` — implementation milestones, blockers, integrations,
                        go-live criteria, training gaps
      ``"hr"``        — employee feedback, performance, career goals, team
                        dynamics, policy questions
      ``"recruiting"`` — candidate qualifications, interview stage,
                        compensation expectations, next steps
      ``"finance"``   — invoices, budgets, reconciliation issues, fraud signals,
                        vendor financial terms
      ``"legal"``     — contract clauses, compliance requirements, liabilities,
                        disputes, filing deadlines
      ``"coding"``    — architecture decisions, bugs, APIs, code review feedback,
                        performance and security concerns
      ``"data_analytics"`` — metrics, SQL, data quality issues, dashboards,
                        ETL and reporting deliverables
      ``"research"``  — hypotheses, sources, findings, methodology choices,
                        evidence gaps and next questions
      ``"cybersecurity"`` — threats, CVEs, affected systems, IOCs,
                        remediation and compliance frameworks
      ``"healthcare"`` — symptoms, medications, triage urgency, care plans,
                        follow-up instructions
      ``"supply_chain"`` — shipments, inventory, suppliers, purchase orders,
                        logistics disruptions and cost changes
      ``"retail"``    — products, SKUs, returns, promotions, preferences,
                        loyalty signals
      ``"operations"`` — scheduling, capacity constraints, SLAs, milestones,
                        resource allocation and bottlenecks

    **Level 2 — Domain hints (low effort)**
    ``focus_on`` and ``ignore`` accept plain-English list items that are
    appended as explicit extraction instructions::

        ExtractionConfig(
            agent_type="sales",
            focus_on=["upsell opportunities", "champion name"],
            ignore=["small talk", "meeting scheduling"],
        )

    **Level 3 — Prompt suffix (medium effort)**
    ``facts_prompt_suffix`` and ``episodes_prompt_suffix`` are arbitrary text
    injected into the prompt right before "Return ONLY the JSON." — use them
    to add domain-specific extraction rules without replacing the base prompt.

    **Level 4 — Full override (escape hatch)**
    ``custom_facts_prompt`` and ``custom_episodes_prompt`` replace the entire
    prompt template.  Your prompt MUST use the same format placeholders as the
    defaults (``{conversation}``, ``{max_facts}``, ``{session_date_line}`` for
    facts; ``{conversation}``, ``{facts_list}``, ``{max_episodes}``,
    ``{session_date_line}`` for episodes) and MUST return JSON in the same
    schema, otherwise storage will silently skip the malformed output.

    Example — pre-sales agent::

        from vektori import Vektori
        from vektori.config import ExtractionConfig

        v = Vektori(
            extraction_config=ExtractionConfig(
                agent_type="presales",
                focus_on=["ICP fit", "executive sponsor"],
            )
        )

    Example — convenience shorthand::

        v = Vektori(agent_type="presales")
    """

    # Built-in preset — see docstring for valid values
    agent_type: str = "general"

    # Level 2 — domain hints
    focus_on: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)

    # Level 3 — suffix appended before "Return ONLY the JSON." in the prompt
    facts_prompt_suffix: str = ""
    episodes_prompt_suffix: str = ""

    # Level 4 — full override (must keep same JSON schema + format placeholders)
    custom_facts_prompt: str | None = None
    custom_episodes_prompt: str | None = None


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
    storage_backend: str = (
        "sqlite"  # "sqlite", "postgres", "memory", "neo4j", "qdrant", "chroma", "lancedb", "milvus"
    )
    qdrant_api_key: str | None = None  # Qdrant Cloud API key (not needed for local)
    milvus_token: str | None = None  # Milvus/Zilliz Cloud token or API key

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

    # Extraction customisation — controls what the LLM focuses on when extracting facts/episodes
    extraction_config: ExtractionConfig = field(default_factory=ExtractionConfig)

    # Processing
    async_extraction: bool = True  # False = block until facts extracted (slower but simpler)
