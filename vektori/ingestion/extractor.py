from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.base import StorageBackend

if TYPE_CHECKING:
    from vektori.config import ExtractionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent-type presets — domain-specific extraction guidance injected into prompts
# ---------------------------------------------------------------------------

_AGENT_FACTS_GUIDANCE: dict[str, str] = {
    # ── Revenue & GTM ─────────────────────────────────────────────────────────
    "presales": """
Domain focus (pre-sales agent):
Prioritise extracting:
- Prospect pain points and current challenges they described
- Budget signals and spending capacity indicators
- Purchase timeline and urgency level
- Decision-maker names, titles, and their role in the buying process
- Objections, blockers, and concerns raised
- Current tools, vendors, or competitors in use
- Specific product features or capabilities the prospect expressed interest in
- ICP qualification signals (company size, industry, tech stack)
Deprioritise: general greetings, scheduling logistics unrelated to deal timeline, small talk.""",
    "sales": """
Domain focus (sales agent):
Prioritise extracting:
- Deal stage and progression updates
- Pricing, discounts, or contract values discussed
- Expected close date or decision timeline
- Stakeholder names, roles, and level of influence
- Agreed-upon next steps and who owns them
- Deal blockers, open risks, and unresolved questions
- Legal, procurement, or security review concerns raised
- Champion and economic buyer identification
Deprioritise: general greetings, small talk.""",
    "account_management": """
Domain focus (account management agent):
Prioritise extracting:
- Renewal dates and contract terms mentioned
- Upsell and expansion opportunities surfaced
- Customer health signals (satisfaction, adoption, usage concerns)
- Escalations or at-risk account indicators
- Stakeholder changes (new contacts, departures, reorgs)
- Feature requests and product feedback from the customer
- Commitments made by the account team
Deprioritise: general pleasantries, unrelated small talk.""",
    # ── Customer experience ───────────────────────────────────────────────────
    "support": """
Domain focus (customer support agent):
Prioritise extracting:
- Issue descriptions, symptoms, and exact error messages reported
- Product area, feature, or platform affected
- Steps the customer already tried and their outcomes
- Resolution or workaround provided by the agent
- Customer satisfaction signals and frustration level
- Escalation triggers or SLA/ticket references
- Follow-up commitments and deadlines
Deprioritise: scripted greetings, hold messages, boilerplate closing pleasantries.""",
    "onboarding": """
Domain focus (customer onboarding agent):
Prioritise extracting:
- Customer's technical environment (stack, infrastructure, integrations needed)
- Onboarding milestones reached or blocked
- Implementation blockers and dependencies
- Key contacts on the customer side (technical lead, project owner)
- Training gaps or knowledge areas requiring follow-up
- Go-live dates and success criteria defined
- Configuration choices and setup decisions made
Deprioritise: generic greetings, off-topic conversation.""",
    # ── People & talent ───────────────────────────────────────────────────────
    "hr": """
Domain focus (HR agent):
Prioritise extracting:
- Employee satisfaction, engagement, or feedback signals
- Performance achievements or concerns mentioned
- Career goals, aspirations, and development interests
- Team dynamics and interpersonal issues raised
- HR policy questions and compliance topics
- Leave, absence, or schedule-related details
- Compensation or benefits questions raised
Deprioritise: casual small talk, water-cooler conversation, off-topic personal chat.""",
    "recruiting": """
Domain focus (recruiting / talent acquisition agent):
Prioritise extracting:
- Candidate name, role applied for, and current employer
- Skills, experience, and qualifications mentioned
- Compensation expectations and availability / notice period
- Interview stage and feedback given
- Candidate interest level and motivations for changing
- Red flags or concerns noted by the recruiter
- Next steps and scheduled interviews
Deprioritise: small talk, generic introductions.""",
    # ── Finance & legal ───────────────────────────────────────────────────────
    "finance": """
Domain focus (finance / accounting agent):
Prioritise extracting:
- Invoice numbers, amounts, and payment terms discussed
- Budget approvals or rejections and the amounts involved
- Expense categories and cost-centre details
- Fraud indicators or anomalies flagged
- Reconciliation discrepancies and their resolution
- Vendor names and contract financial terms
- Reporting deadlines and period-close milestones
Deprioritise: general chat, non-financial pleasantries.""",
    "legal": """
Domain focus (legal agent):
Prioritise extracting:
- Contract clauses, obligations, and terms discussed
- Compliance requirements and regulatory references (GDPR, HIPAA, SOC 2, etc.)
- Risk items, liabilities, and indemnification points raised
- Deadlines for filings, signatures, or regulatory submissions
- Parties involved in the agreement and their roles
- Dispute details, allegations, or litigation references
- Legal advice or guidance provided
Deprioritise: casual conversation, non-legal pleasantries.""",
    # ── Technical ────────────────────────────────────────────────────────────
    "coding": """
Domain focus (software engineering / coding agent):
Prioritise extracting:
- Programming languages, frameworks, and libraries mentioned
- Architecture decisions and design patterns discussed
- Bugs, errors, and their root causes identified
- Code review feedback and requested changes
- APIs, endpoints, or data models referenced
- Performance or security concerns raised
- Task assignments, PR numbers, and deadlines
Deprioritise: generic greetings, non-technical small talk.""",
    "data_analytics": """
Domain focus (data / analytics agent):
Prioritise extracting:
- Metrics, KPIs, and data sources discussed
- SQL queries, schemas, or tables referenced
- Data quality issues or anomalies surfaced
- Dashboard or report requests and their owners
- Analysis findings and key insights shared
- Data pipeline or ETL issues mentioned
- Deadlines for reports or model deliverables
Deprioritise: general pleasantries, off-topic conversation.""",
    "research": """
Domain focus (research agent):
Prioritise extracting:
- Research questions and hypotheses being investigated
- Sources, papers, or datasets referenced
- Key findings and evidence cited
- Conflicting viewpoints or gaps in the literature noted
- Methodology choices and their rationale
- Next research steps and open questions
- Authors, institutions, or publications mentioned
Deprioritise: administrative filler, generic acknowledgements.""",
    "cybersecurity": """
Domain focus (cybersecurity agent):
Prioritise extracting:
- Threat types, attack vectors, and CVE identifiers mentioned
- Affected systems, services, or IP ranges
- Indicators of compromise (IOCs) and evidence cited
- Remediation steps taken or recommended
- Vulnerabilities discovered and their severity
- Compliance frameworks referenced (ISO 27001, NIST, SOC 2)
- Incident timeline and containment actions
Deprioritise: routine greetings, non-security small talk.""",
    # ── Operations & industry ────────────────────────────────────────────────
    "healthcare": """
Domain focus (healthcare / clinical agent):
Prioritise extracting:
- Patient-reported symptoms, conditions, and medical history
- Medications, dosages, and allergies mentioned
- Appointment details, referrals, and care-team members
- Triage urgency signals and red-flag symptoms
- Treatment plans, procedures, or diagnoses discussed
- Insurance, billing, or authorisation questions raised
- Follow-up instructions and care commitments
Deprioritise: general small talk, non-clinical pleasantries.
Note: treat all health information with appropriate sensitivity.""",
    "supply_chain": """
Domain focus (supply chain / logistics agent):
Prioritise extracting:
- Shipment IDs, tracking numbers, and carrier details
- Inventory levels, stockout risks, and reorder points
- Supplier names, lead times, and delivery delays
- Purchase order numbers and order quantities
- Warehouse or fulfilment centre locations involved
- Disruption signals (port delays, weather, supplier issues)
- Cost-per-unit or freight cost changes mentioned
Deprioritise: general pleasantries, unrelated conversation.""",
    "retail": """
Domain focus (retail / e-commerce agent):
Prioritise extracting:
- Product names, SKUs, and categories discussed
- Customer purchase history and return requests
- Pricing changes, promotions, and discount codes applied
- Inventory availability and shelf or warehouse location
- Customer preferences, sizes, and style interests
- Loyalty programme status and points
- Competitor products or pricing mentioned
Deprioritise: generic greetings, unrelated small talk.""",
    "operations": """
Domain focus (operations / scheduling agent):
Prioritise extracting:
- Resource names (people, equipment, rooms) and their availability
- Appointment or shift bookings, changes, and cancellations
- Deadlines, project milestones, and blockers
- Capacity constraints and overbooked resources flagged
- SLA breaches or at-risk deliverables
- Vendor or contractor commitments and delivery dates
- Process bottlenecks and improvement suggestions
Deprioritise: casual small talk, non-operational chat.""",
}

_AGENT_EPISODES_GUIDANCE: dict[str, str] = {
    "presales": (
        "For pre-sales conversations, episodes should capture the prospect's "
        "qualification status, key pain points surfaced, and the next steps agreed."
    ),
    "sales": (
        "For sales conversations, episodes should capture the deal's current stage, "
        "key commitments or pricing discussed, blockers identified, and next actions."
    ),
    "account_management": (
        "For account management conversations, episodes should capture the customer's "
        "health signal, any expansion or renewal discussion, and commitments made."
    ),
    "support": (
        "For support conversations, episodes should capture the issue reported, "
        "the resolution or outcome reached, and any follow-up agreed."
    ),
    "onboarding": (
        "For onboarding conversations, episodes should capture the milestone reached or "
        "blocked, the blocker details, and the agreed next step toward go-live."
    ),
    "hr": (
        "For HR conversations, episodes should capture the employee concern raised, "
        "the guidance provided, and any actions or commitments made."
    ),
    "recruiting": (
        "For recruiting conversations, episodes should capture the candidate's current "
        "stage, key strengths or concerns noted, and the next interview or decision step."
    ),
    "finance": (
        "For finance conversations, episodes should capture the financial decision or "
        "issue discussed, the amount involved, and the resolution or next approval step."
    ),
    "legal": (
        "For legal conversations, episodes should capture the legal topic or risk raised, "
        "the guidance or clause discussed, and any deadline or action required."
    ),
    "coding": (
        "For engineering conversations, episodes should capture the technical problem or "
        "decision discussed, the solution or approach agreed, and any open action items."
    ),
    "data_analytics": (
        "For data conversations, episodes should capture the analysis question, "
        "the key finding or data issue surfaced, and the next analytical step."
    ),
    "research": (
        "For research conversations, episodes should capture the research question explored, "
        "key evidence or findings cited, and the next investigative step."
    ),
    "cybersecurity": (
        "For security conversations, episodes should capture the threat or vulnerability "
        "identified, the impact scope, and remediation steps taken or planned."
    ),
    "healthcare": (
        "For healthcare conversations, episodes should capture the clinical concern raised, "
        "the triage or care decision made, and follow-up instructions given."
    ),
    "supply_chain": (
        "For supply chain conversations, episodes should capture the logistics issue or "
        "risk surfaced, supplier or shipment details involved, and the resolution path."
    ),
    "retail": (
        "For retail conversations, episodes should capture the customer's product interest "
        "or service issue, the outcome reached, and any preference or loyalty signal noted."
    ),
    "operations": (
        "For operations conversations, episodes should capture the scheduling or resource "
        "issue discussed, the resolution agreed, and any SLA or deadline risk flagged."
    ),
}

# ── Prompts ──────────────────────────────────────────────────────────────────

FACTS_PROMPT = """Extract facts from this conversation — both facts about the USER and notable things the ASSISTANT said.

{session_date_line}CONVERSATION:
{conversation}

Return JSON:
{{
  "facts": [
    {{
      "text": "short factual statement under 20 words",
      "source": "'user' for facts about the user, 'assistant' for notable things the assistant said",
      "subject": "'user' or a named person/entity (for user facts); 'assistant' for assistant facts",
      "confidence": 0.95,
      "source_quotes": ["verbatim text copied from the turn this fact came from"],
      "temporal_expr": "optional — only if the fact has an explicit time anchor, e.g. '3 years ago', 'since 2021', 'last month'. Omit if no temporal info."
    }}
  ]
}}

USER facts (source: "user"):
- Extract facts about the USER or entities they explicitly mention, including preferences ("I love X", "I prefer Y", "I always use Z", "I hate X"), habits, and opinions
- When the user gives a short response ("yes", "yeah", "nope", "exactly"), use the preceding ASSISTANT turn to understand what they confirmed or denied, but source_quotes must still be the USER's words
- confidence:
    0.9–1.0  → user volunteers information unprompted ("I work at Google")
    0.7–0.89 → user confirms or agrees with assistant's suggestion ("yeah exactly", "yes that's right")
    0.5–0.69 → inferred from indirect user statement

ASSISTANT facts (source: "assistant"):
- Extract only notable things the assistant said that are worth remembering: recommendations made, advice given, specific information provided, named resources or options presented
- The fact text MUST include the actual specific content — names, values, titles, quantities. Do NOT summarize that something was said without saying what.
  BAD: "Assistant recommended a restaurant in Bandung."
  GOOD: "Assistant recommended Miss Bee Providore for Nasi Goreng in Cihampelas Walk, Bandung."
  BAD: "Assistant provided a list of language learning apps."
  GOOD: "Assistant recommended Memrise (uses mnemonics), Duolingo (gamified), and Babbel for language learning."
- Do NOT extract generic filler ("Sure, I can help", "Great question", "Let me know if you need anything")
- confidence: always 1.0
- source_quotes: verbatim text from the ASSISTANT turn

General:
- One fact per statement. Short and crisp.
- subject: 'user' when about the person speaking; a named entity when about someone/something they mention; 'assistant' for assistant facts
- Extract at most {max_facts} facts total — prioritize high-confidence, significant ones
- If nothing factual was stated, return {{"facts": []}}
- Dates in `text`: if CONVERSATION DATE is provided, replace relative time references with the actual date.
  "today" → "on YYYY-MM-DD", "yesterday" → "on YYYY-MM-DD", "last week" → "on week of YYYY-MM-DD", etc.
  Do NOT use relative expressions if you know the absolute date.
{domain_guidance}
Return ONLY the JSON."""


EPISODES_PROMPT = """You are writing episodic memory records. Given this conversation and the facts extracted from it, write a concise third-person narrative episode describing what happened.

{session_date_line}CONVERSATION:
{conversation}

EXTRACTED FACTS (numbered):
{facts_list}

Return JSON:
{{
  "episodes": [
    {{
      "text": "third-person narrative, 2-4 sentences",
      "fact_indices": [0, 2]
    }}
  ]
}}

Rules:
- Write in third person — use "the user" throughout, never "I", "you", or unresolved pronouns
- If a session date is provided above, open with "On YYYY-MM-DD, the user..."
- Name entities explicitly — not "a place in Connecticut" but "a shelter in Stamford, Connecticut"
- Resolve all pronouns — "they said they wanted to go" → "the user stated they wanted to visit Tokyo"
- 2–4 sentences max — dense, grounded, no padding
- Each episode must reference at least 1 fact via fact_indices (0-based)
- Episodes must describe what actually happened, not themes or patterns
  GOOD: "On 2025-08-20, the user reported sustaining a Grade II ankle sprain during a badminton session. A doctor confirmed the diagnosis and provided preliminary treatment. The user requested a recovery plan."
  BAD:  "User seems to be dealing with a sports injury" ← that is a theme, not an episode
  BAD:  "User said X, assistant said Y" ← that is a transcript summary, not an episode
- One episode per distinct topic in the batch; {max_episodes} maximum
- Return {{"episodes": []}} if nothing notable
{domain_guidance}
Return ONLY the JSON."""

EPISODES_FALLBACK_PROMPT = """You are writing a brief episodic memory record summarising what was discussed in this conversation.

{session_date_line}EXTRACTED FACTS (numbered):
{facts_list}

Write exactly ONE episode that captures the main topic. It must:
- Be in third person ("the user"), 2-3 sentences
- Reference at least one fact via fact_indices (0-based index into the list above)
- Be concrete — name the actual topics, preferences, or events mentioned

Return JSON with exactly this structure:
{{"episodes": [{{"text": "...", "fact_indices": [0]}}]}}

Return ONLY the JSON."""


# ── Extractor ─────────────────────────────────────────────────────────────────


class FactExtractor:
    """
    Extracts facts (L0) from conversations using an LLM.

    One LLM call per session (facts only):
      - No existing facts sent to LLM — avoids token explosion and hallucination feedback loops
      - No `contradicts` field — deduplication runs in code via embedding similarity
        after each fact batch is embedded, using the same vectors already computed for storage

    Write-time semantic dedup:
      - Same session + sim > 0.92 → skip insert, increment mentions on existing
      - Different session + sim > 0.85 → insert new fact, leave old fact unchanged
        (old fact must NOT get a mentions boost — it may have been superseded)
      - Everything else → insert normally
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        max_facts: int = 8,
        max_input_tokens: int = 4000,
        max_output_tokens: int = 8192,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.max_facts = max_facts
        # ~4 chars/token → 4000 tokens ≈ 16k chars per chunk
        # Long conversations are chunked (not truncated) so facts are extracted
        # from the full history, not just the most recent window.
        self._max_chunk_chars = max_input_tokens * 4
        self._max_output_tokens = max_output_tokens
        self._extraction_config = extraction_config

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_domain_guidance_facts(self) -> str:
        """Assemble the domain_guidance block for the facts prompt."""
        cfg = self._extraction_config
        if cfg is None:
            return ""

        parts: list[str] = []

        preset = _AGENT_FACTS_GUIDANCE.get(cfg.agent_type, "")
        if preset:
            parts.append(preset.strip())

        if cfg.focus_on:
            parts.append("Also prioritise extracting: " + ", ".join(cfg.focus_on) + ".")
        if cfg.ignore:
            parts.append("Do NOT extract facts about: " + ", ".join(cfg.ignore) + ".")

        if cfg.facts_prompt_suffix:
            parts.append(cfg.facts_prompt_suffix.strip())

        return ("\n\n" + "\n".join(parts)) if parts else ""

    def _build_domain_guidance_episodes(self) -> str:
        """Assemble the domain_guidance block for the episodes prompt."""
        cfg = self._extraction_config
        if cfg is None:
            return ""

        parts: list[str] = []

        preset = _AGENT_EPISODES_GUIDANCE.get(cfg.agent_type, "")
        if preset:
            parts.append(preset.strip())

        if cfg.episodes_prompt_suffix:
            parts.append(cfg.episodes_prompt_suffix.strip())

        return ("\n\n" + "\n".join(parts)) if parts else ""

    def _facts_prompt(self, conversation: str, session_date_line: str) -> str:
        """Return the complete facts extraction prompt, respecting ExtractionConfig."""
        cfg = self._extraction_config
        if cfg is not None and cfg.custom_facts_prompt:
            return cfg.custom_facts_prompt.format(
                conversation=conversation,
                max_facts=self.max_facts,
                session_date_line=session_date_line,
                domain_guidance=self._build_domain_guidance_facts(),
            )
        return FACTS_PROMPT.format(
            conversation=conversation,
            max_facts=self.max_facts,
            session_date_line=session_date_line,
            domain_guidance=self._build_domain_guidance_facts(),
        )

    def _episodes_prompt(
        self, conversation: str, facts_list: str, max_episodes: int, session_date_line: str
    ) -> str:
        """Return the complete episodes extraction prompt, respecting ExtractionConfig."""
        cfg = self._extraction_config
        if cfg is not None and cfg.custom_episodes_prompt:
            return cfg.custom_episodes_prompt.format(
                conversation=conversation,
                facts_list=facts_list,
                max_episodes=max_episodes,
                session_date_line=session_date_line,
                domain_guidance=self._build_domain_guidance_episodes(),
            )
        return EPISODES_PROMPT.format(
            conversation=conversation,
            facts_list=facts_list,
            max_episodes=max_episodes,
            session_date_line=session_date_line,
            domain_guidance=self._build_domain_guidance_episodes(),
        )

    async def extract(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        sentence_ids: list[str] | None = None,
        session_time: datetime | None = None,
        _capture_out: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        One LLM call: extract facts, run dedup in code, link to sentences.
        Returns {"facts_inserted": N}.

        session_time: when the conversation happened (session started_at).
        Stored as event_time on each fact for temporal filtering at retrieval.
        """
        conversation = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)

        # ── Extract facts — chunk long conversations so early facts aren't lost ──
        try:
            if len(conversation) <= self._max_chunk_chars:
                new_facts = await self._extract_facts(conversation, session_time)
            else:
                new_facts = await self._extract_facts_chunked(messages, conversation, session_time)
        except Exception as e:
            logger.error("Fact extraction failed for session %s: %s", session_id, e)
            return {"facts_inserted": 0, "error": str(e)}

        inserted_facts: list[tuple[str, str]] = []  # (fact_id, text)
        facts_inserted = await self._process_facts(
            new_facts,
            session_id,
            user_id,
            agent_id,
            conversation,
            session_time,
            _capture_out=_capture_out,
            _inserted_facts_out=inserted_facts,
        )

        episodes_created = 0
        if inserted_facts:
            try:
                episodes_created = await self._extract_episodes(
                    inserted_facts,
                    conversation,
                    session_id,
                    user_id,
                    agent_id,
                    session_time=session_time,
                )
            except Exception as e:
                logger.warning("Episode extraction failed for session %s: %s", session_id, e)

        logger.info(
            "Extraction complete for session %s: %d facts, %d episodes",
            session_id,
            facts_inserted,
            episodes_created,
        )
        return {"facts_inserted": facts_inserted, "episodes_created": episodes_created}

    # ── LLM Call ──────────────────────────────────────────────────────────────

    async def _extract_facts(
        self, conversation: str, session_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        session_date_line = (
            f"CONVERSATION DATE: {session_time.strftime('%Y-%m-%d')}\n\n" if session_time else ""
        )
        prompt = self._facts_prompt(conversation, session_date_line)
        response = await self.llm.generate(prompt, max_tokens=self._max_output_tokens)
        return _parse_json_response(response).get("facts", [])

    async def _extract_facts_chunked(
        self,
        messages: list[dict[str, str]],
        full_conversation: str,
        session_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Chunk a long conversation and extract facts from each chunk.

        Chunking at message boundaries (not character boundaries) preserves
        turn structure. The last message of each chunk is carried into the
        next chunk as overlap to maintain context across boundaries.

        All chunks share the same max_facts budget to avoid penalising long
        histories — total facts are deduplicated by the vector-space dedup
        in _process_facts, not here.
        """
        chunks = self._chunk_messages(messages)
        logger.debug(
            "Long conversation (%d chars) split into %d chunks for extraction",
            len(full_conversation),
            len(chunks),
        )
        all_facts: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_conv = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in chunk)
            try:
                chunk_facts = await self._extract_facts(chunk_conv, session_time)
                all_facts.extend(chunk_facts)
            except Exception as e:
                logger.warning("Chunk extraction failed (%d messages): %s", len(chunk), e)
        return all_facts

    def _chunk_messages(self, messages: list[dict[str, str]]) -> list[list[dict[str, str]]]:
        """Split messages into chunks where each chunk's text fits in one LLM call.

        Carries the last message of each chunk into the next as overlap so the
        model has context when a turn spans a chunk boundary.
        """
        chunks: list[list[dict[str, str]]] = []
        current: list[dict[str, str]] = []
        current_chars = 0

        for msg in messages:
            # +10 for "ROLE: \n" overhead
            msg_chars = len(msg.get("role", "")) + len(msg.get("content", "")) + 10
            if current and current_chars + msg_chars > self._max_chunk_chars:
                chunks.append(current)
                overlap = current[-1]
                current = [overlap, msg]
                current_chars = (
                    len(overlap.get("role", "")) + len(overlap.get("content", "")) + 10 + msg_chars
                )
            else:
                current.append(msg)
                current_chars += msg_chars

        if current:
            chunks.append(current)
        return chunks

    # ── Processing ────────────────────────────────────────────────────────────

    async def _process_facts(
        self,
        fact_list: list[dict[str, Any]],
        session_id: str,
        user_id: str,
        agent_id: str | None,
        conversation: str,
        session_time: datetime | None = None,
        _capture_out: list[dict[str, Any]] | None = None,
        _inserted_facts_out: list[tuple[str, str]] | None = None,
    ) -> int:
        """
        For each fact:
          1. Batch embed all fact texts (one embed call for the whole list)
          2. Write-time semantic dedup:
             - same session + sim > 0.92 → skip, increment mentions on existing
             - diff session + sim > 0.85 → insert new fact, leave old untouched
          3. Insert fact with event_time = session_time
          4. Link to source sentences
        """
        if not fact_list:
            return 0

        facts_inserted = 0

        # Batch embed — one call instead of N
        texts = [f["text"] for f in fact_list]
        try:
            embeddings = await self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error("Batch embed failed for %d facts: %s", len(texts), e)
            return 0

        for fact_data, fact_embedding in zip(fact_list, embeddings):
            try:
                subject = fact_data.get("subject") or None

                # Write-time semantic dedup
                dedup = await self._check_dedup(
                    fact_data["text"], fact_embedding, session_id, user_id, agent_id, subject
                )

                if dedup is not None:
                    existing_id, same_session = dedup
                    if same_session:
                        # Pure dedup: same conversation, skip insert entirely
                        await self.db.increment_fact_mentions(existing_id)
                        continue
                    # Cross-session near-dup: the newer fact supersedes the older one.
                    # Insert the new fact (below), then deactivate the old one with a
                    # superseded_by pointer so it's excluded from default active_only queries.

                # Carry temporal expression and source role in metadata
                meta: dict[str, Any] = {}
                if fact_data.get("temporal_expr"):
                    meta["temporal_expr"] = fact_data["temporal_expr"]
                if fact_data.get("source"):
                    meta["source"] = fact_data["source"]

                fact_id = await self.db.insert_fact(
                    text=fact_data["text"],
                    embedding=fact_embedding,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    subject=subject,
                    confidence=fact_data.get("confidence", 1.0),
                    metadata=meta or None,
                    event_time=session_time,
                )
                facts_inserted += 1

                # If this replaced a cross-session near-dup, deactivate the old fact now
                # that we have the new fact_id to use as the superseded_by pointer.
                if dedup is not None:
                    old_id, _ = dedup
                    await self.db.deactivate_fact(old_id, superseded_by=fact_id)

                if _inserted_facts_out is not None:
                    _inserted_facts_out.append((fact_id, fact_data["text"]))

                if _capture_out is not None:
                    _capture_out.append(
                        {
                            "text": fact_data["text"],
                            "subject": subject,
                            "confidence": fact_data.get("confidence", 1.0),
                            "metadata": meta or {},
                            "source_quotes": fact_data.get("source_quotes") or [],
                        }
                    )

                if fact_data.get("source_quotes"):
                    linked = await self._link_to_source_sentences(
                        fact_data["source_quotes"], session_id, conversation
                    )
                    for sent_id in linked:
                        await self.db.insert_fact_source(fact_id, sent_id)

                # Build similarity edges for PPR graph
                await self._build_fact_edges(fact_id, fact_embedding, user_id, agent_id)

            except Exception as e:
                logger.warning("Failed to insert fact '%s': %s", fact_data.get("text"), e)

        return facts_inserted

    # ── Cache Replay ──────────────────────────────────────────────────────────

    async def replay_facts(
        self,
        cached_facts: list[dict[str, Any]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        session_time: datetime | None = None,
        _inserted_facts_out: list[tuple[str, str]] | None = None,
    ) -> int:
        """
        Insert pre-extracted facts from the session cache. No LLM call.
        Re-embeds fact texts (local model, cheap) so embeddings are fresh.
        Used by the benchmark runner for cache-hit sessions.

        _inserted_facts_out: if provided, (fact_id, text) pairs for every
        successfully inserted fact are appended — used by the caller to
        drive episode extraction after replay.
        """
        if not cached_facts:
            return 0

        texts = [f["text"] for f in cached_facts]
        try:
            embeddings = await self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error("Batch embed failed during fact replay for session %s: %s", session_id, e)
            return 0

        facts_inserted = 0
        for fact_data, fact_emb in zip(cached_facts, embeddings):
            try:
                subject = fact_data.get("subject") or None

                # Dedup check — in a fresh user context this always returns None,
                # but we run it anyway for correctness if sessions overlap.
                dedup = await self._check_dedup(
                    fact_data["text"], fact_emb, session_id, user_id, agent_id, subject
                )
                if dedup is not None:
                    existing_id, same_session = dedup
                    await self.db.increment_fact_mentions(existing_id)
                    if same_session:
                        continue

                meta = fact_data.get("metadata") or {}
                fact_id = await self.db.insert_fact(
                    text=fact_data["text"],
                    embedding=fact_emb,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    subject=subject,
                    confidence=fact_data.get("confidence", 1.0),
                    metadata=meta or None,
                    event_time=session_time,
                )
                facts_inserted += 1

                # Match fresh-path behaviour: deactivate the cross-session near-dup
                # now that we have the new fact_id to use as the superseded_by pointer.
                if dedup is not None:
                    old_id, _ = dedup
                    await self.db.deactivate_fact(old_id, superseded_by=fact_id)

                if _inserted_facts_out is not None:
                    _inserted_facts_out.append((fact_id, fact_data["text"]))

                source_quotes = fact_data.get("source_quotes") or []
                if source_quotes:
                    linked = await self._link_to_source_sentences(
                        source_quotes, session_id, conversation=None
                    )
                    for sent_id in linked:
                        await self.db.insert_fact_source(fact_id, sent_id)

            except Exception as e:
                logger.warning("Failed to replay fact '%s': %s", fact_data.get("text"), e)

        return facts_inserted

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _check_dedup(
        self,
        fact_text: str,
        fact_embedding: list[float],
        session_id: str,
        user_id: str,
        agent_id: str | None,
        subject: str | None = None,
    ) -> tuple[str, bool] | None:
        """
        Check if a near-duplicate fact already exists.

        Returns (existing_fact_id, is_same_session) or None.

        Thresholds:
          - same session + sim > 0.92 → strong dedup signal (same conversation re-stating same fact)
          - diff session + sim > 0.85 → cross-session near-dup (fact seen before)
        Subject-scoped when available to avoid cross-entity false positives.
        """
        try:
            candidates = await self.db.search_facts(
                embedding=fact_embedding,
                user_id=user_id,
                agent_id=agent_id,
                subject=subject,
                limit=3,
                active_only=True,
            )
            if not candidates:
                return None
            best = candidates[0]
            sim = 1.0 - best.get("distance", 1.0)
            same_session = best.get("session_id") == session_id

            if same_session and sim > 0.92:
                return (best["id"], True)
            if not same_session and sim > 0.85:
                return (best["id"], False)

            # Contradiction LLM check for near-misses
            # If the fact wasn't deduped by bare similarity, check if it contradicts.
            # Only consider same-subject facts that are reasonably close (sim > 0.65).
            plausible_conflicts = [c for c in candidates if (1.0 - c.get("distance", 1.0)) > 0.65]
            if plausible_conflicts:
                facts_text = "\n".join(f"- [{c['id']}] {c['text']}" for c in plausible_conflicts)
                prompt = CONTRADICTION_PROMPT.format(fact_text=fact_text, existing_facts=facts_text)
                try:
                    response = await self.llm.generate(prompt, max_tokens=512)
                    data = _parse_json_response(response)
                    supersedes_id = data.get("supersedes_id")
                    if supersedes_id and any(c["id"] == supersedes_id for c in plausible_conflicts):
                        return (supersedes_id, False)
                except Exception as e:
                    logger.debug("Contradiction check failed: %s", e)

        except Exception as e:
            logger.warning("Dedup lookup failed: %s", e)
        return None

    async def _build_fact_edges(
        self,
        fact_id: str,
        fact_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        sim_threshold: float = 0.75,
        limit: int = 10,
    ) -> None:
        """Find similar existing facts and insert PPR graph edges.

        Lower threshold than dedup (0.75 vs 0.85) so we capture "related but
        not duplicate" facts — the edges that let PPR multi-hop from matched
        facts to relevant-but-unmatched ones.
        """
        try:
            candidates = await self.db.search_facts(
                embedding=fact_embedding,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
                active_only=True,
            )
            for c in candidates:
                if c["id"] == fact_id:
                    continue
                sim = 1.0 - c.get("distance", 1.0)
                if sim >= sim_threshold:
                    await self.db.insert_fact_edge(fact_id, c["id"], user_id, weight=round(sim, 4))
        except Exception as e:
            logger.debug("_build_fact_edges failed for %s: %s", fact_id, e)

    async def _extract_episodes(
        self,
        inserted_facts: list[tuple[str, str]],
        conversation: str,
        session_id: str,
        user_id: str,
        agent_id: str | None,
        max_episodes: int = 5,
        session_time: datetime | None = None,
    ) -> int:
        """Extract episodic memory narratives from the conversation and its facts.

        Episodes are concise third-person narratives of what happened in this
        session batch — grounded stories with resolved entities and dates. They
        are vector-embedded so they can be found both via graph traversal (fact →
        episode_facts → episodes) and direct cosine search at retrieval.

        Returns the number of episodes inserted.
        """
        if not inserted_facts:
            return 0

        facts_list = "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(inserted_facts))
        fact_id_list = [fid for fid, _ in inserted_facts]

        # Truncate conversation to keep prompt manageable (~3000 chars ≈ 750 tokens)
        conv_snippet = conversation[:3000]

        session_date_line = (
            f"SESSION DATE: {session_time.strftime('%Y-%m-%d')}\n\n" if session_time else ""
        )
        prompt = self._episodes_prompt(conv_snippet, facts_list, max_episodes, session_date_line)
        try:
            response = await self.llm.generate(prompt, max_tokens=1024)
            data = _parse_json_response(response)
        except Exception as e:
            logger.warning("Episode LLM call failed: %s", e)
            return 0

        raw_episodes = data.get("episodes", [])[:max_episodes]

        # Fallback: if the main prompt returned nothing but we have facts, retry
        # with a simpler prompt that doesn't require a "notable event" — any
        # session with extracted facts deserves at least one episode record.
        if not raw_episodes:
            logger.debug(
                "Episode pass returned empty for session with %d facts — retrying with fallback prompt",
                len(inserted_facts),
            )
            fallback_prompt = EPISODES_FALLBACK_PROMPT.format(
                facts_list=facts_list,
                session_date_line=session_date_line,
            )
            try:
                fallback_response = await self.llm.generate(fallback_prompt, max_tokens=512)
                raw_episodes = _parse_json_response(fallback_response).get("episodes", [])[:1]
            except Exception as e:
                logger.warning("Episode fallback LLM call failed: %s", e)

        if not raw_episodes:
            return 0

        # Batch embed all episode texts in one call
        episode_texts = [(ep.get("text") or "").strip() for ep in raw_episodes]
        episode_texts = [t for t in episode_texts if t]
        if not episode_texts:
            return 0

        try:
            embeddings = await self.embedder.embed_batch(episode_texts)
        except Exception as e:
            logger.warning("Batch embed failed for episodes: %s", e)
            return 0

        episodes_created = 0
        text_iter = iter(zip(episode_texts, embeddings))
        for episode_data in raw_episodes:
            text = (episode_data.get("text") or "").strip()
            if not text:
                continue
            indices = episode_data.get("fact_indices") or []

            try:
                _, embedding = next(text_iter)
            except StopIteration:
                break

            # Map indices → fact IDs from this session's inserted facts
            linked_ids = [
                fact_id_list[i]
                for i in indices
                if isinstance(i, int) and 0 <= i < len(fact_id_list)
            ]
            if not linked_ids:
                continue

            try:
                episode_id = await self.db.insert_episode(
                    text, embedding, user_id, agent_id, session_id
                )
                for fid in linked_ids:
                    await self.db.insert_episode_fact(episode_id, fid)
                episodes_created += 1
            except Exception as e:
                logger.warning("Failed to insert episode '%s': %s", text, e)

        return episodes_created

    async def _link_to_source_sentences(
        self,
        source_quotes: list[str],
        session_id: str,
        conversation: str | None,
    ) -> list[str]:
        """
        Link extracted content to source sentences.
        Step 1: hallucination guard — skip quotes not present in conversation
                (skipped when conversation=None, e.g. cached fact replay)
        Step 2: exact substring match via find_sentence_containing()
        Step 3: embedding vector fallback for paraphrased/unmatched quotes
        """
        linked_ids: list[str] = []
        unmatched: list[str] = []

        for quote in source_quotes:
            # Step 1: guard against hallucinated quotes (skip for trusted cached quotes)
            if conversation is not None and quote.lower() not in conversation.lower():
                continue

            # Step 2: exact substring match
            exact = await self.db.find_sentence_containing(session_id, quote)
            if exact:
                linked_ids.append(exact["id"])
            else:
                unmatched.append(quote)

        # Step 3: embedding-based fallback for paraphrased or split quotes
        if unmatched:
            try:
                embeddings = await self.embedder.embed_batch(unmatched)
                seen: set[str] = set(linked_ids)
                for emb in embeddings:
                    ids = await self.db.search_sentences_in_session(emb, session_id)
                    for sid in ids:
                        if sid not in seen:
                            linked_ids.append(sid)
                            seen.add(sid)
            except Exception as e:
                logger.debug("Embedding fallback failed for source quotes: %s", e)

        return linked_ids


# ── Utilities ─────────────────────────────────────────────────────────────────


def _parse_json_response(response: str) -> dict[str, Any]:
    """Parse LLM JSON response, stripping markdown code fences if present.

    Handles three common model output patterns:
    1. Raw JSON
    2. ```json ... ``` fenced block at the start
    3. Prose followed by a fenced or bare JSON block anywhere in the response
    """
    import re

    text = response.strip()

    # Pattern 1: starts with a fence — strip it
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end]).strip()

    # Try direct parse first (handles pattern 1 and raw JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pattern 3: prose with an embedded fenced block — extract the last JSON object/array
    # Try fenced blocks first (```...```)
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response)
    for block in reversed(fenced):
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue

    # Last resort: find the last {...} or [...] in the response
    for match in reversed(list(re.finditer(r"(\{[\s\S]*\}|\[[\s\S]*\])", response))):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue

    logger.error("Failed to parse extraction JSON: %s\nResponse: %.500s", "no valid JSON found", response)
    return {"facts": []}


# ── Contradiction prompt ──────────────────────────────────────────────────────
CONTRADICTION_PROMPT = """Do any of the existing facts contradict or get superseded by the new fact?
New fact: "{fact_text}"

Existing facts:
{existing_facts}

If the new fact updates, contradicts, or replaces an existing fact, return its ID. Otherwise return null.

Return ONLY JSON:
{{
  "supersedes_id": "fact_id_here" or null
}}"""
