from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

FACTS_PROMPT = """Extract facts from this conversation — both facts about the USER and notable things the ASSISTANT said.

CONVERSATION:
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
- Extract facts about the USER or entities they explicitly mention
- When the user gives a short response ("yes", "yeah", "nope", "exactly"), use the preceding ASSISTANT turn to understand what they confirmed or denied, but source_quotes must still be the USER's words
- confidence:
    0.9–1.0  → user volunteers information unprompted ("I work at Google")
    0.7–0.89 → user confirms or agrees with assistant's suggestion ("yeah exactly", "yes that's right")
    0.5–0.69 → inferred from indirect user statement

ASSISTANT facts (source: "assistant"):
- Extract only notable things the assistant said that are worth remembering: recommendations made, advice given, information provided, resources or options presented
- Do NOT extract generic filler ("Sure, I can help", "Great question", "Let me know if you need anything")
- confidence: always 1.0
- source_quotes: verbatim text from the ASSISTANT turn

General:
- One fact per statement. Short and crisp.
- subject: 'user' when about the person speaking; a named entity when about someone/something they mention; 'assistant' for assistant facts
- Extract at most {max_facts} facts total — prioritize high-confidence, significant ones
- If nothing factual was stated, return {{"facts": []}}

Return ONLY the JSON."""


CROSS_SESSION_INSIGHTS_PROMPT = """Analyze facts across multiple sessions to extract CROSS-SESSION PATTERNS.

FACTS BY SESSION (each fact has a stable ID like [F1], [F2] ...):
{facts_by_session}

EXISTING INSIGHTS (do not repeat or rephrase these):
{existing_insights}

Return JSON:
{{
  "insights": [
    {{
      "text": "actionable pattern — must have evidence from 2+ different sessions",
      "confidence": 0.80,
      "derived_from_fact_ids": ["F3", "F7"]
    }}
  ]
}}

Rules:
- Only extract patterns with clear evidence spanning 2+ sessions
- Must be actionable — what should the agent do differently because of this?
- Do not repeat, rephrase, or contradict anything in EXISTING INSIGHTS
- derived_from_fact_ids must reference IDs from the list above
- Extract at most {max_insights} insights
- If no clear cross-session pattern exists, return {{"insights": []}}

Return ONLY the JSON."""


# ── Extractor ─────────────────────────────────────────────────────────────────

class FactExtractor:
    """
    Extracts facts (L0) and insights (L1) from conversations using an LLM.

    One LLM call per session (facts only):
      - No existing facts sent to LLM — avoids token explosion and hallucination feedback loops
      - No `contradicts` field — deduplication runs in code via embedding similarity
        after each fact batch is embedded, using the same vectors already computed for storage

    Cross-session insights (separate, triggered every Nth session):
      - Facts grouped by session with stable IDs ([F1], [F2], ...)
      - LLM returns derived_from_fact_ids — ID-based, no fragile text matching
      - Existing insights passed in to prevent duplication

    Write-time semantic dedup (PHASE2 item 3):
      - Same session + sim > 0.92 → skip insert, increment mentions on existing
      - Different session + sim > 0.85 → insert + increment mentions on older (cross-session IDF signal)
      - Everything else → insert normally
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        max_facts: int = 8,
        max_insights: int = 3,
        max_input_tokens: int = 4000,
        max_output_tokens: int = 1024,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.llm = llm
        self.max_facts = max_facts
        self.max_insights = max_insights
        # ~4 chars/token → 4000 tokens ≈ 16k chars per chunk
        # Long conversations are chunked (not truncated) so facts are extracted
        # from the full history, not just the most recent window.
        self._max_chunk_chars = max_input_tokens * 4
        self._max_output_tokens = max_output_tokens

    async def extract(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        sentence_ids: list[str] | None = None,
        session_time: datetime | None = None,
    ) -> dict[str, Any]:
        """
        One LLM call: extract facts, run dedup in code, link to sentences.
        Returns {"facts_inserted": N}.
        Cross-session insights trigger every 3rd session.

        session_time: when the conversation happened (session started_at).
        Stored as event_time on each fact for temporal filtering at retrieval.
        """
        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
        )

        # ── Extract facts — chunk long conversations so early facts aren't lost ──
        try:
            if len(conversation) <= self._max_chunk_chars:
                new_facts = await self._extract_facts(conversation)
            else:
                new_facts = await self._extract_facts_chunked(messages, conversation)
        except Exception as e:
            logger.error("Fact extraction failed for session %s: %s", session_id, e)
            return {"facts_inserted": 0, "error": str(e)}

        facts_inserted = await self._process_facts(
            new_facts, session_id, user_id, agent_id, conversation, session_time
        )

        # ── Cross-session trigger: every 3rd session ──
        try:
            session_count = await self.db.count_sessions(user_id, agent_id)
            if session_count % 3 == 0 and session_count > 1:
                await self.extract_cross_session_insights(user_id, agent_id)
        except Exception as e:
            logger.warning("Cross-session insight trigger failed: %s", e)

        logger.info("Extraction complete for session %s: %d facts", session_id, facts_inserted)
        return {"facts_inserted": facts_inserted}

    # ── LLM Call ──────────────────────────────────────────────────────────────

    async def _extract_facts(self, conversation: str) -> list[dict[str, Any]]:
        prompt = FACTS_PROMPT.format(conversation=conversation, max_facts=self.max_facts)
        response = await self.llm.generate(prompt, max_tokens=self._max_output_tokens)
        return _parse_json_response(response).get("facts", [])

    async def _extract_facts_chunked(
        self,
        messages: list[dict[str, str]],
        full_conversation: str,
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
            len(full_conversation), len(chunks),
        )
        all_facts: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_conv = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in chunk
            )
            try:
                chunk_facts = await self._extract_facts(chunk_conv)
                all_facts.extend(chunk_facts)
            except Exception as e:
                logger.warning("Chunk extraction failed (%d messages): %s", len(chunk), e)
        return all_facts

    def _chunk_messages(
        self, messages: list[dict[str, str]]
    ) -> list[list[dict[str, str]]]:
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
                    len(overlap.get("role", "")) + len(overlap.get("content", "")) + 10
                    + msg_chars
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
    ) -> int:
        """
        For each fact:
          1. Batch embed all fact texts (one embed call for the whole list)
          2. Write-time semantic dedup:
             - same session + sim > 0.92 → skip, increment mentions on existing
             - diff session + sim > 0.85 → insert + increment mentions on older
          3. Insert fact
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
                    fact_embedding, session_id, user_id, agent_id, subject
                )

                if dedup is not None:
                    existing_id, same_session = dedup
                    if same_session:
                        # Pure dedup: same conversation, skip insert entirely
                        await self.db.increment_fact_mentions(existing_id)
                        continue
                    else:
                        # Cross-session near-dup: insert new fact, note recurrence on older
                        await self.db.increment_fact_mentions(existing_id)
                        # Fall through to insert

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

                if fact_data.get("source_quotes"):
                    linked = await self._link_to_source_sentences(
                        fact_data["source_quotes"], session_id, conversation
                    )
                    for sent_id in linked:
                        await self.db.insert_fact_source(fact_id, sent_id)

            except Exception as e:
                logger.warning("Failed to insert fact '%s': %s", fact_data.get("text"), e)

        return facts_inserted

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _check_dedup(
        self,
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
        except Exception as e:
            logger.warning("Dedup lookup failed: %s", e)
        return None

    async def _link_to_source_sentences(
        self,
        source_quotes: list[str],
        session_id: str,
        conversation: str,
    ) -> list[str]:
        """
        Link extracted content to source sentences.
        Step 1: hallucination guard — skip quotes not present in conversation
        Step 2: exact substring match via find_sentence_containing()
        Step 3: trgm batch fallback via find_sentences_by_similarity() for unmatched quotes
        """
        linked_ids: list[str] = []
        unmatched: list[str] = []

        for quote in source_quotes:
            # Step 1: guard against hallucinated quotes
            if quote.lower() not in conversation.lower():
                continue

            # Step 2: exact substring match
            exact = await self.db.find_sentence_containing(session_id, quote)
            if exact:
                linked_ids.append(exact["id"])
            else:
                unmatched.append(quote)

        # Step 3: trgm batch fallback for all unmatched quotes in one query
        if unmatched:
            try:
                trgm_ids = await self.db.find_sentences_by_similarity(unmatched, session_id)
                linked_ids.extend(trgm_ids)
            except Exception as e:
                logger.debug("trgm fallback failed for quotes: %s", e)

        return linked_ids

    # ── Cross-session ─────────────────────────────────────────────────────────

    async def extract_cross_session_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Analyze facts across all sessions to find cross-session patterns.
        Called automatically every 3rd session, or manually via client.generate_insights().

        Facts are given stable IDs ([F1], [F2], ...) in the prompt so the LLM returns
        derived_from_fact_ids — ID-based linking, no fragile text matching.
        Existing insights are passed in to prevent duplication.
        """
        all_facts = await self.db.get_active_facts(user_id, agent_id, limit=200)
        if not all_facts:
            return {"insights_inserted": 0}

        # Build stable ID map: "F1" → fact dict
        fact_id_map: dict[str, dict[str, Any]] = {
            f"F{i + 1}": fact for i, fact in enumerate(all_facts)
        }

        # Group by session for the prompt
        facts_by_session: dict[str, list[str]] = {}
        for fid, fact in fact_id_map.items():
            sid = fact.get("session_id") or "unknown"
            facts_by_session.setdefault(sid, []).append(f"[{fid}] {fact['text']}")

        if len(facts_by_session) < 2:
            return {"insights_inserted": 0}

        session_summary = "\n".join(
            f"Session {sid}:\n" + "\n".join(f"  {line}" for line in lines)
            for sid, lines in facts_by_session.items()
        )

        # Fetch existing insights for dedup
        existing_insights = await self.db.get_active_insights(user_id, agent_id)
        existing_text = (
            "\n".join(f"- {ins['text']}" for ins in existing_insights)
            if existing_insights else "None yet."
        )

        prompt = CROSS_SESSION_INSIGHTS_PROMPT.format(
            facts_by_session=session_summary,
            existing_insights=existing_text,
            max_insights=self.max_insights,
        )

        try:
            response = await self.llm.generate(prompt, max_tokens=self._max_output_tokens)
            insight_list = _parse_json_response(response).get("insights", [])
        except Exception as e:
            logger.error("Cross-session extraction failed: %s", e)
            return {"insights_inserted": 0, "error": str(e)}

        insights_inserted = 0

        # Batch embed insight texts
        if insight_list:
            insight_texts = [ins["text"] for ins in insight_list]
            try:
                insight_embeddings = await self.embedder.embed_batch(insight_texts)
            except Exception as e:
                logger.error("Batch embed failed for insights: %s", e)
                return {"insights_inserted": 0, "error": str(e)}

            for insight_data, insight_embedding in zip(insight_list, insight_embeddings):
                try:
                    insight_id = await self.db.insert_insight(
                        text=insight_data["text"],
                        embedding=insight_embedding,
                        user_id=user_id,
                        agent_id=agent_id,
                        confidence=insight_data.get("confidence", 1.0),
                    )
                    insights_inserted += 1

                    # ID-based linking — no text matching, no paraphrase failures
                    for fid in insight_data.get("derived_from_fact_ids", []):
                        fact = fact_id_map.get(fid)
                        if fact:
                            await self.db.insert_insight_fact(insight_id, fact["id"])

                except Exception as e:
                    logger.warning("Failed to insert cross-session insight: %s", e)

        logger.info(
            "Cross-session extraction for user %s: %d insights", user_id, insights_inserted
        )
        return {"insights_inserted": insights_inserted}


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_json_response(response: str) -> dict[str, Any]:
    """Parse LLM JSON response, stripping markdown code fences if present."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction JSON: %s\nResponse: %.500s", e, response)
        return {"facts": [], "insights": []}
