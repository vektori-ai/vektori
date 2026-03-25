from __future__ import annotations

import json
import logging
from typing import Any

from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────────

FACTS_PROMPT = """Extract FACTS from this conversation.

CONVERSATION:
{conversation}

Return JSON:
{{
  "facts": [
    {{
      "text": "short, explicit factual statement (under 20 words)",
      "subject": "entity this is about — 'user', or a specific named person/entity",
      "confidence": 0.95,
      "source_quotes": ["verbatim substring copied from the conversation above"]
    }}
  ]
}}

Rules:
- Only explicit, verifiable information stated in the conversation. No inference.
- One fact per statement. Short and crisp.
- subject: always identify — use 'user' when the fact is about the person speaking
- source_quotes: copy-paste exact text, do NOT paraphrase or summarize
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
- If no clear cross-session pattern exists, return {{"insights": []}}

Return ONLY the JSON."""


# ── Extractor ─────────────────────────────────────────────────────────────────

class FactExtractor:
    """
    Extracts facts (L0) and insights (L1) from conversations using an LLM.

    One LLM call per session (facts only):
      - No existing facts sent to LLM — avoids token explosion and hallucination feedback loops
      - No `contradicts` field — contradiction detection runs in code via embedding similarity
        after each fact is embedded, using the same vector already computed for storage

    Cross-session insights (separate, triggered every Nth session):
      - Facts grouped by session with stable IDs ([F1], [F2], ...)
      - LLM returns derived_from_fact_ids — ID-based, no fragile text matching
      - Existing insights passed in to prevent duplication
    """

    def __init__(self, db: StorageBackend, embedder: EmbeddingProvider, llm: LLMProvider) -> None:
        self.db = db
        self.embedder = embedder
        self.llm = llm

    async def extract(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """
        One LLM call: extract facts, run contradiction detection in code, link to sentences.
        Returns {"facts_inserted": N}.
        Cross-session insights trigger every 3rd session.
        """
        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
        )

        # ── Extract facts (one LLM call, no existing facts in prompt) ──
        try:
            new_facts = await self._extract_facts(conversation)
        except Exception as e:
            logger.error("Fact extraction failed for session %s: %s", session_id, e)
            return {"facts_inserted": 0, "error": str(e)}

        facts_inserted = await self._process_facts(
            new_facts, session_id, user_id, agent_id, conversation
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
        prompt = FACTS_PROMPT.format(conversation=conversation)
        response = await self.llm.generate(prompt)
        return _parse_json_response(response).get("facts", [])

    # ── Processing ────────────────────────────────────────────────────────────

    async def _process_facts(
        self,
        fact_list: list[dict[str, Any]],
        session_id: str,
        user_id: str,
        agent_id: str | None,
        conversation: str,
    ) -> int:
        """
        For each fact:
          1. Embed
          2. Contradiction check via embedding similarity (no LLM, threshold 0.85)
             Narrows to same subject if available — avoids cross-entity false positives
          3. Insert fact
          4. Link to source sentences
        """
        facts_inserted = 0

        for fact_data in fact_list:
            try:
                fact_embedding = await self.embedder.embed(fact_data["text"])
                subject = fact_data.get("subject") or None

                # Contradiction detection: purely in code, reuses already-computed embedding
                supersedes_id = None
                old_fact = await self._find_contradicted_fact(
                    fact_embedding, user_id, agent_id, subject
                )
                if old_fact:
                    supersedes_id = old_fact["id"]
                    await self.db.deactivate_fact(old_fact["id"])

                fact_id = await self.db.insert_fact(
                    text=fact_data["text"],
                    embedding=fact_embedding,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    subject=subject,
                    confidence=fact_data.get("confidence", 1.0),
                    superseded_by_target=supersedes_id,
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

    async def _find_contradicted_fact(
        self,
        fact_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        subject: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Find an existing fact that the new fact likely supersedes.
        Uses the already-computed embedding — no extra embed call.
        Narrows to same subject when available to prevent cross-entity false positives.
        Threshold 0.85 — must be clearly the same claim.
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
            if (1 - best["distance"]) >= 0.85:
                return best
        except Exception as e:
            logger.warning("Contradiction lookup failed: %s", e)
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
        )

        try:
            response = await self.llm.generate(prompt)
            insight_list = _parse_json_response(response).get("insights", [])
        except Exception as e:
            logger.error("Cross-session extraction failed: %s", e)
            return {"insights_inserted": 0, "error": str(e)}

        insights_inserted = 0
        for insight_data in insight_list:
            try:
                insight_embedding = await self.embedder.embed(insight_data["text"])
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
