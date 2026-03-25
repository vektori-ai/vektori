

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

EXISTING KNOWN FACTS ABOUT THIS USER:
{existing_facts}

Return JSON:
{{
  "facts": [
    {{
      "text": "clear, standalone factual statement",
      "confidence": 0.95,
      "source_quotes": ["exact quote from conversation"],
      "contradicts": "text of existing fact this contradicts, or null"
    }}
  ]
}}

Rules:
- Explicit, verifiable information stated in the conversation
- Short, crisp statements (under 20 words)
- One fact per statement
- Check EVERY new fact against existing facts for contradictions
- source_quotes must be EXACT substrings from the conversation above

Return ONLY the JSON."""

INSIGHTS_PROMPT = """Given this conversation and known facts, extract INSIGHTS.

CONVERSATION:
{conversation}

ALL KNOWN FACTS (including newly extracted):
{all_facts}

Return JSON:
{{
  "insights": [
    {{
      "text": "inferred pattern or observation, actionable for the agent",
      "confidence": 0.80,
      "derived_from_facts": ["fact text this insight is based on"],
      "source_quotes": ["exact quotes supporting this inference"]
    }}
  ]
}}

Rules:
- Inferred patterns NOT explicitly stated in the conversation
- Must be actionable
- Can reference facts from current AND past sessions
- Higher confidence bar than facts
- source_quotes must be EXACT substrings from the conversation

Return ONLY the JSON."""

CROSS_SESSION_INSIGHTS_PROMPT = """Analyze facts across multiple sessions to find CROSS-SESSION PATTERNS.

FACTS BY SESSION:
{facts_by_session}

Extract insights that span MULTIPLE sessions — patterns, behavioral trends,
what approaches work/don't work, preference evolution over time.

Return JSON:
{{
  "insights": [
    {{
      "text": "pattern description — must reference multiple sessions",
      "confidence": 0.80,
      "derived_from_facts": ["fact text 1", "fact text 2"]
    }}
  ]
}}

Rules:
- ONLY extract patterns that span 2+ sessions
- Must be actionable
- Higher confidence bar — evidence required from multiple data points

Return ONLY the JSON."""


# ── Extractor ─────────────────────────────────────────────────────────────────

class FactExtractor:
    """
    Extracts facts (L0) and insights (L1) from conversations using an LLM.

    Two separate LLM calls per session:
      Call 1 — facts only (with contradiction checking against existing facts)
      Call 2 — insights only (receives just-extracted facts as context)

    Contradiction matching uses embedding similarity, not exact text match.
    Source quote linking uses substring match first, embedding fallback second.
    Cross-session insights trigger every 3rd session automatically.
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
        Orchestrates two-call extraction + cross-session trigger.
        Returns {"facts_inserted": N, "insights_inserted": M}.
        Degrades gracefully: if facts fail, returns early; if insights fail, returns facts result.
        """
        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
        )
        existing_facts = await self.db.get_active_facts(user_id, agent_id, limit=50)

        # ── Call 1: Facts ──
        try:
            new_facts = await self._extract_facts(conversation, existing_facts)
        except Exception as e:
            logger.error("Fact extraction failed for session %s: %s", session_id, e)
            return {"facts_inserted": 0, "insights_inserted": 0, "error": str(e)}

        inserted_fact_ids, facts_inserted = await self._process_facts(
            new_facts, session_id, user_id, agent_id, conversation
        )

        # ── Call 2: Insights ──
        try:
            new_insights = await self._extract_insights(
                conversation, new_facts, existing_facts
            )
        except Exception as e:
            logger.warning("Insight extraction failed for session %s: %s", session_id, e)
            new_insights = []  # graceful degradation — facts already stored

        insights_inserted = await self._process_insights(
            new_insights, session_id, user_id, agent_id, inserted_fact_ids, conversation
        )

        # ── Cross-session trigger: every 3rd session ──
        try:
            session_count = await self.db.count_sessions(user_id, agent_id)
            if session_count % 3 == 0 and session_count > 1:
                await self.extract_cross_session_insights(user_id, agent_id)
        except Exception as e:
            logger.warning("Cross-session insight trigger failed: %s", e)

        logger.info(
            "Extraction complete for session %s: %d facts, %d insights",
            session_id, facts_inserted, insights_inserted,
        )
        return {"facts_inserted": facts_inserted, "insights_inserted": insights_inserted}

    # ── LLM Calls ─────────────────────────────────────────────────────────────

    async def _extract_facts(
        self,
        conversation: str,
        existing_facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        existing_text = (
            "\n".join(f"- {f['text']}" for f in existing_facts)
            if existing_facts else "None yet."
        )
        prompt = FACTS_PROMPT.format(
            conversation=conversation,
            existing_facts=existing_text,
        )
        response = await self.llm.generate(prompt)
        return _parse_json_response(response).get("facts", [])

    async def _extract_insights(
        self,
        conversation: str,
        new_facts: list[dict[str, Any]],
        existing_facts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        all_facts_lines = (
            [f"- {f['text']}" for f in existing_facts] +
            [f"- [NEW] {f['text']}" for f in new_facts]
        )
        all_facts = "\n".join(all_facts_lines) if all_facts_lines else "None yet."
        prompt = INSIGHTS_PROMPT.format(
            conversation=conversation,
            all_facts=all_facts,
        )
        response = await self.llm.generate(prompt)
        return _parse_json_response(response).get("insights", [])

    # ── Processing ────────────────────────────────────────────────────────────

    async def _process_facts(
        self,
        fact_list: list[dict[str, Any]],
        session_id: str,
        user_id: str,
        agent_id: str | None,
        conversation: str,
    ) -> tuple[dict[str, str], int]:
        """Store facts, resolve contradictions, link to source sentences."""
        inserted_fact_ids: dict[str, str] = {}
        facts_inserted = 0

        for fact_data in fact_list:
            try:
                fact_embedding = await self.embedder.embed(fact_data["text"])

                supersedes_id = None
                if fact_data.get("contradicts"):
                    old_fact = await self._find_contradicted_fact(
                        fact_data["contradicts"], user_id, agent_id
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
                    confidence=fact_data.get("confidence", 1.0),
                    superseded_by_target=supersedes_id,
                )
                inserted_fact_ids[fact_data["text"]] = fact_id
                facts_inserted += 1

                if fact_data.get("source_quotes"):
                    linked = await self._link_to_source_sentences(
                        fact_data["source_quotes"], session_id, user_id, agent_id, conversation
                    )
                    for sent_id in linked:
                        await self.db.insert_fact_source(fact_id, sent_id)

            except Exception as e:
                logger.warning("Failed to insert fact '%s': %s", fact_data.get("text"), e)

        return inserted_fact_ids, facts_inserted

    async def _process_insights(
        self,
        insight_list: list[dict[str, Any]],
        session_id: str,
        user_id: str,
        agent_id: str | None,
        inserted_fact_ids: dict[str, str],
        conversation: str,
    ) -> int:
        """Store insights, link to facts (L1↔L0 bridge) and source sentences."""
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

                # Link insight → facts (LLM returns verbatim text → exact match fine here)
                for fact_text in insight_data.get("derived_from_facts", []):
                    if fact_text in inserted_fact_ids:
                        await self.db.insert_insight_fact(
                            insight_id, inserted_fact_ids[fact_text]
                        )
                    else:
                        existing = await self.db.find_fact_by_text(
                            user_id, fact_text, agent_id
                        )
                        if existing:
                            await self.db.insert_insight_fact(insight_id, existing["id"])

                if insight_data.get("source_quotes"):
                    linked = await self._link_to_source_sentences(
                        insight_data["source_quotes"], session_id, user_id, agent_id, conversation
                    )
                    for sent_id in linked:
                        await self.db.insert_insight_source(insight_id, sent_id)

            except Exception as e:
                logger.warning("Failed to insert insight '%s': %s", insight_data.get("text"), e)

        return insights_inserted

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _find_contradicted_fact(
        self,
        contradiction_text: str,
        user_id: str,
        agent_id: str | None,
    ) -> dict[str, Any] | None:
        """
        Find the existing fact that contradicts. Uses embedding similarity (not exact text)
        so paraphrases like 'User likes email' still match 'User prefers email'.
        Threshold 0.85 — must be clearly the same fact.
        """
        if not contradiction_text:
            return None
        try:
            embedding = await self.embedder.embed(contradiction_text)
            candidates = await self.db.search_facts(
                embedding=embedding,
                user_id=user_id,
                agent_id=agent_id,
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
        user_id: str,
        agent_id: str | None,
        conversation: str,
    ) -> list[str]:
        """
        Link extracted content to source sentences.
        Step 1: hallucination guard — skip quotes not in conversation
        Step 2: exact substring match via find_sentence_containing()
        Step 3: embedding similarity fallback via search_sentences()
        """
        linked_ids: list[str] = []

        for quote in source_quotes:
            # Step 1: guard against hallucinated quotes
            if quote.lower() not in conversation.lower():
                continue

            # Step 2: exact substring match (fast, no embedding needed)
            exact = await self.db.find_sentence_containing(session_id, quote)
            if exact:
                linked_ids.append(exact["id"])
                continue

            # Step 3: embedding similarity fallback
            try:
                q_emb = await self.embedder.embed(quote)
                candidates = await self.db.search_sentences(
                    q_emb, user_id, agent_id, limit=5
                )
                for c in candidates:
                    if (
                        c.get("session_id") == session_id
                        and (1 - c["distance"]) >= 0.80
                    ):
                        linked_ids.append(c["id"])
            except Exception as e:
                logger.debug("Similarity fallback failed for quote: %s", e)

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
        """
        all_facts = await self.db.get_active_facts(user_id, agent_id, limit=100)
        if not all_facts:
            return {"insights_inserted": 0}

        # Group by session_id — only sessions with facts can yield patterns
        facts_by_session: dict[str, list[str]] = {}
        for fact in all_facts:
            sid = fact.get("session_id") or "unknown"
            facts_by_session.setdefault(sid, []).append(fact["text"])

        if len(facts_by_session) < 2:
            return {"insights_inserted": 0}

        session_summary = "\n".join(
            f"Session {sid}:\n" + "\n".join(f"  - {f}" for f in facts)
            for sid, facts in facts_by_session.items()
        )
        prompt = CROSS_SESSION_INSIGHTS_PROMPT.format(facts_by_session=session_summary)

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

                for fact_text in insight_data.get("derived_from_facts", []):
                    existing = await self.db.find_fact_by_text(user_id, fact_text, agent_id)
                    if existing:
                        await self.db.insert_insight_fact(insight_id, existing["id"])

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
