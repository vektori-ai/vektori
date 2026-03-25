"""LLM-based fact and insight extraction with conflict detection."""

from __future__ import annotations

import json
import logging
from typing import Any

from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Analyze this conversation and extract two types of information: FACTS and INSIGHTS.

CONVERSATION:
{conversation}

EXISTING KNOWN FACTS ABOUT THIS USER:
{existing_facts}

Return JSON with two arrays:

{{
  "facts": [
    {{
      "text": "clear, standalone factual statement",
      "confidence": 0.95,
      "source_quotes": ["exact quote from conversation this fact comes from"],
      "contradicts": "text of existing fact this contradicts, or null"
    }}
  ],
  "insights": [
    {{
      "text": "inferred pattern or observation, actionable for the agent",
      "confidence": 0.80,
      "derived_from_facts": ["fact text this insight is based on"],
      "source_quotes": ["exact quotes supporting this inference"]
    }}
  ]
}}

FACT rules:
- Explicit, verifiable information stated in the conversation
- Short, crisp statements (under 20 words ideally)
- One fact per statement — don't combine multiple pieces of info
- Check EVERY new fact against existing facts for contradictions

INSIGHT rules:
- Inferred patterns NOT explicitly stated in the conversation
- Must be actionable — "borrower gets defensive when X" is good, "conversation happened" is worthless
- Can reference multiple facts and span cross-session patterns
- Higher confidence bar — only extract insights you're reasonably sure about

Return ONLY the JSON, nothing else."""


class FactExtractor:
    """
    Extracts facts (L0) and insights (L1) from conversations using an LLM.

    Facts are explicit, verifiable statements extracted from the conversation.
    Insights are inferred patterns discovered across facts and sessions.
    Both are linked back to source sentences via join tables.
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
        Extract facts and insights. Handle conflict resolution.

        Returns {"facts_inserted": N, "insights_inserted": M}.
        """
        # Retrieve relevant existing facts to check for contradictions.
        # We only pull 50 — enough to catch likely contradictions without overloading the prompt.
        existing_facts = await self.db.get_active_facts(user_id, agent_id, limit=50)
        existing_facts_text = (
            "\n".join(f"- {f['text']}" for f in existing_facts)
            if existing_facts
            else "None yet."
        )

        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
        )

        prompt = EXTRACTION_PROMPT.format(
            conversation=conversation,
            existing_facts=existing_facts_text,
        )

        try:
            response = await self.llm.generate(prompt)
            extracted = _parse_json_response(response)
        except Exception as e:
            logger.error("Fact extraction failed for session %s: %s", session_id, e)
            return {"facts_inserted": 0, "insights_inserted": 0, "error": str(e)}

        inserted_fact_ids: dict[str, str] = {}
        facts_inserted = 0
        insights_inserted = 0

        # ── Process FACTS ──
        for fact_data in extracted.get("facts", []):
            try:
                fact_embedding = await self.embedder.embed(fact_data["text"])

                # Handle contradiction: deactivate old fact, link new one
                supersedes_id = None
                if fact_data.get("contradicts"):
                    old_fact = await self.db.find_fact_by_text(
                        user_id, fact_data["contradicts"], agent_id
                    )
                    if old_fact:
                        supersedes_id = old_fact["id"]
                        await self.db.deactivate_fact(old_fact["id"])

                fact_id = await self.db.insert_fact(
                    text=fact_data["text"],
                    embedding=fact_embedding,
                    user_id=user_id,
                    agent_id=agent_id,
                    confidence=fact_data.get("confidence", 1.0),
                    superseded_by_target=supersedes_id,
                )
                inserted_fact_ids[fact_data["text"]] = fact_id
                facts_inserted += 1

                # Link fact → source sentences via fact_sources table
                if fact_data.get("source_quotes"):
                    source_sents = await self.db.find_sentences_by_similarity(
                        fact_data["source_quotes"], session_id, threshold=0.75
                    )
                    for sent_id in source_sents:
                        await self.db.insert_fact_source(fact_id, sent_id)

            except Exception as e:
                logger.warning("Failed to insert fact '%s': %s", fact_data.get("text"), e)

        # ── Process INSIGHTS ──
        for insight_data in extracted.get("insights", []):
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

                # Link insight → related facts via insight_facts (the key bridge)
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

                # Link insight → source sentences via insight_sources
                if insight_data.get("source_quotes"):
                    source_sents = await self.db.find_sentences_by_similarity(
                        insight_data["source_quotes"], session_id, threshold=0.75
                    )
                    for sent_id in source_sents:
                        await self.db.insert_insight_source(insight_id, sent_id)

            except Exception as e:
                logger.warning("Failed to insert insight '%s': %s", insight_data.get("text"), e)

        logger.info(
            "Extraction complete for session %s: %d facts, %d insights",
            session_id, facts_inserted, insights_inserted,
        )
        return {"facts_inserted": facts_inserted, "insights_inserted": insights_inserted}


def _parse_json_response(response: str) -> dict[str, Any]:
    """Parse LLM JSON response, stripping markdown code fences if present."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Strip opening fence (```json or ```) and closing ```
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse extraction JSON: %s\nResponse: %.500s", e, response)
        return {"facts": [], "insights": []}
