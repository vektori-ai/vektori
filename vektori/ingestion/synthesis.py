import logging

from vektori.ingestion.extractor import _parse_json_response
from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """You are an AI tasked with finding overarching patterns, habits, or aggregate facts about a user over time based on their individual session facts.

Below are facts extracted from the user's past conversations. Look for things that aggregate across multiple sessions (e.g., "User has been to the gym 4 times since Jan 2024", "User consistently prefers evening meetings", "User changed their diet from vegetarian to vegan around March").

Facts:
{facts_list}

Extract up to {max_facts} aggregate facts. Do NOT regurgitate single-session events. Only output if a pattern spans multiple sessions or shows an accumulated count/trend.

Return JSON:
{{
  "facts": [
    {{
      "text": "short factual statement under 20 words",
      "confidence": 0.95
    }}
  ]
}}
If nothing aggregates meaningfully, return {{"facts": []}}.
Return ONLY the JSON.
"""


class Synthesizer:
    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.llm = llm

    async def synthesize(self, user_id: str, agent_id: str | None = None) -> int:
        # Get active facts for the user
        facts = await self.db.get_active_facts(user_id=user_id, agent_id=agent_id, limit=300)

        # Filter out existing synthesis facts so we don't synthesize the syntheses too much,
        # or maybe we do want to? For now, let's keep it simple: filter them out to avoid feedback loops unless needed.
        # Actually, let's just feed them all, but maybe limit to non-synthesis for base patterns.
        base_facts = [f for f in facts if f.get("metadata", {}).get("source") != "synthesis"]

        if len(base_facts) < 5:
            return 0  # Not enough facts to form a pattern

        facts_list = "\n".join(
            f"- {f['text']} (Session: {f.get('session_id', 'unknown')}, Date: {f.get('created_at', 'unknown')})"
            for f in base_facts
        )

        prompt = SYNTHESIS_PROMPT.format(facts_list=facts_list, max_facts=5)
        try:
            response = await self.llm.generate(prompt, max_tokens=1000)
            data = _parse_json_response(response)
            new_facts = data.get("facts", [])
        except Exception as e:
            logger.warning("Synthesis LLM call failed: %s", e)
            return 0

        if not new_facts:
            return 0

        valid_facts: list[dict] = []
        for item in new_facts:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            valid_facts.append({**item, "text": text.strip()})

        if not valid_facts:
            return 0

        texts = [f["text"] for f in valid_facts]
        try:
            embeddings = await self.embedder.embed_batch(texts)
        except Exception as e:
            logger.error("Synthesis batch embed failed: %s", e)
            return 0

        inserted = 0
        source_fact_ids = [f["id"] for f in base_facts if f.get("id")]

        for fact_dict, emb in zip(valid_facts, embeddings):
            # Check dedup against existing synthesis
            existing = await self.db.search_syntheses(
                embedding=emb, user_id=user_id, agent_id=agent_id, limit=5
            )

            skip = False
            if existing:
                best = existing[0]
                sim = 1.0 - best.get("distance", 1.0)
                if sim > 0.85:
                    # We already have this synthesis, let's just skip it
                    skip = True

            if skip:
                continue

            try:
                synthesis_id = await self.db.insert_synthesis(
                    text=fact_dict["text"],
                    embedding=emb,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=None,
                )
                for fact_id in source_fact_ids:
                    await self.db.insert_synthesis_fact(synthesis_id, fact_id)
                inserted += 1
            except Exception as e:
                logger.warning("Failed to insert synthesis: %s", e)

        return inserted
