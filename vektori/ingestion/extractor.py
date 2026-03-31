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

Return ONLY the JSON."""


INSIGHTS_PROMPT = """You are writing episodic memory records. Given this conversation and the facts extracted from it, write a concise third-person narrative episode describing what happened.

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
- One episode per distinct topic in the batch; {max_insights} maximum
- Return {{"episodes": []}} if nothing notable

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
        conversation = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in messages
        )

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
            new_facts, session_id, user_id, agent_id, conversation, session_time,
            _capture_out=_capture_out,
            _inserted_facts_out=inserted_facts,
        )

        insights_created = 0
        if inserted_facts:
            try:
                insights_created = await self._extract_insights(
                    inserted_facts, conversation, session_id, user_id, agent_id,
                    session_time=session_time,
                )
            except Exception as e:
                logger.warning("Insight extraction failed for session %s: %s", session_id, e)

        logger.info(
            "Extraction complete for session %s: %d facts, %d insights",
            session_id, facts_inserted, insights_created,
        )
        return {"facts_inserted": facts_inserted, "insights_created": insights_created}

    # ── LLM Call ──────────────────────────────────────────────────────────────

    async def _extract_facts(
        self, conversation: str, session_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        if session_time:
            session_date_line = f"CONVERSATION DATE: {session_time.strftime('%Y-%m-%d')}\n\n"
        else:
            session_date_line = ""
        prompt = FACTS_PROMPT.format(
            conversation=conversation,
            max_facts=self.max_facts,
            session_date_line=session_date_line,
        )
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
            len(full_conversation), len(chunks),
        )
        all_facts: list[dict[str, Any]] = []
        for chunk in chunks:
            chunk_conv = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in chunk
            )
            try:
                chunk_facts = await self._extract_facts(chunk_conv, session_time)
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
                    fact_embedding, session_id, user_id, agent_id, subject
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
                    _capture_out.append({
                        "text": fact_data["text"],
                        "subject": subject,
                        "confidence": fact_data.get("confidence", 1.0),
                        "metadata": meta or {},
                        "source_quotes": fact_data.get("source_quotes") or [],
                    })

                if fact_data.get("source_quotes"):
                    linked = await self._link_to_source_sentences(
                        fact_data["source_quotes"], session_id, conversation
                    )
                    for sent_id in linked:
                        await self.db.insert_fact_source(fact_id, sent_id)

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
                dedup = await self._check_dedup(fact_emb, session_id, user_id, agent_id, subject)
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

    async def _extract_insights(
        self,
        inserted_facts: list[tuple[str, str]],
        conversation: str,
        session_id: str,
        user_id: str,
        agent_id: str | None,
        max_insights: int = 5,
        session_time: datetime | None = None,
    ) -> int:
        """Extract episodic memory narratives from the conversation and its facts.

        Episodes are concise third-person narratives of what happened in this
        session batch — grounded stories with resolved entities and dates. They
        are vector-embedded so they can be found both via graph traversal (fact →
        insight_facts → insights) and direct cosine search at retrieval.

        Returns the number of episodes inserted.
        """
        if not inserted_facts:
            return 0

        facts_list = "\n".join(f"{i}. {text}" for i, (_, text) in enumerate(inserted_facts))
        fact_id_list = [fid for fid, _ in inserted_facts]

        # Truncate conversation to keep prompt manageable (~3000 chars ≈ 750 tokens)
        conv_snippet = conversation[:3000]

        session_date_line = (
            f"SESSION DATE: {session_time.strftime('%Y-%m-%d')}\n\n"
            if session_time else ""
        )
        prompt = INSIGHTS_PROMPT.format(
            conversation=conv_snippet,
            facts_list=facts_list,
            max_insights=max_insights,
            session_date_line=session_date_line,
        )
        try:
            response = await self.llm.generate(prompt, max_tokens=1024)
            data = _parse_json_response(response)
        except Exception as e:
            logger.warning("Insight LLM call failed: %s", e)
            return 0

        raw_insights = data.get("episodes", data.get("insights", []))[:max_insights]
        if not raw_insights:
            return 0

        # Batch embed all insight texts in one call
        insight_texts = [(ins.get("text") or "").strip() for ins in raw_insights]
        insight_texts = [t for t in insight_texts if t]
        if not insight_texts:
            return 0

        try:
            embeddings = await self.embedder.embed_batch(insight_texts)
        except Exception as e:
            logger.warning("Batch embed failed for insights: %s", e)
            return 0

        insights_created = 0
        text_iter = iter(zip(insight_texts, embeddings))
        for insight_data in raw_insights:
            text = (insight_data.get("text") or "").strip()
            if not text:
                continue
            indices = insight_data.get("fact_indices") or []

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
                insight_id = await self.db.insert_insight(
                    text, embedding, user_id, agent_id, session_id
                )
                for fid in linked_ids:
                    await self.db.insert_insight_fact(insight_id, fid)
                insights_created += 1
            except Exception as e:
                logger.warning("Failed to insert insight '%s': %s", text, e)

        return insights_created

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
        return {"facts": []}
