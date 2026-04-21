"""Shared QA prompt builder and answer generation for retrieved memory."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class SupportsGenerate(Protocol):
    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate a completion for the given prompt."""
        ...


QA_PROMPT = """You are a memory QA assistant. Your job is to answer the question by extracting the answer from the provided long-term memory context.

{date_line}CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. Use only the provided context. Do not use outside knowledge or guesses.
2. First, silently find every fact, episode, synthesis, or transcript line that directly relates to the question. Do not print this evidence list.
3. For counting, frequency, list, "all", "total", or aggregation questions:
   - Consider every relevant item across all sessions, not just the first match.
   - Deduplicate only if the context clearly describes the same item more than once.
   - Return the complete set or total. A compact calculation is allowed when it helps.
4. For date, time, order, recency, or "when" questions:
   - Use exact absolute dates from the context whenever available.
   - Do not answer with relative time words like "yesterday", "today", "last week", or "recently" when an absolute date is available.
   - If the context includes a temporal note, use it to resolve relative wording.
5. For changed or updated information:
   - Prefer the most recent value when later context overrides earlier context.
   - Mention older values only if the question asks for history or change over time.
6. Copy critical names, dates, places, titles, quantities, and field names exactly from the context. Do not blur them into a generic paraphrase.
7. Say "I don't have that information" only when no context item supports an answer.
8. For answers expressed as "N days/weeks/months before/after DATE": use the temporal note in the context to compute the actual calendar date and give it as an absolute date (e.g. "18 May 2023"). Do not echo the anchor date as the answer.
9. If the context contains facts about multiple named people, only use facts about the person the question asks about. Facts labeled "User" or "Assistant" refer to the primary conversation participant — if other facts establish that person's name (e.g. "User's name is Caroline"), treat "User" facts as belonging to that person.

ANSWER:
"""


def build_qa_prompt(
    question: str,
    context: str,
    *,
    question_date: str = "",
    question_type: str = "",
    prompt_template: str | None = None,
) -> str:
    """Build the QA prompt used by memory benchmark answer generation."""
    date_line = f"TODAY'S DATE: {question_date}\n\n" if question_date else ""
    type_line = f"QUESTION TYPE: {question_type}\n" if question_type else ""
    template = prompt_template or QA_PROMPT
    return template.format(
        date_line=f"{date_line}{type_line}",
        context=context,
        question=question,
    )


async def generate_answer(
    *,
    question: str,
    context: str,
    question_date: str = "",
    question_type: str = "",
    llm: SupportsGenerate | None = None,
    model: str | None = None,
    prompt_template: str | None = None,
    max_tokens: int = 500,
) -> str:
    """Generate an answer from retrieved context using the shared QA prompt."""
    if "No relevant context" in context:
        return "I don't have that information"

    if llm is None:
        if not model:
            raise ValueError("Either llm or model must be provided")
        from vektori.models.factory import create_llm

        llm = create_llm(model)

    prompt = build_qa_prompt(
        question,
        context,
        question_date=question_date,
        question_type=question_type,
        prompt_template=prompt_template,
    )
    try:
        answer = (await llm.generate(prompt, max_tokens=max_tokens)).strip()
        # Unwrap {"answer": "..."} JSON that some models (e.g. gemini) occasionally return
        try:
            import json as _json
            parsed = _json.loads(answer)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = str(parsed["answer"]).strip()
        except Exception:
            pass
        return answer
    except Exception as e:
        logger.warning("Answer generation failed: %s", e)
        return "Unable to generate answer due to API error."
