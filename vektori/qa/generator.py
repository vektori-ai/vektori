"""Shared QA prompt builder and answer generation for retrieved memory."""

from __future__ import annotations

import logging
import re
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
1. Use only the provided context. Logical inference and reasoning from context facts is expected and correct — this is NOT guessing. Do not introduce facts that have no support in any context item.
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
7. If the context contains any relevant evidence, commit to the most supported answer — even if it requires reasoning. Reserve "I don't have that information" strictly for when the context has zero relevant facts about the subject. Committing to a reasoned answer is preferred over abstaining.
8. For answers expressed as "N days/weeks/months before/after DATE": use the temporal note in the context to compute the actual calendar date and give it as an absolute date (e.g. "18 May 2023"). Do not echo the anchor date as the answer.
9. If the context contains facts about multiple named people, only use facts about the person the question asks about. Facts labeled "User" or "Assistant" refer to the primary conversation participant — if other facts establish that person's name (e.g. "User's name is Caroline"), treat "User" facts as belonging to that person.

ANSWER:
"""


_TYPE3_SCAFFOLD = (
    "\nNOTE: This is an inference question (\"Would X...?\", \"Is X likely...?\", "
    "\"What might X...?\"). Step 1: silently list what the context reveals about "
    "the subject's relevant preferences, habits, and situation. Step 2: use that "
    "evidence to commit to a Yes/No answer with a one-sentence justification. "
    "Refusing to answer is only acceptable if the context has zero facts about the subject."
)


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
    prompt = template.format(
        date_line=f"{date_line}{type_line}",
        context=context,
        question=question,
    )
    if str(question_type) == "3":
        prompt = prompt.rstrip() + _TYPE3_SCAFFOLD + "\n"
    return prompt


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
        # Unwrap JSON-wrapped answers (safety net — should not trigger after json_mode fix)
        try:
            import json as _json
            parsed = _json.loads(answer)
            if isinstance(parsed, dict) and "answer" in parsed:
                answer = str(parsed["answer"]).strip()
            elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "answer" in parsed[0]:
                answer = str(parsed[0]["answer"]).strip()
        except Exception:
            # Fallback: regex extraction handles malformed/truncated JSON
            m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', answer)
            if m:
                answer = m.group(1).replace('\\"', '"').replace("\\\\", "\\").strip()
        return answer
    except Exception as e:
        logger.warning("Answer generation failed: %s", e)
        return "I don't have that information"
