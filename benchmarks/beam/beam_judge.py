"""
BEAM Judge
=================
Evaluates the 10 ability types from the BEAM benchmark using an LLM-as-judge.
Falls back to substring match if the LLM call fails.
"""
import json
import logging
import re
from typing import Any, Dict

from vektori.models.factory import create_llm

logger = logging.getLogger(__name__)

ABILITY_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

ABSTENTION_PHRASES = (
    "i don't have that information",
    "i do not have that information",
    "i don't have any information",
    "i do not have any information",
    "i don't have enough information",
    "i do not have enough information",
    "i cannot answer",
    "i can't answer",
    "cannot answer this",
    "not enough information",
    "no information in the provided context",
    "information not available",
    "i don't know",
    "i do not know",
    "not mentioned",
    "never mentioned",
)

JUDGE_PROMPT = """You are evaluating a memory-augmented AI assistant on the BEAM benchmark.

ABILITY TYPE: {ability_type}
QUESTION: {question}
EXPECTED ANSWER: {expected}
MODEL ANSWER: {actual}

Score the model answer. Respond ONLY with valid JSON, no markdown fences.

Scoring rules:
- CORRECT (1.0): model answer matches or clearly contains the expected answer.
- PARTIALLY_CORRECT (0.5): right idea but incomplete or imprecise.
- WRONG (0.0): incorrect, unrelated, or contradicts expected answer.
- ABSTAINED (0.0): model says it does not know / lacks information (unless ability_type is "abstention", in which case ABSTAINED scores 1.0).

Additional rules:
- For "abstention": if the model correctly refuses to answer, score CORRECT. If it hallucinates, score WRONG.
- For "knowledge_update": the model must use the LATEST version of a fact, not an outdated one.
- For "temporal_reasoning": accept equivalent date formats. "2023-10-21" = "October 21, 2023" = CORRECT.
- For "event_ordering": order matters — partial credit only if most events are correctly sequenced.
- If EXPECTED ANSWER is 5 words or fewer, check whether the model answer CONTAINS that core fact. Extra surrounding phrasing is fine.
- Short correct answers (under 25 words) should not be penalized for missing elaboration.
- Extra correct details do NOT make an answer wrong.

JSON schema:
{{"verdict": "CORRECT|PARTIALLY_CORRECT|WRONG|ABSTAINED", "explanation": "one sentence"}}"""

VERDICT_SCORES = {
    "CORRECT": 1.0,
    "PARTIALLY_CORRECT": 0.5,
    "WRONG": 0.0,
    "ABSTAINED": 0.0,
}


def _is_abstention(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def _substring_fallback(expected: str, actual: str) -> float:
    if expected.strip().lower() in actual.strip().lower():
        return 1.0
    return 0.0


class BeamJudge:
    def __init__(self, eval_model: str):
        self.eval_model = eval_model
        self._llm = create_llm(eval_model, json_mode=False)

    async def evaluate_answer(
        self,
        ability_type: str,
        question: str,
        expected: str,
        actual: str,
    ) -> float:
        # Fast-path: abstention ability type — check for abstention phrases directly
        if ability_type == "abstention":
            return 1.0 if _is_abstention(actual) else 0.0

        prompt = JUDGE_PROMPT.format(
            ability_type=ability_type,
            question=question,
            expected=expected,
            actual=actual,
        )
        try:
            raw = await self._llm.generate(prompt, max_tokens=256)
            text = raw.strip()

            parsed: dict[str, Any] | None = None
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass

            if parsed:
                verdict = str(parsed.get("verdict", "")).upper()
                # For abstention ability type with ABSTAINED verdict → correct
                if ability_type == "abstention" and verdict == "ABSTAINED":
                    return 1.0
                return VERDICT_SCORES.get(verdict, 0.0)

        except Exception:
            logger.warning("LLM judge failed for ability_type=%s; falling back to substring match", ability_type, exc_info=True)

        return _substring_fallback(expected, actual)

    def compute_summary(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        metrics: dict[str, dict[str, Any]] = {
            at: {"correct": 0, "total": 0, "score_sum": 0.0, "score": 0.0}
            for at in ABILITY_TYPES
        }

        for res in results:
            atype_raw = res.get("ability_type")
            atype = str(atype_raw) if atype_raw is not None else "unknown"

            if atype not in metrics:
                metrics[atype] = {"correct": 0, "total": 0, "score_sum": 0.0, "score": 0.0}

            score = float(res.get("score", 0.0))
            metrics[atype]["total"] += 1
            metrics[atype]["score_sum"] += score
            if score >= 0.5:
                metrics[atype]["correct"] += 1

        for stats in metrics.values():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
                stats["score"] = stats["score_sum"] / stats["total"]
            del stats["score_sum"]

        overall_total = sum(s["total"] for s in metrics.values())
        overall_correct = sum(s["correct"] for s in metrics.values())
        overall_score = sum(s.get("score", 0.0) * s["total"] for s in metrics.values())

        return {
            "per_ability": metrics,
            "overall": {
                "total": overall_total,
                "correct": overall_correct,
                "accuracy": overall_correct / overall_total if overall_total else 0.0,
                "score": overall_score / overall_total if overall_total else 0.0,
            },
        }
