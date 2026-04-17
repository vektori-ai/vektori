"""Optimize the LoCoMo QA synthesis prompt with GEPA.

This script intentionally freezes retrieval by using retrieved_context values
from an existing LoCoMo *_full_results.json file. GEPA therefore optimizes only
the answer synthesis prompt, matching the observed failure mode where retrieval
surfaces relevant evidence but QA synthesis misses the answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vektori.models.factory import create_llm  # noqa: E402
from vektori.qa.generator import QA_PROMPT  # noqa: E402

PROMPT_COMPONENT = "qa_prompt"

ABSTENTION_MARKERS = (
    "i don't have that information",
    "i do not have that information",
    "i don't have enough information",
    "i do not have enough information",
    "not enough information",
    "i cannot answer",
    "i can't answer",
    "i don't know",
    "i do not know",
    "not mentioned",
    "unable to answer",
)

RELATIVE_TIME_MARKERS = (
    "yesterday",
    "today",
    "tomorrow",
    "last week",
    "next week",
    "last month",
    "next month",
    "recently",
)

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass(frozen=True)
class LocomoExample:
    question_id: str
    question: str
    question_type: str
    question_date: str
    context: str
    expected_answer: str


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    token_f1: float
    exact_match: float
    containment: float
    groundedness: float
    abstained: bool
    used_relative_time: bool
    expected_in_context: bool


def normalize_answer(text: str) -> str:
    """Normalize answer text with light date canonicalization."""
    text = _canonicalize_dates(text.lower())
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_f1(prediction: str, expected: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    expected_tokens = normalize_answer(expected).split()
    if not pred_tokens and not expected_tokens:
        return 1.0
    if not pred_tokens or not expected_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(expected_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, expected: str, context: str) -> ScoreBreakdown:
    pred_norm = normalize_answer(prediction)
    expected_norm = normalize_answer(expected)
    context_norm = normalize_answer(context)

    f1 = token_f1(prediction, expected)
    exact = 1.0 if pred_norm and pred_norm == expected_norm else 0.0
    containment = 0.0
    if pred_norm and expected_norm and (pred_norm in expected_norm or expected_norm in pred_norm):
        containment = 1.0

    groundedness = 0.0
    if pred_norm and pred_norm in context_norm:
        groundedness = 1.0
    elif expected_norm and expected_norm in context_norm and f1 > 0:
        groundedness = min(1.0, f1 + 0.2)

    abstained = is_abstention(prediction)
    used_relative_time = contains_relative_time(prediction)
    expected_in_context = bool(expected_norm and expected_norm in context_norm)

    # Primary GEPA score: deterministic, LoCoMo-style answer correctness.
    # Grounding is intentionally a small tie-breaker, not a judge replacement.
    correctness = max(f1, exact, containment)
    score = 0.9 * correctness + 0.1 * groundedness

    if abstained and expected_norm:
        score = min(score, 0.05)
    if used_relative_time and _looks_temporal(expected) and not contains_relative_time(expected):
        score = min(score, 0.5 * score)

    return ScoreBreakdown(
        score=round(max(0.0, min(1.0, score)), 6),
        token_f1=round(f1, 6),
        exact_match=exact,
        containment=containment,
        groundedness=round(groundedness, 6),
        abstained=abstained,
        used_relative_time=used_relative_time,
        expected_in_context=expected_in_context,
    )


def is_abstention(text: str) -> bool:
    normalized = normalize_answer(text)
    return any(marker.replace("'", "") in normalized for marker in ABSTENTION_MARKERS)


def contains_relative_time(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(marker)}\b", lowered) for marker in RELATIVE_TIME_MARKERS)


def _looks_temporal(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", lowered)
        or re.search(r"\b\d{1,2}\s+[a-z]+\s+\d{4}\b", lowered)
        or re.search(r"\b[a-z]+\s+\d{1,2},?\s+\d{4}\b", lowered)
    )


def _canonicalize_dates(text: str) -> str:
    text = re.sub(
        r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
        lambda m: _safe_iso(int(m.group(1)), int(m.group(2)), int(m.group(3))),
        text,
    )
    text = re.sub(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+),?\s+(\d{4})\b",
        lambda m: _safe_iso(int(m.group(3)), MONTHS.get(m.group(2), 0), int(m.group(1))),
        text,
    )
    text = re.sub(
        r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        lambda m: _safe_iso(int(m.group(3)), MONTHS.get(m.group(1), 0), int(m.group(2))),
        text,
    )
    return text


def _safe_iso(year: int, month: int, day: int) -> str:
    try:
        return datetime(year, month, day).date().isoformat()
    except ValueError:
        return f"{year:04d} {month:02d} {day:02d}"


def find_latest_full_results(results_dir: Path) -> Path:
    candidates = sorted(
        results_dir.rglob("*_full_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No *_full_results.json files found under {results_dir}")
    return candidates[0]


def load_dataset_metadata(dataset_path: Path) -> dict[str, dict[str, Any]]:
    if not dataset_path.exists():
        return {}
    with dataset_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["question_id"]: row for row in rows if row.get("question_id")}


def load_examples(full_results_path: Path, dataset_path: Path) -> list[LocomoExample]:
    with full_results_path.open("r", encoding="utf-8") as f:
        full_results = json.load(f)

    metadata_by_qid = load_dataset_metadata(dataset_path)
    examples: list[LocomoExample] = []
    for row in full_results.get("qa_results", []):
        question = row.get("question", "")
        expected = row.get("expected_answer", "")
        context = row.get("retrieved_context", "")
        if not question or not expected or not context:
            continue
        if "No relevant context" in context:
            continue

        qid = row.get("question_id", "")
        metadata = metadata_by_qid.get(qid, {})
        examples.append(
            LocomoExample(
                question_id=qid,
                question=question,
                question_type=str(row.get("question_type") or metadata.get("question_type") or ""),
                question_date=str(row.get("question_date") or metadata.get("question_date") or ""),
                context=context,
                expected_answer=expected,
            )
        )
    return examples


def split_examples(
    examples: list[LocomoExample],
    *,
    train_size: int,
    val_size: int,
    seed: int,
    max_examples: int | None,
) -> tuple[list[LocomoExample], list[LocomoExample]]:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    if max_examples is not None:
        shuffled = shuffled[:max_examples]

    if len(shuffled) < 2:
        raise ValueError("Need at least 2 examples for GEPA train/val split")

    train = shuffled[:train_size]
    val = shuffled[train_size : train_size + val_size]
    if not train:
        raise ValueError("Train split is empty")
    if not val:
        val = shuffled[-min(len(shuffled), val_size or 1) :]
    return train, val


class LocomoQAAdapter:
    """GEPA adapter for frozen-context LoCoMo QA synthesis optimization."""

    def __init__(self, model: str, max_tokens: int = 500) -> None:
        self.llm = create_llm(model)
        self.max_tokens = max_tokens

    def evaluate(
        self,
        batch: list[LocomoExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        try:
            from gepa.core.adapter import EvaluationBatch
        except ImportError as exc:
            raise RuntimeError(
                "GEPA is not installed. Install it with: "
                "python -m pip install git+https://github.com/gepa-ai/gepa.git"
            ) from exc

        prompt_template = candidate.get(PROMPT_COMPONENT, "")
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None

        predictions = asyncio.run(self._generate_batch(prompt_template, batch))
        for example, prediction, prompt, error in predictions:
            if error:
                breakdown = ScoreBreakdown(
                    score=0.0,
                    token_f1=0.0,
                    exact_match=0.0,
                    containment=0.0,
                    groundedness=0.0,
                    abstained=False,
                    used_relative_time=False,
                    expected_in_context=False,
                )
                feedback = f"Execution failed before scoring: {error}"
            else:
                breakdown = score_answer(prediction, example.expected_answer, example.context)
                feedback = self._feedback(example, prediction, breakdown)

            outputs.append(
                {
                    "answer": prediction,
                    "prompt": prompt,
                    "question_id": example.question_id,
                    "feedback": feedback,
                }
            )
            scores.append(breakdown.score)
            objective_scores.append(
                {
                    "answer_score": breakdown.score,
                    "token_f1": breakdown.token_f1,
                    "exact_match": breakdown.exact_match,
                    "containment": breakdown.containment,
                    "groundedness": breakdown.groundedness,
                }
            )
            if trajectories is not None:
                trajectories.append(
                    {
                        "question_id": example.question_id,
                        "question": example.question,
                        "question_type": example.question_type,
                        "question_date": example.question_date,
                        "expected_answer": example.expected_answer,
                        "generated_answer": prediction,
                        "retrieved_context": example.context,
                        "prompt": prompt,
                        "score_breakdown": breakdown.__dict__,
                        "feedback": feedback,
                    }
                )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    async def _generate_batch(
        self,
        prompt_template: str,
        batch: list[LocomoExample],
    ) -> list[tuple[LocomoExample, str, str, str | None]]:
        async def one(example: LocomoExample) -> tuple[LocomoExample, str, str, str | None]:
            try:
                date_line = (
                    f"TODAY'S DATE: {example.question_date}\n\n"
                    if example.question_date
                    else ""
                )
                if example.question_type:
                    date_line = f"{date_line}QUESTION TYPE: {example.question_type}\n"
                prompt = prompt_template.format(
                    date_line=date_line,
                    context=example.context,
                    question=example.question,
                )
            except Exception as exc:
                return example, "", "", f"Prompt formatting error: {exc}"

            try:
                answer = await self.llm.generate(prompt, max_tokens=self.max_tokens)
                return example, answer.strip(), prompt, None
            except Exception as exc:
                return example, "", prompt, f"LLM generation error: {exc}"

        return await asyncio.gather(*(one(example) for example in batch))

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        del candidate
        records: dict[str, list[dict[str, Any]]] = {component: [] for component in components_to_update}
        trajectories = eval_batch.trajectories
        if trajectories is None:
            raise ValueError("Trajectories are required for GEPA reflection")

        for component in components_to_update:
            if component != PROMPT_COMPONENT:
                continue
            for traj in trajectories:
                records[component].append(
                    {
                        "Question ID": traj["question_id"],
                        "Question": traj["question"],
                        "Question type": traj["question_type"],
                        "Question date": traj["question_date"],
                        "Expected answer": traj["expected_answer"],
                        "Generated answer": traj["generated_answer"],
                        "Score breakdown": traj["score_breakdown"],
                        "Feedback": traj["feedback"],
                        "Retrieved context": traj["retrieved_context"],
                        "Prompt used": traj["prompt"],
                    }
                )
        return records

    def _feedback(
        self,
        example: LocomoExample,
        prediction: str,
        breakdown: ScoreBreakdown,
    ) -> str:
        lines = [
            f"Score: {breakdown.score:.3f}",
            f"Token F1: {breakdown.token_f1:.3f}",
            f"Exact match: {bool(breakdown.exact_match)}",
            f"Containment match: {bool(breakdown.containment)}",
            f"Groundedness: {breakdown.groundedness:.3f}",
        ]
        if breakdown.abstained:
            lines.append("Failure signal: model abstained even though an expected answer exists.")
        if breakdown.used_relative_time:
            lines.append("Failure signal: answer used relative time wording; prefer absolute dates.")
        if not breakdown.expected_in_context:
            lines.append(
                "Context signal: exact expected answer string was not found after normalization; "
                "the prompt may need paraphrase/list/date extraction discipline."
            )
        if breakdown.score < 1:
            lines.extend(
                [
                    f"Question: {example.question}",
                    f"Expected answer: {example.expected_answer}",
                    f"Generated answer: {prediction}",
                    "Improve the QA prompt so the model extracts exact names, dates, quantities, "
                    "and complete lists from the retrieved context.",
                ]
            )
        return "\n".join(lines)


def run_gepa(args: argparse.Namespace) -> None:
    try:
        import gepa
    except ImportError as exc:
        raise SystemExit(
            "GEPA is not installed in this environment.\n"
            "Install it first, for example:\n"
            "  py -m ensurepip --upgrade\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install "
            "git+https://github.com/gepa-ai/gepa.git"
        ) from exc

    full_results_path = (
        Path(args.full_results)
        if args.full_results
        else find_latest_full_results(Path(args.results_dir))
    )
    examples = load_examples(full_results_path, Path(args.dataset))
    trainset, valset = split_examples(
        examples,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        max_examples=args.max_examples,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using full results: {full_results_path}")
    print(f"Loaded examples: {len(examples)}")
    print(f"Train/val: {len(trainset)}/{len(valset)}")
    print(f"Task model: {args.task_model}")
    print(f"Reflection LM: {args.reflection_lm}")
    print(f"Run dir: {run_dir}")

    adapter = LocomoQAAdapter(model=args.task_model, max_tokens=args.max_tokens)
    result = gepa.optimize(
        seed_candidate={PROMPT_COMPONENT: QA_PROMPT},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=args.reflection_lm,
        reflection_lm_kwargs=json.loads(args.reflection_lm_kwargs),
        candidate_selection_strategy=args.candidate_selection_strategy,
        frontier_type=args.frontier_type,
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        perfect_score=1.0,
        run_dir=str(run_dir),
        cache_evaluation=True,
        seed=args.seed,
        display_progress_bar=True,
        use_merge=args.use_merge,
    )

    best_candidate = result.best_candidate
    if not isinstance(best_candidate, dict):
        raise TypeError(f"Expected dict best_candidate, got {type(best_candidate)}")
    best_prompt = best_candidate[PROMPT_COMPONENT]
    best_prompt_path = run_dir / "best_qa_prompt.txt"
    best_prompt_path.write_text(best_prompt, encoding="utf-8")

    summary = {
        "full_results": str(full_results_path),
        "dataset": args.dataset,
        "task_model": args.task_model,
        "reflection_lm": args.reflection_lm,
        "train_size": len(trainset),
        "val_size": len(valset),
        "best_idx": result.best_idx,
        "best_val_score": result.val_aggregate_scores[result.best_idx],
        "total_metric_calls": result.total_metric_calls,
        "best_prompt_path": str(best_prompt_path),
    }
    (run_dir / "locomo_gepa_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GEPA on the LoCoMo QA prompt")
    parser.add_argument("--full-results", default="", help="Path to *_full_results.json")
    parser.add_argument(
        "--results-dir",
        default="benchmark_results",
        help="Used to find latest *_full_results.json when --full-results is omitted",
    )
    parser.add_argument("--dataset", default="data/locomo10_cooked.json")
    parser.add_argument("--run-dir", default="benchmark_results/gepa_locomo_qa_v1")
    parser.add_argument(
        "--task-model",
        default="vllm:Qwen/Qwen3-8B",
        help=(
            "Vektori model string used for QA candidate evaluation. "
            "Examples: vllm:Qwen/Qwen3-8B, ollama:qwen3:8b, gemini:gemini-2.5-flash-lite"
        ),
    )
    parser.add_argument(
        "--reflection-lm",
        default="hosted_vllm/Qwen/Qwen3-8B",
        help=(
            "LiteLLM model string used by GEPA for reflection. "
            "For local vLLM, use hosted_vllm/<served-model-name> with --reflection-lm-kwargs."
        ),
    )
    parser.add_argument(
        "--reflection-lm-kwargs",
        default='{"api_base":"http://localhost:8000/v1","api_key":"local"}',
        help="JSON kwargs passed to GEPA/LiteLLM reflection LM",
    )
    parser.add_argument("--train-size", type=int, default=60)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--reflection-minibatch-size", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--candidate-selection-strategy",
        choices=["pareto", "current_best", "epsilon_greedy", "top_k_pareto"],
        default="pareto",
    )
    parser.add_argument(
        "--frontier-type",
        choices=["instance", "objective", "hybrid", "cartesian"],
        default="instance",
    )
    parser.add_argument("--use-merge", action="store_true")
    return parser


if __name__ == "__main__":
    run_gepa(build_arg_parser().parse_args())
