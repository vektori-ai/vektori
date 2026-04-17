"""LLM-as-a-judge evaluation for LoCoMo benchmark full-results files.

This judge is diagnostic, not the primary GEPA optimization metric. It helps
separate QA synthesis failures from retrieval failures by checking whether the
retrieved context contained enough evidence for the expected answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vektori.models.factory import create_llm

logger = logging.getLogger("locomo_judge")

VALID_VERDICTS = {"CORRECT", "PARTIALLY_CORRECT", "WRONG", "ABSTAINED"}
DEFAULT_JUDGE_MODEL = "vllm:Qwen/Qwen3-8B"

ABSTENTION_PHRASES = (
    "i don't have that information",
    "i do not have that information",
    "i don't have enough information",
    "i do not have enough information",
    "i cannot answer",
    "i can't answer",
    "cannot answer this",
    "not enough information",
    "information not available",
    "i don't know",
    "i do not know",
    "not mentioned",
    "never mentioned",
)

JUDGE_PROMPT_TEMPLATE = """You are evaluating a memory-augmented AI assistant on the LoCoMo benchmark.

QUESTION: {question}
QUESTION TYPE: {question_type}

MODEL ANSWER: {hypothesis}
EXPECTED ANSWER: {expected_answer}

RETRIEVED CONTEXT:
{retrieved_context}

Evaluate the answer and context. Respond ONLY with valid JSON, no markdown.

Rules:
- CORRECT: the model answer contains or clearly matches the expected answer.
- PARTIALLY_CORRECT: the answer has the right idea but is incomplete or imprecise.
- WRONG: the answer is incorrect, unrelated, or contradicts the expected answer.
- ABSTAINED: the model says it does not know or lacks information.
- context_has_answer: true if the retrieved context contains enough evidence to answer correctly, even if phrased differently from EXPECTED ANSWER.
- failure_mode:
  - null if verdict is CORRECT or PARTIALLY_CORRECT
  - "QA_FAILURE" if verdict is WRONG/ABSTAINED and context_has_answer is true
  - "RETRIEVAL_FAILURE" if verdict is WRONG/ABSTAINED and context_has_answer is false
- For list answers, order does not matter.
- For date answers, accept equivalent formats, e.g. "7 May 2023" and "2023-05-07".
- Extra correct details do not make an answer wrong.

JSON schema:
{{"verdict": "CORRECT|PARTIALLY_CORRECT|WRONG|ABSTAINED", "context_has_answer": true|false, "failure_mode": null|"QA_FAILURE"|"RETRIEVAL_FAILURE", "explanation": "one sentence"}}"""


@dataclass
class JudgeConfig:
    full_results: str
    judge_model: str = DEFAULT_JUDGE_MODEL
    n: int = 100
    seed: int = 42
    output_dir: str = "benchmark_results"
    qids: list[str] = field(default_factory=list)


@dataclass
class JudgeVerdict:
    verdict: str
    context_has_answer: bool
    failure_mode: str | None
    explanation: str
    latency_ms: float
    raw_response: str
    parse_error: bool = False


def build_prompt(entry: dict[str, Any]) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=entry.get("question", ""),
        question_type=entry.get("question_type", "unknown"),
        hypothesis=entry.get("hypothesis", "(no answer)"),
        expected_answer=entry.get("expected_answer", ""),
        retrieved_context=entry.get("retrieved_context", "(none)"),
    )


def is_abstention_answer(answer: str | None) -> bool:
    if not answer:
        return False
    normalized = re.sub(r"\s+", " ", answer.lower()).strip()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def parse_verdict(raw: str, latency_ms: float) -> JudgeVerdict:
    text = raw.strip()
    parsed: dict[str, Any] | None = None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        return JudgeVerdict(
            verdict="WRONG",
            context_has_answer=False,
            failure_mode="RETRIEVAL_FAILURE",
            explanation="Could not parse judge JSON.",
            latency_ms=latency_ms,
            raw_response=raw,
            parse_error=True,
        )

    verdict = str(parsed.get("verdict", "WRONG")).upper()
    if verdict not in VALID_VERDICTS:
        verdict = "WRONG"

    context_has_answer = bool(parsed.get("context_has_answer", False))
    failure_mode = parsed.get("failure_mode")
    if verdict in {"CORRECT", "PARTIALLY_CORRECT"}:
        failure_mode = None
    elif failure_mode not in {"QA_FAILURE", "RETRIEVAL_FAILURE"}:
        failure_mode = "QA_FAILURE" if context_has_answer else "RETRIEVAL_FAILURE"

    return JudgeVerdict(
        verdict=verdict,
        context_has_answer=context_has_answer,
        failure_mode=failure_mode,
        explanation=str(parsed.get("explanation") or "").strip() or "(no explanation)",
        latency_ms=latency_ms,
        raw_response=raw,
        parse_error=False,
    )


def load_entries(full_results_path: Path, qids: list[str], n: int, seed: int) -> list[dict[str, Any]]:
    if not full_results_path.exists():
        raise FileNotFoundError(f"Full results file not found: {full_results_path}")

    with full_results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = list(payload.get("qa_results", []))

    if qids:
        by_qid = {entry.get("question_id"): entry for entry in entries}
        missing = [qid for qid in qids if qid not in by_qid]
        if missing:
            raise ValueError(f"Requested QIDs not found: {', '.join(missing[:5])}")
        return [by_qid[qid] for qid in qids]

    rng = random.Random(seed)
    if n <= 0 or n >= len(entries):
        return entries
    return rng.sample(entries, n)


async def evaluate_entry(entry: dict[str, Any], judge_model: str) -> dict[str, Any]:
    llm = create_llm(judge_model)
    prompt = build_prompt(entry)
    t0 = time.perf_counter()
    try:
        raw = await llm.generate(prompt, max_tokens=350)
        latency_ms = (time.perf_counter() - t0) * 1000
        verdict = parse_verdict(raw, latency_ms)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.warning("Judge generation failed for %s: %s", entry.get("question_id"), exc)
        verdict = JudgeVerdict(
            verdict="WRONG",
            context_has_answer=False,
            failure_mode="RETRIEVAL_FAILURE",
            explanation=f"Judge generation failed: {exc}",
            latency_ms=latency_ms,
            raw_response="",
            parse_error=True,
        )

    if verdict.verdict == "WRONG" and is_abstention_answer(entry.get("hypothesis")):
        verdict.verdict = "ABSTAINED"
        verdict.failure_mode = "QA_FAILURE" if verdict.context_has_answer else "RETRIEVAL_FAILURE"

    return {
        **entry,
        "verdict": verdict.verdict,
        "context_has_answer": verdict.context_has_answer,
        "failure_mode": verdict.failure_mode,
        "explanation": verdict.explanation,
        "judge_latency_ms": round(verdict.latency_ms, 1),
        "parse_error": verdict.parse_error,
        "raw_judge_response": verdict.raw_response,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    partial = sum(1 for r in results if r["verdict"] == "PARTIALLY_CORRECT")
    wrong = sum(1 for r in results if r["verdict"] == "WRONG")
    abstained = sum(1 for r in results if r["verdict"] == "ABSTAINED")
    context_has_answer = sum(1 for r in results if r["context_has_answer"])
    qa_failure = sum(1 for r in results if r["failure_mode"] == "QA_FAILURE")
    retrieval_failure = sum(1 for r in results if r["failure_mode"] == "RETRIEVAL_FAILURE")
    parse_errors = sum(1 for r in results if r.get("parse_error"))

    by_type: dict[str, dict[str, int]] = {}
    for result in results:
        qtype = str(result.get("question_type", "unknown"))
        by_type.setdefault(qtype, {"total": 0, "correct": 0, "partial": 0, "ctx_ok": 0})
        by_type[qtype]["total"] += 1
        if result["verdict"] == "CORRECT":
            by_type[qtype]["correct"] += 1
        if result["verdict"] == "PARTIALLY_CORRECT":
            by_type[qtype]["partial"] += 1
        if result["context_has_answer"]:
            by_type[qtype]["ctx_ok"] += 1

    return {
        "total": total,
        "correct": correct,
        "partially_correct": partial,
        "wrong": wrong,
        "abstained": abstained,
        "correct_rate": round(correct / total, 4) if total else 0.0,
        "combined_rate": round((correct + partial) / total, 4) if total else 0.0,
        "context_has_answer": context_has_answer,
        "context_has_answer_rate": round(context_has_answer / total, 4) if total else 0.0,
        "qa_failure": qa_failure,
        "retrieval_failure": retrieval_failure,
        "parse_errors": parse_errors,
        "by_type": by_type,
    }


async def run(config: JudgeConfig) -> dict[str, Any]:
    entries = load_entries(Path(config.full_results), config.qids, config.n, config.seed)
    print(f"Judging {len(entries)} LoCoMo answers with {config.judge_model}")

    results: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, 1):
        result = await evaluate_entry(entry, config.judge_model)
        results.append(result)
        print(
            f"[{idx}/{len(entries)}] {result.get('question_id')} "
            f"{result['verdict']} ctx={result['context_has_answer']} "
            f"failure={result['failure_mode']}"
        )

    summary = summarize(results)
    output = {"config": asdict(config), "summary": summary, "results": results}

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"locomo_judge_{stamp}.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Results saved -> {out_path}")
    return output


def _parse_qids(qids_arg: str, qids_file: str) -> list[str]:
    qids: list[str] = []
    if qids_arg:
        qids.extend(part.strip() for part in qids_arg.split(",") if part.strip())
    if qids_file:
        for line in Path(qids_file).read_text(encoding="utf-8").splitlines():
            candidate = line.split("#", 1)[0].strip()
            if candidate:
                qids.append(candidate)
    return list(dict.fromkeys(qids))


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for LoCoMo full results")
    parser.add_argument("--full-results", required=True, help="Path to LoCoMo *_full_results.json")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--n", type=int, default=100, help="Number to sample; <=0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--qids", default="", help="Comma-separated question IDs")
    parser.add_argument("--qids-file", default="", help="One question ID per line")
    args = parser.parse_args()

    try:
        qids = _parse_qids(args.qids, args.qids_file)
    except ValueError as exc:
        parser.error(str(exc))

    config = JudgeConfig(
        full_results=args.full_results,
        judge_model=args.judge_model,
        n=args.n,
        seed=args.seed,
        output_dir=args.output_dir,
        qids=qids,
    )
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
