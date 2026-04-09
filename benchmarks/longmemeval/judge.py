"""LLM-as-a-Judge evaluation for LongMemEval benchmark results.

Evaluates completed benchmark entries using a 3-signal approach:
  - hypothesis  (model's answer)
  - expected_answer (ground truth)
  - retrieved_context (facts from memory)

This lets us distinguish two failure modes:
  QA_FAILURE       — context had the answer but the model didn't extract it
  RETRIEVAL_FAILURE — context lacked the answer entirely

Providers:
  gemini   — GeminiLLM (gemini-2.5-flash-lite, API call)
  lmstudio — AsyncOpenAI pointed at localhost:1234 (meta-llama-3.1-8b-instruct)

Usage:
  python benchmarks/longmemeval/judge.py --provider gemini --n 5
  python benchmarks/longmemeval/judge.py --provider lmstudio --n 5
  python benchmarks/longmemeval/judge.py --provider gemini --n 500
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("judge")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_LMSTUDIO_MODEL = "meta-llama-3.1-8b-instruct"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

VALID_VERDICTS = {"CORRECT", "PARTIALLY_CORRECT", "WRONG", "ABSTAINED"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    provider: str = "gemini"
    model: str = ""  # auto-filled below if empty
    checkpoint_path: str = "benchmark_results/longmemeval_s_run_checkpoint.json"
    n: int = 5
    output_dir: str = "benchmark_results"
    seed: int = 42
    qids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = (
                DEFAULT_GEMINI_MODEL
                if self.provider == "gemini"
                else DEFAULT_LMSTUDIO_MODEL
            )


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    verdict: str
    context_has_answer: bool
    failure_mode: str | None
    explanation: str
    latency_ms: float
    raw_response: str
    parse_error: bool = False


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a memory-augmented AI assistant. Your task is to assess \
whether the model answered a question correctly, and whether its retrieved \
memory context contained enough information to answer correctly.

QUESTION: {question}
QUESTION TYPE: {question_type}

MODEL ANSWER (hypothesis): {hypothesis}
EXPECTED ANSWER: {expected_answer}

RETRIEVED CONTEXT (facts extracted from conversation history):
{retrieved_context}

Evaluate the following and respond ONLY with valid JSON (no markdown, no extra text):

Rules:
- CORRECT: the model answer contains or clearly matches the expected answer
- PARTIALLY_CORRECT: the model has the right idea but is imprecise or incomplete
- WRONG: the model answer is clearly incorrect, unrelated, or contradicts expected
- ABSTAINED: the model said it doesn't know / information not available
- context_has_answer: true if the retrieved context contains enough info to answer correctly
- failure_mode:
    null             — if verdict is CORRECT or PARTIALLY_CORRECT
    "QA_FAILURE"     — if verdict is WRONG/ABSTAINED but context_has_answer is true
    "RETRIEVAL_FAILURE" — if verdict is WRONG/ABSTAINED and context_has_answer is false
- For temporal-reasoning questions: off-by-one errors in day/time counts = PARTIALLY_CORRECT
- For knowledge-update questions: accept updated answer if it matches expected
- For questions where QUESTION TYPE ends with "_abs": the question is designed to be unanswerable. If the model's answer is an abstention ("I don't have that information", "I cannot answer", "not enough information", etc.), verdict must be CORRECT — the model correctly identified the question as unanswerable. The expected_answer text is only an example explanation, not a required phrase to match.

Respond with exactly this JSON structure:
{{"verdict": "CORRECT|PARTIALLY_CORRECT|WRONG|ABSTAINED", "context_has_answer": true|false, "failure_mode": null|"QA_FAILURE"|"RETRIEVAL_FAILURE", "explanation": "one sentence"}}"""


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


def build_prompt(entry: dict[str, Any]) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=entry.get("question", ""),
        question_type=entry.get("question_type", "unknown"),
        hypothesis=entry.get("hypothesis", "(no answer)"),
        expected_answer=entry.get("expected_answer", ""),
        retrieved_context=entry.get("retrieved_context", "(none)"),
    )


def is_abs_question_type(question_type: str | None) -> bool:
    return bool(question_type and question_type.endswith("_abs"))


def is_abstention_answer(answer: str | None) -> bool:
    if not answer:
        return False
    normalized = re.sub(r"\s+", " ", answer.lower()).strip()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


def _parse_qid_inputs(qids_arg: str, qids_file: str) -> list[str]:
    qids: list[str] = []

    if qids_arg:
        qids.extend(part.strip() for part in qids_arg.split(",") if part.strip())

    if qids_file:
        path = Path(qids_file)
        if not path.exists():
            raise ValueError(f"QID file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            candidate = line.split("#", 1)[0].strip()
            if candidate:
                qids.append(candidate)

    # Deduplicate while preserving first-seen order
    return list(dict.fromkeys(qids))


def _select_entries(
    completed_entries: dict[str, dict[str, Any]],
    *,
    n: int,
    seed: int,
    qids: list[str] | None,
) -> list[dict[str, Any]]:
    if not completed_entries:
        raise ValueError("no completed entries in checkpoint")

    if qids:
        missing = [qid for qid in qids if qid not in completed_entries]
        if missing:
            preview = ", ".join(missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            raise ValueError(f"requested QIDs not found in checkpoint: {preview}{suffix}")
        return [completed_entries[qid] for qid in qids]

    all_entries = list(completed_entries.values())
    rng = random.Random(seed)
    return rng.sample(all_entries, min(n, len(all_entries)))


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------


async def _call_gemini(prompt: str, model: str) -> str:
    """Call Gemini via existing GeminiLLM provider."""
    # Import here so lmstudio-only users don't need google-generativeai
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vektori.models.gemini import GeminiLLM

    llm = GeminiLLM(model=model, api_key=os.environ.get("GOOGLE_API_KEY"))
    return await llm.generate(prompt, max_tokens=300)


async def _call_lmstudio(prompt: str, model: str, base_url: str) -> str:
    """Call LM Studio via OpenAI-compatible API."""
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError("openai package required: pip install openai") from e

    client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300,
    )
    return response.choices[0].message.content or ""


async def call_judge(
    prompt: str,
    provider: str,
    model: str,
    lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL,
) -> tuple[str, float]:
    """Call the judge LLM and return (raw_response, latency_ms)."""
    t0 = time.perf_counter()
    if provider == "gemini":
        raw = await _call_gemini(prompt, model)
    elif provider == "lmstudio":
        raw = await _call_lmstudio(prompt, model, lmstudio_base_url)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'gemini' or 'lmstudio'.")
    latency_ms = (time.perf_counter() - t0) * 1000
    return raw, latency_ms


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_verdict(raw: str, latency_ms: float) -> JudgeVerdict:
    """Parse LLM response into a JudgeVerdict.

    Tries strict JSON first, then regex-extracts first {...} block as fallback.
    """
    text = raw.strip()

    parsed: dict[str, Any] | None = None

    # 1) Direct parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Extract first {...} block (handles markdown fences)
    if parsed is None:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    # 3) Fallback
    if parsed is None:
        logger.warning("Could not parse judge response: %s", text[:200])
        return JudgeVerdict(
            verdict="WRONG",
            context_has_answer=False,
            failure_mode="RETRIEVAL_FAILURE",
            explanation="(parse error — could not extract JSON from response)",
            latency_ms=latency_ms,
            raw_response=raw,
            parse_error=True,
        )

    # Normalise verdict
    verdict = str(parsed.get("verdict", "WRONG")).upper()
    if verdict not in VALID_VERDICTS:
        verdict = "WRONG"

    context_has_answer = bool(parsed.get("context_has_answer", False))

    # Derive failure_mode if not provided or inconsistent
    failure_mode = parsed.get("failure_mode")
    if verdict in {"CORRECT", "PARTIALLY_CORRECT"}:
        failure_mode = None
    elif failure_mode not in {"QA_FAILURE", "RETRIEVAL_FAILURE"}:
        failure_mode = "QA_FAILURE" if context_has_answer else "RETRIEVAL_FAILURE"

    explanation = str(parsed.get("explanation", "")).strip() or "(no explanation)"

    return JudgeVerdict(
        verdict=verdict,
        context_has_answer=context_has_answer,
        failure_mode=failure_mode,
        explanation=explanation,
        latency_ms=latency_ms,
        raw_response=raw,
        parse_error=False,
    )


# ---------------------------------------------------------------------------
# Per-entry evaluation
# ---------------------------------------------------------------------------


async def evaluate_entry(
    entry: dict[str, Any],
    provider: str,
    model: str,
) -> dict[str, Any]:
    prompt = build_prompt(entry)
    raw, latency_ms = await call_judge(prompt, provider, model)
    verdict = parse_verdict(raw, latency_ms)

    if is_abs_question_type(entry.get("question_type")) and is_abstention_answer(
        entry.get("hypothesis")
    ):
        verdict.verdict = "CORRECT"
        verdict.context_has_answer = False
        verdict.failure_mode = None

    return {
        **entry,
        "verdict": verdict.verdict,
        "context_has_answer": verdict.context_has_answer,
        "failure_mode": verdict.failure_mode,
        "explanation": verdict.explanation,
        "latency_ms": round(latency_ms, 1),
        "parse_error": verdict.parse_error,
        "raw_judge_response": verdict.raw_response,
    }


# ---------------------------------------------------------------------------
# Console output helpers
# ---------------------------------------------------------------------------

_VERDICT_COLORS = {
    "CORRECT": "\033[92m",       # green
    "PARTIALLY_CORRECT": "\033[93m",  # yellow
    "WRONG": "\033[91m",         # red
    "ABSTAINED": "\033[94m",     # blue
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _colored(text: str, color_code: str) -> str:
    return f"{color_code}{text}{_RESET}"


def print_entry_result(idx: int, result: dict[str, Any], total: int) -> None:
    verdict = result["verdict"]
    color = _VERDICT_COLORS.get(verdict, "")
    ctx = "YES" if result["context_has_answer"] else "NO"
    fm = result["failure_mode"] or "—"
    parse_warn = "  [PARSE ERR]" if result.get("parse_error") else ""

    print(f"\n{'─'*70}")
    print(f"{_BOLD}[{idx}/{total}] {result['question_id']}  |  {result['question_type']}{_RESET}{parse_warn}")
    print(f"  Question : {result['question']}")
    print(f"  Expected : {result['expected_answer']}")
    print(f"  Hypothesis: {result['hypothesis']}")
    print(f"  Verdict  : {_colored(verdict, color)}  |  Context had answer: {ctx}  |  Failure: {fm}")
    print(f"  Explain  : {result['explanation']}")
    print(f"  Latency  : {result['latency_ms']:.0f} ms")


def print_summary(results: list[dict[str, Any]], config: JudgeConfig, total_s: float) -> None:
    n = len(results)
    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    partial = sum(1 for r in results if r["verdict"] == "PARTIALLY_CORRECT")
    wrong = sum(1 for r in results if r["verdict"] == "WRONG")
    abstained = sum(1 for r in results if r["verdict"] == "ABSTAINED")
    ctx_ok = sum(1 for r in results if r["context_has_answer"])
    qa_fail = sum(1 for r in results if r["failure_mode"] == "QA_FAILURE")
    ret_fail = sum(1 for r in results if r["failure_mode"] == "RETRIEVAL_FAILURE")
    parse_errs = sum(1 for r in results if r.get("parse_error"))
    avg_lat = sum(r["latency_ms"] for r in results) / n if n else 0

    pct = lambda x: f"{x}/{n} ({100*x//n}%)" if n else "0/0"

    print(f"\n{'═'*70}")
    print(f"{_BOLD}JUDGE SUMMARY  |  provider={config.provider}  model={config.model}{_RESET}")
    print(f"{'═'*70}")
    print(f"  Accuracy   (CORRECT)          : {_colored(pct(correct), _VERDICT_COLORS['CORRECT'])}")
    print(f"  Partial    (PARTIALLY_CORRECT) : {_colored(pct(partial), _VERDICT_COLORS['PARTIALLY_CORRECT'])}")
    print(f"  Wrong                          : {_colored(pct(wrong), _VERDICT_COLORS['WRONG'])}")
    print(f"  Abstained                      : {pct(abstained)}")
    print(f"  Combined (CORRECT+PARTIAL)     : {_colored(pct(correct+partial), _VERDICT_COLORS['CORRECT'])}")
    print()
    print(f"  Retrieval quality (ctx had answer): {pct(ctx_ok)}")
    print(f"  QA failures   (retrieval OK)      : {qa_fail}")
    print(f"  Retrieval failures                : {ret_fail}")
    if parse_errs:
        print(f"  Parse errors (JSON malformed)     : {parse_errs}")
    print()
    print(f"  Total time : {total_s:.1f}s  |  Avg/question : {avg_lat:.0f} ms")
    print(f"{'═'*70}")

    # Breakdown by question type
    types: dict[str, dict[str, int]] = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        types.setdefault(qt, {"total": 0, "correct": 0, "ctx_ok": 0})
        types[qt]["total"] += 1
        if r["verdict"] in {"CORRECT", "PARTIALLY_CORRECT"}:
            types[qt]["correct"] += 1
        if r["context_has_answer"]:
            types[qt]["ctx_ok"] += 1

    if len(types) > 1:
        print(f"\n{_BOLD}By question type:{_RESET}")
        for qt, s in sorted(types.items()):
            t = s["total"]
            c = s["correct"]
            ck = s["ctx_ok"]
            print(f"  {qt:<35} acc={c}/{t}  ctx_ok={ck}/{t}")


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


async def run(config: JudgeConfig) -> list[dict[str, Any]]:
    # Load checkpoint
    cp_path = Path(config.checkpoint_path)
    if not cp_path.exists():
        print(f"ERROR: checkpoint not found: {cp_path}", file=sys.stderr)
        sys.exit(1)

    raw_data = json.loads(cp_path.read_text(encoding="utf-8"))
    completed_entries = raw_data.get("completed", {})

    try:
        sample = _select_entries(
            completed_entries,
            n=config.n,
            seed=config.seed,
            qids=config.qids,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{_BOLD}LLM-as-a-Judge  |  provider={config.provider}  model={config.model}{_RESET}")
    if config.qids:
        print(f"Evaluating {len(sample)} questions from {cp_path.name} (explicit QIDs)\n")
    else:
        print(f"Evaluating {len(sample)} questions from {cp_path.name}\n")

    results: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for i, entry in enumerate(sample, 1):
        result = await evaluate_entry(entry, config.provider, config.model)
        results.append(result)
        print_entry_result(i, result, len(sample))

    total_s = time.perf_counter() - t_start
    print_summary(results, config, total_s)

    # Save output
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"judge_{config.provider}_{ts}.json"

    summary = {
        "total": len(results),
        "correct": sum(1 for r in results if r["verdict"] == "CORRECT"),
        "partially_correct": sum(1 for r in results if r["verdict"] == "PARTIALLY_CORRECT"),
        "wrong": sum(1 for r in results if r["verdict"] == "WRONG"),
        "abstained": sum(1 for r in results if r["verdict"] == "ABSTAINED"),
        "context_has_answer": sum(1 for r in results if r["context_has_answer"]),
        "qa_failure": sum(1 for r in results if r["failure_mode"] == "QA_FAILURE"),
        "retrieval_failure": sum(1 for r in results if r["failure_mode"] == "RETRIEVAL_FAILURE"),
        "parse_errors": sum(1 for r in results if r.get("parse_error")),
        "total_time_s": round(total_s, 2),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1) if results else 0,
    }

    output = {
        "config": asdict(config),
        "results": results,
        "summary": summary,
    }
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved → {out_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge evaluation for LongMemEval benchmark results"
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "lmstudio"],
        default="gemini",
        help="Judge LLM provider (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name override. Defaults: gemini=gemini-2.5-flash-lite, lmstudio=meta-llama-3.1-8b-instruct",
    )
    parser.add_argument(
        "--checkpoint",
        default="benchmark_results/longmemeval_s_run_checkpoint.json",
        help="Path to benchmark checkpoint JSON",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions to evaluate (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for output JSON (default: benchmark_results)",
    )
    parser.add_argument(
        "--qids",
        default="",
        help="Comma-separated question IDs to evaluate (disables random sampling)",
    )
    parser.add_argument(
        "--qids-file",
        default="",
        help="Path to file containing question IDs (one per line; '#' comments allowed)",
    )
    args = parser.parse_args()

    try:
        qids = _parse_qid_inputs(args.qids, args.qids_file)
    except ValueError as e:
        parser.error(str(e))

    config = JudgeConfig(
        provider=args.provider,
        model=args.model,
        checkpoint_path=args.checkpoint,
        n=args.n,
        seed=args.seed,
        output_dir=args.output_dir,
        qids=qids,
    )

    asyncio.run(run(config))


if __name__ == "__main__":
    main()
