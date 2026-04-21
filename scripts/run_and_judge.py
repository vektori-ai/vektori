"""
run_and_judge.py — Run LoCoMo eval then immediately judge the output.

Usage (smoke test, 2 questions):
    python scripts/run_and_judge.py --max-questions 2

Usage (full run, PPR on — default):
    python scripts/run_and_judge.py

Usage (full run, PPR off — for AWS ablation):
    python scripts/run_and_judge.py --no-ppr --run-name locomo_ppr_off

Required env vars:
    GOOGLE_API_KEY          — Google AI Studio key (for Gemini eval + judge)
    CLOUDFLARE_API_TOKEN    — Cloudflare Workers AI token (for embeddings)
    CLOUDFLARE_ACCOUNT_ID   — Cloudflare account ID
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path regardless of where the script is invoked from
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Env var check ─────────────────────────────────────────────────────────────

REQUIRED_ENV = {
    "GOOGLE_API_KEY": "Google AI Studio — aistudio.google.com → Get API key",
    "CLOUDFLARE_API_TOKEN": "Cloudflare dashboard → Workers AI → API tokens",
    "CLOUDFLARE_ACCOUNT_ID": "Cloudflare dashboard → top-right account ID",
}


def _check_env() -> None:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print("ERROR: Missing required environment variables:\n")
        for k in missing:
            print(f"  {k}\n    → {REQUIRED_ENV[k]}")
        print("\nSet them and re-run.")
        sys.exit(1)


# ── Args ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LoCoMo eval + LLM-as-judge in one shot"
    )
    # Eval args (passed through to run_locomo)
    p.add_argument("--max-questions", type=int, default=0,
                   help="Limit questions (0 = full dataset, 2 = smoke test)")
    p.add_argument("--depth", default="l1", choices=["l0", "l1", "l2"])
    p.add_argument("--no-ppr", action="store_true",
                   help="Disable PPR (ablation run)")
    p.add_argument("--run-name", default=None,
                   help="Custom run name (auto-timestamped if omitted)")
    p.add_argument("--output-dir", default="benchmark_results")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval-model", default="gemini:gemini-2.5-flash-lite")
    p.add_argument("--extraction-model", default="gemini:gemini-2.5-flash-lite")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-namespace", default=None)
    # Judge args
    p.add_argument("--judge-model", default="gemini:gemini-2.5-flash-lite")
    p.add_argument("--judge-n", type=int, default=0,
                   help="How many answers to judge (0 = all, default)")
    return p.parse_args()


# ── Step 1: Eval ──────────────────────────────────────────────────────────────

async def _run_eval(args: argparse.Namespace, run_name: str, run_output_dir: str) -> Path:
    """Run the LoCoMo benchmark. Returns path to full_results.json."""
    from benchmarks.locomo.locomo_runner import LoCoMoConfig, LoCoMoBenchmark

    max_q = args.max_questions if args.max_questions and args.max_questions > 0 else None

    config = LoCoMoConfig(
        dataset_name="locomo10_cooked",
        embedding_model="cloudflare:@cf/baai/bge-m3",
        extraction_model=args.extraction_model,
        eval_model=args.eval_model,
        retrieval_depth=args.depth,
        output_dir=run_output_dir,
        data_dir=args.data_dir,
        run_name=run_name,
        max_questions=max_q,
        use_cache=not args.no_cache,
        cache_namespace=args.cache_namespace,
        use_ppr=not args.no_ppr,
    )

    print(f"\n{'='*60}")
    print(f"STEP 1 — EVAL")
    print(f"  run_name : {run_name}")
    print(f"  output   : {run_output_dir}")
    print(f"  PPR      : {'OFF' if args.no_ppr else 'ON'}")
    print(f"  questions: {'ALL' if not max_q else max_q}")
    print(f"{'='*60}\n")

    await LoCoMoBenchmark(config).run()

    full_results = Path(run_output_dir) / f"{run_name}_full_results.json"
    if not full_results.exists():
        print(f"ERROR: Expected full results at {full_results} but file not found.")
        sys.exit(1)

    return full_results


# ── Step 2: Judge ─────────────────────────────────────────────────────────────

async def _run_judge(
    full_results: Path,
    run_output_dir: str,
    judge_model: str,
    judge_n: int,
) -> dict:
    from benchmarks.locomo.locomo_judge import JudgeConfig, run as judge_run

    print(f"\n{'='*60}")
    print(f"STEP 2 — JUDGE")
    print(f"  input  : {full_results}")
    print(f"  model  : {judge_model}")
    print(f"  sample : {'ALL' if judge_n <= 0 else judge_n}")
    print(f"{'='*60}\n")

    config = JudgeConfig(
        full_results=str(full_results),
        judge_model=judge_model,
        n=judge_n if judge_n > 0 else 0,
        output_dir=run_output_dir,
    )

    return await judge_run(config)


# ── Step 3: Summary ───────────────────────────────────────────────────────────

def _print_scorecard(judge_output: dict, run_name: str, ppr_on: bool) -> None:
    s = judge_output.get("summary", {})
    total = s.get("total", 0)
    correct = s.get("correct", 0)
    partial = s.get("partially_correct", 0)
    wrong = s.get("wrong", 0)
    abstained = s.get("abstained", 0)
    combined = correct + partial

    correct_pct = 100 * s.get("correct_rate", 0)
    combined_pct = 100 * s.get("combined_rate", 0)
    ctx_pct = 100 * s.get("context_has_answer_rate", 0)
    qa_fail = s.get("qa_failure", 0)
    ret_fail = s.get("retrieval_failure", 0)

    print(f"\n{'='*60}")
    print(f"FINAL SCORECARD — {run_name}")
    print(f"  PPR: {'ON' if ppr_on else 'OFF'}")
    print(f"{'='*60}")
    print(f"  Total judged  : {total}")
    print(f"  Correct       : {correct:>4}  ({correct_pct:.1f}%)")
    print(f"  Partial       : {partial:>4}  ({100*partial/total:.1f}%)" if total else "")
    print(f"  Combined C+PC : {combined:>4}  ({combined_pct:.1f}%)")
    print(f"  Wrong         : {wrong:>4}")
    print(f"  Abstained     : {abstained:>4}")
    print(f"  Context OK    : {s.get('context_has_answer'):>4}  ({ctx_pct:.1f}%)")
    print(f"  QA failures   : {qa_fail}")
    print(f"  Retrieval fail: {ret_fail}")

    by_type = s.get("by_type", {})
    if by_type:
        print(f"\n  By question type:")
        for qtype, counts in sorted(by_type.items()):
            t = counts["total"]
            c = counts["correct"]
            pc = counts["partial"]
            pct = 100 * (c + pc) / t if t else 0
            print(f"    {qtype:<20} {c+pc:>3}/{t:<3}  ({pct:.0f}% C+PC)")
    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    _check_env()
    args = _parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ppr_tag = "ppr_off" if args.no_ppr else "ppr_on"
    run_name = args.run_name or f"locomo_{ppr_tag}_{timestamp}"
    run_output_dir = str(Path(args.output_dir) / run_name)

    full_results = await _run_eval(args, run_name, run_output_dir)
    judge_output = await _run_judge(
        full_results,
        run_output_dir,
        args.judge_model,
        args.judge_n,
    )
    _print_scorecard(judge_output, run_name, ppr_on=not args.no_ppr)


if __name__ == "__main__":
    asyncio.run(main())
