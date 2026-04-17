"""
LoCoMo Benchmark Entry Point
==============================

Runs Vektori on the LoCoMo-10 dataset using:
  - Cloudflare Workers AI: BGE-M3 embeddings (zero local GPU)
  - Gemini 2.5 Flash-Lite: fact/episode extraction by default
  - Qwen3-8B via local vLLM: QA evaluation by default

Required environment variables (set in .env or shell):
    CLOUDFLARE_API_TOKEN   — Workers AI token (Account > Workers AI > Read)
    CLOUDFLARE_ACCOUNT_ID  — Cloudflare account ID from dash.cloudflare.com
    GOOGLE_API_KEY         — Google AI Studio / Gemini API key

Quick start:
    # Step 1: cook the dataset (once)
    python -m benchmarks.locomo.cook_locomo \\
        --output data/locomo10_cooked.json --download-if-missing

    # Step 2: pilot run (5 questions)
    python run_locomo.py

    # Step 3: full run (remove max_questions below or set to None)
    python run_locomo.py --max-questions 0   # 0 = no limit (see below)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Load .env if present (python-dotenv optional — graceful fallback)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _check_env() -> None:
    missing = []
    for var in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID", "GOOGLE_API_KEY"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        print(
            "ERROR: The following required environment variables are not set:\n"
            + "\n".join(f"  {v}" for v in missing)
            + "\n\nSet them in your .env file or shell before running.",
            file=sys.stderr,
        )
        sys.exit(1)


async def _main() -> None:
    import argparse
    from datetime import datetime as _dt

    from benchmarks.locomo.locomo_runner import LoCoMoBenchmark, LoCoMoConfig

    parser = argparse.ArgumentParser(description="Run Vektori on LoCoMo-10")
    parser.add_argument(
        "--max-questions", type=int, default=5,
        help="Limit to first N questions for pilot run. "
             "Pass 0 to run the full dataset."
    )
    parser.add_argument("--depth", choices=["l0", "l1", "l2"], default="l1")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--extraction-model", default="gemini:gemini-2.5-flash-lite",
        help="LLM for fact/episode extraction"
    )
    parser.add_argument(
        "--eval-model", default="vllm:Qwen/Qwen3-8B",
        help="LLM for QA answer generation"
    )
    parser.add_argument(
        "--qa-prompt-file", default=None,
        help="Optional path to GEPA-optimized QA prompt text"
    )
    parser.add_argument(
        "--max-extraction-output-tokens", type=int, default=32768,
        help="Max output tokens for extraction LLM calls"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable session extract cache for this run"
    )
    parser.add_argument(
        "--cache-namespace", default=None,
        help="Optional cache namespace override to isolate cached extractions"
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Custom run name. Defaults to 'locomo10_cooked_YYYYMMDD_HHMMSS'."
    )
    args = parser.parse_args()

    max_q = args.max_questions if args.max_questions and args.max_questions > 0 else None

    # Auto-generate timestamped run name so each run gets its own output folder
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"locomo10_cooked_{timestamp}"

    # Results land in benchmark_results/locomo10_cooked_YYYYMMDD_HHMMSS/
    output_dir = str(Path(args.output_dir) / run_name)

    config = LoCoMoConfig(
        dataset_name="locomo10_cooked",
        embedding_model="cloudflare:@cf/baai/bge-m3",
        extraction_model=args.extraction_model,
        eval_model=args.eval_model,
        qa_prompt_path=args.qa_prompt_file,
        max_extraction_output_tokens=args.max_extraction_output_tokens,
        retrieval_depth=args.depth,
        output_dir=output_dir,
        data_dir=args.data_dir,
        run_name=run_name,
        max_questions=max_q,
        use_cache=not args.no_cache,
        cache_namespace=args.cache_namespace,
    )

    if max_q:
        print(f"[PILOT] Running first {max_q} questions to verify the pipeline.")
        print("        Pass --max-questions 0 (or 0) for full dataset.\n")

    await LoCoMoBenchmark(config).run()


if __name__ == "__main__":
    _check_env()
    asyncio.run(_main())
