"""
BEAM Benchmark Entry Point
==========================

Runs Vektori on the BEAM dataset (HuggingFace: Mohammadta/BEAM).

Requirements:
  - HuggingFace datasets library
  - API keys configured for models

Example:
  python -m benchmarks.beam.run_beam --dataset 500K
"""

import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path

from benchmarks.beam.beam_runner import BeamBenchmark, BeamConfig

logger = logging.getLogger(__name__)

async def _main():
    parser = argparse.ArgumentParser(description="Run Vektori on BEAM benchmark")
    parser.add_argument("--dataset", choices=["100K", "500K", "1M"], default="500K")
    parser.add_argument("--embedding-model", default="cloudflare:@cf/baai/bge-m3")
    parser.add_argument("--extraction-model", default="gemini:gemini-2.5-flash-lite")
    parser.add_argument("--reranker", default="bge:BAAI/bge-reranker-v2-m3")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--reranker-top-n", type=int, default=30)
    parser.add_argument("--context-window", type=int, default=5)
    parser.add_argument("--depth", choices=["l0", "l1", "l2"], default="l1")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-questions", type=int, default=None, help="Pilot run cap")
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"beam_{args.dataset}_{timestamp}"
    output_dir = str(Path(args.output_dir) / run_name)

    config = BeamConfig(
        dataset_split=args.dataset,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        reranker_model=args.reranker,
        top_k=args.top_k,
        reranker_top_n=args.reranker_top_n,
        context_window=args.context_window,
        retrieval_depth=args.depth,
        output_dir=output_dir,
        run_name=run_name,
        max_questions=args.max_questions
    )

    runner = BeamBenchmark(config)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(_main())
