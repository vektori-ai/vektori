"""
Filesystem Memory benchmark — recall@k on a document corpus.

Ingests a folder of documents, runs a Q&A set, reports recall@1/3/5.
Uses vektori's own docs as the default corpus (no external data needed).

Usage:
    # default: ingest ./docs + README.md, run built-in Q&A set
    uv run python benchmarks/eval_fsmemory.py

    # custom corpus + model
    uv run python benchmarks/eval_fsmemory.py --corpus ./my-notes --embedding-model bge:BAAI/bge-m3

    # skip LLM extraction (raw RAG baseline, no API cost)
    uv run python benchmarks/eval_fsmemory.py --no-extract

    # compare both modes side by side
    uv run python benchmarks/eval_fsmemory.py --compare
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vektori.fsmemory import FilesystemMemory

# ── built-in Q&A set (answers are in vektori's README + docs) ─────────────
# Format: (question, keywords_that_must_appear_in_retrieved_facts)
# A fact "hits" if ANY of the keywords appear in it (case-insensitive).

QA_SET = [
    (
        "What storage backends does Vektori support?",
        ["sqlite", "postgres", "postgresql", "neo4j", "qdrant", "milvus", "chroma", "lancedb"],
    ),
    (
        "What embedding model does Vektori use by default?",
        ["bge", "bge-m3", "text-embedding", "openai"],
    ),
    (
        "What are the three layers in Vektori's memory architecture?",
        ["l0", "l1", "l2", "fact", "episode", "sentence", "layer"],
    ),
    (
        "How does Vektori handle fact conflicts between sessions?",
        ["supersed", "conflict", "deactivat", "overrid"],
    ),
    (
        "What benchmark scores has Vektori achieved?",
        ["locomo", "longmemeval", "66", "73", "benchmark"],
    ),
    (
        "How do you install Vektori?",
        ["pip install", "vektori", "pypi"],
    ),
    (
        "What is Personalized Pagerank used for in Vektori?",
        ["ppr", "pagerank", "graph", "retrieval", "traversal"],
    ),
    (
        "How does Vektori extract facts from conversations?",
        ["llm", "extract", "gpt", "litellm", "fact"],
    ),
    (
        "What is the episode layer in Vektori?",
        ["episode", "l1", "pattern", "narrative", "cross-session"],
    ),
    (
        "How does Vektori handle duplicate or repeated facts?",
        ["dedup", "mention", "hash", "upsert", "duplicate"],
    ),
]


@dataclass
class BenchmarkResult:
    mode: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    files_ingested: int
    facts_stored: int
    ingest_time_s: float
    search_time_s: float
    hits_per_question: list[dict] = field(default_factory=list)


async def run_benchmark(
    corpus_paths: list[str],
    embedding_model: str,
    extraction_model: str,
    extract_facts: bool,
    db_suffix: str = "",
) -> BenchmarkResult:
    mode = "extract" if extract_facts else "raw-chunks"
    db_path = Path(f"/tmp/fsmemory_bench_{mode}{db_suffix}.db")
    db_path.unlink(missing_ok=True)

    print(f"\n{'─'*60}")
    print(f"  Mode: {mode}")
    print(f"  Embedding: {embedding_model}")
    if extract_facts:
        print(f"  Extraction LLM: {extraction_model}")
    print(f"{'─'*60}")

    async with FilesystemMemory(
        user_id="bench",
        database_url=f"sqlite:///{db_path}",
        embedding_model=embedding_model,
        extraction_model=extraction_model,
        extract_facts=extract_facts,
    ) as fs:

        # ── ingest ────────────────────────────────────────────────────────
        print("Ingesting corpus...")
        t0 = time.perf_counter()
        all_results = []
        for corpus_path in corpus_paths:
            p = Path(corpus_path).expanduser()
            if p.is_dir():
                results = await fs.add_directory(str(p), glob="**/*.md")
                all_results.extend(results)
            elif p.is_file():
                all_results.append(await fs.add_file(str(p)))

        ingest_time = time.perf_counter() - t0
        stats = await fs.get_stats()

        ingested = [r for r in all_results if not r.skipped and r.error is None]
        errors = [r for r in all_results if r.error]
        total_facts = sum(r.facts_inserted for r in ingested)

        print(f"  Files ingested : {len(ingested)}")
        if errors:
            print(f"  Errors         : {len(errors)}")
        print(f"  Facts stored   : {total_facts}")
        print(f"  Ingest time    : {ingest_time:.1f}s")

        if total_facts == 0:
            print("\n  ⚠ No facts stored — check corpus path and API key.")
            return BenchmarkResult(
                mode=mode, recall_at_1=0, recall_at_3=0, recall_at_5=0,
                files_ingested=0, facts_stored=0,
                ingest_time_s=ingest_time, search_time_s=0,
            )

        # ── search ────────────────────────────────────────────────────────
        print(f"\nRunning {len(QA_SET)} queries...")
        hits_at_1 = hits_at_3 = hits_at_5 = 0
        t1 = time.perf_counter()
        hits_per_question = []

        for question, keywords in QA_SET:
            out = await fs.search(question, limit=5)
            facts = out.get("facts", [])

            hit_1 = _hits(facts[:1], keywords)
            hit_3 = _hits(facts[:3], keywords)
            hit_5 = _hits(facts[:5], keywords)

            if hit_1: hits_at_1 += 1
            if hit_3: hits_at_3 += 1
            if hit_5: hits_at_5 += 1

            hits_per_question.append({
                "question": question,
                "hit@1": hit_1,
                "hit@3": hit_3,
                "hit@5": hit_5,
                "top_fact": facts[0]["text"][:100] if facts else "(no results)",
            })

        search_time = time.perf_counter() - t1
        n = len(QA_SET)
        r1 = hits_at_1 / n
        r3 = hits_at_3 / n
        r5 = hits_at_5 / n

        print(f"\n  Recall@1 : {r1:.0%}  ({hits_at_1}/{n})")
        print(f"  Recall@3 : {r3:.0%}  ({hits_at_3}/{n})")
        print(f"  Recall@5 : {r5:.0%}  ({hits_at_5}/{n})")
        print(f"  Search time : {search_time*1000/n:.0f}ms avg per query")

        return BenchmarkResult(
            mode=mode,
            recall_at_1=r1, recall_at_3=r3, recall_at_5=r5,
            files_ingested=len(ingested),
            facts_stored=total_facts,
            ingest_time_s=ingest_time,
            search_time_s=search_time,
            hits_per_question=hits_per_question,
        )


def _hits(facts: list[dict], keywords: list[str]) -> bool:
    for fact in facts:
        text = fact.get("text", "").lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def _print_per_question(result: BenchmarkResult) -> None:
    print(f"\n  Per-question breakdown ({result.mode}):")
    print(f"  {'Q':<55} @1   @3   @5  top fact")
    print(f"  {'─'*55} ─── ─── ───  {'─'*40}")
    for h in result.hits_per_question:
        q = h["question"][:54]
        h1 = "✓" if h["hit@1"] else "✗"
        h3 = "✓" if h["hit@3"] else "✗"
        h5 = "✓" if h["hit@5"] else "✗"
        print(f"  {q:<55} {h1}    {h3}    {h5}   {h['top_fact'][:50]}")


def _print_comparison(r_raw: BenchmarkResult, r_ext: BenchmarkResult) -> None:
    print(f"\n{'═'*60}")
    print("  COMPARISON: raw chunks vs LLM extraction")
    print(f"{'═'*60}")
    print(f"  {'Metric':<20} {'raw-chunks':>12} {'extract':>12} {'delta':>8}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*8}")

    def row(label, a, b, fmt=".0%"):
        delta = b - a
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<20} {a:{fmt}>12} {b:{fmt}>12} {sign}{delta:{fmt}>7}")

    row("Recall@1", r_raw.recall_at_1, r_ext.recall_at_1)
    row("Recall@3", r_raw.recall_at_3, r_ext.recall_at_3)
    row("Recall@5", r_raw.recall_at_5, r_ext.recall_at_5)
    row("Facts stored", r_raw.facts_stored, r_ext.facts_stored, "d")
    print(f"  {'Ingest time':<20} {r_raw.ingest_time_s:>11.1f}s {r_ext.ingest_time_s:>11.1f}s")


async def main() -> None:
    parser = argparse.ArgumentParser(description="FilesystemMemory recall benchmark")
    parser.add_argument(
        "--corpus", nargs="+",
        default=["./docs", "./README.md"],
        help="Paths to ingest (files or directories)",
    )
    parser.add_argument(
        "--embedding-model", default="openai:text-embedding-3-small",
        help="Embedding model string (e.g. openai:text-embedding-3-small, bge:BAAI/bge-m3)",
    )
    parser.add_argument(
        "--extraction-model", default="openai:gpt-4o-mini",
        help="LLM for fact extraction",
    )
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip LLM extraction — embed raw chunks directly (fast, no LLM cost)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both modes and print side-by-side comparison",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show per-question breakdown",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") and "openai" in args.embedding_model:
        print("⚠  OPENAI_API_KEY not set. Set it or pass --embedding-model bge:BAAI/bge-m3")
        sys.exit(1)

    print(f"\nFilesystemMemory Benchmark")
    print(f"Corpus : {args.corpus}")
    print(f"Q&A set: {len(QA_SET)} questions")

    if args.compare:
        r_raw = await run_benchmark(
            args.corpus, args.embedding_model, args.extraction_model,
            extract_facts=False, db_suffix="_raw",
        )
        r_ext = await run_benchmark(
            args.corpus, args.embedding_model, args.extraction_model,
            extract_facts=True, db_suffix="_ext",
        )
        _print_comparison(r_raw, r_ext)
        if args.verbose:
            _print_per_question(r_raw)
            _print_per_question(r_ext)
    else:
        result = await run_benchmark(
            args.corpus, args.embedding_model, args.extraction_model,
            extract_facts=not args.no_extract,
        )
        if args.verbose:
            _print_per_question(result)


if __name__ == "__main__":
    asyncio.run(main())
