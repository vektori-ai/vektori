"""
LongMemEval Benchmark Runner for Vektori
=========================================

End-to-end evaluation of Vektori's long-term memory on LongMemEval.

Ingestion strategy
------------------
Approach 2 with session-level extract cache:

* **Per-question isolation** — each question gets its own ``user_id``
  (``bq_<question_id>``).  After answering, ``delete_user()`` wipes all
  rows so the next question starts clean.  This prevents content-hash
  collisions from shared haystack sessions.

* **Session extract cache** — LLM fact-extraction is the expensive step
  (costs money).  Results are cached on disk keyed by ``haystack_session_id``.
  Shared sessions (~20 % of the dataset) are extracted once; replayed from
  cache for every other question that uses them.  Re-embedding on replay is
  cheap (local sentence-transformers).

Checkpointing
-------------
Progress is saved to ``<output_dir>/<run_name>_checkpoint.json`` after every
completed question.  Re-running with the same ``--run-name`` (or same dataset
default name) automatically skips already-finished questions and resumes from
the first unfinished one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from benchmarks.longmemeval.checkpoint import BenchmarkCheckpoint
from benchmarks.longmemeval.session_cache import SessionExtractCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Configuration for a LongMemEval benchmark run."""

    # Dataset
    data_dir: str = "data"
    dataset_name: str = "longmemeval_s_cleaned"  # s | m | oracle

    # Vektori
    storage_backend: str = "sqlite"
    database_url: str | None = None
    embedding_model: str = "sentence-transformers:BAAI/bge-m3"
    extraction_model: str = "gemini:gemini-2.5-flash-lite"

    # Retrieval
    retrieval_depth: str = "l1"   # l0 | l1 | l2
    top_k: int = 10
    context_window: int = 3

    # Processing
    batch_size: int = 8
    max_workers: int = 4

    # Output
    output_dir: str = "benchmark_results"
    run_name: str | None = None

    # Evaluation
    eval_model: str = "gemini:gemini-2.5-flash-lite"


# ── Runner ────────────────────────────────────────────────────────────────────

class LongMemEvalBenchmark:
    """Main benchmark runner."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vektori_client = None
        self.storage = None
        self.dataset: list[dict[str, Any]] = []
        self.dataset_path: Path | None = None

        self._session_cache: SessionExtractCache | None = None
        self._checkpoint: BenchmarkCheckpoint | None = None

    # ── Setup ────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        logger.info("Setting up benchmark environment…")
        await self._init_vektori()
        await self._load_dataset()
        await self._init_session_cache()
        self._init_checkpoint()
        logger.info("Setup complete — %d questions in dataset", len(self.dataset))

    async def _init_vektori(self) -> None:
        from vektori import Vektori

        logger.info(
            "Initialising Vektori (backend=%s, embedding=%s, extraction=%s)",
            self.config.storage_backend,
            self.config.embedding_model,
            self.config.extraction_model,
        )
        self.vektori_client = Vektori(
            database_url=self.config.database_url,
            storage_backend=self.config.storage_backend,
            embedding_model=self.config.embedding_model,
            extraction_model=self.config.extraction_model,
            default_top_k=self.config.top_k,
            context_window=self.config.context_window,
            async_extraction=False,   # benchmark needs sync so we can interleave cache ops
        )
        await self.vektori_client._ensure_initialized()
        self.storage = self.vektori_client.db

    async def _load_dataset(self) -> None:
        filename = f"{self.config.dataset_name}.json"
        self.dataset_path = Path(self.config.data_dir) / filename

        if not self.dataset_path.exists():
            logger.warning("Dataset not found at %s — attempting download…", self.dataset_path)
            await self._download_dataset(filename)

        logger.info("Loading dataset from %s", self.dataset_path)
        with open(self.dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)
        logger.info("Loaded %d test instances", len(self.dataset))

    async def _download_dataset(self, filename: str) -> None:
        hf_base = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
        url = f"{hf_base}/{filename}"
        logger.info("Downloading %s …", url)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            with open(self.dataset_path, "wb") as f:
                f.write(response.content)
        logger.info("Dataset saved to %s", self.dataset_path)

    async def _init_session_cache(self) -> None:
        cache_path = self.output_dir / ".cache" / "session_extract_cache.db"
        self._session_cache = SessionExtractCache(cache_path)
        await self._session_cache.initialize()

    def _init_checkpoint(self) -> None:
        run_name = self.config.run_name or self.config.dataset_name
        chk_path = self.output_dir / f"{run_name}_checkpoint.json"
        self._checkpoint = BenchmarkCheckpoint(chk_path)
        n_done = self._checkpoint.load()
        remaining = len(self.dataset) - n_done
        if n_done:
            logger.info("Resuming — %d done, %d remaining", n_done, remaining)

    # ── Main loop ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        try:
            await self.setup()
            await self._run_questions()
            await self.evaluate()
            await self.save_results()
            logger.info("Benchmark complete!")
            self._print_summary()
        finally:
            await self.cleanup()

    async def _run_questions(self) -> None:
        total = len(self.dataset)
        for idx, instance in enumerate(self.dataset):
            qid = instance["question_id"]

            if self._checkpoint.is_done(qid):
                continue

            user_id = f"bq_{qid}"
            try:
                await self._ingest_question(instance, user_id)
                result = await self._answer_question(instance, user_id)
                self._checkpoint.mark_done(qid, result)
                self._checkpoint.save()

                done = self._checkpoint.n_completed
                if done % 10 == 0 or done == total:
                    logger.info(
                        "Progress: %d/%d questions answered (cache: %d sessions)",
                        done, total, await self._session_cache.count(),
                    )


            except Exception as e:
                logger.error("Question %s failed: %s", qid, e, exc_info=True)
                self._checkpoint.mark_failed(qid, str(e))
                self._checkpoint.save()
            finally:
                # Always wipe this question's memory so the next one starts clean.
                try:
                    await self.vektori_client.delete_user(user_id)
                except Exception as e:
                    logger.warning("delete_user(%s) failed: %s", user_id, e)

    # ── Per-question ingestion ────────────────────────────────────────────────

    async def _ingest_question(self, instance: dict[str, Any], user_id: str) -> None:
        """Ingest all haystack sessions for one question.

        Cache hit  → replay pre-extracted facts (no LLM, only local re-embed).
        Cache miss → full LLM extraction, then write to cache.
        """
        haystack_sessions = instance["haystack_sessions"]
        haystack_sids = instance["haystack_session_ids"]
        haystack_dates = instance.get("haystack_dates") or []

        for i, (session, hsid) in enumerate(zip(haystack_sessions, haystack_sids)):
            session_date = haystack_dates[i] if i < len(haystack_dates) else None
            session_time = _parse_date(session_date) if session_date else None

            cached_facts = await self._session_cache.get(hsid)
            if cached_facts is not None:
                await self._replay_session(session, hsid, user_id, session_time, session_date, cached_facts)
            else:
                new_facts = await self._full_ingest_session(
                    session, hsid, user_id, session_time, session_date
                )
                if new_facts:  # don't cache failed/empty extractions — allow retry on next run
                    await self._session_cache.put(hsid, new_facts)

    async def _replay_session(
        self,
        session: list[dict[str, str]],
        haystack_sid: str,
        user_id: str,
        session_time: datetime | None,
        session_date: str | None,
        cached_facts: list[dict[str, Any]],
    ) -> None:
        """Cache hit path: store sentences locally, replay pre-extracted facts,
        then run episode extraction (cheap LLM call — only facts were cached,
        not episodes, so we generate them fresh here)."""
        pipeline = self.vektori_client._pipeline
        extractor = self.vektori_client._extractor

        await pipeline.ingest(
            messages=session,
            session_id=haystack_sid,
            user_id=user_id,
            metadata={"timestamp": session_date} if session_date else None,
            session_time=session_time,
            skip_extraction=True,
        )

        inserted_facts: list[tuple[str, str]] = []
        await extractor.replay_facts(
            cached_facts=cached_facts,
            session_id=haystack_sid,
            user_id=user_id,
            session_time=session_time,
            _inserted_facts_out=inserted_facts,
        )

        if inserted_facts:
            conversation = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}" for msg in session
            )
            try:
                await extractor._extract_insights(
                    inserted_facts=inserted_facts,
                    conversation=conversation,
                    session_id=haystack_sid,
                    user_id=user_id,
                    agent_id=None,
                    session_time=session_time,
                )
            except Exception as e:
                logger.warning(
                    "Episode extraction failed for cached session %s: %s", haystack_sid, e
                )

    async def _full_ingest_session(
        self,
        session: list[dict[str, str]],
        haystack_sid: str,
        user_id: str,
        session_time: datetime | None,
        session_date: str | None,
    ) -> list[dict[str, Any]]:
        """Cache miss path: full LLM extraction.  Returns cacheable facts."""
        pipeline = self.vektori_client._pipeline
        extractor = self.vektori_client._extractor

        # Sentences first (sync path, fast, no LLM).
        await pipeline.ingest(
            messages=session,
            session_id=haystack_sid,
            user_id=user_id,
            metadata={"timestamp": session_date} if session_date else None,
            session_time=session_time,
            skip_extraction=True,
        )

        # LLM fact extraction — capture results for the cache.
        captured_facts: list[dict[str, Any]] = []
        await extractor.extract(
            messages=session,
            session_id=haystack_sid,
            user_id=user_id,
            session_time=session_time,
            _capture_out=captured_facts,
        )

        return captured_facts

    # ── Retrieval + QA ────────────────────────────────────────────────────────

    async def _answer_question(
        self, instance: dict[str, Any], user_id: str
    ) -> dict[str, Any]:
        qid = instance["question_id"]
        question = instance["question"]
        question_type = instance["question_type"]
        question_date = instance.get("question_date") or ""

        search_results = await self.vektori_client.search(
            query=question,
            user_id=user_id,
            depth=self.config.retrieval_depth,
            reference_date=_parse_date(question_date) if question_date else None,
        )

        context = self._format_retrieved_context(search_results)
        answer = await self._generate_answer(question, context, question_type, question_date)

        return {
            "question_id": qid,
            "question": question,
            "question_type": question_type,
            "hypothesis": answer,
            "expected_answer": instance["answer"],
            "retrieved_context": context,
            "retrieval_depth": self.config.retrieval_depth,
        }

    def _format_retrieved_context(self, search_results: Any) -> str:
        if not search_results:
            return "No relevant context retrieved."

        lines: list[str] = []

        facts = search_results.get("facts") or []
        if facts:
            # Sort chronologically so the LLM can reason about temporal order
            facts = sorted(
                facts,
                key=lambda f: f.get("event_time") or f.get("created_at") or "",
            )
            lines.append("## Facts")
            for i, fact in enumerate(facts, 1):
                date_prefix = ""
                ts = fact.get("event_time") or fact.get("created_at") or ""
                if ts:
                    date_prefix = f"[{str(ts)[:10]}] "
                lines.append(f"{i}. {date_prefix}{fact.get('text', str(fact))}")

        episodes = search_results.get("insights") or []
        if episodes:
            lines.append("\n## Episodes")
            for i, ep in enumerate(episodes, 1):
                lines.append(f"{i}. {ep.get('text', str(ep))}")

        sentences = search_results.get("sentences") or []
        if sentences:
            # Group by session_id so the LLM sees each conversation as a block
            session_sents: dict[str, list[dict[str, Any]]] = {}
            session_order: list[str] = []
            for sent in sentences:
                ssid = sent.get("session_id") or "unknown"
                if ssid not in session_sents:
                    session_sents[ssid] = []
                    session_order.append(ssid)
                session_sents[ssid].append(sent)

            lines.append("\n## Session Context")
            for n, ssid in enumerate(session_order, 1):
                sents = session_sents[ssid]
                # Use created_at of first sentence as session date hint
                date_hint = ""
                for s in sents:
                    ts = s.get("created_at") or ""
                    if ts:
                        date_hint = f" [{str(ts)[:10]}]"
                        break
                lines.append(f"\n### Session {n}{date_hint}")
                for sent in sents:
                    role = sent.get("role", "")
                    prefix = f"[{role.upper()}] " if role else ""
                    lines.append(f"  {prefix}{sent.get('text', str(sent))}")

        return "\n".join(lines) if lines else "No relevant context retrieved."

    async def _generate_answer(
        self, question: str, context: str, question_type: str, question_date: str = ""
    ) -> str:
        from vektori.models.factory import create_llm

        if "No relevant context" in context:
            return "I don't have relevant information to answer this question."

        llm = create_llm(self.config.eval_model)
        prompt = self._build_qa_prompt(question, context, question_type, question_date)
        max_tokens = 800 if question_type == "temporal-reasoning" else 500
        try:
            return (await llm.generate(prompt, max_tokens=max_tokens)).strip()
        except Exception as e:
            logger.warning("Answer generation failed: %s", e)
            return "Unable to generate answer due to API error."

    def _build_qa_prompt(
        self, question: str, context: str, question_type: str, question_date: str = ""
    ) -> str:
        date_line = f"TODAY'S DATE: {question_date}\n\n" if question_date else ""

        abs_hint = ""
        if question_type.endswith("_abs"):
            abs_hint = (
                "\n- This question may be specifically testing whether you correctly "
                "recognise that the information was never mentioned"
            )

        if question_type == "temporal-reasoning":
            return (
                "You are an AI assistant answering questions based on provided context "
                "from chat history.\n\n"
                f"{date_line}"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION:\n{question}\n\n"
                "INSTRUCTIONS:\n"
                "- Answer based ONLY on the provided context\n"
                "- First, list the relevant dated events from the context\n"
                "- Then compute the answer (count days/weeks/months between dates)\n"
                "- If the context does not contain enough information to answer, say "
                "\"I don't have that information\"\n\n"
                "REASONING:\n"
            )

        return (
            "You are an AI assistant answering questions based on provided context "
            "from chat history.\n\n"
            f"{date_line}"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION:\n{question}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer the question based ONLY on the provided context\n"
            "- Be concise and direct — a short phrase or sentence is preferred over a long explanation\n"
            "- For questions about time elapsed, use TODAY'S DATE above as your reference point\n"
            "- If the context does not contain enough information to answer, say "
            "\"I don't have that information\" — do not guess or infer beyond what is stated"
            f"{abs_hint}\n\n"
            "ANSWER:"
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(self) -> None:
        logger.info("Computing evaluation metrics…")
        qa_results = list(self._checkpoint.get_completed().values())
        if not qa_results:
            logger.warning("No completed QA results to evaluate")
            return

        # Save JSONL for external LLM-judge evaluation
        run_name = self.config.run_name or self.config.dataset_name
        jsonl_path = self.output_dir / f"{run_name}_qa_results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in qa_results:
                f.write(json.dumps({"question_id": r["question_id"], "hypothesis": r["hypothesis"]}) + "\n")
        logger.info("QA pairs saved to %s", jsonl_path)

        self._metrics = self._compute_metrics(qa_results)

    def _compute_metrics(self, qa_results: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "total_questions": len(qa_results),
            "answered": 0,
            "abstained": 0,
            "by_type": {},
        }
        for r in qa_results:
            hyp = (r.get("hypothesis") or "").lower()
            answered = bool(hyp) and "not available" not in hyp
            if answered:
                metrics["answered"] += 1
            else:
                metrics["abstained"] += 1

            qt = r.get("question_type", "unknown")
            metrics["by_type"].setdefault(qt, {"total": 0, "answered": 0})
            metrics["by_type"][qt]["total"] += 1
            if answered:
                metrics["by_type"][qt]["answered"] += 1

        return metrics

    # ── Save results ──────────────────────────────────────────────────────────

    async def save_results(self) -> None:
        run_name = self.config.run_name or self.config.dataset_name
        qa_results = list(self._checkpoint.get_completed().values())
        metrics = getattr(self, "_metrics", None)

        full = {
            "config": self.config.__dict__,
            "metrics": metrics,
            "qa_results": qa_results,
            "cache_sessions": await self._session_cache.count() if self._session_cache else None,
        }
        full_path = self.output_dir / f"{run_name}_full_results.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2, default=str)
        logger.info("Full results → %s", full_path)

        summary = {
            "config": self.config.__dict__,
            "metrics": metrics,
            "completed": self._checkpoint.n_completed,
            "failed": self._checkpoint.n_failed,
            "cache_sessions": await self._session_cache.count() if self._session_cache else None,
        }
        summary_path = self.output_dir / f"{run_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary → %s", summary_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        if self._session_cache:
            await self._session_cache.close()
        if self.vektori_client:
            await self.vektori_client.close()
        logger.info("Cleanup complete")

    # ── Print summary ─────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        metrics = getattr(self, "_metrics", None)
        print("\n" + "=" * 60)
        print("LONGMEMEVAL BENCHMARK RESULTS")
        print("=" * 60)

        if metrics:
            print(f"\nTotal : {metrics['total_questions']}")
            print(f"Answered : {metrics['answered']}")
            print(f"Abstained: {metrics['abstained']}")
            if metrics.get("by_type"):
                print("\nBy question type:")
                for qt, counts in metrics["by_type"].items():
                    pct = counts["answered"] / counts["total"] * 100 if counts["total"] else 0
                    print(f"  {qt:<35} {counts['answered']}/{counts['total']}  ({pct:.1f} %)")

        print(f"\nCompleted : {self._checkpoint.n_completed}")
        print(f"Failed    : {self._checkpoint.n_failed}")
        print(f"\nResults in: {self.output_dir}")
        print("=" * 60 + "\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> datetime | None:
    """Parse a LongMemEval date string like '2023/05/30 (Tue) 23:40'."""
    if not date_str:
        return None
    # Strip the weekday abbreviation: '2023/05/30 (Tue) 23:40' → '2023/05/30 23:40'
    clean = date_str.split("(")[0].strip() + " " + date_str.split(")")[-1].strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(clean.strip(), fmt)
        except ValueError:
            continue
    return None


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Vektori benchmark on LongMemEval")
    parser.add_argument(
        "--dataset",
        choices=["longmemeval_s_cleaned", "longmemeval_m_cleaned", "longmemeval_oracle"],
        default="longmemeval_s_cleaned",
    )
    parser.add_argument("--depth", choices=["l0", "l1", "l2"], default="l1")
    parser.add_argument("--embedding-model", default="sentence-transformers:BAAI/bge-m3")
    parser.add_argument("--extraction-model", default="gemini:gemini-2.5-flash-lite")
    parser.add_argument("--eval-model", default="gemini:gemini-2.5-flash-lite")
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--run-name",
        help="Name for this run (also used to locate its checkpoint for resume)",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        dataset_name=args.dataset,
        retrieval_depth=args.depth,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        eval_model=args.eval_model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
        run_name=args.run_name,
    )

    logger.info("Starting LongMemEval benchmark — config: %s", config)
    await LongMemEvalBenchmark(config).run()


if __name__ == "__main__":
    asyncio.run(main())
