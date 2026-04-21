"""
LoCoMo Benchmark Runner for Vektori
=====================================

Evaluates Vektori's persistent agent memory on the LoCoMo-10 dataset.

Ingestion strategy
------------------
Unlike LongMemEval (per-question isolation), LoCoMo groups all QA items by
sample — every QA item in the same sample shares the **exact same** haystack
sessions.  We therefore ingest once per sample, answer all pending questions
for that sample, then delete the user.

* **Per-sample isolation** — ``user_id = "locomo_{sample_id}"``.
  After all questions for a sample are answered, ``delete_user()`` wipes rows.

* **No extract cache needed** — sessions are never shared across samples, so
  there is nothing to cache between samples.

Checkpointing
-------------
Progress is saved per-question (same BenchmarkCheckpoint as LongMemEval).
Re-running resumes from the first unfinished question.  A sample whose
questions are ALL already done is skipped entirely (no re-ingestion).

Pilot mode
----------
Set ``LoCoMoConfig.max_questions`` to a small number (e.g. 5) to run a quick
sanity-check before committing to the full dataset.
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from benchmarks.locomo.locomo_judge import is_abstention_answer
from benchmarks.longmemeval.checkpoint import BenchmarkCheckpoint
from benchmarks.longmemeval.session_cache import SessionExtractCache
from vektori.qa import generate_answer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LoCoMoConfig:
    """Configuration for a LoCoMo benchmark run."""

    # Dataset
    data_dir: str = "data"
    dataset_name: str = "locomo10_cooked"

    # Vektori
    storage_backend: str = "sqlite"
    database_url: str | None = None
    embedding_model: str = "cloudflare:@cf/baai/bge-m3"
    extraction_model: str = "gemini:gemini-2.5-flash-lite"
    max_extraction_output_tokens: int = 32768

    # Retrieval
    retrieval_depth: str = "l1"
    top_k: int = 10
    context_window: int = 3
    enable_retrieval_gate: bool = False

    # Output
    output_dir: str = "benchmark_results"
    run_name: str | None = None

    # Evaluation
    eval_model: str = "gemini:gemini-2.5-flash-lite"
    qa_prompt_path: str | None = None

    # Pilot mode — set to a small number to test before full run
    max_questions: int | None = None

    # Extraction cache
    use_cache: bool = True
    cache_namespace: str | None = None

    # Retrieval ablation
    use_ppr: bool = True


# ── Runner ────────────────────────────────────────────────────────────────────

class LoCoMoBenchmark:
    """Main LoCoMo benchmark runner."""

    def __init__(self, config: LoCoMoConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vektori_client = None
        self.storage = None
        self.dataset: list[dict[str, Any]] = []
        self.dataset_path: Path | None = None
        self._session_cache: SessionExtractCache | None = None
        self._checkpoint: BenchmarkCheckpoint | None = None
        self._qa_prompt_override = _load_qa_prompt_override(config.qa_prompt_path)

    # ── Setup ────────────────────────────────────────────────────────────────

    async def setup(self) -> None:
        logger.info("Setting up LoCoMo benchmark…")
        await self._init_vektori()
        await self._load_dataset()
        await self._init_session_cache()
        self._init_checkpoint()
        pilot_note = (
            f" [PILOT MODE: first {self.config.max_questions} questions only]"
            if self.config.max_questions else ""
        )
        logger.info(
            "Setup complete — %d questions in dataset%s", len(self.dataset), pilot_note
        )

    async def _init_vektori(self) -> None:
        from vektori import Vektori
        from vektori.config import VektoriConfig

        logger.info(
            "Initialising Vektori (backend=%s, embedding=%s, extraction=%s)",
            self.config.storage_backend,
            self.config.embedding_model,
            self.config.extraction_model,
        )
        self.vektori_client = Vektori(
            config=VektoriConfig(
                database_url=self.config.database_url,
                storage_backend=self.config.storage_backend,
                embedding_model=self.config.embedding_model,
                extraction_model=self.config.extraction_model,
                default_top_k=self.config.top_k,
                context_window=self.config.context_window,
                enable_retrieval_gate=self.config.enable_retrieval_gate,
                async_extraction=False,
                max_extraction_output_tokens=self.config.max_extraction_output_tokens,
                use_ppr=self.config.use_ppr,
            )
        )
        await self.vektori_client._ensure_initialized()
        self.storage = self.vektori_client.db

    async def _load_dataset(self) -> None:
        filename = f"{self.config.dataset_name}.json"
        self.dataset_path = Path(self.config.data_dir) / filename

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"LoCoMo cooked dataset not found at: {self.dataset_path}\n"
                "Cook it first with:\n"
                "  python -m benchmarks.locomo.cook_locomo "
                "--output data/locomo10_cooked.json --download-if-missing"
            )

        logger.info("Loading dataset from %s", self.dataset_path)
        with open(self.dataset_path, encoding="utf-8") as f:
            all_items = json.load(f)

        # Apply pilot cap if set
        if self.config.max_questions is not None:
            all_items = all_items[: self.config.max_questions]
            logger.info(
                "Pilot mode: using %d/%d questions",
                len(all_items), self.config.max_questions,
            )

        self.dataset = all_items
        logger.info("Loaded %d QA items", len(self.dataset))

    async def _init_session_cache(self) -> None:
        if not self.config.use_cache:
            logger.info("Session extract cache disabled (--no-cache)")
            self._session_cache = None
            return

        # Shared with LongMemEval cache dir so sessions extracted by either
        # benchmark can be reused by the other (same session_id namespace).
        cache_path = Path(self.config.output_dir).parent / ".cache" / "session_extract_cache.db"
        self._session_cache = SessionExtractCache(cache_path)
        await self._session_cache.initialize()
        logger.info("Session cache namespace: %s", self._cache_namespace())

    def _cache_namespace(self) -> str:
        if self.config.cache_namespace:
            return self.config.cache_namespace
        raw = (
            f"locomo|{self.config.extraction_model}|"
            f"out={self.config.max_extraction_output_tokens}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _cache_key(self, session_id: str) -> str:
        return f"{self._cache_namespace()}::{session_id}"

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
            await self._run_samples()
            await self.evaluate()
            await self.save_results()
            logger.info("LoCoMo benchmark complete!")
            self._print_summary()
        finally:
            await self.cleanup()

    async def _run_samples(self) -> None:
        """Group dataset by sample_id; ingest once per sample, answer all QA items."""
        # Group questions by sample_id (preserves order within each sample)
        samples: dict[str, list[dict[str, Any]]] = {}
        sample_order: list[str] = []
        for item in self.dataset:
            sid = _sample_id_from_question_id(item["question_id"])
            if sid not in samples:
                samples[sid] = []
                sample_order.append(sid)
            samples[sid].append(item)

        total_qs = len(self.dataset)
        logger.info("Processing %d samples (%d total questions)", len(sample_order), total_qs)

        for sample_id in sample_order:
            qa_items = samples[sample_id]
            pending = [qa for qa in qa_items if not self._checkpoint.is_done(qa["question_id"])]

            if not pending:
                logger.debug("Sample %s: all questions done, skipping ingestion", sample_id)
                continue

            user_id = f"locomo_{sample_id}"
            logger.info(
                "Sample %s — ingesting sessions, then answering %d/%d questions",
                sample_id, len(pending), len(qa_items),
            )

            try:
                ingest_t0 = time.perf_counter()
                await self._ingest_sample(qa_items[0], user_id)
                ingestion_ms = (time.perf_counter() - ingest_t0) * 1000
                logger.info("Sample %s ingested in %.0f ms", sample_id, ingestion_ms)

                for qa_item in pending:
                    qid = qa_item["question_id"]
                    try:
                        q_t0 = time.perf_counter()
                        result = await self._answer_question(qa_item, user_id)
                        total_ms = (time.perf_counter() - q_t0) * 1000

                        result["ingestion_ms"] = round(ingestion_ms, 1)
                        result["total_question_ms"] = round(total_ms, 1)

                        self._checkpoint.mark_done(qid, result)
                        self._checkpoint.save()

                        done = self._checkpoint.n_completed
                        logger.info(
                            "  [%d/%d] %s → retrieval=%.0fms  qa=%.0fms  total=%.0fms",
                            done, total_qs, qid,
                            result["retrieval_ms"], result["qa_ms"], result["total_question_ms"],
                        )
                    except Exception as e:
                        logger.error("Question %s failed: %s", qid, e, exc_info=True)
                        self._checkpoint.mark_failed(qid, str(e))
                        self._checkpoint.save()

            except Exception as e:
                logger.error("Sample %s ingestion failed: %s", sample_id, e, exc_info=True)
                for qa_item in pending:
                    self._checkpoint.mark_failed(qa_item["question_id"], f"ingestion_failed: {e}")
                self._checkpoint.save()
            finally:
                try:
                    await self.vektori_client.delete_user(user_id)
                except Exception as e:
                    logger.warning("delete_user(%s) failed: %s", user_id, e)

    # ── Ingestion ────────────────────────────────────────────────────────────

    async def _ingest_sample(self, reference_item: dict[str, Any], user_id: str) -> None:
        """Ingest all haystack sessions for a sample.

        Cache hit  → replay pre-extracted facts (no LLM, only local re-embed).
        Cache miss → full LLM extraction, write to cache.

        This enables crash recovery: if the run is interrupted mid-sample,
        sessions already extracted are cached and won't cost another LLM call
        on the next run.
        """
        haystack_sessions = reference_item["haystack_sessions"]
        haystack_sids = reference_item["haystack_session_ids"]
        haystack_dates = reference_item.get("haystack_dates") or []
        n_sessions = len(haystack_sessions)

        for i, (session, hsid) in enumerate(zip(haystack_sessions, haystack_sids)):
            session_date = haystack_dates[i] if i < len(haystack_dates) else None
            session_time = _parse_date(session_date) if session_date else None

            sess_t0 = time.perf_counter()
            cached_entry = (
                await self._session_cache.get(self._cache_key(hsid))
                if self._session_cache
                else None
            )
            cached_facts = cached_entry.get("facts", []) if isinstance(cached_entry, dict) else cached_entry
            if cached_facts is not None:
                await self._replay_session(session, hsid, user_id, session_time, session_date, cached_facts)
                elapsed = (time.perf_counter() - sess_t0) * 1000
                logger.info(
                    "  Session %d/%d [CACHE HIT ] %s — %.0f ms (%d facts replayed)",
                    i + 1, n_sessions, hsid, elapsed, len(cached_facts),
                )
            else:
                new_facts = await self._full_ingest_session(
                    session, hsid, user_id, session_time, session_date
                )
                elapsed = (time.perf_counter() - sess_t0) * 1000
                logger.info(
                    "  Session %d/%d [CACHE MISS] %s — %.0f ms (%d facts extracted)",
                    i + 1, n_sessions, hsid, elapsed, len(new_facts),
                )
                if new_facts and self._session_cache:
                    await self._session_cache.put(self._cache_key(hsid), new_facts)

    async def _replay_session(
        self,
        session: list[dict[str, str]],
        haystack_sid: str,
        user_id: str,
        session_time: datetime | None,
        session_date: str | None,
        cached_facts: list[dict[str, Any]],
    ) -> None:
        """Cache hit: store sentences + replay pre-extracted facts + run episode extraction."""
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
                await extractor._extract_episodes(
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
        """Cache miss: full LLM extraction. Returns cacheable facts."""
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
        self, item: dict[str, Any], user_id: str
    ) -> dict[str, Any]:
        qid = item["question_id"]
        question = item["question"]
        question_type = item["question_type"]
        question_date = item.get("question_date") or ""

        retrieval_t0 = time.perf_counter()
        search_results = await self.vektori_client.search(
            query=question,
            user_id=user_id,
            depth=self.config.retrieval_depth,
            reference_date=_parse_date(question_date) if question_date else None,
        )
        retrieval_ms = (time.perf_counter() - retrieval_t0) * 1000

        context = _format_retrieved_context(search_results)

        qa_t0 = time.perf_counter()
        answer = await self._generate_answer(question, context, question_date)
        qa_ms = (time.perf_counter() - qa_t0) * 1000

        return {
            "question_id": qid,
            "question": question,
            "question_type": question_type,
            "question_date": question_date,
            "hypothesis": answer,
            "expected_answer": item["answer"],
            "retrieved_context": context,
            "retrieval_depth": self.config.retrieval_depth,
            "retrieval_ms": round(retrieval_ms, 1),
            "qa_ms": round(qa_ms, 1),
        }

    async def _generate_answer(
        self, question: str, context: str, question_date: str = ""
    ) -> str:
        return await generate_answer(
            question=question,
            context=context,
            question_date=question_date,
            model=self.config.eval_model,
            prompt_template=self._qa_prompt_override,
            max_tokens=2048,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    async def evaluate(self) -> None:
        logger.info("Computing evaluation metrics…")
        qa_results = list(self._checkpoint.get_completed().values())
        if not qa_results:
            logger.warning("No completed QA results to evaluate")
            return

        run_name = self.config.run_name or self.config.dataset_name
        jsonl_path = self.output_dir / f"{run_name}_qa_results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in qa_results:
                f.write(
                    json.dumps({
                        "question_id": r["question_id"],
                        "hypothesis": r["hypothesis"],
                    }) + "\n"
                )
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
            hyp = r.get("hypothesis") or ""
            answered = bool(hyp) and not is_abstention_answer(hyp)
            if answered:
                metrics["answered"] += 1
            else:
                metrics["abstained"] += 1

            qt = r.get("question_type", "unknown")
            metrics["by_type"].setdefault(qt, {"total": 0, "answered": 0})
            metrics["by_type"][qt]["total"] += 1
            if answered:
                metrics["by_type"][qt]["answered"] += 1

        def _collect(field: str) -> list[float]:
            return [float(r[field]) for r in qa_results if isinstance(r.get(field), (int, float))]

        def _avg(vals: list[float]) -> float | None:
            return round(sum(vals) / len(vals), 1) if vals else None

        retrieval_vals = _collect("retrieval_ms")
        qa_vals = _collect("qa_ms")
        total_vals = _collect("total_question_ms")

        metrics["latency_ms"] = {
            "retrieval_avg": _avg(retrieval_vals),
            "qa_avg": _avg(qa_vals),
            "total_question_avg": _avg(total_vals),
        }

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
        print("LOCOMO BENCHMARK RESULTS")
        print("=" * 60)

        if metrics:
            print(f"\nTotal     : {metrics['total_questions']}")
            print(f"Answered  : {metrics['answered']}")
            print(f"Abstained : {metrics['abstained']}")
            lat = metrics.get("latency_ms") or {}
            if lat:
                print("\nLatency (ms):")
                print(f"  Retrieval      avg={lat.get('retrieval_avg')}")
                print(f"  QA generation  avg={lat.get('qa_avg')}")
                print(f"  Total/question avg={lat.get('total_question_avg')}")
            if metrics.get("by_type"):
                print("\nBy question type:")
                for qt, counts in metrics["by_type"].items():
                    pct = counts["answered"] / counts["total"] * 100 if counts["total"] else 0
                    print(f"  {qt:<35} {counts['answered']}/{counts['total']}  ({pct:.1f} %)")

        print(f"\nCompleted : {self._checkpoint.n_completed}")
        print(f"Failed    : {self._checkpoint.n_failed}")
        print(f"\nResults in: {self.output_dir}")
        print("=" * 60 + "\n")


# ── Module-level helpers ──────────────────────────────────────────────────────

_QID_RE = re.compile(r"^locomo_(.+)_q\d+$")


def _sample_id_from_question_id(qid: str) -> str:
    """Extract sample_id from 'locomo_{sample_id}_q{N}'."""
    m = _QID_RE.match(qid)
    if not m:
        raise ValueError(f"Cannot parse sample_id from question_id: {qid!r}")
    return m.group(1)


def _parse_date(date_str: str) -> datetime | None:
    """Parse LoCoMo session date strings.

    Supports:
    - LoCoMo native format: '9:55 am on 22 October, 2023'
    - ISO-ish strings like '2023-05-30' or '2023-05-30T14:30:00'
    - LongMemEval-style fallback strings
    """
    if not date_str:
        return None

    clean = date_str.strip()

    # LoCoMo native format
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(clean.upper(), fmt)
        except ValueError:
            continue

    # Try ISO formats first
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            continue

    # LongMemEval-style fallback: '2023/05/30 (Tue) 23:40'
    clean = date_str.split("(")[0].strip() + " " + date_str.split(")")[-1].strip()
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d"):
        try:
            return datetime.strptime(clean.strip(), fmt)
        except ValueError:
            continue

    logger.debug("Could not parse date string: %r", date_str)
    return None


def _load_qa_prompt_override(path: str | None) -> str | None:
    if not path:
        return None

    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"QA prompt override file not found: {prompt_path}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    required = ("{date_line}", "{context}", "{question}")
    missing = [field for field in required if field not in prompt]
    if missing:
        raise ValueError(
            f"QA prompt override {prompt_path} is missing required placeholders: {missing}"
        )
    logger.info("Loaded QA prompt override from %s", prompt_path)
    return prompt


def _coerce_datetime(value: Any) -> datetime | None:
    """Convert common timestamp shapes into a naive datetime."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if not value:
        return None

    clean = str(value).strip()
    if not clean:
        return None
    if clean.endswith("Z"):
        clean = f"{clean[:-1]}+00:00"

    try:
        return datetime.fromisoformat(clean).replace(tzinfo=None)
    except ValueError:
        return _parse_date(clean)


def _timestamp_for_context(item: dict[str, Any]) -> Any:
    # Prefer event_time (real session date). created_at is ingestion time — only
    # use it as a last resort for relative-time note computation, NOT for date hints
    # shown to the model (which would show today's date instead of the session date).
    return item.get("event_time") or item.get("created_at") or ""


def _event_time_only(item: dict[str, Any]) -> Any:
    """Return event_time only — never fall back to created_at (ingestion date)."""
    return item.get("event_time") or ""


def _date_prefix(timestamp: Any) -> str:
    parsed = _coerce_datetime(timestamp)
    if parsed:
        return parsed.date().isoformat()
    text = str(timestamp).strip()
    return text[:10] if text else ""


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fact_relevance_score(fact: dict[str, Any]) -> float:
    score = _as_float(fact.get("score"))
    if score is not None:
        return score

    distance = _as_float(fact.get("distance"))
    if distance is not None:
        return 1.0 - distance

    return 0.0


def _fact_specificity_score(fact: dict[str, Any]) -> int:
    text = str(fact.get("text", ""))
    metadata = fact.get("metadata") or {}
    if isinstance(metadata, str):
        import json
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}
    
    score = 0
    if _timestamp_for_context(fact):
        score += 3
    if metadata.get("temporal_expr"):
        score += 2
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        score += 2
    score += min(len(re.findall(r"\b\d[\w:/.-]*\b", text)), 3)
    score += min(len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text)), 3)
    if len(text) <= 240:
        score += 1
    return score


def _fact_context_sort_key(fact: dict[str, Any]) -> tuple[float, int, float]:
    timestamp = _coerce_datetime(_timestamp_for_context(fact))
    timestamp_value = timestamp.timestamp() if timestamp else 0.0
    return (
        -_fact_relevance_score(fact),
        -_fact_specificity_score(fact),
        -timestamp_value,
    )


def _add_months(dt: datetime, n: int) -> datetime:
    """Add n months to dt, clamping day to last day of target month."""
    month = dt.month - 1 + n
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


_TEMPORAL_WORD_TO_N: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "a few": 3, "several": 4, "couple": 2,
}

_AGO_PAT = re.compile(
    r"\b(a few|several|a|an|\d+|one|two|three|four|five|six|seven|couple)"
    r"\s+(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)

_DOW_PAT = re.compile(
    r"\b(last|next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _relative_time_note(text: str, timestamp: Any) -> str:
    reference_dt = _coerce_datetime(timestamp)
    if not text or not reference_dt:
        return ""

    reference_date = reference_dt.date()
    lower = text.lower()
    notes: list[str] = []
    for phrase, offset_days in (
        ("yesterday", -1),
        ("today", 0),
        ("tomorrow", 1),
    ):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            resolved = (reference_dt + timedelta(days=offset_days)).date().isoformat()
            notes.append(
                f'"{phrase}" resolves to {resolved} from session date {reference_date.isoformat()}'
            )

    # Fix 3: "N days/weeks/months/years ago" — resolve to actual date
    for m in _AGO_PAT.finditer(lower):
        qty_str = m.group(1).lower()
        unit = m.group(2).lower()
        n = _TEMPORAL_WORD_TO_N.get(qty_str) or (int(qty_str) if qty_str.isdigit() else None)
        if n is None:
            continue
        if unit == "day":
            resolved = (reference_dt - timedelta(days=n)).date()
            notes.append(f'"{m.group(0)}" → {resolved.isoformat()} (session date: {reference_date.isoformat()})')
        elif unit == "week":
            resolved = (reference_dt - timedelta(weeks=n)).date()
            notes.append(f'"{m.group(0)}" → week of {resolved.isoformat()} (session date: {reference_date.isoformat()})')
        elif unit == "month":
            resolved = _add_months(reference_dt, -n).date()
            notes.append(f'"{m.group(0)}" → {resolved.strftime("%B %Y")} (session date: {reference_date.isoformat()})')
        elif unit == "year":
            resolved = reference_dt.replace(year=reference_dt.year - n).date()
            notes.append(f'"{m.group(0)}" → {resolved.year} (session date: {reference_date.isoformat()})')

    # Fix 4: "last/next/this <weekday>" — resolve to actual date
    for m in _DOW_PAT.finditer(lower):
        direction = m.group(1).lower()
        day_name = m.group(2).lower()
        target_dow = _DAYS.index(day_name)   # 0=Mon … 6=Sun
        current_dow = reference_dt.weekday()
        if direction == "last":
            delta = (current_dow - target_dow) % 7 or 7
            resolved = (reference_dt - timedelta(days=delta)).date()
        elif direction == "next":
            delta = (target_dow - current_dow) % 7 or 7
            resolved = (reference_dt + timedelta(days=delta)).date()
        else:  # "this"
            delta = (target_dow - current_dow) % 7
            resolved = (reference_dt + timedelta(days=delta)).date()
        notes.append(f'"{m.group(0)}" → {resolved.isoformat()} (session date: {reference_date.isoformat()})')

    # Fix 1: named offset phrases — resolve to actual date/period
    _OFFSET_PHRASES: list[tuple[str, str]] = [
        ("last week",  "week"),
        ("next week",  "week"),
        ("last month", "month"),
        ("next month", "month"),
        ("last year",  "year"),
        ("next year",  "year"),
        ("this week",  "anchor"),
        ("this month", "anchor"),
        ("this year",  "anchor"),
        ("recently",   "anchor"),
        ("earlier this week", "anchor"),
        ("later this week",   "anchor"),
    ]
    for phrase, kind in _OFFSET_PHRASES:
        if phrase not in lower:
            continue
        if kind == "anchor":
            notes.append(f'"{phrase}" anchored to session date {reference_date.isoformat()}')
        elif kind == "year":
            delta_y = -1 if phrase.startswith("last") else 1
            resolved_year = reference_dt.replace(year=reference_dt.year + delta_y).year
            notes.append(f'"{phrase}" → {resolved_year} (session date: {reference_date.isoformat()})')
        elif kind == "month":
            delta_m = -1 if phrase.startswith("last") else 1
            resolved_dt = _add_months(reference_dt, delta_m)
            notes.append(f'"{phrase}" → {resolved_dt.strftime("%B %Y")} (session date: {reference_date.isoformat()})')
        else:  # week
            delta_w = -1 if phrase.startswith("last") else 1
            resolved_dt = reference_dt + timedelta(weeks=delta_w)
            notes.append(f'"{phrase}" → week of {resolved_dt.date().isoformat()} (session date: {reference_date.isoformat()})')

    if not notes:
        return ""
    return f" [Temporal note: {'; '.join(notes)}.]"


def _format_retrieved_context(search_results: Any) -> str:
    if not search_results:
        return "No relevant context retrieved."

    lines: list[str] = []

    facts = search_results.get("facts") or []
    if facts:
        facts = sorted(
            facts,
            key=_fact_context_sort_key,
        )
        lines.append("## Facts (ranked by relevance and specificity)")
        for i, fact in enumerate(facts, 1):
            timestamp = _timestamp_for_context(fact)
            date = _date_prefix(timestamp)
            date_prefix = f"[{date}] " if date else ""
            text = str(fact.get("text", str(fact))).strip()
            text = f"{text}{_relative_time_note(text, timestamp)}"
            lines.append(f"{i}. {date_prefix}{text}")

    episodes = search_results.get("episodes") or []
    if episodes:
        lines.append("\n## Episodes")
        for i, ep in enumerate(episodes, 1):
            timestamp = _timestamp_for_context(ep)
            date = _date_prefix(timestamp)
            date_prefix = f"[{date}] " if date else ""
            text = str(ep.get("text", str(ep))).strip()
            text = f"{text}{_relative_time_note(text, timestamp)}"
            lines.append(f"{i}. {date_prefix}{text}")

    syntheses = search_results.get("syntheses") or []
    if syntheses:
        lines.append("\n## Syntheses")
        for i, sy in enumerate(syntheses, 1):
            timestamp = _timestamp_for_context(sy)
            date = _date_prefix(timestamp)
            date_prefix = f"[{date}] " if date else ""
            text = str(sy.get("text", str(sy))).strip()
            text = f"{text}{_relative_time_note(text, timestamp)}"
            lines.append(f"{i}. {date_prefix}{text}")

    syntheses = search_results.get("syntheses") or []
    if syntheses:
        lines.append("\n## Syntheses")
        for i, sy in enumerate(syntheses, 1):
            lines.append(f"{i}. {sy.get('text', str(sy))}")

    sentences = search_results.get("sentences") or []
    if sentences:
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
            date_hint = ""
            for s in sents:
                ts = _event_time_only(s)
                if ts:
                    date = _date_prefix(ts)
                    date_hint = f" [{date}]" if date else ""
                    break
            lines.append(f"\n### Session {n}{date_hint}")
            for sent in sents:
                role = sent.get("role", "")
                prefix = f"[{role.upper()}] " if role else ""
                timestamp = _timestamp_for_context(sent)
                text = str(sent.get("text", str(sent))).strip()
                text = f"{text}{_relative_time_note(text, timestamp)}"
                lines.append(f"  {prefix}{text}")

    return "\n".join(lines) if lines else "No relevant context retrieved."


# ── CLI entry point ───────────────────────────────────────────────────────────

async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Vektori benchmark on LoCoMo-10")
    parser.add_argument("--depth", choices=["l0", "l1", "l2"], default="l1")
    parser.add_argument(
        "--embedding-model", default="cloudflare:@cf/baai/bge-m3",
        help="Embedding model string (provider:model_name)"
    )
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
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--run-name", help="Run name for checkpointing/output files")
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Limit to first N questions (pilot mode; omit for full run)"
    )

    args = parser.parse_args()

    config = LoCoMoConfig(
        retrieval_depth=args.depth,
        embedding_model=args.embedding_model,
        extraction_model=args.extraction_model,
        eval_model=args.eval_model,
        qa_prompt_path=args.qa_prompt_file,
        max_extraction_output_tokens=args.max_extraction_output_tokens,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        top_k=args.top_k,
        run_name=args.run_name,
        max_questions=args.max_questions,
        use_cache=not args.no_cache,
        cache_namespace=args.cache_namespace,
    )

    logger.info("Starting LoCoMo benchmark — config: %s", config)
    await LoCoMoBenchmark(config).run()


if __name__ == "__main__":
    asyncio.run(main())
