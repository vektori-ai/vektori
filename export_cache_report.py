"""Export session extract cache as a dashboard-friendly JSON report.

Usage (PowerShell):
  python export_cache_report.py \
    --cache-db benchmark_results/.cache/session_extract_cache.db \
    --dataset data/longmemeval_s_cleaned.json \
    --output benchmark_results/longmemeval_s_cache_report.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_question_index(dataset_path: Path) -> dict[str, list[dict[str, Any]]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        qid = item.get("question_id", "")
        qtype = item.get("question_type", "")
        question = item.get("question", "")
        answer_session_ids = set(item.get("answer_session_ids", []) or [])
        for sid in item.get("haystack_session_ids", []) or []:
            by_session[sid].append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "question": question,
                    "contains_answer": sid in answer_session_ids,
                }
            )
    return dict(by_session)


def load_cache_rows(cache_db: Path) -> list[tuple[str, str, str | None]]:
    conn = sqlite3.connect(str(cache_db))
    try:
        rows = conn.execute(
            "SELECT session_id, facts_json, created_at FROM session_cache ORDER BY session_id"
        ).fetchall()
        return rows
    finally:
        conn.close()


def build_report(cache_db: Path, dataset_path: Path) -> dict[str, Any]:
    question_index = load_question_index(dataset_path)
    rows = load_cache_rows(cache_db)

    cache_records: list[dict[str, Any]] = []
    linked_sessions = 0
    linked_answer_sessions = 0
    total_facts = 0

    for session_id, facts_json, created_at in rows:
        logical_session_id = session_id.split("::", 1)[1] if "::" in session_id else session_id
        try:
            facts = json.loads(facts_json) if facts_json else []
        except json.JSONDecodeError:
            facts = []

        linked_questions = question_index.get(session_id, []) or question_index.get(logical_session_id, [])
        if linked_questions:
            linked_sessions += 1
        if any(bool(q.get("contains_answer")) for q in linked_questions):
            linked_answer_sessions += 1

        fact_texts: list[str] = []
        for fact in facts:
            if isinstance(fact, dict):
                text = str(fact.get("text", "")).strip()
                if text:
                    fact_texts.append(text)

        fact_count = len(fact_texts)
        total_facts += fact_count

        answer_qids = [q["question_id"] for q in linked_questions if q.get("contains_answer")]
        linked_qids = [q["question_id"] for q in linked_questions if not q.get("contains_answer")]
        ordered_preview_ids = answer_qids[:2] + linked_qids[:4]
        question_preview = " ; ".join(ordered_preview_ids[:4])
        hypothesis_preview = fact_texts[0] if fact_texts else "(no facts cached)"

        cache_records.append(
            {
                "question_id": session_id,
                "question_type": "cache-session",
                "question": (
                    f"Linked questions: {question_preview}" if question_preview else "No linked benchmark question"
                ),
                "hypothesis": hypothesis_preview,
                "expected_answer": f"{fact_count} facts cached",
                "retrieval_depth": "cache",
                "cache_meta": {
                    "session_id": session_id,
                    "logical_session_id": logical_session_id,
                    "fact_count": fact_count,
                    "created_at": created_at,
                    "contains_answer_for_any": any(bool(q.get("contains_answer")) for q in linked_questions),
                    "linked_questions": linked_questions,
                    "all_facts": fact_texts,
                    "sample_facts": fact_texts[:12],
                },
            }
        )

    total = len(cache_records)
    summary = {
        "total_cache_sessions": total,
        "linked_sessions": linked_sessions,
        "linked_answer_sessions": linked_answer_sessions,
        "unlinked_sessions": max(0, total - linked_sessions),
        "avg_facts_per_session": round((total_facts / total), 2) if total else 0.0,
    }

    return {
        "cache_report_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_db": str(cache_db),
        "dataset": str(dataset_path),
        "summary": summary,
        "cache_records": cache_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cache DB with links to benchmark questions")
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=Path("benchmark_results/.cache/session_extract_cache.db"),
        help="Path to session_extract_cache.db",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/longmemeval_s_cleaned.json"),
        help="Path to LongMemEval dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/longmemeval_s_cache_report.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not args.cache_db.exists():
        raise FileNotFoundError(f"Cache DB not found: {args.cache_db}")
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    report = build_report(args.cache_db, args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print(f"Wrote: {args.output}")
    print(
        "Cache sessions: {total} | linked: {linked} | unlinked: {unlinked} | avg facts/session: {avg}".format(
            total=s["total_cache_sessions"],
            linked=s["linked_sessions"],
            unlinked=s["unlinked_sessions"],
            avg=s["avg_facts_per_session"],
        )
    )


if __name__ == "__main__":
    main()
