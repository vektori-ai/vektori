import json

from benchmarks.locomo.locomo_judge import load_entries, parse_verdict, summarize


def test_parse_locomo_judge_verdict_json():
    verdict = parse_verdict(
        '{"verdict":"CORRECT","context_has_answer":true,"failure_mode":null,"explanation":"ok"}',
        latency_ms=12.3,
    )

    assert verdict.verdict == "CORRECT"
    assert verdict.context_has_answer is True
    assert verdict.failure_mode is None
    assert verdict.parse_error is False


def test_load_entries_from_full_results_samples_by_qid(tmp_path):
    full_results = tmp_path / "run_full_results.json"
    full_results.write_text(
        json.dumps(
            {
                "qa_results": [
                    {"question_id": "q1", "question": "one"},
                    {"question_id": "q2", "question": "two"},
                ]
            }
        ),
        encoding="utf-8",
    )

    entries = load_entries(full_results, qids=["q2"], n=1, seed=0)

    assert [entry["question_id"] for entry in entries] == ["q2"]


def test_summarize_locomo_judge_results():
    summary = summarize(
        [
            {
                "verdict": "CORRECT",
                "context_has_answer": True,
                "failure_mode": None,
                "question_type": "1",
            },
            {
                "verdict": "WRONG",
                "context_has_answer": True,
                "failure_mode": "QA_FAILURE",
                "question_type": "1",
            },
        ]
    )

    assert summary["correct"] == 1
    assert summary["qa_failure"] == 1
    assert summary["context_has_answer_rate"] == 1.0
    assert summary["by_type"]["1"]["total"] == 2
