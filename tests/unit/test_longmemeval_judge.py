"""Unit tests for LongMemEval judge behavior."""

import asyncio

import pytest

from benchmarks.longmemeval import judge


def _entry(question_type: str, hypothesis: str) -> dict:
    return {
        "question_id": "q-1",
        "question": "What was the launch date?",
        "question_type": question_type,
        "hypothesis": hypothesis,
        "expected_answer": "The launch date was never mentioned.",
        "retrieved_context": "No launch date appears in the sessions.",
    }


def _judge_response(verdict: str = "ABSTAINED") -> str:
    return (
        "{"
        f'"verdict":"{verdict}",'
        '"context_has_answer":false,'
        '"failure_mode":"RETRIEVAL_FAILURE",'
        '"explanation":"model abstained"'
        "}"
    )


def test_abs_question_with_abstention_is_forced_correct(monkeypatch):
    async def fake_call_judge(prompt: str, provider: str, model: str, lmstudio_base_url: str = ""):
        return _judge_response("WRONG"), 12.3

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)

    result = asyncio.run(
        judge.evaluate_entry(
            _entry("single-session_abs", "I don't have that information."),
            provider="lmstudio",
            model="meta-llama-3.1-8b-instruct",
        )
    )

    assert result["verdict"] == "CORRECT"
    assert result["failure_mode"] is None
    assert result["raw_judge_response"] == _judge_response("WRONG")


def test_abs_question_without_abstention_not_promoted(monkeypatch):
    async def fake_call_judge(
        prompt: str,
        provider: str,
        model: str,
        lmstudio_base_url: str = "",
    ):
        return _judge_response("WRONG"), 9.4

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)

    result = asyncio.run(
        judge.evaluate_entry(
            _entry("single-session_abs", "The launch date was April 9, 2026."),
            provider="lmstudio",
            model="meta-llama-3.1-8b-instruct",
        )
    )

    assert result["verdict"] == "WRONG"
    assert result["failure_mode"] == "RETRIEVAL_FAILURE"


def test_non_abs_question_abstention_is_not_promoted(monkeypatch):
    async def fake_call_judge(prompt: str, provider: str, model: str, lmstudio_base_url: str = ""):
        return _judge_response("ABSTAINED"), 8.5

    monkeypatch.setattr(judge, "call_judge", fake_call_judge)

    result = asyncio.run(
        judge.evaluate_entry(
            _entry("single-session", "I don't have that information."),
            provider="lmstudio",
            model="meta-llama-3.1-8b-instruct",
        )
    )

    assert result["verdict"] == "ABSTAINED"
    assert result["failure_mode"] == "RETRIEVAL_FAILURE"


def test_is_abstention_answer_detects_common_phrases():
    assert judge.is_abstention_answer("Not enough information to answer.")
    assert judge.is_abstention_answer("I cannot answer this question from the provided context.")
    assert not judge.is_abstention_answer("The answer is April 9, 2026.")


def test_parse_qid_inputs_combines_file_and_inline_and_dedupes(tmp_path):
    qids_file = tmp_path / "qids.txt"
    qids_file.write_text("q2\nq3\n# comment\nq1\n", encoding="utf-8")

    parsed = judge._parse_qid_inputs("q1,q2", str(qids_file))

    assert parsed == ["q1", "q2", "q3"]


def test_select_entries_uses_explicit_qids_in_order():
    completed = {
        "q1": {"question_id": "q1"},
        "q2": {"question_id": "q2"},
        "q3": {"question_id": "q3"},
    }

    selected = judge._select_entries(completed, n=1, seed=42, qids=["q3", "q1"])

    assert [entry["question_id"] for entry in selected] == ["q3", "q1"]


def test_select_entries_errors_for_missing_qids():
    completed = {"q1": {"question_id": "q1"}}

    with pytest.raises(ValueError, match="requested QIDs not found"):
        judge._select_entries(completed, n=5, seed=42, qids=["q1", "q-missing"])
