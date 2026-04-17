from datetime import datetime

import pytest

from benchmarks.locomo.locomo_runner import (
    LoCoMoBenchmark,
    LoCoMoConfig,
    _format_retrieved_context,
    _load_qa_prompt_override,
    _parse_date,
)
from vektori.qa import build_qa_prompt, generate_answer


def test_locomo_defaults_disable_retrieval_gate():
    config = LoCoMoConfig()

    assert config.enable_retrieval_gate is False


def test_parse_date_supports_locomo_native_format():
    parsed = _parse_date("9:55 am on 22 October, 2023")

    assert parsed == datetime(2023, 10, 22, 9, 55)


def test_parse_date_supports_locomo_native_short_month_format():
    parsed = _parse_date("1:56 pm on 8 May, 2023")

    assert parsed == datetime(2023, 5, 8, 13, 56)


def test_build_qa_prompt_contains_grounding_and_aggregation_controls():
    prompt = build_qa_prompt(
        "How many fields did Caroline consider?",
        "Caroline considered psychology and counseling certification.",
        question_date="2023-05-08",
    )

    assert "TODAY'S DATE: 2023-05-08" in prompt
    assert "Consider every relevant item across all sessions" in prompt
    assert "Prefer the most recent value" in prompt
    assert "Do not answer with relative time words" in prompt


async def test_generate_answer_uses_shared_prompt_with_supplied_llm():
    class RecordingLLM:
        prompt = ""
        max_tokens = None

        async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
            self.prompt = prompt
            self.max_tokens = max_tokens
            return "2023-05-07"

    llm = RecordingLLM()

    answer = await generate_answer(
        question="When did Caroline attend the support group?",
        context="Caroline attended the support group on 2023-05-07.",
        question_date="2023-05-08",
        llm=llm,
        max_tokens=123,
    )

    assert answer == "2023-05-07"
    assert "Caroline attended the support group on 2023-05-07." in llm.prompt
    assert "When did Caroline attend the support group?" in llm.prompt
    assert llm.max_tokens == 123


def test_format_retrieved_context_ranks_facts_and_annotates_relative_time():
    context = _format_retrieved_context(
        {
            "facts": [
                {
                    "text": "User likes tea.",
                    "event_time": "2023-05-09T09:00:00",
                    "score": 0.2,
                },
                {
                    "text": "Caroline attended the support group yesterday.",
                    "event_time": "2023-05-08T13:56:00",
                    "score": 0.95,
                },
            ],
            "episodes": [],
            "syntheses": [],
            "sentences": [],
        }
    )

    lines = context.splitlines()

    assert lines[0] == "## Facts (ranked by relevance and specificity)"
    assert "Caroline attended the support group yesterday." in lines[1]
    assert "[2023-05-08]" in lines[1]
    assert '"yesterday" resolves to 2023-05-07' in lines[1]
    assert "User likes tea." in lines[2]


def test_load_qa_prompt_override_validates_placeholders(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("{date_line}{context}{question}", encoding="utf-8")

    assert _load_qa_prompt_override(str(prompt_path)) == "{date_line}{context}{question}"

    bad_path = tmp_path / "bad.txt"
    bad_path.write_text("{context}{question}", encoding="utf-8")

    with pytest.raises(ValueError, match="date_line"):
        _load_qa_prompt_override(str(bad_path))


def test_locomo_runner_keeps_prompt_override_on_instance(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("{date_line}{context}{question}", encoding="utf-8")

    runner = LoCoMoBenchmark(LoCoMoConfig(qa_prompt_path=str(prompt_path)))

    assert runner._qa_prompt_override == "{date_line}{context}{question}"
