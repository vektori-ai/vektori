from benchmarks.longmemeval.longmemeval_runner import BenchmarkConfig, LongMemEvalBenchmark


class RecordingClient:
    def __init__(self) -> None:
        self.expand = None

    async def search(self, **kwargs):
        self.expand = kwargs["expand"]
        return {"facts": [{"text": "User discussed the trip."}]}


async def test_abs_temporal_questions_use_base_type_for_expansion(tmp_path):
    runner = LongMemEvalBenchmark(BenchmarkConfig(output_dir=str(tmp_path)))
    client = RecordingClient()
    runner.vektori_client = client

    async def fake_generate(
        question: str, context: str, question_type: str, question_date: str = ""
    ) -> str:
        return "answer"

    runner._generate_answer = fake_generate

    await runner._answer_question(
        {
            "question_id": "q1",
            "question": "How long ago was the trip?",
            "question_type": "temporal-reasoning_abs",
            "question_date": "2025-01-10",
            "answer": "I don't have that information",
        },
        user_id="u1",
    )

    assert client.expand is True


def test_abs_temporal_prompt_keeps_temporal_and_abstention_instructions(tmp_path):
    runner = LongMemEvalBenchmark(BenchmarkConfig(output_dir=str(tmp_path)))

    prompt = runner._build_qa_prompt(
        question="How long ago was the trip?",
        context="No trip date was mentioned.",
        question_type="temporal-reasoning_abs",
        question_date="2025-01-10",
    )

    assert "REASONING:" in prompt
    assert "never mentioned" in prompt
