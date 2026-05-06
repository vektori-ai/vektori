from benchmarks.beam.beam_judge import ABILITY_TYPES
from benchmarks.beam.beam_runner import BeamBenchmark, BeamConfig


def test_beam_parser_flattens_nested_chat_batches():
    runner = BeamBenchmark(BeamConfig())
    raw_chat = [
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ],
        [
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ],
    ]

    assert runner._parse_chat_messages(raw_chat) == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    assert runner._parse_chat_batches(raw_chat) == [
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ],
        [
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ],
    ]


def test_beam_questions_parse_python_repr_and_ideal_response():
    runner = BeamBenchmark(BeamConfig())
    questions = runner._parse_probing_questions(
        "{'summarization': [{'question': 'What happened?', 'ideal_response': 'A useful summary.'}]}"
    )

    item = questions["summarization"][0]

    assert item["question"] == "What happened?"
    assert runner._expected_answer(item) == "A useful summary."


def test_beam_judge_tracks_dataset_ability_names():
    assert "summarization" in ABILITY_TYPES
    assert "temporal_reasoning" in ABILITY_TYPES
    assert "temporal_tracking" not in ABILITY_TYPES
