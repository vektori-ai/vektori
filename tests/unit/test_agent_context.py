from pathlib import Path

from vektori.context import AgentContextLoader


def test_markdown_context_loader(tmp_path: Path):
    path = tmp_path / "agents.md"
    path.write_text(
        "# Agent\n\n"
        "## Persona\n"
        "You are concise.\n\n"
        "## Instructions\n"
        "- Prefer direct answers.\n"
        "- Ask only when needed.\n\n"
        "## Memory Policy\n"
        "- Persist explicit preferences.\n"
    )

    loaded = AgentContextLoader(str(path)).load()

    assert loaded.persona == "You are concise."
    assert loaded.instructions == ["Prefer direct answers.", "Ask only when needed."]
    assert "Persist explicit preferences." in loaded.memory_policy["notes"]


def test_markdown_context_loader_parses_key_value_sections(tmp_path: Path):
    path = tmp_path / "agents.md"
    path.write_text(
        "# Agent\n\n"
        "## Response Style\n"
        "- Verbosity: short\n"
        "- Tone: direct\n\n"
        "## Memory Policy\n"
        "- Persist Preferences: true\n"
        "- Do not store secrets.\n"
    )

    loaded = AgentContextLoader(str(path)).load()

    assert loaded.response_style["verbosity"] == "short"
    assert loaded.response_style["tone"] == "direct"
    assert loaded.memory_policy["persist_preferences"] == "true"
    assert "Do not store secrets." in loaded.memory_policy["notes"]
