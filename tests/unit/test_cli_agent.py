from typer.testing import CliRunner

from vektori.cli import app


def test_agent_chat_help_includes_native_harness_options():
    runner = CliRunner()

    result = runner.invoke(app, ["agent", "chat", "--help"])

    assert result.exit_code == 0
    assert "--chat-model" in result.stdout
    assert "--context-path" in result.stdout
    assert "Start an interactive chat loop using `VektoriAgent`." in result.stdout
