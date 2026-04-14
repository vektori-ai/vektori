from pathlib import Path


def test_harness_spec_checklist_exists():
    path = Path("docs/AGENT_HARNESS_CHECKLIST.md")
    assert path.exists()
    text = path.read_text()
    assert "Implemented" in text
    assert "Missing / Next" in text
