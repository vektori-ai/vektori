"""Unit tests for built-in agent-type extraction presets."""

from unittest.mock import MagicMock

from vektori.config import ExtractionConfig
from vektori.ingestion.extractor import (
    _AGENT_EPISODES_GUIDANCE,
    _AGENT_FACTS_GUIDANCE,
    FactExtractor,
)
from vektori.storage.memory import MemoryBackend


def _build_extractor(agent_type: str) -> FactExtractor:
    return FactExtractor(
        db=MemoryBackend(),
        embedder=MagicMock(),
        llm=MagicMock(),
        extraction_config=ExtractionConfig(agent_type=agent_type),
    )


def test_agent_type_guidance_catalog_stays_in_sync():
    assert _AGENT_FACTS_GUIDANCE
    assert _AGENT_EPISODES_GUIDANCE
    assert set(_AGENT_FACTS_GUIDANCE) == set(_AGENT_EPISODES_GUIDANCE)


def test_all_agent_type_presets_inject_fact_and_episode_guidance():
    for agent_type in _AGENT_FACTS_GUIDANCE:
        extractor = _build_extractor(agent_type)

        facts_prompt = extractor._facts_prompt("USER: hello", "")
        episodes_prompt = extractor._episodes_prompt("USER: hello", "- test fact", 3, "")

        assert _AGENT_FACTS_GUIDANCE[agent_type].strip() in facts_prompt
        assert _AGENT_EPISODES_GUIDANCE[agent_type].strip() in episodes_prompt


def test_general_agent_type_has_no_domain_guidance():
    extractor = _build_extractor("general")

    assert extractor._build_domain_guidance_facts() == ""
    assert extractor._build_domain_guidance_episodes() == ""
