"""Unit tests for model and storage factories."""

import pytest

from vektori.models.anthropic import AnthropicLLM
from vektori.models.factory import LLM_REGISTRY, create_embedder, create_llm
from vektori.models.nvidia import DEFAULT_EMBEDDING_MODEL, NvidiaEmbedder, NvidiaLLM
from vektori.models.ollama import OllamaEmbedder, OllamaLLM
from vektori.models.openai import OpenAIEmbedder, OpenAILLM


def test_create_openai_embedder():
    embedder = create_embedder("openai:text-embedding-3-small")
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == "text-embedding-3-small"


def test_create_openai_embedder_default_model():
    embedder = create_embedder("openai")
    assert isinstance(embedder, OpenAIEmbedder)


def test_create_ollama_embedder():
    embedder = create_embedder("ollama:nomic-embed-text")
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "nomic-embed-text"


def test_create_sentence_transformer_embedder():
    from vektori.models.sentence_transformers import SentenceTransformerEmbedder

    embedder = create_embedder("sentence-transformers:all-MiniLM-L6-v2")
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_unknown_embedding_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedder("nonexistent:model")


def test_create_openai_llm():
    llm = create_llm("openai:gpt-4o-mini")
    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o-mini"


def test_create_ollama_llm():
    llm = create_llm("ollama:llama3")
    assert isinstance(llm, OllamaLLM)


def test_create_anthropic_llm():
    llm = create_llm("anthropic:claude-haiku-4-5-20251001")
    assert isinstance(llm, AnthropicLLM)


def test_unknown_llm_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm("nonexistent:model")


def test_create_nvidia_embedder_default_model():
    embedder = create_embedder("nvidia")
    assert isinstance(embedder, NvidiaEmbedder)
    assert embedder.model == DEFAULT_EMBEDDING_MODEL

def test_create_nvidia_embedder_custom_dimensions():
    embedder = create_embedder("nvidia:llama-nemotron-embed-1b-v2", dimensions=1024)
    assert isinstance(embedder, NvidiaEmbedder)
    assert embedder.dimension == 1024  # Matryoshka support

def test_create_nvidia_llm():
    llm = create_llm("nvidia:llama-3.3-nemotron-super-49b-v1")
    assert isinstance(llm, NvidiaLLM)
    assert llm.model == "nvidia/llama-3.3-nemotron-super-49b-v1"

def test_create_nvidia_llm_default_model():
    llm = create_llm("nvidia")
    assert isinstance(llm, NvidiaLLM)
    assert "nvidia/llama-3.3-nemotron-super-49b-v1" == llm.model

def test_nvidia_llm_registered():
    """Verify NVIDIA LLM is registered in factory."""
    assert "nvidia" in LLM_REGISTRY, "nvidia should be registered in LLM_REGISTRY"
    llm = create_llm("nvidia")
    assert llm is not None, "create_llm('nvidia') should return a valid instance"
    assert "Nvidia" in llm.__class__.__name__, (
        f"LLM class name should include 'Nvidia', got {llm.__class__.__name__}"
    )
