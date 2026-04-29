"""Unit tests for model and storage factories."""

import pytest

from vektori.models.factory import CHAT_REGISTRY, create_chat_model, create_embedder, create_llm


def test_create_openai_embedder():
    from vektori.models.openai import OpenAIEmbedder

    embedder = create_embedder("openai:text-embedding-3-small")
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == "text-embedding-3-small"


def test_create_openai_embedder_default_model():
    from vektori.models.openai import OpenAIEmbedder

    embedder = create_embedder("openai")
    assert isinstance(embedder, OpenAIEmbedder)


def test_create_ollama_embedder():
    from vektori.models.ollama import OllamaEmbedder

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
    from vektori.models.openai import OpenAILLM

    llm = create_llm("openai:gpt-4o-mini")
    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4o-mini"


def test_create_ollama_llm():
    from vektori.models.ollama import OllamaLLM

    llm = create_llm("ollama:llama3")
    assert isinstance(llm, OllamaLLM)


def test_create_anthropic_llm():
    from vektori.models.anthropic import AnthropicLLM

    llm = create_llm("anthropic:claude-haiku-4-5-20251001")
    assert isinstance(llm, AnthropicLLM)


def test_unknown_llm_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm("nonexistent:model")


def test_create_chat_model_litellm():
    from vektori.models.litellm_provider import LiteLLMChatModel

    model = create_chat_model("litellm:gpt-4o-mini")
    assert isinstance(model, LiteLLMChatModel)
    assert model.model == "gpt-4o-mini"


def test_create_chat_model_openai():
    from vektori.models.openai import OpenAIChatModel

    model = create_chat_model("openai:gpt-4o-mini")
    assert isinstance(model, OpenAIChatModel)
    assert model.model == "gpt-4o-mini"


def test_create_chat_model_litellm_is_in_registry():
    assert "litellm" in CHAT_REGISTRY


def test_create_chat_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown chat provider"):
        create_chat_model("nonexistent:model")
