"""Config-driven model provider resolution."""

from __future__ import annotations

import importlib

from vektori.models.base import EmbeddingProvider, LLMProvider

# Format: "provider_key": "module.path.ClassName"
EMBEDDING_REGISTRY: dict[str, str] = {
    "openai": "vektori.models.openai.OpenAIEmbedder",
    "anthropic": "vektori.models.anthropic.AnthropicEmbedder",
    "ollama": "vektori.models.ollama.OllamaEmbedder",
    "sentence-transformers": "vektori.models.sentence_transformers.SentenceTransformerEmbedder",
    # BGE-M3: fully local, multilingual, 1024-dim — recommended default
    "bge": "vektori.models.bge.BGEEmbedder",
}

LLM_REGISTRY: dict[str, str] = {
    "openai": "vektori.models.openai.OpenAILLM",
    "anthropic": "vektori.models.anthropic.AnthropicLLM",
    "ollama": "vektori.models.ollama.OllamaLLM",
    "gemini": "vektori.models.gemini.GeminiLLM",  # Direct Gemini API
    # LiteLLM: single interface for 100+ providers — recommended for extraction
    "litellm": "vektori.models.litellm_provider.LiteLLMProvider",
}


def create_embedder(model_string: str, **kwargs) -> EmbeddingProvider:
    """
    Resolve 'provider:model_name' string into an EmbeddingProvider instance.

    Examples:
        "openai:text-embedding-3-small"
        "ollama:nomic-embed-text"
        "sentence-transformers:all-MiniLM-L6-v2"
        "anthropic:voyage-3"
    """
    provider, _, model_name = model_string.partition(":")
    if provider not in EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            f"Available: {list(EMBEDDING_REGISTRY.keys())}"
        )
    cls = _import_class(EMBEDDING_REGISTRY[provider])
    return cls(model=model_name or None, **kwargs)


def create_llm(model_string: str, **kwargs) -> LLMProvider:
    """
    Resolve 'provider:model_name' string into an LLMProvider instance.

    Examples:
        "openai:gpt-4o-mini"
        "anthropic:claude-haiku-4-5-20251001"
        "ollama:llama3"
    """
    provider, _, model_name = model_string.partition(":")
    if provider not in LLM_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            f"Available: {list(LLM_REGISTRY.keys())}"
        )
    cls = _import_class(LLM_REGISTRY[provider])
    return cls(model=model_name or None, **kwargs)


def _import_class(dotted_path: str):
    """Dynamically import a class from a dotted path string."""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
