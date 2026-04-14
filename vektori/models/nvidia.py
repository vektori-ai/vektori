"""NVIDIA NIM embedding and LLM providers.

NVIDIA NIM (NVIDIA Inference Microservices) provides GPU-optimized models
for embedding and text generation via an OpenAI-compatible API.

Get API key: https://build.nvidia.com

Usage:
v = Vektori(
    embedding_model="nvidia:llama-nemotron-embed-1b-v2",
    extraction_model="nvidia:llama-3.3-nemotron-super-49b-v1",
)

Environment:
NVIDIA_API_KEY - Your NVIDIA API key (required)

Models:
Embeddings (llama-nemotron-embed-1b-v2 is recommended default):
- llama-nemotron-embed-1b-v2: 2048 dim, 8192 tokens, multilingual, Matryoshka
- llama-3.2-nv-embedqa-1b-v2: 2048 dim, 8192 tokens, QA-optimized
- baai/bge-m3: 1024 dim, 8192 tokens, multilingual, dense+sparse+multi-vector
- nv-embed-v1: 4096 dim, 32k tokens, generalist
- nv-embedqa-e5-v5: 1024 dim, 512 tokens, fast

LLMs:
- llama-3.3-nemotron-super-49b-v1: High quality extraction (default)
- llama-3.1-nemotron-ultra-253b-v1: Largest NVIDIA model
- nemotron-4-mini-hindi-4b-instruct: Fast, efficient
- z-ai/glm5: GLM-5, multilingual LLM
- z-ai/glm4.7: GLM-4.7, multilingual LLM
- google/gemma-4-31b-it: Google's Gemma 4 31B instruction tuned
- minimaxai/minimax-m2.7: MiniMax M2.7 model
- moonshotai/kimi-k2.5: Moonshot Kimi K2.5
- moonshotai/kimi-k2-instruct: Moonshot Kimi K2 instruction
- deepseek-ai/deepseek-v3.1: DeepSeek V3.1
- qwen/qwen3-coder-480b-a35b-instruct: Qwen3 Coder 480B
"""

from __future__ import annotations

import logging
import os
from typing import Any

from vektori.models.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Embedding dimensions for known NVIDIA models
# Matryoshka models support multiple dimensions; max is shown
EMBEDDING_DIMENSIONS = {
    # Nemotron family (Matryoshka: 384, 512, 768, 1024, 2048)
    "nvidia/llama-nemotron-embed-1b-v2": 2048,
    "nvidia/llama-nemotron-embed-vl-1b-v2": 2048,
    # NeMo Retriever family (Matryoshka: 384, 512, 768, 1024, 2048)
    "nvidia/llama-3.2-nv-embedqa-1b-v2": 2048,
    "nvidia/llama-3.2-nv-embedqa-1b-v1": 2048,
    "nvidia/llama-3.2-nemoretriever-300m-embed-v2": 1024,
    "nvidia/llama-3.2-nemoretriever-300m-embed-v1": 1024,
    "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1": 2048,
    # BAAI BGE-M3 (hosted on NVIDIA NIM)
    "baai/bge-m3": 1024,
    # Other models
    "nvidia/nv-embed-v1": 4096,
    "nvidia/nv-embedqa-e5-v5": 1024,
    "nvidia/nv-embedcode-7b-v1": 4096,
    "nvidia/nv-embedqa-e5-v4": 1024,
    "nvidia/nvclip": 512,
}

# Models that support Matryoshka embeddings (configurable dimensions)
MATRYOSHKA_MODELS = {
    "nvidia/llama-nemotron-embed-1b-v2",
    "nvidia/llama-nemotron-embed-vl-1b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "nvidia/llama-3.2-nv-embedqa-1b-v1",
}

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"
DEFAULT_LLM_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"


class NvidiaEmbedder(EmbeddingProvider):
    """NVIDIA NIM embedding models via OpenAI-compatible API.

    Supports all NVIDIA embedding models including:
    - llama-nemotron-embed-1b-v2 (recommended): multilingual, 2048-dim, Matryoshka
    - llama-3.2-nv-embedqa-1b-v2: QA-optimized, 2048-dim, Matryoshka
    - nv-embed-v1: generalist, 4096-dim, 32k context
    - nv-embedqa-e5-v5: fast, 1024-dim, 512 context

    Matryoshka models support configurable output dimensions:
    384, 512, 768, 1024, or 2048. Use the dimensions parameter or
    embedding_dimension config to set custom dimensions.

    Args:
        model: Model name from NVIDIA catalog. Defaults to llama-nemotron-embed-1b-v2.
        api_key: NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
        dimensions: Optional custom embedding dimensions for Matryoshka models.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        raw_model = model or DEFAULT_EMBEDDING_MODEL
        # NVIDIA NIM requires full path with namespace prefix (e.g., "nvidia/" or "baai/")
        # Add "nvidia/" prefix if model doesn't already have a namespace
        if "/" not in raw_model:
            raw_model = f"nvidia/{raw_model}"
        self.model = raw_model
        self._api_key = api_key
        self._client = None

        # Validate dimensions parameter
        if dimensions is not None:
            if not isinstance(dimensions, int):
                raise ValueError(f"dimensions must be an integer, got {type(dimensions).__name__}")
            if dimensions <= 0:
                raise ValueError(f"dimensions must be positive, got {dimensions}")
            # Validate that dimensions is a supported size for Matryoshka models
            supported_dims = {384, 512, 768, 1024, 2048}
            if dimensions not in supported_dims:
                raise ValueError(
                    f"dimensions must be one of {sorted(supported_dims)}, got {dimensions}"
                )
        self._custom_dims = dimensions

    def _get_client(self):
        """Lazy initialization of OpenAI client configured for NVIDIA API."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package required for NVIDIA NIM: pip install openai>=1.12"
                ) from e

            api_key = self._api_key or os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError(
                    "NVIDIA API key required. Set NVIDIA_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=NVIDIA_BASE_URL,
            )
        return self._client

    @property
    def dimension(self) -> int:
        """Return embedding dimension.

        If custom dimensions were specified and the model supports Matryoshka
        embeddings, returns the custom dimension. Otherwise returns the
        model's default dimension.
        """
        # Validate _custom_dims in case __init__ was bypassed
        if self._custom_dims is not None:
            if not isinstance(self._custom_dims, int):
                raise ValueError(
                    f"_custom_dims must be an integer, got {type(self._custom_dims).__name__}"
                )
            if self._custom_dims <= 0:
                raise ValueError(f"_custom_dims must be positive, got {self._custom_dims}")
            supported_dims = {384, 512, 768, 1024, 2048}
            if self._custom_dims not in supported_dims:
                raise ValueError(
                    f"_custom_dims must be one of {sorted(supported_dims)}, got {self._custom_dims}"
                )
            if self.model in MATRYOSHKA_MODELS:
                return self._custom_dims
        return EMBEDDING_DIMENSIONS.get(self.model, 2048)

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, preserving input order.
        """
        if not texts:
            return []

        client = self._get_client()

        # Build extra_body for NVIDIA-specific parameters
        extra_body: dict[str, Any] = {}

        # Matryoshka models support custom dimensions
        # Validate dimensions before consuming (fail-fast if __init__ was bypassed)
        if self._custom_dims is not None:
            if not isinstance(self._custom_dims, int):
                raise ValueError(
                    f"_custom_dims must be an integer, got {type(self._custom_dims).__name__}"
                )
            if self._custom_dims <= 0:
                raise ValueError(f"_custom_dims must be positive, got {self._custom_dims}")
            supported_dims = {384, 512, 768, 1024, 2048}
            if self._custom_dims not in supported_dims:
                raise ValueError(
                    f"_custom_dims must be one of {sorted(supported_dims)}, got {self._custom_dims}"
                )
            if self.model in MATRYOSHKA_MODELS:
                extra_body["dimensions"] = self._custom_dims

        # Some NVIDIA embedding models require input_type parameter
        # "passage" is used for indexing, "query" for querying
        # We default to "passage" for general embedding use
        if self.model in {
            "nvidia/llama-nemotron-embed-1b-v2",
            "nvidia/llama-nemotron-embed-vl-1b-v2",
        }:
            extra_body["input_type"] = "passage"

        response = await client.embeddings.create(
            input=texts,
            model=self.model,
            extra_body=extra_body if extra_body else None,
        )

        # Sort by index to preserve order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class NvidiaLLM(LLMProvider):
    """NVIDIA NIM LLM for fact and episode extraction.

    Supports all NVIDIA chat completion models and third-party models hosted
    on NVIDIA NIM, including:
    - nvidia/llama-3.3-nemotron-super-49b-v1 (default): High quality extraction
    - nvidia/llama-3.1-nemotron-ultra-253b-v1: Largest NVIDIA model
    - nvidia/nemotron-4-mini-hindi-4b-instruct: Fast, efficient
    - z-ai/glm5: GLM-5 multilingual model
    - z-ai/glm4.7: GLM-4.7 multilingual model
    - google/gemma-4-31b-it: Google's Gemma 4 31B instruction tuned
    - minimaxai/minimax-m2.7: MiniMax M2.7 model
    - moonshotai/kimi-k2.5: Moonshot Kimi K2.5
    - moonshotai/kimi-k2-instruct: Moonshot Kimi K2 instruction
    - deepseek-ai/deepseek-v3.1: DeepSeek V3.1
    - qwen/qwen3-coder-480b-a35b-instruct: Qwen3 Coder 480B

    Uses OpenAI-compatible API with NVIDIA-specific base URL.

    Args:
        model: Model name from NVIDIA catalog (e.g., "llama-3.3-nemotron-super-49b-v1"
        for NVIDIA models, or full path like "z-ai/glm5" for third-party models).
        api_key: NVIDIA API key. Falls back to NVIDIA_API_KEY env var.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        raw_model = model or DEFAULT_LLM_MODEL
        # NVIDIA NIM requires full path with namespace prefix (e.g., "nvidia/")
        # Add "nvidia/" prefix if model doesn't already have a namespace
        if "/" not in raw_model:
            raw_model = f"nvidia/{raw_model}"
        self.model = raw_model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client configured for NVIDIA API."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package required for NVIDIA NIM: pip install openai>=1.12"
                ) from e

            api_key = self._api_key or os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise ValueError(
                    "NVIDIA API key required. Set NVIDIA_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=NVIDIA_BASE_URL,
            )
        return self._client

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """Generate text completion for the given prompt.

        Attempts to use JSON mode for structured outputs. Falls back to
        regular chat completion if the model doesn't support JSON mode.

        Args:
            prompt: The prompt text to send to the model.
            max_tokens: Optional maximum tokens in the response.

        Returns:
            Generated text response.
        """
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Try JSON mode first (for structured extraction)
        kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            # Only fallback if the error indicates JSON mode is unsupported
            error_str = str(e).lower()
            is_json_mode_error = (
                "response_format" in error_str
                or "json" in error_str
                or "json_object" in error_str
                or hasattr(e, "code")
                and (
                    "unsupported" in str(getattr(e, "code", "")).lower()
                    or "invalid" in str(getattr(e, "code", "")).lower()
                )
            )
            if is_json_mode_error:
                # Fallback if model doesn't support JSON mode
                del kwargs["response_format"]
                response = await client.chat.completions.create(**kwargs)
            else:
                # Re-raise for auth, network, rate-limit, etc.
                raise

        return response.choices[0].message.content or ""
