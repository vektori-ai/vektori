"""Hermes Agent memory provider integration for Vektori."""

from typing import Any

# Assuming a mock BaseMemoryProvider that Hermes Agent uses
try:
    from hermes_agent.memory.base import BaseMemoryProvider
except ImportError:
    class BaseMemoryProvider:
        pass

from vektori import Vektori


class VektoriHermesMemory(BaseMemoryProvider):
    """
    Vektori's episodic graph memory as a Hermes Memory Provider.
    Provides semantic recall that feeds into Hermes' self-improvement loop.
    """

    def __init__(self, vektori_instance: Vektori, user_id: str):
        self.memory = vektori_instance
        self.user_id = user_id

    async def add_memory(self, text: str, metadata: dict[str, Any] = None) -> str:
        """Store a new fact/episode from the Hermes agent into Vektori."""
        result = await self.memory.add(
            messages=[{"role": "user", "content": text}],
            user_id=self.user_id,
            metadata=metadata
        )
        return result.get('status', 'success')

    async def search_memory(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Retrieve L1 (Facts + Episodes) depth context for Hermes prompt injection."""
        results = await self.memory.search(
            query=query,
            user_id=self.user_id,
            depth="l1",
            top_k=limit
        )

        # Hermes usually expects a flat list of memory dicts
        memories = []
        for f in results.get("facts", []):
            memories.append({"type": "fact", "content": f["text"], "score": f.get("score")})
        for ep in results.get("episodes", []):
            memories.append({"type": "episode", "content": ep["text"], "score": 1.0})

        return memories

    async def clear(self) -> None:
        """Clear memory for the current user."""
        # Vektori users would need an explicit delete endpoint,
        pass
