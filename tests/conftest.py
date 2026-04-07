"""Shared test fixtures."""

import pytest

from vektori.storage.memory import MemoryBackend


@pytest.fixture
def memory_backend():
    return MemoryBackend()


@pytest.fixture
async def initialized_backend():
    backend = MemoryBackend()
    await backend.initialize()
    return backend
