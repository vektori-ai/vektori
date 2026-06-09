"""Filesystem memory — standalone file-based memory for AI agents."""

from vektori.fsmemory.memory import FilesystemMemory
from vektori.fsmemory.models import FileChunk, IngestResult

__all__ = ["FilesystemMemory", "FileChunk", "IngestResult"]
