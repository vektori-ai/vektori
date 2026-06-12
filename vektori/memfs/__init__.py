"""MemFS — filesystem-native memory. Files are canonical; the index is a cache."""

from vektori.memfs.core import MemFS
from vektori.memfs.models import (
    CompactReport,
    IngestReport,
    Note,
    RecallItem,
    RecallResult,
    SyncReport,
    VerifyReport,
)
from vektori.memfs.secrets import SecretsFoundError

__all__ = [
    "MemFS", "Note", "RecallItem", "RecallResult", "SyncReport",
    "VerifyReport", "CompactReport", "IngestReport", "SecretsFoundError",
]
