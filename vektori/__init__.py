"""Vektori — Open-source memory engine for AI agents."""

from vektori.client import Vektori
from vektori.config import ExtractionConfig, FSMemoryConfig, QualityConfig, VektoriConfig

try:
    from vektori._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

try:
    from vektori.fsmemory import FilesystemMemory
    __all__ = ["Vektori", "VektoriConfig", "QualityConfig", "ExtractionConfig", "FSMemoryConfig", "FilesystemMemory"]
except ImportError:
    __all__ = ["Vektori", "VektoriConfig", "QualityConfig", "ExtractionConfig", "FSMemoryConfig"]
