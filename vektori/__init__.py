"""Vektori — Open-source memory engine for AI agents."""

from vektori.agent import AgentConfig, AgentTurnResult, VektoriAgent
from vektori.client import Vektori
from vektori.config import ExtractionConfig, QualityConfig, VektoriConfig

try:
    from vektori._version import version as __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "AgentConfig",
    "AgentTurnResult",
    "ExtractionConfig",
    "QualityConfig",
    "Vektori",
    "VektoriAgent",
    "VektoriConfig",
]
