"""
Vektori × Pipecat integration.

Install the optional dependency first:
    pip install "vektori[pipecat]"

Usage
-----
    from vektori.integrations.pipecat import VektoriMemoryProcessor, VektoriStorageProcessor
"""

try:
    from .processor import VektoriMemoryProcessor, VektoriStorageProcessor

    __all__ = ["VektoriMemoryProcessor", "VektoriStorageProcessor"]
except ImportError as exc:  # pipecat not installed
    raise ImportError("Pipecat is not installed. Run: pip install 'vektori[pipecat]'") from exc
