"""
Argus - Crypto Market Monitor
=============================

24/7 market monitoring system for detecting trading opportunities.
"""

__version__ = "0.1.0"
__author__ = "Argus"

__all__ = ["ArgusOrchestrator", "main"]


def __getattr__(name: str):
    if name in __all__:
        from .orchestrator import ArgusOrchestrator, main
        return {"ArgusOrchestrator": ArgusOrchestrator, "main": main}[name]
    raise AttributeError(f"module 'src' has no attribute {name}")
