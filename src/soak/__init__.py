"""
Argus Soak-Test Hardening Module
=================================

Provides observability, fail-fast guards, tape capture, and resource
monitoring for 24/7 soak runs.

Components
----------
* :class:`SoakGuardian` — threshold-based guards with rate-limited alerts
* :class:`TapeRecorder` — optional rolling capture of quotes/ticks
* :class:`ResourceMonitor` — process/disk/log health tracking
* :func:`build_soak_summary` — single-payload soak status view
"""

from .guards import SoakGuardian
from .tape import TapeRecorder
from .resource_monitor import ResourceMonitor
from .summary import build_soak_summary

__all__ = [
    "SoakGuardian",
    "TapeRecorder",
    "ResourceMonitor",
    "build_soak_summary",
]
