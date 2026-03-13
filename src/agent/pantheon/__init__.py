"""Pantheon Intelligence Engine — structured research agents."""

from .factory import EvidenceGrader, FactoryPipe
from .hermes import HadesBacktestConfig, HermesRouter
from .roles import (
    PantheonRole,
    ContextInjector,
    PROMETHEUS,
    ARES,
    ATHENA,
    get_role_for_stage,
    build_stage_prompt,
)

__all__ = [
    "EvidenceGrader",
    "FactoryPipe",
    "HadesBacktestConfig",
    "HermesRouter",
    "PantheonRole",
    "ContextInjector",
    "PROMETHEUS",
    "ARES",
    "ATHENA",
    "get_role_for_stage",
    "build_stage_prompt",
]
