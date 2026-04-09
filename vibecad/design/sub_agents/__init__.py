"""
VibeCAD SubAgent Architecture.

Decomposes the monolithic DesignAgent into focused specialists:
  - ComponentCheckAgent — single v4 spec/BOM/component audit agent
  - NetAssignAgent      — infer pad-level nets and trigger autorouting
  - Orchestrator        — phase-aware delegation to the right subagent
"""

from .base import SubAgent, SubAgentResult
from .component_check import ComponentCheckAgent
from .net import NetAssignAgent
from .orchestrator import Orchestrator, DesignPhase

__all__ = [
    "SubAgent",
    "SubAgentResult",
    "ComponentCheckAgent",
    "NetAssignAgent",
    "Orchestrator",
    "DesignPhase",
]
