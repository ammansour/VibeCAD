"""
VibeCAD SubAgent Architecture.

Decomposes the monolithic DesignAgent into focused specialists:
  - InfoGatheringAgent  — component search, datasheets, web lookup
  - PlacementAgent      — smart component placement with spatial optimization
  - RoutingAgent        — net assignment, track drawing, autorouting
  - VerificationAgent   — DRC/ERC and iterative fix proposals
  - Orchestrator        — phase-aware delegation to the right subagent
"""

from .base import SubAgent, SubAgentResult
from .info_gathering import InfoGatheringAgent
from .placement import PlacementAgent
from .routing import RoutingAgent
from .verification import VerificationAgent
from .orchestrator import Orchestrator, DesignPhase

__all__ = [
    "SubAgent",
    "SubAgentResult",
    "InfoGatheringAgent",
    "PlacementAgent",
    "RoutingAgent",
    "VerificationAgent",
    "Orchestrator",
    "DesignPhase",
]
