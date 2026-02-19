"""
VibeCAD Design Module - Phase 4

This module provides design assistance capabilities:
- Symbol/footprint library management (download from SnapEDA, etc.)
- Connection drawing (user-specified wiring)
- BOM generation and export
- Design action execution with preview
- Sub-agent architecture for reliable, phase-aware design automation
"""

from .library_manager import LibraryManager, LibrarySource, LibraryItem
from .connection_manager import ConnectionManager, ConnectionRequest
from .bom_exporter import BOMExporter, BOMFormat, BOMEntry
from .design_agent import DesignAgent, DesignAction, DesignActionType
from .agent_loop import AgentLoop, AgentState, AgentLoopConfig
from .circuit_context import CircuitContextBuilder, CircuitSnapshot
from .component_search import ComponentWebSearch, ComponentInfo

# Sub-agent architecture (optional — only imported if the sub_agents package exists).
try:
    from .sub_agents import (
        SubAgent,
        SubAgentResult,
        InfoGatheringAgent,
        PlacementAgent,
        RoutingAgent,
        VerificationAgent,
        Orchestrator,
        DesignPhase,
    )
    _SUBAGENTS = [
        'SubAgent', 'SubAgentResult',
        'InfoGatheringAgent', 'PlacementAgent',
        'RoutingAgent', 'VerificationAgent',
        'Orchestrator', 'DesignPhase',
    ]
except ImportError:
    _SUBAGENTS = []

__all__ = [
    'LibraryManager',
    'LibrarySource', 
    'LibraryItem',
    'ConnectionManager',
    'ConnectionRequest',
    'BOMExporter',
    'BOMFormat',
    'BOMEntry',
    'DesignAgent',
    'DesignAction',
    'DesignActionType',
    'AgentLoop',
    'AgentState',
    'AgentLoopConfig',
    'CircuitContextBuilder',
    'CircuitSnapshot',
    'ComponentWebSearch',
    'ComponentInfo',
] + _SUBAGENTS
