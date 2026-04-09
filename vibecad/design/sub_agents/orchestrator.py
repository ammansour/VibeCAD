"""Minimal v4 orchestrator with SPEC, PLACE, and NET subagents."""

from __future__ import annotations

import json
import logging
from enum import Enum, auto
from typing import Any, Dict, Optional

from .base import SubAgent, SubAgentResult
from .component_check import ComponentCheckAgent

logger = logging.getLogger(__name__)


class DesignPhase(Enum):
    SPEC = auto()
    PLACE = auto()
    NET = auto()
    DONE = auto()


_PHASE_AGENT_MAP = {
    DesignPhase.SPEC: "component_check",
    DesignPhase.PLACE: "place",
    DesignPhase.NET: "net",
}

_PHASE_ORDER = [
    DesignPhase.SPEC,
    DesignPhase.PLACE,
    DesignPhase.NET,
    DesignPhase.DONE,
]


class Orchestrator:
    def __init__(self, llm_client=None):
        from .place import ComponentPlaceAgent
        from .net import NetAssignAgent
        self._agents: Dict[str, SubAgent] = {
            "component_check": ComponentCheckAgent(llm_client),
            "place": ComponentPlaceAgent(llm_client),
            "net": NetAssignAgent(llm_client),
        }
        self._phase = DesignPhase.SPEC

    @property
    def phase(self) -> DesignPhase:
        return self._phase

    @phase.setter
    def phase(self, value: DesignPhase) -> None:
        self._phase = value

    def get_agent(self, name: str) -> Optional[SubAgent]:
        return self._agents.get(name)

    def reset(self) -> None:
        self._phase = DesignPhase.SPEC

    def advance_to(self, phase: DesignPhase) -> None:
        self._phase = phase

    def _advance(self) -> None:
        try:
            idx = _PHASE_ORDER.index(self._phase)
        except ValueError:
            self._phase = DesignPhase.DONE
            return
        if idx + 1 < len(_PHASE_ORDER):
            self._phase = _PHASE_ORDER[idx + 1]
        else:
            self._phase = DesignPhase.DONE

    def _build_phase_goal(
        self,
        goal: str,
        feedback: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        phase_prompts = {
            DesignPhase.SPEC: "PHASE: SPEC - Build and verify complete BOM/component set.",
            DesignPhase.PLACE: "PHASE: PLACE - Place all grouped components onto valid board positions based on the manifest.",
            DesignPhase.NET: "PHASE: NET - Assign all required PCB nets based on manifest/spec intent, then autoroute with Freerouting.",
            DesignPhase.DONE: "PHASE: DONE - Design complete.",
        }
        pfx = phase_prompts.get(self._phase, "")
        text = f"{pfx}\n\nUSER GOAL:\n{goal or ''}".strip()
        quality_constraints = (
            context.get("quality_constraints")
            if isinstance(context, dict)
            else None
        )
        if isinstance(quality_constraints, dict) and quality_constraints:
            try:
                text += "\n\nQUALITY_CONSTRAINTS_JSON:\n" + json.dumps(
                    quality_constraints,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except Exception:
                pass
        if feedback:
            text += f"\n\nLAST ACTION FEEDBACK:\n{feedback}"
        return text

    def step(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
    ) -> SubAgentResult:
        if self._phase == DesignPhase.DONE:
            return SubAgentResult(
                message="DESIGN_COMPLETE",
                actions=[],
                confidence=0.95,
                phase_complete=True,
                thinking="orchestrator done",
            )

        agent_name = _PHASE_AGENT_MAP.get(self._phase, "component_check")
        agent = self._agents.get(agent_name)
        if agent is None:
            return SubAgentResult(
                message=f"No subagent registered for phase {self._phase.name}.",
                actions=[],
                confidence=0.0,
            )

        phase_goal = self._build_phase_goal(goal, feedback, context=context)
        logger.info("Orchestrator: phase=%s agent=%s", self._phase.name, agent_name)
        result = agent.plan(phase_goal, context, board_snapshot)
        if bool(getattr(result, "phase_complete", False)):
            self._advance()
        return result
