"""
Orchestrator — phase-aware delegation to the right subagent.

The Orchestrator owns the high-level design pipeline:

  GATHER → PLACE → OUTLINE → ARRANGE → NET_ASSIGN → ROUTE → VERIFY → DONE

It examines the current board state and feedback to determine which phase
is active, delegates goal+context to the appropriate subagent, and
collects their proposed actions.

The existing AgentLoop keeps its approve/execute machinery; the
Orchestrator just replaces the single monolithic LLM call with
targeted subagent calls.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from .base import SubAgent, SubAgentResult
from .info_gathering import InfoGatheringAgent
from .placement import PlacementAgent
from .routing import RoutingAgent
from .verification import VerificationAgent

logger = logging.getLogger(__name__)

_SUBAGENT_LLM_RETRIES = 2  # initial try + 1 repair attempt


class DesignPhase(Enum):
    """High-level phases of the design pipeline."""
    GATHER = auto()        # Search for parts, datasheets
    PLACE = auto()         # Add components to the board
    OUTLINE = auto()       # Define board outline & mounting holes
    ARRANGE = auto()       # Intelligent layout arrangement (resolve overlaps)
    NET_ASSIGN = auto()    # Wire logical nets to physical pads
    ROUTE = auto()         # Draw copper traces
    VERIFY = auto()        # Run DRC / ERC
    FIX = auto()           # Fix verification errors (may loop back)
    DONE = auto()


# Map each phase to the subagent that handles it.
_PHASE_AGENT_MAP = {
    DesignPhase.GATHER: "info_gathering",
    DesignPhase.PLACE: "placement",
    DesignPhase.OUTLINE: "placement",
    DesignPhase.ARRANGE: "placement",
    DesignPhase.NET_ASSIGN: "routing",
    DesignPhase.ROUTE: "routing",
    DesignPhase.VERIFY: "verification",
    DesignPhase.FIX: "verification",
}


class Orchestrator:
    """Phase-aware delegation to specialised subagents.

    Usage (inside AgentLoop._run_loop)::

        orch = Orchestrator(llm_client)
        result = orch.step(goal, context, board_snapshot)
        # result.actions → preview/approve as usual
        # result.phase_complete → advance to next phase
    """

    def __init__(self, llm_client=None):
        self._agents: Dict[str, SubAgent] = {
            "info_gathering": InfoGatheringAgent(llm_client),
            "placement": PlacementAgent(llm_client),
            "routing": RoutingAgent(llm_client),
            "verification": VerificationAgent(llm_client),
        }
        self._phase = DesignPhase.GATHER
        self._phase_attempts: Dict[DesignPhase, int] = {}

    # ── Public API ──────────────────────────────────────────────

    @property
    def phase(self) -> DesignPhase:
        return self._phase

    @phase.setter
    def phase(self, value: DesignPhase) -> None:
        self._phase = value

    def get_agent(self, name: str) -> Optional[SubAgent]:
        """Retrieve a subagent by name (for direct access)."""
        return self._agents.get(name)

    def step(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
        feedback: Optional[str] = None,
    ) -> SubAgentResult:
        """Execute one orchestration step.

        1. Determine the current phase from board state (or advance if previous
           phase is complete).
        2. Delegate to the right subagent.
        3. Return the subagent's result (actions + message).
        """
        # Auto-detect phase from board state if possible.
        self._maybe_advance_phase(context, board_snapshot, feedback)

        agent_name = _PHASE_AGENT_MAP.get(self._phase, "placement")
        agent = self._agents.get(agent_name)
        if agent is None:
            return SubAgentResult(
                message=f"No subagent registered for phase {self._phase.name}.",
                confidence=0.0,
            )

        # Inject phase context into the goal so the subagent knows what to do.
        phase_goal = self._build_phase_goal(goal, feedback)

        # Track attempts to prevent infinite loops in one phase.
        attempts = self._phase_attempts.get(self._phase, 0) + 1
        self._phase_attempts[self._phase] = attempts

        logger.info(
            "Orchestrator: phase=%s  agent=%s  attempt=%d",
            self._phase.name, agent_name, attempts,
        )

        # Subagents are strict about JSON schema. If the LLM emits malformed JSON
        # or unknown action types, retry once with explicit feedback rather than
        # aborting the whole AgentLoop. If it still fails, return no actions so
        # AgentLoop can fall through to the monolithic LLM (still LLM-driven).
        try:
            from ...llm.client import LLMError
        except Exception:  # pragma: no cover
            LLMError = Exception  # type: ignore

        last_err: Optional[Exception] = None
        result: Optional[SubAgentResult] = None
        for retry_idx in range(_SUBAGENT_LLM_RETRIES):
            try:
                result = agent.plan(phase_goal, context, board_snapshot)
                last_err = None
                break
            except LLMError as e:  # type: ignore[misc]
                last_err = e
                logger.warning(
                    "Orchestrator: subagent=%s phase=%s failed (LLM/format) retry=%d/%d: %s",
                    agent_name, self._phase.name, retry_idx + 1, _SUBAGENT_LLM_RETRIES, e,
                )
                if retry_idx + 1 >= _SUBAGENT_LLM_RETRIES:
                    break

                # Add explicit validation feedback to steer the next response.
                allowed = ""
                try:
                    if getattr(agent, "SYSTEM_PROMPT", ""):
                        allowed = "Follow your system prompt exactly."
                except Exception:
                    allowed = ""
                repair_feedback = "\n".join(
                    p for p in [
                        (feedback or "").strip(),
                        f"SUBAGENT_OUTPUT_INVALID: {e}",
                        "Return ONLY a JSON array. Do not wrap it in an object or markdown.",
                        "Every element must be an object with non-empty 'action_type', 'description', and 'parameters'.",
                        allowed or "",
                    ] if p
                ).strip()
                phase_goal = self._build_phase_goal(goal, repair_feedback)

        if result is None:
            msg = f"Subagent '{agent_name}' failed to produce a valid plan: {last_err}"
            return SubAgentResult(
                message=msg,
                actions=[],
                confidence=0.0,
                phase_complete=False,
                thinking=str(last_err or msg),
            )

        # If the subagent says its phase is done, advance.
        if result and result.phase_complete:
            self._advance()

        return result

    def advance_to(self, phase: DesignPhase) -> None:
        """Manually advance to a specific phase (e.g. when the user requests it)."""
        self._phase = phase
        self._phase_attempts[phase] = 0

    def reset(self) -> None:
        """Reset to the beginning of the pipeline."""
        self._phase = DesignPhase.GATHER
        self._phase_attempts.clear()

    # ── Post-placement optimization entry point ────────────────

    def run_placement_optimization(
        self,
        components: List[Dict[str, Any]],
        board_width_mm: float,
        board_height_mm: float,
        board_origin_x_mm: float = 0.0,
        board_origin_y_mm: float = 0.0,
        net_connections: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper around PlacementAgent.optimize_layout()."""
        return PlacementAgent.optimize_layout(
            components,
            board_width_mm,
            board_height_mm,
            board_origin_x_mm,
            board_origin_y_mm,
            net_connections,
        )

    def run_overlap_resolution(
        self,
        components: List[Dict[str, Any]],
        clearance_mm: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper around PlacementAgent.resolve_overlaps()."""
        return PlacementAgent.resolve_overlaps(components, clearance_mm)

    # ── Phase-advancement logic ─────────────────────────────────

    def _advance(self) -> None:
        """Move to the next phase in the pipeline."""
        order = list(DesignPhase)
        idx = order.index(self._phase)
        if idx + 1 < len(order):
            self._phase = order[idx + 1]
            self._phase_attempts[self._phase] = 0
            logger.info("Orchestrator advanced to phase: %s", self._phase.name)

    def _maybe_advance_phase(
        self,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]],
        feedback: Optional[str],
    ) -> None:
        """Heuristic auto-advance based on board state.

        This prevents the orchestrator from getting stuck in a completed phase.
        """
        snap = board_snapshot or {}
        attempts = self._phase_attempts.get(self._phase, 0)

        if self._phase == DesignPhase.GATHER:
            # After 3 iterations in GATHER, move on.
            if attempts >= 3:
                self._phase = DesignPhase.PLACE
                self._phase_attempts[DesignPhase.PLACE] = 0

        elif self._phase == DesignPhase.PLACE:
            # If components already placed, advance after 2 attempts.
            # Even without components, don't stay stuck — after 5 attempts
            # the monolithic agent may have placed them without updating
            # the snapshot, or nothing useful will happen.
            comp_count = len(snap.get("components", []))
            if comp_count > 0 and attempts >= 2:
                self._phase = DesignPhase.OUTLINE
                self._phase_attempts[DesignPhase.OUTLINE] = 0
            elif attempts >= 5:
                # Unconditional escape — don't loop forever.
                self._phase = DesignPhase.OUTLINE
                self._phase_attempts[DesignPhase.OUTLINE] = 0

        elif self._phase == DesignPhase.OUTLINE:
            # Prefer the loop's explicit phase tracking when available.
            # AgentLoop reliably sets outline_defined=True after a successful
            # DEFINE_BOARD_OUTLINE action, even when board dimension keys are
            # absent from the context snapshot.
            if snap.get("outline_defined") or (snap.get("board_width") and snap.get("board_height")):
                self._phase = DesignPhase.ARRANGE
                self._phase_attempts[DesignPhase.ARRANGE] = 0
            elif attempts >= 3:
                self._phase = DesignPhase.ARRANGE
                self._phase_attempts[DesignPhase.ARRANGE] = 0

        elif self._phase == DesignPhase.ARRANGE:
            # One pass of arrangement is usually enough.
            if attempts >= 2:
                self._phase = DesignPhase.NET_ASSIGN
                self._phase_attempts[DesignPhase.NET_ASSIGN] = 0

        elif self._phase == DesignPhase.NET_ASSIGN:
            nets_assigned = snap.get("nets_assigned", 0)
            if nets_assigned > 0 and attempts >= 2:
                self._phase = DesignPhase.ROUTE
                self._phase_attempts[DesignPhase.ROUTE] = 0
            elif attempts >= 5:
                self._phase = DesignPhase.ROUTE
                self._phase_attempts[DesignPhase.ROUTE] = 0

        elif self._phase == DesignPhase.ROUTE:
            if attempts >= 3:
                self._phase = DesignPhase.VERIFY
                self._phase_attempts[DesignPhase.VERIFY] = 0

        elif self._phase == DesignPhase.VERIFY:
            # If DRC passed, done.
            if feedback and "drc_status: pass" in feedback.lower():
                self._phase = DesignPhase.DONE
            elif attempts >= 3:
                self._phase = DesignPhase.FIX
                self._phase_attempts[DesignPhase.FIX] = 0

        elif self._phase == DesignPhase.FIX:
            # After fix attempts, re-verify.
            if attempts >= 3:
                self._phase = DesignPhase.VERIFY
                self._phase_attempts[DesignPhase.VERIFY] = 0

    def _build_phase_goal(self, user_goal: str, feedback: Optional[str]) -> str:
        """Prepend phase-specific instructions to the user's goal."""
        phase_hints = {
            DesignPhase.GATHER: (
                "PHASE: GATHER — Find missing parts and required datasheets.\n"
            ),
            DesignPhase.PLACE: (
                "PHASE: PLACE — Add components with explicit package names.\n"
            ),
            DesignPhase.OUTLINE: (
                "PHASE: OUTLINE — Define board outline (and mounting holes if needed).\n"
            ),
            DesignPhase.ARRANGE: (
                "PHASE: ARRANGE — Move parts inside outline and clear overlaps.\n"
            ),
            DesignPhase.NET_ASSIGN: (
                "PHASE: NET_ASSIGN — Assign power nets first, then signal nets.\n"
            ),
            DesignPhase.ROUTE: (
                "PHASE: ROUTE — Route assigned nets.\n"
            ),
            DesignPhase.VERIFY: (
                "PHASE: VERIFY — Run ERC/DRC and review errors.\n"
            ),
            DesignPhase.FIX: (
                "PHASE: FIX — Apply targeted fixes, then re-run checks.\n"
            ),
            DesignPhase.DONE: "The design is complete.",
        }
        hint = phase_hints.get(self._phase, "")
        parts = [hint, f"USER GOAL: {user_goal}"]
        if feedback:
            parts.append(f"\nFEEDBACK FROM PREVIOUS STEP:\n{feedback}")
        return "\n".join(parts)
