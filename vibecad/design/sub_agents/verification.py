"""
VerificationAgent — DRC / ERC verification and iterative fix proposals.

Handles RUN_DRC, RUN_ERC.  When errors are found it proposes targeted fixes
such as DELETE_TRACKS + re-route, or MOVE_COMPONENT for clearance issues.

This agent also acts as a feedback loop: it analyses DRC/ERC error messages
and produces fix actions that get routed back to the appropriate subagent
(Placement for overlap fixes, Routing for short/open fixes).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class VerificationAgent(SubAgent):
    NAME = "verification"

    SYSTEM_PROMPT = (
        "You are the Verification sub-agent.\n"
        "Run ERC/DRC, analyze errors, and propose targeted fixes.\n"
        "Use MOVE_COMPONENT for overlap/edge/clearance spacing issues.\n"
        "Use DELETE_TRACKS + AUTOROUTE_BOARD for shorts/crossings when routing exists.\n"
        "Use DEFINE_NET/ASSIGN_NETS for connectivity issues.\n"
        "End fix sets with RUN_DRC or RUN_ERC to verify.\n"
        "Parameter schema (use exactly these keys; do not use aliases like 'id'):\n"
        "- MOVE_COMPONENT.parameters = {\"ref\":\"U1\",\"strategy\":\"resolve_overlap\"} OR {\"ref\":\"U1\",\"location\":{\"x\":10.0,\"y\":20.0}}\n"
        "- ROTATE_COMPONENT.parameters = {\"ref\":\"U1\",\"angle\":90}\n"
        "Return JSON array only.\n"
    )

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        from ..design_agent import DesignActionType
        self.HANDLED_ACTION_TYPES = frozenset({
            DesignActionType.RUN_DRC,
            DesignActionType.RUN_ERC,
        })

    # ── plan ────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        from ..design_agent import DesignAction, DesignActionType
        from ...llm.client import LLMError

        if not self._llm_available():
            raise LLMError("verification: LLM is required but is not available/configured.")

        raw = self._llm_chat(self._build_prompt(goal, context, board_snapshot))
        actions = self._parse_actions(raw)
        if actions:
            return SubAgentResult(
                message=f"Proposing {len(actions)} verification action(s).",
                actions=actions,
                confidence=0.85,
            )

        return SubAgentResult(
            message="No verification actions proposed.",
            actions=[],
            confidence=0.3,
            phase_complete=False,
            thinking="LLM proposed no verification actions",
        )

    # ── Error analysis ──────────────────────────────────────────

    def _analyse_errors(self, error_text: str, board_snapshot: Optional[Dict[str, Any]] = None) -> list:
        """Parse DRC/ERC error text and return targeted fix actions."""
        from ..design_agent import DesignAction, DesignActionType

        lower = error_text.lower()
        fixes: list = []

        # Heuristics from board state (when available). If we don't have a
        # snapshot, keep legacy behavior (assume routing may exist).
        has_routing = True
        can_autoroute = True
        if board_snapshot is not None:
            try:
                tracks = int(board_snapshot.get("tracks_count") or 0)
                vias = int(board_snapshot.get("vias_count") or 0)
                has_routing = (tracks + vias) > 0 or bool(board_snapshot.get("routing_attempted"))
            except Exception:
                has_routing = bool(board_snapshot.get("routing_attempted"))
            try:
                can_autoroute = bool(board_snapshot.get("outline_defined")) and int(board_snapshot.get("nets_assigned") or 0) > 0
            except Exception:
                can_autoroute = bool(board_snapshot.get("outline_defined"))

        # Courtyard overlap — extract refs if possible.
        if "courtyard" in lower and "overlap" in lower:
            refs = self._extract_refs(error_text)
            for ref in refs[:4]:
                fixes.append(DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=f"Move {ref} to clear courtyard overlap",
                    parameters={"ref": ref, "strategy": "resolve_overlap"},
                    requires_approval=True,
                ))

        # Board edge clearance / silkscreen clipping.
        if "board edge" in lower or "edge clearance" in lower or "silkscreen clipped" in lower:
            refs = self._extract_refs(error_text)
            for ref in refs[:4]:
                fixes.append(DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=f"Move {ref} away from board edge",
                    parameters={"ref": ref, "strategy": "edge_inset"},
                    requires_approval=True,
                ))

        # Hole clearance issues usually mean duplicated/overlapping holes or
        # footprints packed too tightly. Prefer spacing fixes.
        if "hole clearance" in lower or "drilled hole" in lower:
            refs = self._extract_refs(error_text)
            for ref in refs[:4]:
                fixes.append(DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=f"Move {ref} to fix hole/keepout clearance",
                    parameters={"ref": ref, "strategy": "resolve_overlap"},
                    requires_approval=True,
                ))

        # Short circuit / track crossing: only delete/re-route when routing exists.
        if "short" in lower or "crossing" in lower:
            if has_routing:
                fixes.append(DesignAction(
                    action_type=DesignActionType.DELETE_TRACKS,
                    description="Delete all tracks to clear shorts/crossings",
                    parameters={},
                    requires_approval=True,
                ))
                if can_autoroute:
                    fixes.append(DesignAction(
                        action_type=DesignActionType.AUTOROUTE_BOARD,
                        description="Re-route the board after clearing tracks",
                        parameters={},
                        requires_approval=True,
                    ))

        # Clearance violations are ambiguous (tracks vs. footprints). If we
        # don't have routing yet, treat them as placement spacing issues.
        if "clearance violation" in lower and not has_routing:
            refs = self._extract_refs(error_text)
            for ref in refs[:4]:
                fixes.append(DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=f"Move {ref} to improve clearance",
                    parameters={"ref": ref, "strategy": "resolve_overlap"},
                    requires_approval=True,
                ))

        # Missing connections.
        if "missing connection" in lower or "unconnected" in lower:
            fixes.append(DesignAction(
                action_type=DesignActionType.AUTOROUTE_BOARD,
                description="Autoroute to resolve missing connections",
                parameters={},
                requires_approval=True,
            ))

        # Solder mask bridge — usually spacing issue.
        if "solder mask" in lower and "bridge" in lower:
            refs = self._extract_refs(error_text)
            for ref in refs[:4]:
                fixes.append(DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=f"Move {ref} to fix solder mask bridge",
                    parameters={"ref": ref, "strategy": "increase_spacing"},
                    requires_approval=True,
                ))

        # Always re-run DRC after fixes.
        if fixes:
            fixes.append(DesignAction(
                action_type=DesignActionType.RUN_DRC,
                description="Re-run DRC to verify fixes",
                parameters={},
                requires_approval=False,
            ))

        return fixes

    @staticmethod
    def _extract_refs(text: str) -> List[str]:
        """Extract component reference designators from error text."""
        refs = re.findall(r"\b([A-Z]{1,3}\d{1,4})\b", text)
        # Deduplicate while preserving order.
        seen: set = set()
        unique: list = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                unique.append(r)
        return unique

    # ── Internal ────────────────────────────────────────────────

    def _build_prompt(self, goal, context, board_snapshot) -> str:
        parts = [f"USER GOAL:\n{goal}\n"]
        errors = context.get("drc_errors") or context.get("erc_errors")
        if errors:
            parts.append(f"CURRENT ERRORS:\n{errors}\n")
        parts.append("Propose verification and/or fix actions. Return a JSON array.")
        return "\n".join(parts)

    def _parse_actions(self, raw: str) -> list:
        from ..design_agent import DesignAction, DesignActionType
        from ...llm.client import LLMError
        if not raw:
            raise LLMError("verification: empty LLM response.")
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start < 0 or end <= start:
                raise LLMError("verification: expected a JSON array.")
            items = json.loads(raw[start:end])
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"verification: failed to parse JSON array: {e}") from e

        type_map = {
            "RUN_DRC": DesignActionType.RUN_DRC,
            "RUN_ERC": DesignActionType.RUN_ERC,
            "DELETE_TRACKS": DesignActionType.DELETE_TRACKS,
            "AUTOROUTE_BOARD": DesignActionType.AUTOROUTE_BOARD,
            "MOVE_COMPONENT": DesignActionType.MOVE_COMPONENT,
            "ASSIGN_NETS": DesignActionType.ASSIGN_NETS,
            "DEFINE_NET": DesignActionType.DEFINE_NET,
        }
        actions: list = []
        for item in items:
            if not isinstance(item, dict):
                raise LLMError("verification: each action must be an object.")
            atype_str = str(item.get("action_type", "")).upper()
            atype = type_map.get(atype_str)
            if atype is None:
                raise LLMError(f"verification: unknown action_type: {atype_str!r}")
            params = item.get("parameters") or {}
            if not isinstance(params, dict):
                raise LLMError(f"verification: parameters must be an object for {atype.name}.")
            actions.append(DesignAction(
                action_type=atype,
                description=str(item.get("description", "")),
                parameters=params,
                requires_approval=(atype not in {DesignActionType.RUN_DRC, DesignActionType.RUN_ERC}),
            ))
        return actions
