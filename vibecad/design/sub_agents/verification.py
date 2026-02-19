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
        "You are the Verification sub-agent of VibeCAD.\n\n"
        "Your job: run DRC / ERC, analyse errors, and propose specific fixes.\n"
        "Available actions: RUN_DRC, RUN_ERC.\n"
        "You can also propose fix actions FROM OTHER DOMAINS when needed:\n"
        "  - DELETE_TRACKS + AUTOROUTE_BOARD for short / crossing errors.\n"
        "  - MOVE_COMPONENT for courtyard / clearance / edge errors.\n"
        "  - ASSIGN_NETS / DEFINE_NET for missing-connection errors.\n\n"
        "Fix strategy (apply in order):\n"
        "1. Courtyard overlap → MOVE_COMPONENT to separate.\n"
        "2. Board edge clearance → MOVE_COMPONENT inward.\n"
        "3. Track clearance / short → DELETE_TRACKS then AUTOROUTE_BOARD.\n"
        "4. Missing connection → check net assignments, then re-route.\n"
        "5. Solder mask bridge → MOVE_COMPONENT to increase spacing.\n\n"
        "IMPORTANT: After proposing fixes, always include a final RUN_DRC or RUN_ERC\n"
        "so the loop can verify the fix worked.\n\n"
        "Return ONLY a JSON array of actions.\n"
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

        # If we have DRC/ERC results in context, analyse them.
        errors = context.get("drc_errors") or context.get("erc_errors") or ""
        if errors:
            fixes = self._analyse_errors(str(errors))
            if fixes:
                return SubAgentResult(
                    message=f"Proposing {len(fixes)} fix(es) for verification errors.",
                    actions=fixes,
                    confidence=0.8,
                    thinking=f"Analysed errors → {len(fixes)} fixes",
                )

        # LLM-backed planning.
        if self._llm_available():
            raw = self._llm_chat(self._build_prompt(goal, context, board_snapshot))
            actions = self._parse_actions(raw)
            if actions:
                return SubAgentResult(
                    message=f"Proposing {len(actions)} verification action(s).",
                    actions=actions,
                    confidence=0.85,
                )

        # Fallback: just run DRC.
        return SubAgentResult(
            message="Running DRC to check the design.",
            actions=[
                DesignAction(
                    action_type=DesignActionType.RUN_DRC,
                    description="Run Design Rule Check",
                    parameters={},
                    requires_approval=False,
                ),
            ],
            confidence=0.7,
        )

    # ── Error analysis ──────────────────────────────────────────

    def _analyse_errors(self, error_text: str) -> list:
        """Parse DRC/ERC error text and return targeted fix actions."""
        from ..design_agent import DesignAction, DesignActionType

        lower = error_text.lower()
        fixes: list = []

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

        # Board edge clearance.
        if "board edge" in lower or "edge clearance" in lower:
            refs = self._extract_refs(error_text)
            for ref in refs[:4]:
                fixes.append(DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=f"Move {ref} away from board edge",
                    parameters={"ref": ref, "strategy": "edge_inset"},
                    requires_approval=True,
                ))

        # Short circuit / track crossing.
        if "short" in lower or "crossing" in lower or "clearance violation" in lower:
            fixes.append(DesignAction(
                action_type=DesignActionType.DELETE_TRACKS,
                description="Delete all tracks to clear shorts/crossings",
                parameters={},
                requires_approval=True,
            ))
            fixes.append(DesignAction(
                action_type=DesignActionType.AUTOROUTE_BOARD,
                description="Re-route the board after clearing tracks",
                parameters={},
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
        if not raw:
            return []
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start < 0 or end <= start:
                return []
            items = json.loads(raw[start:end])
        except Exception:
            return []

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
                continue
            atype_str = str(item.get("action_type", "")).upper()
            atype = type_map.get(atype_str)
            if atype is None:
                continue
            actions.append(DesignAction(
                action_type=atype,
                description=str(item.get("description", "")),
                parameters=item.get("parameters") or {},
                requires_approval=(atype not in {DesignActionType.RUN_DRC, DesignActionType.RUN_ERC}),
            ))
        return actions
