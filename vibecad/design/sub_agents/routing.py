"""
RoutingAgent — net assignment, track drawing, and autorouting.

Handles DEFINE_NET, ASSIGN_NETS, DRAW_TRACK, ADD_VIA, AUTOROUTE_BOARD,
SET_LAYER_COUNT, and DELETE_TRACKS.

This subagent focuses solely on connectivity: making sure pads are assigned
to the correct nets and that copper traces connect them.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .base import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class RoutingAgent(SubAgent):
    NAME = "routing"

    SYSTEM_PROMPT = (
        "You are the Routing sub-agent of VibeCAD.\n\n"
        "Your ONLY job is to assign nets to pads and route copper traces.\n"
        "Available actions: DEFINE_NET, ASSIGN_NETS, DRAW_TRACK, ADD_VIA, "
        "AUTOROUTE_BOARD, SET_LAYER_COUNT, DELETE_TRACKS.\n\n"
        "Strategy:\n"
        "1. Before any routing, ensure ALL pads have nets assigned.\n"
        "   - Use DEFINE_NET for logical connections (e.g. GND, VCC, SDA).\n"
        "   - Use ASSIGN_NETS for bulk pad-to-net mapping.\n"
        "2. If pad assignments report 'pad not found', the footprint has the\n"
        "   wrong pin count.  Report this — do NOT proceed to routing.\n"
        "3. For routing: prefer AUTOROUTE_BOARD when ≥ 5 nets need routing.\n"
        "   Use DRAW_TRACK for individual critical nets (short, matched-length).\n"
        "4. If the autorouter reports unrouted nets, try DELETE_TRACKS then\n"
        "   re-route after verifying net assignments are correct.\n"
        "5. Use SET_LAYER_COUNT if the board needs more than 2 layers.\n\n"
        "Routing rules:\n"
        "- All tracks follow cardinal (H/V) or 45° directions.\n"
        "- Default track width: 0.25 mm (signal), 0.5 mm (power).\n"
        "- Never invent coordinates.  Use ref/pad notation (e.g. U1/14).\n\n"
        "CRITICAL: Never propose placement, search, or verification actions.\n\n"
        "Return ONLY a JSON array of actions.\n"
    )

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        from ..design_agent import DesignActionType
        self.HANDLED_ACTION_TYPES = frozenset({
            DesignActionType.DEFINE_NET,
            DesignActionType.ASSIGN_NETS,
            DesignActionType.DRAW_TRACK,
            DesignActionType.DRAW_WIRE,
            DesignActionType.ROUTE_NET,
            DesignActionType.ADD_VIA,
            DesignActionType.AUTOROUTE_BOARD,
            DesignActionType.SET_LAYER_COUNT,
            DesignActionType.DELETE_TRACKS,
        })

    # ── plan ────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        from ..design_agent import DesignAction, DesignActionType

        if self._llm_available():
            raw = self._llm_chat(self._build_prompt(goal, context, board_snapshot))
            actions = self._parse_actions(raw)
            if actions:
                return SubAgentResult(
                    message=f"Proposing {len(actions)} routing/net-assignment action(s).",
                    actions=actions,
                    confidence=0.85,
                    thinking=f"LLM proposed {len(actions)} routing actions",
                )

        # Minimal fallback: if pads have no nets, suggest AUTOROUTE_BOARD.
        return SubAgentResult(
            message="Could not generate routing actions. Ensure nets are assigned first.",
            confidence=0.2,
            phase_complete=True,  # Don't get stuck — let orchestrator advance
        )

    # ── Internal ────────────────────────────────────────────────

    def _build_prompt(self, goal, context, board_snapshot) -> str:
        parts = [f"USER GOAL:\n{goal}\n"]

        if board_snapshot:
            # Summarize existing net state.
            nets = board_snapshot.get("nets", {})
            if nets:
                parts.append(f"Nets defined: {len(nets)}")
                for name, info in list(nets.items())[:20]:
                    pads = info.get("pads", [])
                    parts.append(f"  {name}: {len(pads)} pad(s)")
            comps = board_snapshot.get("components", [])
            if comps:
                refs = [c.get("reference", "?") for c in comps[:30]]
                parts.append(f"Components: {', '.join(refs)}")
            unrouted = board_snapshot.get("unrouted_count", None)
            if unrouted is not None:
                parts.append(f"Unrouted connections: {unrouted}")

        parts.append(
            "\nPropose net assignment and/or routing actions. Return a JSON array."
        )
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
            "DEFINE_NET": DesignActionType.DEFINE_NET,
            "ASSIGN_NETS": DesignActionType.ASSIGN_NETS,
            "DRAW_TRACK": DesignActionType.DRAW_TRACK,
            "DRAW_WIRE": DesignActionType.DRAW_WIRE,
            "ROUTE_NET": DesignActionType.ROUTE_NET,
            "ADD_VIA": DesignActionType.ADD_VIA,
            "AUTOROUTE_BOARD": DesignActionType.AUTOROUTE_BOARD,
            "SET_LAYER_COUNT": DesignActionType.SET_LAYER_COUNT,
            "DELETE_TRACKS": DesignActionType.DELETE_TRACKS,
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
                requires_approval=True,
            ))
        return actions
