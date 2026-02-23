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
        "You are the Routing sub-agent.\n"
        "Only propose: DEFINE_NET, ASSIGN_NETS, DRAW_TRACK, ADD_VIA, AUTOROUTE_BOARD, SET_LAYER_COUNT, DELETE_TRACKS.\n"
        "Assign nets before routing. If pad-not-found appears, do not route.\n"
        "Prefer AUTOROUTE_BOARD for many nets; use DRAW_TRACK for specific nets.\n"
        "Use ref/pad notation instead of invented coordinates.\n"
        "Parameter schema (use exactly these keys):\n"
        "- DEFINE_NET.parameters = {\"net\":\"GND\",\"pads\":[\"U1:3\",\"J1:4\"]} (pads optional)\n"
        "- ASSIGN_NETS.parameters = {\"assignments\":[{\"net\":\"GND\",\"ref\":\"U1\",\"pad\":\"3\"}, ...]}\n"
        "- DRAW_TRACK.parameters = {\"from_point\":\"U1/3\",\"to_point\":\"J1/4\"}\n"
        "Never propose placement/search/verification actions.\n"
        "Return JSON array only.\n"
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

        raw = self._llm_chat(self._build_prompt(goal, context, board_snapshot))
        actions = self._parse_actions(raw)
        if actions:
            return SubAgentResult(
                message=f"Proposing {len(actions)} routing/net-assignment action(s).",
                actions=actions,
                confidence=0.85,
                thinking=f"LLM proposed {len(actions)} routing actions",
            )

        return SubAgentResult(
            message="No routing/net-assignment actions proposed.",
            confidence=0.3,
            phase_complete=False,
            thinking="LLM proposed no routing actions",
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
        from ...llm.client import LLMError
        if not raw:
            raise LLMError("routing: empty LLM response.")
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start < 0 or end <= start:
                raise LLMError("routing: expected a JSON array.")
            items = json.loads(raw[start:end])
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"routing: failed to parse JSON array: {e}") from e

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
            params = item.get("parameters") or {}
            if not isinstance(params, dict):
                raise LLMError(f"routing: parameters must be an object for {atype.name}.")
            actions.append(DesignAction(
                action_type=atype,
                description=str(item.get("description", "")),
                parameters=params,
                requires_approval=True,
            ))
        return actions
