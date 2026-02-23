"""
InfoGatheringAgent — handles component search, datasheet lookup, and web queries.

This subagent is non-destructive (read-only) and runs without user approval.
It enriches the design context so downstream agents (Placement, Routing) have
accurate footprint names, pin-counts, and package information.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .base import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)


class InfoGatheringAgent(SubAgent):
    NAME = "info_gathering"

    SYSTEM_PROMPT = (
        "You are the InfoGathering sub-agent.\n"
        "Only propose SEARCH_PART, SEARCH_WEB, or LOOKUP_DATASHEET actions.\n"
        "SEARCH_PART/SEARCH_WEB require parameters.query.\n"
        "LOOKUP_DATASHEET requires parameters.mpn; skip it if no real MPN.\n"
        "If the user goal implies adding/placing any components, propose SEARCH_PART actions for the key components first so later phases can use real local footprint/symbol names.\n"
        "Never propose placement/routing actions.\n"
        "Return JSON array only.\n"
    )

    HANDLED_ACTION_TYPES: frozenset = frozenset()  # set in __init_subclass__

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        # Lazy import to avoid circular deps at module level.
        from ..design_agent import DesignActionType
        self.HANDLED_ACTION_TYPES = frozenset({
            DesignActionType.SEARCH_PART,
            DesignActionType.SEARCH_WEB,
            DesignActionType.LOOKUP_DATASHEET,
        })

    # ── plan ────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        """Identify needed components and produce search actions."""
        goal_text = self._extract_primary_goal_text(goal)
        raw = self._llm_chat(self._build_prompt(goal_text, context, board_snapshot))
        actions = self._parse_actions(raw)
        if actions:
            return SubAgentResult(
                message="Searching for components and datasheets…",
                actions=actions,
                phase_complete=True,
                confidence=0.85,
                thinking=f"LLM proposed {len(actions)} info-gathering actions",
            )
        return SubAgentResult(
            message="No specific components detected to search for.",
            phase_complete=True,
            confidence=0.3,
            thinking="LLM proposed no info-gathering actions",
        )

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _extract_primary_goal_text(goal: str) -> str:
        """Extract user intent from phase+feedback wrapper text."""
        text = str(goal or "").strip()
        if not text:
            return ""

        # Orchestrator appends prior step logs under this marker.
        feedback_marker = "FEEDBACK FROM PREVIOUS STEP:"
        if feedback_marker in text:
            text = text.split(feedback_marker, 1)[0].strip()

        m = re.search(r"USER GOAL:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip()

        # Remove optional phase hint prefix.
        text = re.sub(r"^PHASE:\s*[A-Z_]+\s*[—-]\s*.*\n", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _build_prompt(self, goal, context, board_snapshot) -> str:
        existing = ""
        if board_snapshot and board_snapshot.get("components"):
            refs = [c.get("reference", "?") for c in board_snapshot["components"][:30]]
            existing = f"\nAlready on the board: {', '.join(refs)}"

        return (
            f"USER GOAL:\n{goal}\n{existing}\n\n"
            "Identify every component that needs to be researched / searched and "
            "return a JSON array of actions. Return [] if nothing to search."
        )

    def _parse_actions(self, raw: str) -> list:
        from ..design_agent import DesignAction, DesignActionType

        from ...llm.client import LLMError
        if not raw:
            raise LLMError("info_gathering: empty LLM response.")
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start < 0 or end <= start:
                raise LLMError("info_gathering: expected a JSON array.")
            items = json.loads(raw[start:end])
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"info_gathering: failed to parse JSON array: {e}") from e

        actions: list = []
        type_map = {
            "SEARCH_PART": DesignActionType.SEARCH_PART,
            "SEARCH_WEB": DesignActionType.SEARCH_WEB,
            "LOOKUP_DATASHEET": DesignActionType.LOOKUP_DATASHEET,
        }
        for item in items:
            if not isinstance(item, dict):
                continue
            atype_str = str(item.get("action_type", "")).upper()
            atype = type_map.get(atype_str)
            if atype is None:
                continue
            params = item.get("parameters") or {}
            if not isinstance(params, dict):
                params = {}
            desc = str(item.get("description", "") or "").strip()
            if atype in (DesignActionType.SEARCH_PART, DesignActionType.SEARCH_WEB):
                if not str(params.get("query", "") or "").strip():
                    raise LLMError(f"info_gathering: missing required parameters.query for {atype.name}.")
            if atype == DesignActionType.LOOKUP_DATASHEET:
                if not str(params.get("mpn", "") or "").strip():
                    raise LLMError("info_gathering: missing required parameters.mpn for LOOKUP_DATASHEET.")
            actions.append(DesignAction(
                action_type=atype,
                description=str(item.get("description", "")),
                parameters=params,
                requires_approval=False,
            ))
        return actions
