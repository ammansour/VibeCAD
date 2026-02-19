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
        "You are the InfoGathering sub-agent of VibeCAD, a KiCad PCB design assistant.\n\n"
        "Your ONLY job is to identify which components and information are needed for the "
        "user's design and produce SEARCH_PART / SEARCH_WEB / LOOKUP_DATASHEET actions.\n\n"
        "Rules:\n"
        "- For each distinct component the user mentions (or that the design requires), "
        "  emit one SEARCH_PART action with a specific MPN or short description.\n"
        "- SEARCH_PART actions MUST include parameters: {\"query\": \"...\"}.\n"
        "- SEARCH_WEB actions MUST include parameters: {\"query\": \"...\"}.\n"
        "- LOOKUP_DATASHEET actions MUST include parameters: {\"mpn\": \"...\"} (an actual MPN).\n"
        "- If you don't have an actual MPN, do NOT emit LOOKUP_DATASHEET.\n"
        "- If you need pinout or package information for an IC, emit LOOKUP_DATASHEET using its MPN.\n"
        "- Never propose placement, routing, or board-outline actions.\n"
        "- Return ONLY a JSON array of actions. No markdown, no extra text.\n"
        "- Each action: {\"action_type\": \"SEARCH_PART\"|\"SEARCH_WEB\"|\"LOOKUP_DATASHEET\", "
        "  \"description\": \"...\", \"parameters\": {...}}\n"
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
        from ..design_agent import DesignAction, DesignActionType

        # Try LLM-based planning first.
        if self._llm_available():
            raw = self._llm_chat(self._build_prompt(goal, context, board_snapshot))
            actions = self._parse_actions(raw)
            if actions:
                return SubAgentResult(
                    message="Searching for components and datasheets…",
                    actions=actions,
                    confidence=0.85,
                    thinking=f"LLM proposed {len(actions)} info-gathering actions",
                )

        # Fallback: extract component-like tokens and create SEARCH_PART actions.
        actions = self._fallback_extract(goal)
        if actions:
            return SubAgentResult(
                message="Searching for mentioned components…",
                actions=actions,
                confidence=0.5,
                thinking="Fallback regex extraction",
            )

        return SubAgentResult(
            message="No specific components detected to search for.",
            phase_complete=True,
            confidence=0.3,
        )

    # ── Internal ────────────────────────────────────────────────

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
            # Self-heal common LLM omission: if required keys are missing, try
            # to recover a query from description.
            if atype == DesignActionType.SEARCH_PART and not str(params.get("query", "") or "").strip():
                if desc:
                    params["query"] = desc
            if atype == DesignActionType.SEARCH_WEB and not str(params.get("query", "") or "").strip():
                if desc:
                    params["query"] = desc
            if atype == DesignActionType.LOOKUP_DATASHEET and not str(params.get("mpn", "") or "").strip():
                # For datasheets, only accept an obvious MPN-like token.
                m = re.search(r"\b([A-Za-z]{2,}\d{2,}[A-Za-z0-9_+\-./]{0,60})\b", desc)
                if m:
                    params["mpn"] = m.group(1)
                else:
                    # Skip invalid datasheet lookup actions rather than erroring.
                    continue
            actions.append(DesignAction(
                action_type=atype,
                description=str(item.get("description", "")),
                parameters=params,
                requires_approval=False,
            ))
        return actions

    def _fallback_extract(self, goal: str) -> list:
        """Regex-based extraction of component tokens → SEARCH_PART actions.

        Handles both explicit MPNs (ATmega328P, LM7805) and high-level board
        names (Arduino UNO, ESP32 DevKit) by searching for MPN-like tokens
        first, then falling back to recognisable board / module names.
        """
        from ..design_agent import DesignAction, DesignActionType

        # 1. Explicit MPN patterns: ATmega328P-PU, STM32F103, LM7805, USB-C, etc.
        tokens = re.findall(
            r"\b([A-Z]{2,}[0-9][A-Z0-9\-]{2,})\b",
            goal,
            re.IGNORECASE,
        )

        # 2. Well-known board / module names (case-insensitive).
        board_patterns = [
            r"\b(arduino\s+\w+)\b",
            r"\b(esp32[\w\-]*)\b",
            r"\b(esp8266[\w\-]*)\b",
            r"\b(raspberry\s+pi[\w\s]*?(?:pico|zero|[0-9]+)?)\b",
            r"\b(stm32[\w\-]+)\b",
            r"\b(teensy[\w\-]*)\b",
            r"\b(nrf52[\w\-]*)\b",
        ]
        for pat in board_patterns:
            for m in re.finditer(pat, goal, re.IGNORECASE):
                tokens.append(m.group(1).strip())

        # 3. Quoted component names: "LM7805", 'ATmega328P'
        for m in re.finditer(r"""['"]([^'"]{2,30})['"]""", goal):
            tokens.append(m.group(1).strip())

        seen: set = set()
        actions: list = []
        for tok in tokens:
            key = tok.upper().strip()
            if key in seen or len(key) < 2:
                continue
            seen.add(key)
            actions.append(DesignAction(
                action_type=DesignActionType.SEARCH_PART,
                description=f"Search for {tok}",
                parameters={"query": tok},
                requires_approval=False,
            ))
        return actions
