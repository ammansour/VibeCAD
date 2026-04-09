# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UNIVERSAL PLUGIN — NO BOARD-SPECIFIC HARDCODING IN THIS FILE      ║
# ║  Prompts must use only goal_str / context variables.               ║
# ║  Never embed specific MPNs, part names, board names, or            ║
# ║  design-specific quantities in prompt strings or system prompts.   ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""LLM-based intent routing for the Design tab.

Goal: avoid starting the autonomous AgentLoop for simple informational Q&A.

Primary routing is done via an LLM classification call. If the LLM is not
available or returns an invalid response, routing fails fast (no deterministic
fallbacks).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Protocol


class LLMClientLike(Protocol):
    is_available: bool

    def chat(self, messages, system_prompt: Optional[str] = None):  # pragma: no cover
        ...


@dataclass(frozen=True)
class IntentDecision:
    route: str  # 'qa' | 'direct_tool' | 'agent'
    reason: str


_SYSTEM_PROMPT = (
    "You are an intent router for a PCB design assistant UI. "
    "Classify the user's message as one of:\n"
    "- 'qa' (informational question; answer directly; do NOT start autonomous multi-step agent)\n"
    "- 'direct_tool' (focused tool tasks, multi-tool usage, querying board state, and making targeted changes:  "
    "move one component, run/check wiring/DRC for a specific item, lookup part price/data)\n"
    "- 'agent' (high-level build workflow requiring multi-phase autonomous planning: "
    "SPEC + PLACE + NET)\n\n"
    "Examples:\n"
    "- 'move J1 right 5 in' -> direct_tool\n"
    "- 'is the wiring of F1 correct' -> direct_tool\n"
    "- 'what is the unit price of ADS1256' -> qa\n"
    "- 'build an ESP32 board' -> agent\n\n"
    "Hybrid precedence rules (single final route):\n"
    "- qa + direct_tool -> direct_tool\n"
    "- qa + agent/build -> agent\n"
    "- direct_tool + agent/build -> agent\n\n"
    "Return ONLY a compact JSON object like "
    "{\"route\":\"qa\"|\"direct_tool\"|\"agent\",\"reason\":\"...\"}. "
    "No extra keys, no markdown."
)


def _extract_json_object(text: str) -> Optional[dict]:
    if not text:
        return None

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    blob = match.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def decide_route(llm_client: Optional[LLMClientLike], message: str) -> IntentDecision:
    """Decide whether to route the message to Q&A or AgentLoop."""

    from vibecad.llm.client import LLMError

    if llm_client is None or not getattr(llm_client, "is_available", False):
        raise LLMError("LLM is required for intent routing but is not available/configured.")

    try:
        from vibecad.llm.client import LLMMessage

        resp = llm_client.chat(
            [LLMMessage(role="user", content=(message or "").strip())],
            system_prompt=_SYSTEM_PROMPT,
        )
        raw = (getattr(resp, "content", "") or "").strip()
        parsed = _extract_json_object(raw)
        if not parsed:
            raise LLMError("Intent router returned invalid/empty JSON.")

        route = str(parsed.get("route", "")).strip().lower()
        route_aliases = {
            "tool": "direct_tool",
            "tools": "direct_tool",
            "direct": "direct_tool",
            "direct-tool": "direct_tool",
            "directtool": "direct_tool",
            "action": "direct_tool",
            "actions": "direct_tool",
            "full_agent": "agent",
            "workflow": "agent",
            "builder": "agent",
        }
        route = route_aliases.get(route, route)
        reason = str(parsed.get("reason", "")).strip()[:240]
        if route not in ("qa", "direct_tool", "agent"):
            raise LLMError(f"Intent router returned invalid route: {route!r}")

        return IntentDecision(route=route, reason=reason or "llm")
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Intent routing failed: {e}") from e


def should_route_to_qa(llm_client: Optional[LLMClientLike], message: str) -> bool:
    return decide_route(llm_client, message).route == "qa"


def should_route_to_direct_tool(llm_client: Optional[LLMClientLike], message: str) -> bool:
    return decide_route(llm_client, message).route == "direct_tool"
