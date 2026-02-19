"""LLM-based intent routing for the Design tab.

Goal: avoid starting the autonomous AgentLoop for simple informational Q&A.

This module intentionally avoids complex heuristics. Primary routing is done via
an LLM classification call. When the LLM is not available, it falls back to a
minimal, deterministic rule:
- message ending with '?' => Q&A
- otherwise => AgentLoop
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
    route: str  # 'qa' or 'agent'
    reason: str


_SYSTEM_PROMPT = (
    "You are an intent router for a PCB design assistant UI. "
    "Classify the user's message as either:\n"
    "- 'qa' (informational question; answer directly; do NOT start autonomous multi-step agent)\n"
    "- 'agent' (a request to perform design work, generate/modify things, run multi-step tasks)\n\n"
    "Return ONLY a compact JSON object like {\"route\":\"qa\"|\"agent\",\"reason\":\"...\"}. "
    "No extra keys, no markdown."
)


def _minimal_fallback(message: str) -> IntentDecision:
    text = (message or "").strip()
    if text.endswith("?"):
        return IntentDecision(route="qa", reason="fallback:question_mark")
    return IntentDecision(route="agent", reason="fallback:default")


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

    if llm_client is None or not getattr(llm_client, "is_available", False):
        return _minimal_fallback(message)

    try:
        from vibecad.llm.client import LLMMessage

        resp = llm_client.chat(
            [LLMMessage(role="user", content=(message or "").strip())],
            system_prompt=_SYSTEM_PROMPT,
        )
        raw = (getattr(resp, "content", "") or "").strip()
        parsed = _extract_json_object(raw)
        if not parsed:
            return _minimal_fallback(message)

        route = str(parsed.get("route", "")).strip().lower()
        reason = str(parsed.get("reason", "")).strip()[:240]
        if route not in ("qa", "agent"):
            return _minimal_fallback(message)

        return IntentDecision(route=route, reason=reason or "llm")
    except Exception:
        return _minimal_fallback(message)


def should_route_to_qa(llm_client: Optional[LLMClientLike], message: str) -> bool:
    return decide_route(llm_client, message).route == "qa"
