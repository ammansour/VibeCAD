# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UNIVERSAL PLUGIN — NO BOARD-SPECIFIC HARDCODING IN THIS FILE      ║
# ║  Prompts must use only goal_str / context variables.               ║
# ║  Never embed specific MPNs, part names, board names, or            ║
# ║  design-specific quantities in prompt strings or system prompts.   ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""LLM-based workflow routing for AgentLoop.

Goal: decide whether the user's goal implies the *full workflow* (placement +
net assignment + routing) or a narrower layout-only workflow.

This decision is made via an LLM classification call. If the LLM is not
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
class WorkflowDecision:
    require_full_workflow: bool
    reason: str


_SYSTEM_PROMPT = (
    "You are a workflow router for a PCB design assistant.\n\n"
    "Decide whether the user's goal requires the *full workflow*:\n"
    "- full workflow = place components + define/assign nets + route (or autoroute)\n"
    "- layout-only = placement/outline/moves/DRC checks only (no net assignment/routing)\n\n"
    "Return ONLY a compact JSON object like:\n"
    "{\"require_full_workflow\":true|false,\"reason\":\"...\"}\n"
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


def decide_require_full_workflow(llm_client: Optional[LLMClientLike], goal: str) -> WorkflowDecision:
    """Return an LLM decision for whether to run the full workflow."""
    from vibecad.llm.client import LLMError

    if llm_client is None or not getattr(llm_client, "is_available", False):
        raise LLMError("LLM is required for workflow routing but is not available/configured.")

    try:
        from vibecad.llm.client import LLMMessage

        resp = llm_client.chat(
            [LLMMessage(role="user", content=(goal or "").strip())],
            system_prompt=_SYSTEM_PROMPT,
        )
        raw = (getattr(resp, "content", "") or "").strip()
        parsed = _extract_json_object(raw)
        if not parsed:
            raise LLMError("Workflow router returned invalid/empty JSON.")

        require_full_workflow = parsed.get("require_full_workflow", None)
        if not isinstance(require_full_workflow, bool):
            raise LLMError("Workflow router returned invalid require_full_workflow value.")

        reason = str(parsed.get("reason", "")).strip()[:240]
        return WorkflowDecision(require_full_workflow=require_full_workflow, reason=reason or "llm")
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Workflow routing failed: {e}") from e

