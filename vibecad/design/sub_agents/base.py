"""
SubAgent base class — the common interface every specialist must implement.

Each subagent has:
  - A focused system prompt (much shorter than the monolithic one).
  - A list of action types it can generate.
  - A ``plan()`` method that produces actions from context + user goal.
  - A ``can_handle()`` predicate so the Orchestrator knows who to call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..design_agent import DesignAction, DesignActionType

logger = logging.getLogger(__name__)


@dataclass
class SubAgentResult:
    """Outcome of a subagent ``plan()`` call."""

    # Human-readable explanation of what the subagent decided.
    message: str = ""

    # Proposed DesignActions (will be previewed / approved in the normal flow).
    actions: list = field(default_factory=list)  # List[DesignAction]

    # Confidence 0-1 that these actions address the current sub-goal.
    confidence: float = 0.0

    # If True, the subagent thinks its domain is done (hand off to next phase).
    phase_complete: bool = False

    # Optional diagnostic / reasoning trace for the debug panel.
    thinking: str = ""

    # Optional machine-readable artifacts produced by the subagent.
    # This is kept separate from actions so we can enforce hard gates without
    # inventing "store artifact" pseudo-actions.
    artifacts: Dict[str, Any] = field(default_factory=dict)


class SubAgent:
    """Abstract base for every specialised subagent."""

    # Override in subclasses — defines the subdomain.
    NAME: str = "base"
    HANDLED_ACTION_TYPES: frozenset = frozenset()

    # Override with a short, focused system prompt.
    SYSTEM_PROMPT: str = ""

    def __init__(self, llm_client=None):
        self._llm_client = llm_client

    # ── Public API ──────────────────────────────────────────────

    def can_handle(self, action_types: frozenset) -> bool:
        """Return True if this subagent covers any of *action_types*."""
        return bool(self.HANDLED_ACTION_TYPES & action_types)

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        """Produce zero or more DesignActions for the current sub-goal.

        Subclasses must override this.
        """
        raise NotImplementedError

    def refine(
        self,
        feedback: str,
        previous_result: SubAgentResult,
        context: Dict[str, Any],
    ) -> SubAgentResult:
        """Re-plan given feedback (e.g. DRC failures, overlap warnings).

        Default implementation just calls ``plan()`` again with the feedback
        prepended to the goal.
        """
        combined = f"FEEDBACK: {feedback}\nOriginal plan: {previous_result.thinking}"
        return self.plan(combined, context)

    # ── Helpers for subclasses ──────────────────────────────────

    def _llm_available(self) -> bool:
        if self._llm_client is None:
            return False
        return bool(getattr(self._llm_client, "is_available", False))

    def _llm_chat(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """Convenience wrapper around the shared LLM client."""
        from ...llm.client import LLMError
        if not self._llm_available():
            raise LLMError(f"{self.NAME}: LLM is required but is not available/configured.")
        try:
            # Expose the prompt/response for orchestrator debugging on schema failures.
            self._last_llm_user_prompt = user_prompt
            from ...llm.client import LLMMessage
            resp = self._llm_client.chat(
                [LLMMessage(role="user", content=user_prompt)],
                system_prompt=system_prompt or self.SYSTEM_PROMPT,
                # Many OpenAI-compatible providers support this and it dramatically
                # reduces malformed "almost-JSON" outputs from subagents.
                response_format={"type": "json_object"},
            )
            content = (getattr(resp, "content", "") or "").strip()
            self._last_llm_response_text = content
            if not content:
                raise LLMError(f"{self.NAME}: LLM returned empty content.")
            return content
        except LLMError:
            raise
        except Exception as e:
            logger.exception("%s: LLM call failed", self.NAME)
            raise LLMError(f"{self.NAME}: LLM call failed: {e}") from e

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        """Extract the outermost JSON object from *raw_text*."""
        s = (raw_text or "").strip()
        if not s:
            return ""
        start = s.find("{")
        end = s.rfind("}") + 1
        if start < 0 or end <= start:
            return ""
        return s[start:end]

    def _parse_plan_object(self, raw_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Parse the unified plan schema returned by subagents.

        Required schema:
        {
          "assistant_message": "string",
          "actions": [ { "action_type": "...", "description": "...", "parameters": { ... }, ... } ]
        }
        """
        import json
        from ...llm.client import LLMError
        from ..design_agent import sanitize_llm_json_text

        if not raw_text:
            raise LLMError(f"{self.NAME}: empty LLM response.")

        cleaned = sanitize_llm_json_text(raw_text)
        obj_text = self._extract_json_object(cleaned) or cleaned
        try:
            obj = json.loads(obj_text)
        except Exception as e:
            # Parse diagnostics for malformed provider responses (truncation, bad escaping).
            try:
                s = str(obj_text or "")
                tail_bytes = list(s.encode("utf-8", errors="replace")[-40:])
                logger.error(
                    "%s: JSON parse failed: %s | len=%d first80=%r last200=%r tail_bytes=%s",
                    self.NAME,
                    e,
                    len(s),
                    s[:80],
                    s[-200:],
                    tail_bytes,
                )
            except Exception:
                logger.exception("%s: failed to emit JSON parse diagnostics", self.NAME)
            raise LLMError(f"{self.NAME}: failed to parse JSON object: {e}") from e
        if not isinstance(obj, dict):
            raise LLMError(f"{self.NAME}: expected JSON object.")
        if "assistant_message" not in obj:
            raise LLMError(f"{self.NAME}: missing 'assistant_message'.")
        msg = str(obj.get("assistant_message") or "").strip()
        actions = obj.get("actions")
        if not isinstance(actions, list):
            raise LLMError(f"{self.NAME}: missing/invalid 'actions' array.")
        out: List[Dict[str, Any]] = []
        for i, item in enumerate(actions):
            if not isinstance(item, dict):
                raise LLMError(f"{self.NAME}: action[{i}] must be an object.")
            # Normalize common key variants so downstream parsers only deal with
            # the canonical schema.
            if "action_type" not in item or item.get("action_type") in (None, ""):
                for alt in ("actionType", "actiontype", "action", "type", "tool", "tool_name", "toolName"):
                    if alt in item and item.get(alt) not in (None, ""):
                        item["action_type"] = item.get(alt)
                        if alt != "action_type":
                            item.pop(alt, None)
                        break
            if "parameters" not in item and any(k in item for k in ("params", "arguments")):
                item["parameters"] = item.get("parameters") or item.get("params") or item.get("arguments") or {}
                item.pop("params", None)
                item.pop("arguments", None)
            out.append(item)
        return msg, out
