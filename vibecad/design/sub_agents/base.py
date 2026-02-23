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
            from ...llm.client import LLMMessage
            resp = self._llm_client.chat(
                [LLMMessage(role="user", content=user_prompt)],
                system_prompt=system_prompt or self.SYSTEM_PROMPT,
            )
            content = (getattr(resp, "content", "") or "").strip()
            if not content:
                raise LLMError(f"{self.NAME}: LLM returned empty content.")
            return content
        except LLMError:
            raise
        except Exception as e:
            logger.exception("%s: LLM call failed", self.NAME)
            raise LLMError(f"{self.NAME}: LLM call failed: {e}") from e
