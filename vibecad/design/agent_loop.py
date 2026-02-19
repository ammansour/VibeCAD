"""
Agent Loop — orchestrates the iterative plan → execute → observe → replan cycle.

This module transforms VibeCAD from single-turn request/response into a
multi-step autonomous design agent that can:
  - Decompose a high-level goal (e.g., "design an Arduino UNO") into phases
  - Execute actions iteratively (search → place → outline → route → DRC)
  - Pause for user approval on destructive actions
  - Ask clarifying questions when needed
  - Re-run DRC and self-correct until the design passes
  - Be paused/cancelled by the user at any checkpoint

State Machine:
  IDLE → PLANNING → EXECUTING → AWAITING_APPROVAL → OBSERVING → DONE / PAUSED
"""

import logging
import threading
import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .design_agent import DesignAgent, DesignAction, DesignActionType

# Sub-agent orchestrator (optional — graceful fallback if not present).
try:
    from .sub_agents.orchestrator import Orchestrator, DesignPhase
    from .sub_agents.placement import PlacementAgent
    _SUBAGENTS_AVAILABLE = True
except ImportError:
    _SUBAGENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States of the agent loop."""
    IDLE = auto()
    PLANNING = auto()
    EXECUTING = auto()
    AWAITING_APPROVAL = auto()
    AWAITING_INPUT = auto()
    OBSERVING = auto()
    DONE = auto()
    PAUSED = auto()
    ERROR = auto()


# Actions that can auto-execute without user approval
READ_ONLY_ACTIONS = frozenset({
    DesignActionType.SEARCH_PART,
    DesignActionType.EXPORT_BOM,
    DesignActionType.RUN_DRC,
    DesignActionType.RUN_ERC,
    DesignActionType.SEARCH_WEB,
    DesignActionType.LOOKUP_DATASHEET,
})

# Actions that modify the board and need explicit approval
DESTRUCTIVE_ACTIONS = frozenset({
    DesignActionType.ADD_COMPONENT,
    DesignActionType.DRAW_TRACK,
    DesignActionType.DRAW_WIRE,
    DesignActionType.ROUTE_NET,
    DesignActionType.ADD_VIA,
    DesignActionType.ASSIGN_NETS,
    DesignActionType.MOVE_COMPONENT,
    DesignActionType.ROTATE_COMPONENT,
    DesignActionType.ALIGN_COMPONENTS,
    DesignActionType.ADD_POLYGON,
    DesignActionType.ADD_TEXT,
    DesignActionType.ADD_MOUNTING_HOLE,
    DesignActionType.DEFINE_BOARD_OUTLINE,
    DesignActionType.DOWNLOAD_SYMBOL,
    DesignActionType.DOWNLOAD_FOOTPRINT,
    DesignActionType.UPDATE_BOM_FIELDS,
    DesignActionType.AUTOROUTE_BOARD,
    DesignActionType.SET_LAYER_COUNT,
    DesignActionType.DELETE_TRACKS,
    DesignActionType.DELETE_COMPONENT,
})


def is_read_only(action_type: DesignActionType) -> bool:
    """Return True if the action type is non-destructive / informational."""
    return action_type in READ_ONLY_ACTIONS


@dataclass
class AgentLoopConfig:
    """Configuration for the agent loop."""
    max_iterations: int = 50
    max_drc_retries: int = 10
    auto_approve_readonly: bool = True
    yolo_auto_apply: bool = False


@dataclass
class LoopStep:
    """Record of a single step in the agent loop."""
    iteration: int
    action: Optional[DesignAction] = None
    result_success: Optional[bool] = None
    result_message: Optional[str] = None
    assistant_message: Optional[str] = None
    was_auto_executed: bool = False


class AgentLoop:
    """Orchestrates the iterative plan → execute → observe → replan cycle.

    Usage:
        loop = AgentLoop(design_agent, config)
        loop.set_ui_callback(on_message)
        loop.set_approval_callback(request_approval)
        loop.run(goal, context)  # Runs on a background thread
    """

    def __init__(self, design_agent: DesignAgent, config: Optional[AgentLoopConfig] = None):
        self._agent = design_agent
        self._config = config or AgentLoopConfig()

        # State
        self._state = AgentState.IDLE
        self._iteration = 0
        self._drc_retry_count = 0
        self._last_drc_passed: Optional[bool] = None  # None = never run
        self._history: List[LoopStep] = []
        self._goal: str = ""

        # Phase tracking — soft gates to prevent out-of-order operations
        self._phase = {
            'components_placed': 0,     # count of successfully placed components
            'outline_defined': False,
            'nets_assigned': 0,         # count of successful net assignments
            'net_assign_warnings': 0,   # count of net assignment warnings/failures
            'routing_attempted': False,
        }
        # Track add/delete cycles per component to detect death loops
        self._component_add_count: Dict[str, int] = {}   # query/ref -> add attempts
        self._component_delete_count: Dict[str, int] = {} # ref -> delete count
        self._consecutive_fp_failures: int = 0            # consecutive footprint load failures
        self._seen_search_queries: set = set()

        # Threading
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unpaused
        self._approval_event = threading.Event()
        self._input_event = threading.Event()
        self._stop_flag = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Approval state
        self._approval_round_actions: List[DesignAction] = []
        self._approval_round_decisions: Dict[int, bool] = {}
        self._approval_round_reasons: Dict[int, str] = {}

        # User input state (for clarifying questions)
        self._pending_user_input: Optional[str] = None

        # Sub-agent orchestrator (enhances reliability when available).
        self._orchestrator: Optional['Orchestrator'] = None
        self._use_subagents: bool = False
        if _SUBAGENTS_AVAILABLE:
            try:
                llm = getattr(design_agent, '_llm_client', None)
                self._orchestrator = Orchestrator(llm)
                self._use_subagents = True
                logger.info("Sub-agent orchestrator enabled")
            except Exception:
                logger.warning("Failed to initialise sub-agent orchestrator; using monolithic agent")

        # Message deduplication — prevent spamming the same error 50x.
        self._last_emitted_message: str = ""
        self._duplicate_message_count: int = 0

        # Callbacks
        self._ui_message_cb: Optional[Callable[[str], None]] = None
        self._ui_thinking_cb: Optional[Callable[[str], None]] = None
        self._ui_action_preview_cb: Optional[Callable] = None
        self._ui_response_cb: Optional[Callable[[str], None]] = None
        self._state_change_cb: Optional[Callable[[AgentState], None]] = None
        self._get_context_cb: Optional[Callable[[], Dict[str, Any]]] = None
        self._execute_on_gui_cb: Optional[Callable] = None

    # ── Public API ──────────────────────────────────────────────

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state not in (AgentState.IDLE, AgentState.DONE,
                                    AgentState.PAUSED, AgentState.ERROR)

    @property
    def iteration(self) -> int:
        return self._iteration

    def set_ui_message_callback(self, cb: Callable[[str], None]):
        """Callback for assistant chat messages shown to the user."""
        self._ui_message_cb = cb

    def set_ui_thinking_callback(self, cb: Callable[[str], None]):
        """Callback for thinking/status messages (ephemeral)."""
        self._ui_thinking_cb = cb

    def set_ui_action_preview_callback(self, cb: Callable):
        """Callback: (action_type_name, description, preview_text, action) → show preview."""
        self._ui_action_preview_cb = cb

    def set_ui_response_callback(self, cb: Callable[[str], None]):
        """Callback for final / status responses."""
        self._ui_response_cb = cb

    def set_state_change_callback(self, cb: Callable[[AgentState], None]):
        """Callback when agent state transitions."""
        self._state_change_cb = cb

    def set_context_callback(self, cb: Callable[[], Dict[str, Any]]):
        """Callback to refresh design context (board data, etc.)."""
        self._get_context_cb = cb

    def set_execute_on_gui_callback(self, cb: Callable):
        """Callback: (action, context, result_callback) → execute on GUI thread."""
        self._execute_on_gui_cb = cb

    def run(self, goal: str, initial_context: Optional[Dict[str, Any]] = None):
        """Start the agent loop in a background thread.

        Args:
            goal: The user's high-level design goal (natural language).
            initial_context: Initial design context dict.
        """
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("Agent loop already running")
            return

        self._goal = goal
        self._iteration = 0
        self._drc_retry_count = 0
        self._last_drc_passed = None
        self._history.clear()
        self._stop_flag.clear()
        self._pause_event.set()
        # Reset phase tracking
        self._phase = {
            'components_placed': 0,
            'outline_defined': False,
            'nets_assigned': 0,
            'net_assign_warnings': 0,
            'routing_attempted': False,
        }
        self._component_add_count = {}
        self._component_delete_count = {}
        self._consecutive_fp_failures = 0
        self._seen_search_queries = set()
        self._last_feedback: Optional[str] = None
        self._last_emitted_message = ""
        self._duplicate_message_count = 0

        # Reset orchestrator if available.
        if self._orchestrator:
            self._orchestrator.reset()
            self._use_subagents = True

        self._worker_thread = threading.Thread(
            target=self._run_loop,
            args=(goal, initial_context or {}),
            daemon=True,
        )
        self._worker_thread.start()

    def pause(self):
        """Pause the agent loop at the next checkpoint."""
        self._pause_event.clear()
        self._stop_flag.set()
        # If waiting for approval, unblock it
        self._approval_event.set()
        self._input_event.set()
        self._set_state(AgentState.PAUSED)

    def resume(self, user_message: Optional[str] = None):
        """Resume a paused loop, optionally feeding a user message."""
        if self._state == AgentState.PAUSED:
            self._stop_flag.clear()
            self._pause_event.set()
            if user_message:
                self._agent._append_history("user", user_message)
            self._set_state(AgentState.PLANNING)
            # Restart the loop
            self._worker_thread = threading.Thread(
                target=self._run_loop,
                args=(user_message or self._goal, self._get_context() or {}),
                daemon=True,
            )
            self._worker_thread.start()
        elif self._state == AgentState.AWAITING_INPUT:
            if user_message:
                self._pending_user_input = user_message
                self._input_event.set()

    def approve_action(self, action: DesignAction, approved: bool, reason: str = ""):
        """Record approval/rejection for a specific action in the current approval round."""
        if not action:
            return

        action_id = id(action)
        self._approval_round_decisions[action_id] = bool(approved)
        if reason:
            self._approval_round_reasons[action_id] = reason

        # Wake the loop; it will proceed only once all actions are decided.
        self._approval_event.set()

    def stop(self):
        """Fully stop the agent loop."""
        self._stop_flag.set()
        self._pause_event.set()
        self._approval_event.set()
        self._input_event.set()
        self._set_state(AgentState.DONE)

    # ── Internal Loop ──────────────────────────────────────────

    def _set_state(self, new_state: AgentState):
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.info(f"AgentLoop state: {old.name} → {new_state.name}")
            if self._state_change_cb:
                try:
                    self._state_change_cb(new_state)
                except Exception:
                    logger.exception("State change callback error")

    def _emit_thinking(self, text: str):
        if self._ui_thinking_cb:
            try:
                self._ui_thinking_cb(text)
            except Exception:
                pass

    def _emit_message(self, text: str):
        if self._ui_message_cb:
            # Deduplicate consecutive identical messages.
            if text == self._last_emitted_message:
                self._duplicate_message_count += 1
                if self._duplicate_message_count > 2:
                    return  # Suppress after 2 repeats
            else:
                self._duplicate_message_count = 0
            self._last_emitted_message = text
            try:
                self._ui_message_cb(text)
            except Exception:
                pass

    def _emit_response(self, text: str):
        if self._ui_response_cb:
            try:
                self._ui_response_cb(text)
            except Exception:
                pass

    def _get_context(self) -> Dict[str, Any]:
        if self._get_context_cb:
            try:
                return self._get_context_cb()
            except Exception:
                logger.exception("Context callback error")
        return {}

    def _check_stop(self) -> bool:
        """Return True if the loop should stop."""
        return self._stop_flag.is_set()

    def _run_loop(self, goal: str, context: Dict[str, Any]):
        """Main agent loop — runs on a background thread."""
        try:
            self._set_state(AgentState.PLANNING)
            self._emit_thinking("🧠 Planning approach...")

            # The first message is the user's goal.
            current_message = goal

            while self._iteration < self._config.max_iterations:
                if self._check_stop():
                    break

                self._iteration += 1
                self._emit_thinking(f"🔄 Step {self._iteration}...")

                # Refresh context for each iteration
                fresh_context = self._get_context()
                if fresh_context:
                    context.update(fresh_context)

                # ── PLAN: Ask the LLM for next steps ──
                # If sub-agents are enabled, delegate through the Orchestrator.
                # Otherwise fall back to the monolithic DesignAgent.chat().
                self._set_state(AgentState.PLANNING)

                assistant_message = ""
                actions: List[DesignAction] = []

                # ── Sub-agent enhanced planning ──
                # Sub-agents provide phase awareness and spatial optimisation.
                # If they produce actions, great. If not, we ALWAYS fall
                # through to the monolithic agent so the LLM stays in the
                # loop even when regex fallbacks have nothing useful.
                phase_hint = ""

                if self._use_subagents and self._orchestrator is not None:
                    try:
                        phase_label = self._orchestrator.phase.name
                        self._emit_thinking(f"🧠 Phase: {phase_label}")

                        # Build a board snapshot dict for subagent context.
                        board_snapshot = self._build_board_snapshot(context)

                        result = self._orchestrator.step(
                            goal=current_message,
                            context=context,
                            board_snapshot=board_snapshot,
                            feedback=self._last_feedback,
                        )
                        assistant_message = result.message or ""
                        actions = result.actions or []

                        # After ARRANGE phase, run overlap resolution on the
                        # board so MOVE_COMPONENT actions get smart positions.
                        if (
                            self._orchestrator.phase == DesignPhase.ARRANGE  # type: ignore
                            and not actions
                            and board_snapshot.get("components")
                            and board_snapshot.get("board_width")
                        ):
                            optimized = self._run_placement_optimization(
                                board_snapshot, context
                            )
                            if optimized:
                                actions = optimized
                                assistant_message = (
                                    "Optimizing component layout to resolve overlaps "
                                    "and improve spacing…"
                                )

                        if result.phase_complete:
                            self._emit_thinking(
                                f"✅ Phase {phase_label} complete → "
                                f"{self._orchestrator.phase.name}"
                            )

                        # Capture the phase-aware goal so the monolithic
                        # agent can benefit from phase context if needed.
                        phase_hint = self._orchestrator._build_phase_goal(
                            current_message, self._last_feedback
                        )
                    except Exception as e:
                        logger.exception("Orchestrator step failed; falling back to monolithic agent")
                        self._use_subagents = False  # disable for rest of loop

                # ── Fallback to monolithic agent when subagents produced
                #    no *actions*.  A message without actions is not useful
                #    on its own — we need the full LLM to drive the design.
                if not actions:
                    subagent_msg = assistant_message  # preserve for logging
                    assistant_message = ""
                    try:
                        # Use the phase-enhanced prompt so the monolithic
                        # agent knows what step we're on.
                        enhanced = phase_hint or current_message
                        assistant_message, request = self._agent.chat(
                            enhanced, context
                        )
                        actions = request.interpreted_actions if request else []
                    except Exception as e:
                        logger.exception("Agent chat failed")
                        self._emit_message(f"❌ Agent error: {e}")
                        self._set_state(AgentState.ERROR)
                        return
                    # If monolithic also had nothing, restore subagent msg.
                    if not assistant_message and subagent_msg:
                        assistant_message = subagent_msg

                if self._check_stop():
                    break

                # ── Check for clarifying questions ──
                if self._is_question(assistant_message, actions):
                    self._emit_message(assistant_message)
                    self._set_state(AgentState.AWAITING_INPUT)
                    self._input_event.clear()
                    self._pending_user_input = None

                    # Wait for user response
                    self._input_event.wait()
                    if self._check_stop():
                        break

                    user_answer = self._pending_user_input
                    if user_answer:
                        self._agent._append_history("user", user_answer)
                        current_message = user_answer
                        continue
                    else:
                        break

                # ── Show assistant message ──
                if assistant_message:
                    self._emit_message(assistant_message)

                # ── Check for completion signal ──
                if self._is_completion(assistant_message, actions):
                    # Only allow completion if DRC actually passed or was never run
                    if self._last_drc_passed is not False:
                        self._emit_message("✅ Design process complete!")
                        self._set_state(AgentState.DONE)
                        return
                    else:
                        # DRC hasn't passed yet — override completion and keep going
                        self._emit_thinking("⚠️ DRC has not passed yet — continuing fixes...")
                        current_message = (
                            "DRC has NOT passed yet — you cannot declare DESIGN_COMPLETE. "
                            "Use DELETE_TRACKS to clear old routing, then MOVE_COMPONENT "
                            "to fix overlaps, then AUTOROUTE_BOARD or DRAW_TRACK to re-route, "
                            "and finally RUN_DRC again."
                        )
                        continue

                # ── No actions → ask the LLM to continue ──
                if not actions:
                    current_message = (
                        "Continue with the next step of the design. "
                        "If you're done, say DESIGN_COMPLETE."
                    )
                    continue

                # ── EXECUTE actions one by one ──
                self._set_state(AgentState.EXECUTING)
                action_results: List[str] = []

                # Execute actions in order. Once we hit an approval-needed action,
                # gather all remaining approval-needed actions into a single
                # approval round and wait until the user decides all of them.
                approval_round_started = False
                approval_round_action_ids: List[int] = []

                for idx, action in enumerate(actions):
                    if self._check_stop():
                        break

                    if action.action_type == DesignActionType.UNKNOWN:
                        self._emit_message(f"🤔 Skipping unknown action: {action.description}")
                        continue

                    is_ro = is_read_only(action.action_type)

                    if (not approval_round_started) and is_ro and self._config.auto_approve_readonly:
                        # Auto-execute read-only actions
                        self._execute_and_record(action, context, action_results, auto=True, emoji="🔍")

                    else:
                        # Start approval round on the first approval-needed action.
                        if not approval_round_started:
                            approval_round_started = True

                            # YOLO mode: auto-apply everything (no approvals UI).
                            if bool(getattr(self._config, 'yolo_auto_apply', False)):
                                self._emit_message(
                                    "⚠️ YOLO mode is enabled — auto-applying actions without approval."
                                )
                                remaining = actions[idx:]
                                for a in remaining:
                                    if self._check_stop():
                                        break
                                    if a.action_type == DesignActionType.UNKNOWN:
                                        self._emit_message(
                                            f"🤔 Skipping unknown action: {a.description}"
                                        )
                                        continue
                                    emoji = "🔍" if is_read_only(a.action_type) else "⚙️"
                                    auto = is_read_only(a.action_type) and self._config.auto_approve_readonly
                                    self._execute_and_record(a, context, action_results, auto=auto, emoji=emoji)

                                break

                            self._set_state(AgentState.AWAITING_APPROVAL)
                            self._approval_event.clear()
                            self._approval_round_actions = []
                            self._approval_round_decisions = {}
                            self._approval_round_reasons = {}

                            remaining = actions[idx:]
                            for a in remaining:
                                a_is_ro = is_read_only(a.action_type)
                                needs_approval = (not (a_is_ro and self._config.auto_approve_readonly))
                                if not needs_approval:
                                    continue
                                if a.action_type == DesignActionType.UNKNOWN:
                                    continue
                                self._approval_round_actions.append(a)
                                approval_round_action_ids.append(id(a))

                            # Emit previews for the whole round
                            for a in self._approval_round_actions:
                                preview = a.preview_text or self._agent.create_preview(a, context)
                                if self._ui_action_preview_cb:
                                    try:
                                        self._ui_action_preview_cb(
                                            a.action_type.name,
                                            a.description,
                                            preview,
                                            a,
                                        )
                                    except Exception:
                                        logger.exception("Action preview callback error")

                            # Wait for decisions, and execute approved actions as they
                            # become decided (in the original order). The model will
                            # only proceed once all actions in the round are decided.
                            cursor = 0
                            while not self._check_stop() and cursor < len(self._approval_round_actions):
                                a = self._approval_round_actions[cursor]
                                aid = id(a)
                                if aid not in self._approval_round_decisions:
                                    self._approval_event.wait(timeout=0.25)
                                    self._approval_event.clear()
                                    continue

                                approved = bool(self._approval_round_decisions.get(aid, False))
                                if approved:
                                    self._execute_and_record(a, context, action_results, emoji="⚙️")
                                else:
                                    reason = self._approval_round_reasons.get(aid) or "User rejected"
                                    self._emit_message(f"⏭️ Skipped: {a.description} ({reason})")
                                    action_results.append(
                                        f"[{a.action_type.name}] REJECTED: {reason}"
                                    )

                                cursor += 1

                            if self._check_stop():
                                break

                            # After the approval round is fully decided, execute any
                            # remaining auto-approved read-only actions from the same
                            # LLM response (preserving their order).
                            approval_ids = set(approval_round_action_ids)
                            for a in remaining:
                                if self._check_stop():
                                    break
                                if id(a) in approval_ids:
                                    continue
                                if a.action_type == DesignActionType.UNKNOWN:
                                    self._emit_message(
                                        f"🤔 Skipping unknown action: {a.description}"
                                    )
                                    continue

                                a_is_ro = is_read_only(a.action_type)
                                if a_is_ro and self._config.auto_approve_readonly:
                                    self._execute_and_record(a, context, action_results, auto=True, emoji="🔍")

                            # We fully handled the remainder.
                            break

                        # If approval_round_started is already True, we should never
                        # reach this branch because the remainder loop breaks out.
                        continue

                if self._check_stop():
                    break

                # ── OBSERVE: Feed results back to LLM ──
                self._set_state(AgentState.OBSERVING)

                # Check if the last action was DRC and it passed
                last_drc = self._find_last_drc_result(action_results)
                if last_drc and last_drc.get("passed"):
                    self._last_drc_passed = True
                    self._emit_message("✅ DRC passed — no errors found! Design is complete.")
                    self._set_state(AgentState.DONE)
                    return

                # Track DRC failures for escalating guidance
                if last_drc and not last_drc.get("passed"):
                    self._last_drc_passed = False
                    self._drc_retry_count += 1

                # Build feedback for next iteration
                results_summary = self._compact_action_results(action_results)
                if self._drc_retry_count > 0:
                    drc_hint = self._build_drc_hint(action_results)
                else:
                    drc_hint = ""

                if self._drc_retry_count >= self._config.max_drc_retries:
                    self._emit_message(
                        f"⚠️ Reached max DRC retries ({self._config.max_drc_retries}). "
                        "Try adjusting the design manually."
                    )
                    self._set_state(AgentState.DONE)
                    return

                current_message = (
                    f"Results from the previous step:\n{results_summary}"
                    f"{drc_hint}\n\n"
                    "Continue with the next step of the design. "
                    "If the design is complete and DRC passes, say DESIGN_COMPLETE. "
                    "If DRC found errors, propose fixes."
                )
                if len(current_message) > 8000:
                    current_message = current_message[:8000] + "\n...[feedback truncated]"
                # Store feedback for the orchestrator's subagent delegation.
                self._last_feedback = current_message

            # Max iterations reached
            if self._iteration >= self._config.max_iterations:
                self._emit_message(
                    f"⚠️ Reached maximum iterations ({self._config.max_iterations}). "
                    "The design may not be complete. You can continue manually."
                )

            if self._state not in (AgentState.DONE, AgentState.ERROR):
                self._set_state(AgentState.PAUSED if self._stop_flag.is_set() else AgentState.DONE)

        except Exception as e:
            logger.exception(f"Agent loop error: {e}")
            self._emit_message(f"❌ Agent loop error: {e}")
            self._set_state(AgentState.ERROR)

    # ── Subagent helpers ────────────────────────────────────────

    def _build_board_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build a lightweight board snapshot dict for subagent context."""
        snap: Dict[str, Any] = {}
        snap["components_placed"] = self._phase.get("components_placed", 0)
        snap["outline_defined"] = self._phase.get("outline_defined", False)
        snap["nets_assigned"] = self._phase.get("nets_assigned", 0)
        snap["net_assign_warnings"] = self._phase.get("net_assign_warnings", 0)
        snap["routing_attempted"] = self._phase.get("routing_attempted", False)

        # Try to extract component list from pcb_data in context.
        pcb_data = context.get("pcb_data")
        components: List[Dict[str, Any]] = []
        if pcb_data is not None:
            try:
                for fp in getattr(pcb_data, "footprints", []) or []:
                    ref = str(getattr(fp, "reference", "") or "")
                    val = str(getattr(fp, "value", "") or "")
                    x = getattr(fp, "x", 0.0)
                    y = getattr(fp, "y", 0.0)
                    w = getattr(fp, "width", 10.0)
                    h = getattr(fp, "height", 10.0)
                    components.append({
                        "reference": ref, "value": val,
                        "x": float(x), "y": float(y),
                        "width": float(w), "height": float(h),
                    })
            except Exception:
                pass
        snap["components"] = components

        # Board dimensions.
        snap["board_width"] = context.get("board_width")
        snap["board_height"] = context.get("board_height")
        snap["search_part_results"] = context.get("search_part_results", {})

        return snap

    def _run_placement_optimization(
        self,
        board_snapshot: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[DesignAction]:
        """Use PlacementAgent to resolve overlaps and produce MOVE_COMPONENT actions."""
        if not _SUBAGENTS_AVAILABLE:
            return []

        components = board_snapshot.get("components", [])
        bw = board_snapshot.get("board_width")
        bh = board_snapshot.get("board_height")
        if not components or not bw or not bh:
            return []

        try:
            optimized = PlacementAgent.resolve_overlaps(components, clearance_mm=2.0)

            # Compare old vs new positions, generate MOVE_COMPONENT actions.
            move_actions: List[DesignAction] = []
            for old, new in zip(components, optimized):
                dx = abs(float(old.get("x", 0)) - float(new.get("x", 0)))
                dy = abs(float(old.get("y", 0)) - float(new.get("y", 0)))
                if dx > 0.5 or dy > 0.5:  # moved more than 0.5 mm
                    ref = old.get("reference", "?")
                    new_x = round(float(new["x"]), 2)
                    new_y = round(float(new["y"]), 2)
                    move_actions.append(DesignAction(
                        action_type=DesignActionType.MOVE_COMPONENT,
                        description=f"Move {ref} to ({new_x}, {new_y}) to resolve overlap",
                        parameters={"ref": ref, "location": f"{new_x},{new_y}"},
                        requires_approval=True,
                    ))

            return move_actions
        except Exception:
            logger.exception("Placement optimization failed")
            return []

    def _execute_action(self, action: DesignAction, context: Dict[str, Any]) -> DesignAction:
        """Execute a single action.

        If there's a GUI callback for pcbnew-touching actions, use it.
        Otherwise fall back to direct async execution.
        """
        import asyncio

        # For pcbnew-modifying actions, delegate to the GUI thread
        gui_actions = {
            DesignActionType.ADD_COMPONENT,
            DesignActionType.DRAW_TRACK,
            DesignActionType.ASSIGN_NETS,
            DesignActionType.MOVE_COMPONENT,
            DesignActionType.ROTATE_COMPONENT,
            DesignActionType.ADD_VIA,
            DesignActionType.DEFINE_BOARD_OUTLINE,
            DesignActionType.ADD_MOUNTING_HOLE,
            DesignActionType.ALIGN_COMPONENTS,
            DesignActionType.ADD_POLYGON,
            DesignActionType.ADD_TEXT,
            DesignActionType.AUTOROUTE_BOARD,
            DesignActionType.SET_LAYER_COUNT,
            DesignActionType.DELETE_TRACKS,
            DesignActionType.DELETE_COMPONENT,
        }

        if action.action_type in gui_actions and self._execute_on_gui_cb:
            result_event = threading.Event()
            result_holder = [None]

            def on_result(executed_action):
                result_holder[0] = executed_action
                result_event.set()

            try:
                self._execute_on_gui_cb(action, context, on_result)
                result_event.wait(timeout=60)  # 60s timeout for GUI execution
                if result_holder[0]:
                    return result_holder[0]
                else:
                    action.success = False
                    action.result_message = "GUI execution timed out"
                    action.executed = True
                    return action
            except Exception as e:
                action.success = False
                action.result_message = f"GUI execution failed: {e}"
                action.executed = True
                return action

        # Non-GUI actions: execute directly
        try:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    self._agent.execute_action(action, context)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            action.success = False
            action.result_message = f"Execution error: {e}"
            action.executed = True
            return action

    def _execute_and_record(self, action: DesignAction, context: Dict[str, Any],
                            action_results: List[str], auto: bool = False,
                            emoji: str = "🔍") -> None:
        """Execute an action, record the result, and emit status to the UI."""
        # De-duplicate repeated SEARCH_PART loops that flood logs/context.
        if action.action_type == DesignActionType.SEARCH_PART:
            query = ""
            try:
                query = str(self._agent._extract_search_query(action)).strip().lower()  # type: ignore[attr-defined]
            except Exception:
                params = action.parameters or {}
                if isinstance(params, dict):
                    query = str(params.get("query", "") or "").strip().lower()
            if query and query in self._seen_search_queries:
                msg = f"Skipping duplicate SEARCH_PART query: {query}"
                self._emit_message(f"⏭️ {msg}")
                action_results.append(f"[SEARCH_PART] SKIPPED: {msg}")
                return
            if query:
                self._seen_search_queries.add(query)

        # Phase gate: check prerequisites before executing
        gate_msg = self._check_phase_gate(action)
        if gate_msg:
            self._emit_message(f"⛔ {gate_msg}")
            action_results.append(f"[{action.action_type.name}] BLOCKED: {gate_msg}")
            return

        action.approved = True
        self._emit_thinking(f"{emoji} {action.description}...")
        self._execute_action(action, context)

        # Update phase tracking after execution
        self._update_phase_state(action)

        self._history.append(LoopStep(
            iteration=self._iteration,
            action=action,
            result_success=action.success,
            result_message=action.result_message,
            was_auto_executed=auto,
        ))
        result_text = action.result_message or "Done"
        status = "✅" if action.success else "❌"
        self._emit_message(f"{status} {result_text}")
        tag = 'OK' if action.success else 'FAILED'
        action_results.append(f"[{action.action_type.name}] {tag}: {result_text}")

    @staticmethod
    def _compact_action_results(action_results: List[str], max_lines: int = 30) -> str:
        """Keep feedback compact so iterative prompts don't grow unbounded."""
        if not action_results:
            return "No actions executed."
        if len(action_results) <= max_lines:
            return "\n".join(action_results)

        head = action_results[: max_lines // 2]
        tail = action_results[-(max_lines - len(head)) :]
        omitted = len(action_results) - len(head) - len(tail)
        return "\n".join(head + [f"... ({omitted} result line(s) omitted) ..."] + tail)

    # ── Helpers ──────────────────────────────────────────────────

    def _is_question(self, assistant_message: str, actions: List[DesignAction]) -> bool:
        """Detect if the LLM is asking a clarifying question."""
        if actions:
            return False
        if not assistant_message:
            return False
        msg = assistant_message.strip().lower()
        # Heuristic: if the message ends with a question mark or contains
        # explicit question patterns with no actions
        if msg.endswith("?"):
            return True
        question_patterns = [
            r'\bwhich\b.*\bprefer\b',
            r'\bwhat\b.*\bwould you\b',
            r'\bcould you\b.*\bspecify\b',
            r'\bplease\b.*\bchoose\b',
            r'\bdo you want\b',
            r'\bshould i\b',
            r'\bwhat.*\bshould\b',
        ]
        for pat in question_patterns:
            if re.search(pat, msg):
                return True
        return False

    def _is_completion(self, assistant_message: str, actions: List[DesignAction]) -> bool:
        """Detect if the agent is signaling completion."""
        if not assistant_message:
            return False
        msg = assistant_message.strip().upper()
        if "DESIGN_COMPLETE" in msg:
            return True
        # Also check for explicit completion phrases
        lower = assistant_message.strip().lower()
        completion_phrases = [
            "design is complete",
            "design process is complete",
            "all steps have been completed",
            "the design is finished",
        ]
        return any(phrase in lower for phrase in completion_phrases)

    def _find_last_drc_result(self, action_results: List[str]) -> Optional[Dict]:
        """Check if the most recent DRC action passed."""
        for result in reversed(action_results):
            if "RUN_DRC" in result:
                lower = result.lower()
                if "drc_status: pass" in lower:
                    passed = True
                elif "drc_status: fail" in lower:
                    passed = False
                else:
                    passed = "OK" in result and ("0 error" in lower or "no error" in lower or "passed" in lower)
                return {"passed": passed, "text": result}
        return None

    @staticmethod
    def action_type_is_destructive(action_type: DesignActionType) -> bool:
        """Return True if the action type modifies the board."""
        return action_type in DESTRUCTIVE_ACTIONS

    # ── Phase-gate logic ─────────────────────────────────────────

    def _check_phase_gate(self, action: DesignAction) -> Optional[str]:
        """Check if an action's prerequisites are met.

        Returns None if the action is allowed, or a feedback string to inject
        back to the LLM instead of executing the action.
        """
        at = action.action_type
        params = getattr(action, 'parameters', None) or {}

        # ── Death-loop detection: block repeated add/delete cycles ──
        if at == DesignActionType.ADD_COMPONENT:
            query = ''
            for key in ('query', 'part_name', 'mpn', 'part', 'name'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    query = v.strip().lower()
                    break
            if query:
                count = self._component_add_count.get(query, 0)
                if count >= 3:
                    return (
                        f"STOP: already tried to add '{query}' {count} times. "
                        "The correct footprint may not be available. "
                        "Skip this component and move on to the next design phase."
                    )
            # Block if we have had 5+ consecutive footprint load failures
            if self._consecutive_fp_failures >= 5:
                return (
                    "STOP: the last 5 component placements failed due to footprint loading errors. "
                    "The pcbnew library state may be unstable. "
                    "Proceed to DEFINE_BOARD_OUTLINE or net assignment with the components already on the board."
                )

        if at == DesignActionType.DELETE_COMPONENT:
            ref = ''
            for key in ('reference', 'ref', 'component'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    ref = v.strip()
                    break
            if ref:
                count = self._component_delete_count.get(ref, 0)
                if count >= 2:
                    return (
                        f"STOP: already deleted '{ref}' {count} times in this session. "
                        "Do not delete and re-add the same component repeatedly. "
                        "Move on to the next design phase instead."
                    )

        if at == DesignActionType.AUTOROUTE_BOARD:
            if not self._phase.get('outline_defined'):
                return (
                    "Cannot AUTOROUTE_BOARD yet: no board outline has been defined. "
                    "Please use DEFINE_BOARD_OUTLINE first, then ensure all components "
                    "are placed inside the outline and nets are assigned."
                )
            if self._phase.get('nets_assigned', 0) == 0:
                return (
                    "Cannot AUTOROUTE_BOARD yet: no nets have been assigned to pads. "
                    "Use DEFINE_NET or ASSIGN_NETS to connect pads to nets (GND, +5V, signals, etc.) "
                    "before routing."
                )
            warn_ratio = (
                self._phase.get('net_assign_warnings', 0)
                / max(self._phase.get('nets_assigned', 0) + self._phase.get('net_assign_warnings', 0), 1)
            )
            if warn_ratio > 0.5 and self._phase.get('nets_assigned', 0) < 5:
                return (
                    "Cannot AUTOROUTE_BOARD yet: too many net assignment failures "
                    f"({self._phase.get('net_assign_warnings', 0)} warnings vs "
                    f"{self._phase.get('nets_assigned', 0)} successful). "
                    "This usually means the footprints have different pad numbers than expected. "
                    "Check the pad-not-found errors above and fix assignments before routing."
                )

        if at == DesignActionType.RUN_DRC:
            if not self._phase.get('outline_defined'):
                return (
                    "Cannot RUN_DRC yet: no board outline has been defined. "
                    "DRC needs a board outline to check edge clearances. "
                    "Use DEFINE_BOARD_OUTLINE first."
                )

        return None  # action is allowed

    def _update_phase_state(self, action: DesignAction) -> None:
        """Update phase tracking after an action is executed."""
        if not action.executed:
            return
        at = action.action_type
        ok = bool(action.success)
        msg = (action.result_message or '').lower()
        params = getattr(action, 'parameters', None) or {}

        if at == DesignActionType.ADD_COMPONENT:
            # Track add attempts per query (for death-loop detection)
            query = ''
            for key in ('query', 'part_name', 'mpn', 'part', 'name'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    query = v.strip().lower()
                    break
            if query:
                self._component_add_count[query] = self._component_add_count.get(query, 0) + 1
            # Track consecutive footprint failures
            if ok:
                self._phase['components_placed'] = self._phase.get('components_placed', 0) + 1
                self._consecutive_fp_failures = 0  # reset on success
            elif 'failed to load footprint' in msg or 'no footprint found' in msg:
                self._consecutive_fp_failures += 1

        elif at == DesignActionType.DELETE_COMPONENT:
            ref = ''
            for key in ('reference', 'ref', 'component'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    ref = v.strip()
                    break
            if ref:
                self._component_delete_count[ref] = self._component_delete_count.get(ref, 0) + 1

        elif at == DesignActionType.DEFINE_BOARD_OUTLINE and ok:
            self._phase['outline_defined'] = True

        elif at in (DesignActionType.DEFINE_NET, DesignActionType.ASSIGN_NETS):
            # Parse the result message for assigned/warning counts
            import re as _re
            assigned_m = _re.search(r'(\d+)\s+pad', msg)
            warn_m = _re.search(r'(\d+)\s+item\(s\)\s+could not', msg)
            if ok and assigned_m:
                self._phase['nets_assigned'] = (
                    self._phase.get('nets_assigned', 0) + int(assigned_m.group(1))
                )
            if warn_m:
                self._phase['net_assign_warnings'] = (
                    self._phase.get('net_assign_warnings', 0) + int(warn_m.group(1))
                )

        elif at == DesignActionType.AUTOROUTE_BOARD:
            self._phase['routing_attempted'] = True

    def _build_drc_hint(self, action_results: List[str]) -> str:
        """Analyse DRC error text and return targeted fix guidance."""
        # Collect all DRC result text
        drc_text = ''
        for r in action_results:
            if 'RUN_DRC' in r:
                drc_text = r
                break
        lower = drc_text.lower()

        hints: List[str] = []
        hints.append(
            f"\n\nDRC FIX ATTEMPT {self._drc_retry_count}/{self._config.max_drc_retries}:"
        )

        # Categorise errors and give specific guidance
        has_edge = 'board edge clearance' in lower
        has_courtyard = 'courtyard' in lower and 'overlap' in lower
        has_clearance = 'clearance violation' in lower
        has_missing = 'missing connection' in lower
        has_solder_mask = 'solder mask' in lower and 'bridge' in lower
        has_short = 'short' in lower

        if has_edge:
            hints.append(
                "BOARD EDGE: Components or pads are too close to board edge. "
                "MOVE_COMPONENT to bring them at least 1mm inside the outline."
            )
        if has_courtyard:
            hints.append(
                "COURTYARD OVERLAP: Components are physically overlapping. "
                "MOVE_COMPONENT to increase spacing between the overlapping parts "
                "(minimum 1-2mm gap between courtyard edges)."
            )
        if has_clearance or has_solder_mask:
            hints.append(
                "CLEARANCE/SOLDER MASK: Traces or pads are too close together. "
                "DELETE_TRACKS, MOVE_COMPONENT to create more space, then AUTOROUTE_BOARD."
            )
        if has_missing:
            hints.append(
                "MISSING CONNECTIONS: Some nets are not fully routed. "
                "Check that all required nets are assigned with ASSIGN_NETS, then re-route."
            )
        if has_short:
            hints.append(
                "SHORT CIRCUIT: Traces from different nets are touching. "
                "DELETE_TRACKS and re-route carefully."
            )

        # Escalation for repeated failures
        if self._drc_retry_count >= 3:
            hints.append(
                "ESCALATION (attempt {}+): Start fresh - DELETE_TRACKS to clear ALL routing, "
                "then MOVE_COMPONENT to fix ALL overlaps and edge violations, verify placement "
                "is clean, then AUTOROUTE_BOARD, then RUN_DRC.".format(self._drc_retry_count)
            )

        # If no specific errors detected, give generic advice
        if len(hints) == 1:
            hints.append(
                "Fix DRC errors: 1) DELETE_TRACKS 2) MOVE_COMPONENT to fix overlaps "
                "3) AUTOROUTE_BOARD 4) RUN_DRC."
            )

        return ' '.join(hints)
