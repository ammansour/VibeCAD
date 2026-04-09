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
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from .design_agent import DesignAgent, DesignAction, DesignActionType, normalize_action_parameters

# Sub-agent orchestrator (optional — graceful fallback if not present).
try:
    from .sub_agents.orchestrator import Orchestrator
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
    component_by_component_placement: bool = False
    # When placing many parts, execute in batches rather than one-by-one replans.
    # 0 disables batching (default: disabled for orchestrated pre-computed placement).
    placement_batch_size: int = 0
    # If None, the loop asks the LLM to classify whether this goal needs the
    # "full workflow" (placement + netting + routing). This avoids deterministic
    # heuristics for universal behavior.
    require_full_workflow: Optional[bool] = None


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

        # Machine-readable artifacts produced during the run (SPEC/GEOM/BIND).
        # Kept in-memory; may be surfaced by the UI later.
        self._artifacts: Dict[str, Any] = {}

        # State
        self._state = AgentState.IDLE
        self._iteration = 0
        self._drc_retry_count = 0
        # Deterministic grid/packing mode for overlap resolution (ARRANGE).
        # In this mode, allow unlimited DRC retries and throttle UI output.
        self._deterministic_grid_phase_active: bool = False
        self._det_grid_drc_ui_counter: int = 0
        self._last_drc_passed: Optional[bool] = None  # None = never run
        self._nets_dirty_for_connectivity_drc: bool = False
        self._history: List[LoopStep] = []
        self._goal: str = ""

        # Phase tracking — soft gates to prevent out-of-order operations
        self._phase = {
            'components_placed': 0,     # count of successfully placed components
            # True once the PLACE-phase audit declares the component set complete.
            # This avoids freezing placement just because early placement DRC is clean.
            'component_set_verified': False,
            'outline_defined': False,
            'nets_assigned': 0,         # count of successful net assignments
            'net_assign_warnings': 0,   # count of net assignment warnings/failures
            'routing_attempted': False,
            # Last parsed DRC counts keyed by focus ("placement"/"connectivity"/...).
            'drc_last': {},
            # Tracks RUN_DRC focus="placement" results (None = never run).
            'placement_drc_passed': None,
            # Tracks RUN_DRC focus="connectivity" results (None = never run).
            'connectivity_drc_passed': None,
        }
        self._require_full_workflow: bool = False
        self._workflow_router_reason: str = ""
        # Track add/delete cycles per component to detect death loops
        self._component_add_fail_count: Dict[str, int] = {}   # query -> failed add attempts
        self._component_delete_count: Dict[str, int] = {} # ref -> delete count
        self._consecutive_fp_failures: int = 0            # consecutive footprint load failures
        self._search_query_last_iteration: Dict[str, int] = {}
        self._search_query_cooldown_steps: int = 2
        self._max_search_actions_per_step: int = 32
        self._ran_placement_drc_this_iteration: bool = False
        self._ran_routing_drc_this_iteration: bool = False
        self._placements_executed_this_iteration: int = 0
        self._needs_post_placement_drc: bool = False
        self._abort_remaining_actions_this_iteration: bool = False
        # Monotonic board-change counter used to detect unchanged DRC reruns.
        self._board_mutation_epoch: int = 0
        # Per-focus DRC diff/fingerprint tracking.
        self._drc_diff_state: Dict[str, Dict[str, Any]] = {}

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
                # Monolithic fallback is intentionally disabled (see planning loop).
                logger.exception("Failed to initialise sub-agent orchestrator")
                self._orchestrator = None
                self._use_subagents = False

        # Message deduplication — prevent spamming the same error 50x.
        self._last_emitted_message: str = ""
        self._duplicate_message_count: int = 0
        # Per-run ledger used to suppress repeated idempotent actions
        # (e.g. repeated DOWNLOAD_FOOTPRINT messages when the planner loops).
        self._action_ledger: Dict[str, Dict[str, Any]] = {}

        # Callbacks
        self._ui_message_cb: Optional[Callable[[str], None]] = None
        self._ui_thinking_cb: Optional[Callable[[str], None]] = None
        self._ui_action_preview_cb: Optional[Callable] = None
        self._ui_response_cb: Optional[Callable[[str], None]] = None
        self._state_change_cb: Optional[Callable[[AgentState], None]] = None
        self._phase_complete_cb: Optional[Callable[[str, Any], None]] = None
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

    def set_phase_complete_callback(self, cb: Callable[[str, Any], None]):
        """Callback when a sub-agent phase finishes."""
        self._phase_complete_cb = cb

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
        try:
            reset_usage = getattr(self._agent, "reset_llm_usage_stats", None)
            if callable(reset_usage):
                reset_usage()
        except Exception:
            logger.debug("Failed to reset per-run LLM usage stats", exc_info=True)
        # Workflow mode will be determined inside the worker thread (via LLM),
        # unless explicitly overridden by config.
        self._require_full_workflow = False
        self._workflow_router_reason = ""
        self._iteration = 0
        self._drc_retry_count = 0
        self._last_drc_passed = None
        self._history.clear()
        self._stop_flag.clear()
        self._pause_event.set()
        self._artifacts = {}
        # Reset phase tracking
        self._phase = {
            'components_placed': 0,
            'component_set_verified': False,
            'outline_defined': False,
            'nets_assigned': 0,
            'net_assign_warnings': 0,
            'routing_attempted': False,
            'drc_last': {},
            'placement_drc_passed': None,
            'connectivity_drc_passed': None,
        }
        self._component_add_fail_count = {}
        self._component_delete_count = {}
        self._consecutive_fp_failures = 0
        self._search_query_last_iteration = {}
        self._last_feedback: Optional[str] = None
        self._last_emitted_message = ""
        self._duplicate_message_count = 0
        self._action_ledger = {}
        self._board_mutation_epoch = 0
        self._drc_diff_state = {}

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

    @staticmethod
    def _display_phase_label(label: str) -> str:
        """Map internal phase labels to less misleading UI labels."""
        s = str(label or "").strip().upper()
        if s == "FAIL":
            return "RETRY"
        return s or "WORKFLOW"

    def _emit_message(self, text: str):
        # Deduplicate consecutive identical messages (both UI and debug logs).
        suppressed = False
        if text == self._last_emitted_message:
            self._duplicate_message_count += 1
            if self._duplicate_message_count > 2:
                suppressed = True  # Suppress after 2 repeats
        else:
            self._duplicate_message_count = 0
        self._last_emitted_message = text

        # Ensure user-visible error/warning lines also show up in debug logs.
        if (not suppressed) and str(text).startswith(("❌", "⛔", "⏸️")):
            try:
                logger.debug(text)
            except Exception:
                pass

        if self._ui_message_cb and (not suppressed):
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

    def _emit_phase_complete(self, phase_name: str, result: Any):
        if self._phase_complete_cb:
            try:
                self._phase_complete_cb(str(phase_name or ""), result)
            except Exception:
                logger.exception("Phase completion callback error")

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

    def _action_ledger_key(self, action: DesignAction) -> Optional[str]:
        """Return a stable key for idempotent actions we want to de-duplicate."""
        try:
            at = action.action_type
            params = action.parameters or {}
        except Exception:
            return None

        if not isinstance(params, dict):
            params = {}

        # Only de-duplicate idempotent-ish actions where repeating is nearly
        # always noise. Do NOT include ADD_COMPONENT/DELETE_COMPONENT/etc.
        if at == DesignActionType.SEARCH_PART:
            q = str(params.get("query", "") or "").strip()
            if not q:
                # Fall back to the same robust extraction used elsewhere.
                try:
                    q = str(self._agent._extract_search_query(action) or "").strip()  # type: ignore[attr-defined]
                except Exception:
                    q = ""
            qn = self._normalize_search_query_key(q)
            return f"SEARCH_PART:{qn}" if qn else None

        if at == DesignActionType.SEARCH_WEB:
            q = str(params.get("query", "") or "").strip()
            qn = self._normalize_search_query_key(q)
            return f"SEARCH_WEB:{qn}" if qn else None

        if at == DesignActionType.LOOKUP_DATASHEET:
            mpn = self._normalize_query_key(str(params.get("mpn", "") or ""))
            return f"LOOKUP_DATASHEET:{mpn}" if mpn else None

        if at in {DesignActionType.DOWNLOAD_SYMBOL, DesignActionType.DOWNLOAD_FOOTPRINT}:
            pn = self._normalize_query_key(str(params.get("part_name", "") or params.get("query", "") or ""))
            if not pn:
                return None
            if at == DesignActionType.DOWNLOAD_SYMBOL:
                return f"DOWNLOAD_SYMBOL:{pn}"
            pkg = self._normalize_query_key(str(params.get("package", "") or ""))
            return f"DOWNLOAD_FOOTPRINT:{pn}|{pkg}" if pkg else f"DOWNLOAD_FOOTPRINT:{pn}"

        return None

    @staticmethod
    def _normalize_search_query_key(value: str) -> str:
        """Normalize search-like queries so trivial wording changes de-duplicate.

        Intended to collapse variants like:
          - "DC-005 2.1mm barrel jack" vs "... barrel jack symbol"
          - "USB-B Female Through Hole Connector" vs "USB_B symbol"

        This is deliberately more aggressive than _normalize_query_key; it is
        only used for SEARCH_PART/SEARCH_WEB keys and related de-duplication.
        """
        text = str(value or "").strip().lower()
        if not text:
            return ""
        # Normalize separators.
        text = text.replace("_", " ")
        text = text.replace("-", " ")
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize (keep common unit suffixes).
        tokens = re.findall(r"[a-z0-9]+(?:\.\d+)?(?:mm|mil|mhz|khz|ghz)?", text)
        if not tokens:
            return ""

        stop = {
            "symbol",
            "footprint",
            "footprints",
            "library",
            "kicad",
            "part",
            "parts",
            "component",
            "components",
            # Generic connector descriptors that tend to cause repeated searches.
            "connector",
            "connectors",
            "through",
            "hole",
            "throughhole",
            "female",
            "male",
            "type",
            "vertical",
            "horizontal",
            "pth",
            "tht",
            "smd",
        }
        out: List[str] = []
        seen = set()
        for t in tokens:
            if t in stop:
                continue
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return " ".join(out).strip()

    def _ledger_succeeded(self, key: str) -> bool:
        """Return True if *key* (or a safe alias) is marked succeeded in ledger."""
        if not key:
            return False
        try:
            entry = self._action_ledger.get(key) or {}
            status = str(entry.get("status", "") or "").strip().lower()
            if status == "succeeded":
                # Only de-dup searches that actually yielded results; allow follow-up
                # refinement after "No parts found".
                if key.startswith(("SEARCH_PART:", "SEARCH_WEB:")) and not bool(entry.get("has_results", False)):
                    return False
                return True
        except Exception:
            pass

        # For DOWNLOAD_FOOTPRINT/DOWNLOAD_SYMBOL, treat package-hint variations
        # as duplicates *only when* the prior success indicated no download was
        # needed (i.e. already available locally). This avoids blocking genuine
        # attempts to fetch/resolve an alternate footprint when the first one
        # was wrong.
        if "|" in key and key.startswith(("DOWNLOAD_FOOTPRINT:", "DOWNLOAD_SYMBOL:")):
            base = key.split("|", 1)[0]
            try:
                entry = self._action_ledger.get(base) or {}
                if str(entry.get("status", "") or "").strip().lower() == "succeeded" and bool(entry.get("no_download_needed", False)):
                    return True
            except Exception:
                pass

        # Library-prefix alias: for download actions, allow matching on the
        # suffix token (e.g. "Connector_USB:USB_B_TE_..." vs "USB_B_TE_...").
        #
        # Only apply this alias when the suffix looks specific (contains digits
        # and is reasonably long) to avoid accidental collisions.
        if key.startswith(("DOWNLOAD_SYMBOL:", "DOWNLOAD_FOOTPRINT:")):
            payload = key.split(":", 1)[1] if ":" in key else ""
            payload = payload.split("|", 1)[0] if "|" in payload else payload
            if ":" in payload:
                suffix = payload.rsplit(":", 1)[-1].strip()
                if len(suffix) >= 8 and re.search(r"\d", suffix):
                    alt = key.split(":", 1)[0] + ":" + suffix
                    try:
                        entry = self._action_ledger.get(alt) or {}
                        if str(entry.get("status", "") or "").strip().lower() == "succeeded":
                            return True
                    except Exception:
                        pass

        return False

    def _filter_duplicate_planned_actions(
        self, actions: List[DesignAction], context: Dict[str, Any]
    ) -> Tuple[List[DesignAction], int]:
        """Drop actions that already succeeded earlier in this run."""
        if not actions:
            return actions, 0

        # Expose the ledger to planners/previews (best-effort).
        try:
            context["action_ledger"] = self._action_ledger
        except Exception:
            pass

        out: List[DesignAction] = []
        skipped = 0
        seen_in_this_plan: set = set()
        for a in actions:
            key = self._action_ledger_key(a)
            if not key:
                out.append(a)
                continue
            # De-dup within a single proposed plan.
            if key in seen_in_this_plan:
                skipped += 1
                continue
            seen_in_this_plan.add(key)

            if self._ledger_succeeded(key):
                skipped += 1
                continue
            out.append(a)
        return out, skipped

    def _synthesize_net_actions_from_artifacts(
        self,
        *,
        phase_name: str,
        actions: List[DesignAction],
        result: Any,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> List[DesignAction]:
        """Fallback: recover NET execution actions from net_plan artifacts.

        This guards against rare planner/control-path issues where the NET phase
        computes a valid net_plan but returns an empty action list.
        """
        if str(phase_name or "").upper() != "NET":
            return actions
        if actions:
            return actions

        artifacts: Dict[str, Any] = {}
        try:
            if isinstance(getattr(result, "artifacts", None), dict):
                artifacts = dict(getattr(result, "artifacts", {}) or {})
        except Exception:
            artifacts = {}
        if not artifacts:
            artifacts = self._artifacts if isinstance(self._artifacts, dict) else {}
        net_plan = artifacts.get("net_plan") if isinstance(artifacts.get("net_plan"), dict) else {}
        if not net_plan:
            net_plan = context.get("net_plan") if isinstance(context.get("net_plan"), dict) else {}
        if not isinstance(net_plan, dict) or not net_plan:
            return actions

        assignment_rows: List[Dict[str, str]] = []
        for row in list(net_plan.get("assignments") or []):
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref", "") or "").strip()
            pad = str(row.get("pad", "") or "").strip()
            net = str(row.get("net", "") or "").strip()
            if not ref or not pad or not net:
                continue
            assignment_rows.append({"ref": ref, "pad": pad, "net": net})

        routing_ready = bool(net_plan.get("routing_ready", False))
        snap = board_snapshot if isinstance(board_snapshot, dict) else {}
        routing_attempted = bool(snap.get("routing_attempted", False))
        include_autoroute = routing_ready and (not routing_attempted)
        if not assignment_rows and not include_autoroute:
            return actions

        synthesized: List[DesignAction] = []
        chunk_size = 96
        if assignment_rows:
            total_chunks = max(1, (len(assignment_rows) + max(chunk_size, 1) - 1) // max(chunk_size, 1))
            for idx in range(total_chunks):
                chunk = assignment_rows[idx * chunk_size:(idx + 1) * chunk_size]
                nets = sorted({str(row.get("net", "") or "") for row in chunk if str(row.get("net", "") or "")})
                synthesized.append(
                    DesignAction(
                        action_type=DesignActionType.ASSIGN_NETS,
                        description=(
                            f"Assign nets to {len(chunk)} pad(s)"
                            + (f" (chunk {idx + 1}/{total_chunks})" if total_chunks > 1 else "")
                        ),
                        parameters={"assignments": chunk},
                        requires_approval=False,
                        preview_text=(
                            f"Assign {len(chunk)} pad(s) across {len(nets)} net(s): "
                            + ", ".join(nets[:12])
                        ),
                    )
                )
        if include_autoroute:
            synthesized.append(
                DesignAction(
                    action_type=DesignActionType.AUTOROUTE_BOARD,
                    description="Autoroute the assigned nets with Freerouting",
                    parameters={"router": "freerouting"},
                    requires_approval=False,
                )
            )
        if synthesized:
            logger.warning(
                "NET fallback: synthesized %d action(s) from net_plan artifacts (assignments=%d routing_ready=%s attempted=%s)",
                len(synthesized),
                len(assignment_rows),
                "yes" if routing_ready else "no",
                "yes" if routing_attempted else "no",
            )
            self._emit_thinking(
                f"🛠️ NET fallback recovered {len(synthesized)} action(s) from planned net artifacts."
            )
            return synthesized
        return actions

    def _record_ledger_result(self, action: DesignAction) -> None:
        """Update per-run ledger based on an executed action result."""
        key = self._action_ledger_key(action)
        if not key:
            return
        try:
            prev = self._action_ledger.get(key) or {}
            count = int(prev.get("count", 0) or 0) + 1
        except Exception:
            count = 1
        msg = str(getattr(action, "result_message", "") or "").strip()
        no_download_needed = "no download needed" in msg.lower()
        has_results = False
        if getattr(action, "action_type", None) in {DesignActionType.SEARCH_PART, DesignActionType.SEARCH_WEB}:
            m = re.search(r"Found\s+(\d+)\s+matching parts", msg, flags=re.IGNORECASE)
            if m:
                try:
                    has_results = int(m.group(1)) > 0
                except Exception:
                    has_results = True
        self._action_ledger[key] = {
            "status": "succeeded" if bool(getattr(action, "success", False)) else "failed",
            "iteration": int(self._iteration or 0),
            "count": count,
            "result_message": msg,
            "action_type": str(getattr(getattr(action, "action_type", None), "name", "") or ""),
            "no_download_needed": bool(no_download_needed),
            "has_results": bool(has_results),
        }

        # On success, also record safe alias keys so later iterations can skip
        # re-downloading when the LLM changes phrasing (e.g. adds/removes
        # "Lib:" prefix, omits package, or starts with a descriptive name).
        if not bool(getattr(action, "success", False)):
            return

        at = getattr(action, "action_type", None)
        msg = msg or ""

        def _mark_succeeded(alias_key: str) -> None:
            if not alias_key:
                return
            prev2 = self._action_ledger.get(alias_key) or {}
            try:
                c2 = int(prev2.get("count", 0) or 0) + 1
            except Exception:
                c2 = 1
            self._action_ledger[alias_key] = {
                "status": "succeeded",
                "iteration": int(self._iteration or 0),
                "count": c2,
                "result_message": str(getattr(action, "result_message", "") or "").strip(),
                "action_type": str(getattr(getattr(action, "action_type", None), "name", "") or ""),
                "no_download_needed": bool(no_download_needed),
                "alias_for": key,
            }

        aliases: List[str] = []

        # Package-less alias for footprint downloads.
        if key.startswith("DOWNLOAD_FOOTPRINT:") and "|" in key:
            aliases.append(key.split("|", 1)[0])

        # Library-prefix / suffix alias for download actions: store both the
        # full token and its suffix (after the last ':').
        if key.startswith(("DOWNLOAD_SYMBOL:", "DOWNLOAD_FOOTPRINT:")):
            prefix, payload = key.split(":", 1)
            payload_no_pkg = payload.split("|", 1)[0]
            if ":" in payload_no_pkg:
                suffix = payload_no_pkg.rsplit(":", 1)[-1].strip()
                if suffix:
                    aliases.append(f"{prefix}:{suffix}")
                    # Also keep package-less variant if present.
                    if "|" in payload:
                        pkg = payload.split("|", 1)[1]
                        if pkg:
                            aliases.append(f"{prefix}:{suffix}|{pkg}")

        # Parse common success messages to learn the resolved identifiers.
        #
        # DOWNLOAD_* built-in path typically includes:
        #   "Library: SomeLib:Name"
        # and/or "Footprint: Lib:Name" or "Footprint: Name"
        m = re.search(r"(?m)^Library:\s*(.+?)\s*$", msg)
        if m:
            lib_ref = self._normalize_query_key(m.group(1))
            if lib_ref:
                if at == DesignActionType.DOWNLOAD_SYMBOL:
                    aliases.append(f"DOWNLOAD_SYMBOL:{lib_ref}")
                elif at == DesignActionType.DOWNLOAD_FOOTPRINT:
                    aliases.append(f"DOWNLOAD_FOOTPRINT:{lib_ref}")
                # Also suffix-only.
                if ":" in lib_ref:
                    suf = lib_ref.rsplit(":", 1)[-1].strip()
                    if suf:
                        if at == DesignActionType.DOWNLOAD_SYMBOL:
                            aliases.append(f"DOWNLOAD_SYMBOL:{suf}")
                        elif at == DesignActionType.DOWNLOAD_FOOTPRINT:
                            aliases.append(f"DOWNLOAD_FOOTPRINT:{suf}")

        m = re.search(r"(?m)^Footprint:\s*(.+?)\s*$", msg)
        if m:
            fp_ref = self._normalize_query_key(m.group(1))
            if fp_ref:
                aliases.append(f"DOWNLOAD_FOOTPRINT:{fp_ref}")
                if ":" in fp_ref:
                    suf = fp_ref.rsplit(":", 1)[-1].strip()
                    if suf:
                        aliases.append(f"DOWNLOAD_FOOTPRINT:{suf}")

        # "✅ Found XYZ in KiCad's built-in library!" → XYZ
        m = re.search(r"Found\s+(.+?)\s+in KiCad's built-in library", msg, flags=re.IGNORECASE)
        if m:
            found = self._normalize_query_key(m.group(1))
            if found:
                if at == DesignActionType.DOWNLOAD_SYMBOL:
                    aliases.append(f"DOWNLOAD_SYMBOL:{found}")
                elif at == DesignActionType.DOWNLOAD_FOOTPRINT:
                    aliases.append(f"DOWNLOAD_FOOTPRINT:{found}")

        # "✅ Installed XYZ." → XYZ (from download/install path)
        m = re.search(r"(?m)^Installed\s+(.+?)\.\s*$", msg)
        if m:
            installed = self._normalize_query_key(m.group(1))
            if installed:
                if at == DesignActionType.DOWNLOAD_SYMBOL:
                    aliases.append(f"DOWNLOAD_SYMBOL:{installed}")
                elif at == DesignActionType.DOWNLOAD_FOOTPRINT:
                    aliases.append(f"DOWNLOAD_FOOTPRINT:{installed}")

        for ak in dict.fromkeys(aliases):
            if ak != key:
                _mark_succeeded(ak)

        # Special-case: when the download handler reports "No download needed",
        # also mark the package-less base key as succeeded so later requests
        # with different package hints are skipped.
        if no_download_needed and key.startswith(("DOWNLOAD_FOOTPRINT:", "DOWNLOAD_SYMBOL:")) and "|" in key:
            _mark_succeeded(key.split("|", 1)[0])

    def _run_loop(self, goal: str, context: Dict[str, Any]):
        """Main agent loop — runs on a background thread."""
        try:
            self._set_state(AgentState.PLANNING)
            self._emit_thinking("🧠 Planning approach...")

            # The first message is the user's goal.
            current_message = goal

            if bool(context.get("benchmark_mode")):
                if bool(context.get("benchmark_bom_only")):
                    # Benchmark path is SPEC/BOM-only; skip workflow routing entirely.
                    self._require_full_workflow = False
                    self._workflow_router_reason = "benchmark bom-only"
                    context["require_full_workflow"] = False
                    context["workflow_mode"] = "bom_only"
                    context["require_add_component_location"] = True
                    self._run_benchmark_spec_bom_only(goal, context)
                    return
                else:
                    self._require_full_workflow = False
                    self._workflow_router_reason = "benchmark spec+place"
                    context["require_full_workflow"] = False
                    context["workflow_mode"] = "layout_only"
                    context["require_add_component_location"] = True
            # Decide workflow requirements once per run. This must be LLM-driven
            # (no deterministic heuristic) so the plugin remains universal.
            try:
                override = getattr(self._config, "require_full_workflow", None)
                if override is not None:
                    self._require_full_workflow = bool(override)
                    self._workflow_router_reason = "config override"
                else:
                    from .workflow_router import decide_require_full_workflow

                    decision = decide_require_full_workflow(getattr(self._agent, "_llm_client", None), goal)
                    self._require_full_workflow = bool(decision.require_full_workflow)
                    self._workflow_router_reason = decision.reason or "llm"
                context["require_full_workflow"] = self._require_full_workflow
                context["workflow_mode"] = "full" if self._require_full_workflow else "layout_only"
                # Configuration for the monolithic DesignAgent / subagents.
                # Force explicit placement thinking (no default/random centre placement).
                context["require_add_component_location"] = True
                self._emit_thinking(
                    "🧭 Workflow mode: "
                    f"{'full' if self._require_full_workflow else 'layout-only'} "
                    f"({self._workflow_router_reason})"
                )
            except Exception as e:
                self._emit_message(f"❌ Workflow routing failed: {e}")
                self._set_state(AgentState.ERROR)
                return

            while self._iteration < self._config.max_iterations:
                if self._check_stop():
                    break

                self._iteration += 1
                self._emit_thinking(f"🔄 Step {self._iteration}...")

                # Refresh context for each iteration
                fresh_context = self._get_context()
                if fresh_context:
                    context.update(fresh_context)
                # Preserve agent-run invariants even if the UI refresh overwrites keys.
                context["require_add_component_location"] = True
                # Persist de-dup ledger across context refreshes.
                context["action_ledger"] = self._action_ledger

                # ── PLAN: Ask the LLM for next steps ──
                # If sub-agents are enabled, delegate through the Orchestrator.
                self._set_state(AgentState.PLANNING)

                assistant_message = ""
                actions: List[DesignAction] = []

                # ── Sub-agent enhanced planning ──
                # Sub-agents provide phase awareness and spatial optimisation.

                if not (self._use_subagents and self._orchestrator is not None):
                    self._emit_message(
                        "❌ Sub-agents are unavailable/failed to initialize, and monolithic fallback is disabled."
                    )
                    self._set_state(AgentState.ERROR)
                    return

                try:
                    # Build a board snapshot dict for subagent context.
                    board_snapshot = self._build_board_snapshot(context)
                    self._inject_quality_constraints(context=context, board_snapshot=board_snapshot)

                    phase_name = self._orchestrator.phase.name
                    phase_label = self._display_phase_label(phase_name)
                    self._emit_thinking(f"🧠 Phase: {phase_label}")
                    self._deterministic_grid_phase_active = False
                    result = self._orchestrator.step(
                        goal=current_message,
                        context=context,
                        board_snapshot=board_snapshot,
                        feedback=self._last_feedback,
                    )
                    if isinstance(getattr(result, "artifacts", None), dict) and result.artifacts:
                        self._artifacts.update(result.artifacts)
                        for k in (
                            "design_spec_draft",
                            "design_spec",
                            "coverage_checklist",
                            "manifest",
                            "placement_plan",
                            "spec_debug",
                            "net_plan",
                        ):
                            if k in result.artifacts and isinstance(result.artifacts.get(k), dict):
                                context[k] = result.artifacts.get(k)
                    assistant_message = result.message or ""
                    actions = list(result.actions or [])
                    actions = self._synthesize_net_actions_from_artifacts(
                        phase_name=phase_name,
                        actions=actions,
                        result=result,
                        context=context,
                        board_snapshot=board_snapshot,
                    )
                    if result is not None and result.phase_complete:
                        if str(phase_name or "").upper() == "PLACE":
                            self._phase["component_set_verified"] = True
                        self._emit_phase_complete(phase_name, result)

                    if result is not None and result.phase_complete:
                        self._emit_thinking(
                            f"✅ Phase {phase_label} complete → "
                            f"{self._orchestrator.phase.name}"
                        )
                except Exception as e:
                    logger.exception("Orchestrator step failed")
                    self._emit_message(f"❌ Sub-agent planning failed: {e}")
                    self._set_state(AgentState.ERROR)
                    return

                # Subagents may legitimately return no actions for an iteration
                # (e.g. after a clarifying question). In that case, keep the loop
                # LLM-driven by re-prompting through the orchestrator next step
                # rather than using the monolithic agent.
                if not actions and not assistant_message:
                    current_message = "Continue with the next step. If blocked, ask a clarifying question."
                    continue

                # Drop repeated idempotent actions (e.g. repeated DOWNLOAD_* calls)
                # to prevent log spam and wasted work when context truncates.
                actions, skipped = self._filter_duplicate_planned_actions(actions, context)
                if skipped:
                    self._emit_thinking(f"⏭️ Skipping {skipped} already-completed duplicate action(s).")

                if self._check_stop():
                    break

                # ── Check for clarifying questions ──
                has_structured_clarification_request = False
                try:
                    has_structured_clarification_request = isinstance(self._artifacts.get("clarification_request"), dict)
                except Exception:
                    has_structured_clarification_request = False

                if has_structured_clarification_request or self._is_question(assistant_message, actions):
                    self._emit_message(assistant_message)
                    self._set_state(AgentState.AWAITING_INPUT)
                    self._input_event.clear()
                    self._pending_user_input = None
                    clarification_accepted = False

                    # Wait for user response (and keep waiting if a structured
                    # clarification reply cannot be translated to JSON).
                    while True:
                        self._input_event.wait()
                        if self._check_stop():
                            break

                        user_answer = self._pending_user_input
                        if not user_answer:
                            break
                        # Structured clarification path: if a controller/sub-agent
                        # emitted a clarification_request artifact, accept a JSON object
                        # reply and store it as a machine-readable artifact for the
                        # next planning step. For plain text, use the LLM to translate
                        # semantics into the requested schema (retry once, then fail).
                        structured_parse_failed = False
                        try:
                            creq = self._artifacts.get("clarification_request")
                            if isinstance(creq, dict):
                                req_id = str(creq.get("id", "") or "")
                                payload = None
                                s = str(user_answer or "").strip()
                                if s.startswith("{") and s.endswith("}"):
                                    try:
                                        obj = json.loads(s)
                                        if isinstance(obj, dict):
                                            payload = obj
                                    except Exception:
                                        payload = None
                                if payload is None:
                                    payload = self._translate_clarification_with_llm(s, creq)
                                if payload is not None:
                                    try:
                                        logger.info(
                                            "Clarification translated: request_id=%s kind=%s payload=%s",
                                            req_id,
                                            str(creq.get("kind", "") or ""),
                                            json.dumps(payload, ensure_ascii=True, sort_keys=True),
                                        )
                                    except Exception:
                                        logger.info(
                                            "Clarification translated: request_id=%s kind=%s payload=%r",
                                            req_id,
                                            str(creq.get("kind", "") or ""),
                                            payload,
                                        )
                                    self._artifacts["clarification_response"] = {
                                        "request_id": req_id,
                                        "payload": payload,
                                    }
                                    # Clear the pending request after a structured response.
                                    self._artifacts.pop("clarification_request", None)
                                else:
                                    structured_parse_failed = True
                        except Exception:
                            structured_parse_failed = True
                        if structured_parse_failed:
                            try:
                                logger.warning(
                                    "Clarification translation failed: request=%s user_answer=%r",
                                    json.dumps(creq, ensure_ascii=True, sort_keys=True) if isinstance(creq, dict) else creq,
                                    str(user_answer or "")[:500],
                                )
                            except Exception:
                                logger.warning("Clarification translation failed (unable to serialize debug context)")
                            self._emit_message(
                                "❌ I couldn't translate that clarification into the required JSON schema. Please retry (plain text is okay)."
                            )
                            self._input_event.clear()
                            self._pending_user_input = None
                            continue
                        self._agent._append_history("user", user_answer)
                        current_message = user_answer
                        clarification_accepted = True
                        break
                    if self._check_stop():
                        break
                    if clarification_accepted:
                        continue
                    if not self._pending_user_input:
                        break

                # ── Show assistant message ──
                if assistant_message:
                    self._emit_message(assistant_message)

                # ── Check for completion signal ──
                if self._is_completion(assistant_message, actions):
                    # In benchmark mode, accept orchestrator completion unconditionally
                    # once all orchestrated phases are done. Avoids infinite loop
                    # caused by DRC/workflow requirements not being met.
                    try:
                        from .sub_agents.orchestrator import DesignPhase as _DP
                        _orch_done = (
                            self._orchestrator is not None
                            and getattr(self._orchestrator, "phase", None) == _DP.DONE
                        )
                    except Exception:
                        _orch_done = False
                    if bool(context.get("benchmark_mode")):
                        if _orch_done:
                            self._emit_message("✅ Benchmark phases complete.")
                            self._set_state(AgentState.DONE)
                            return
                        # Force benchmark progression across all orchestrated
                        # phases: ignore premature completion messages until the
                        # orchestrator explicitly reaches DONE.
                        self._emit_thinking(
                            "⏭️ Ignoring premature completion signal in benchmark mode; continuing to next phase."
                        )
                        current_message = (
                            "Continue benchmark execution. Do not declare completion yet. "
                            "Advance through remaining orchestrator phases and execute required actions."
                        )
                        continue
                    completion_ok, reason = self._completion_requirements_met()
                    if completion_ok:
                        self._emit_message("✅ Design process complete!")
                        self._set_state(AgentState.DONE)
                        return

                    self._emit_thinking(f"⚠️ Cannot complete yet: {reason}")
                    if self._require_full_workflow:
                        current_message = (
                            f"Cannot declare DESIGN_COMPLETE yet: {reason} "
                            "Continue the board workflow: place missing components, assign nets, "
                            "route, then run ERC and DRC again."
                        )
                    else:
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
                self._ran_placement_drc_this_iteration = False
                self._ran_routing_drc_this_iteration = False
                self._placements_executed_this_iteration = 0
                self._needs_post_placement_drc = False
                self._abort_remaining_actions_this_iteration = False

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
                        if self._abort_remaining_actions_this_iteration:
                            break
                        if self._should_replan_after_action(action):
                            break

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
                                    if self._abort_remaining_actions_this_iteration:
                                        break
                                    if self._should_replan_after_action(a):
                                        break

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
                                    # Atomic batch support: if this is part of a board-mutating batch,
                                    # only execute once the whole batch is approved and prevalidated.
                                    batch_id = ""
                                    try:
                                        if isinstance(a.parameters, dict):
                                            batch_id = str(a.parameters.get("_atomic_batch", "") or "").strip()
                                    except Exception:
                                        batch_id = ""

                                    if (
                                        batch_id
                                        and (a.action_type in DESTRUCTIVE_ACTIONS)
                                        and (not is_read_only(a.action_type))
                                    ):
                                        # Collect the contiguous batch starting at cursor.
                                        batch_actions: List[DesignAction] = []
                                        end = cursor
                                        while end < len(self._approval_round_actions):
                                            b = self._approval_round_actions[end]
                                            bid = ""
                                            try:
                                                if isinstance(b.parameters, dict):
                                                    bid = str(b.parameters.get("_atomic_batch", "") or "").strip()
                                            except Exception:
                                                bid = ""
                                            if bid != batch_id:
                                                break
                                            batch_actions.append(b)
                                            end += 1

                                        # Wait until all actions in the batch have a decision.
                                        if any(id(b) not in self._approval_round_decisions for b in batch_actions):
                                            self._approval_event.wait(timeout=0.25)
                                            self._approval_event.clear()
                                            continue

                                        # If any action in the batch is rejected, skip the entire batch.
                                        if any(not bool(self._approval_round_decisions.get(id(b), False)) for b in batch_actions):
                                            reason = "Batch rejected (one or more actions were rejected)"
                                            self._emit_message(f"⏭️ Skipped {batch_id} batch: {reason}")
                                            for b in batch_actions:
                                                action_results.append(f"[{b.action_type.name}] REJECTED: {reason}")
                                            cursor = end
                                            continue

                                        success = self._execute_mutating_batch_atomic(
                                            batch_actions,
                                            context,
                                            action_results,
                                            batch_label=f"{batch_id} batch",
                                            board_snapshot=self._build_board_snapshot(context),
                                        )
                                        if not success:
                                            # After an atomic batch failure, stop executing further actions and replan.
                                            cursor = len(self._approval_round_actions)
                                            break
                                        cursor = end
                                        continue

                                    # Non-batched action: execute normally.
                                    self._execute_and_record(a, context, action_results, emoji="⚙️")
                                    if self._abort_remaining_actions_this_iteration:
                                        cursor = len(self._approval_round_actions)
                                        break
                                    if self._should_replan_after_action(a):
                                        # Stop executing further queued actions; replan next iteration.
                                        cursor = len(self._approval_round_actions)
                                        break
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
                                    if self._abort_remaining_actions_this_iteration:
                                        break
                                    if self._should_replan_after_action(a):
                                        break

                            # We fully handled the remainder.
                            break

                        # If approval_round_started is already True, we should never
                        # reach this branch because the remainder loop breaks out.
                        continue

                if self._check_stop():
                    break

                # Post-placement DRC: after a batch of placements, run once to catch overlaps/edge
                # clearance issues without spamming DRC for each component.
                if self._needs_post_placement_drc and (not self._ran_placement_drc_this_iteration):
                    drc_action = DesignAction(
                        action_type=DesignActionType.RUN_DRC,
                        description="Run DRC (placement) after placement batch",
                        parameters={"focus": "placement"},
                        requires_approval=False,
                    )
                    self._ran_placement_drc_this_iteration = True
                    self._execute_and_record(drc_action, context, action_results, auto=True, emoji="🔍")
                    # Only use the deterministic nudge fixer for tiny localized misses.
                    # On broad placement failures it rewrites the board into a worse layout
                    # and hides whether the placer itself improved anything.
                    if self._abort_remaining_actions_this_iteration:
                        drc_text = str(getattr(drc_action, "result_message", "") or "")
                        error_match = re.search(r"DRC Results:\s*(\d+) error", drc_text)
                        error_count = int(error_match.group(1)) if error_match else 0
                        fixed = False
                        if 0 < error_count <= 6:
                            fixed = self._apply_targeted_drc_fix(drc_action, context, action_results)
                        if fixed:
                            self._abort_remaining_actions_this_iteration = False
                            self._last_drc_passed = True

                # ── OBSERVE: Feed results back to LLM ──
                self._set_state(AgentState.OBSERVING)

                # Check if the last action was DRC and it passed
                last_drc = self._find_last_drc_result(action_results)
                if last_drc and last_drc.get("passed"):
                    self._last_drc_passed = True
                    completion_ok, reason = self._completion_requirements_met()
                    if completion_ok:
                        self._emit_message("✅ DRC passed — no errors found.")
                        current_message = (
                            "DRC is clean. Continue with the next required design step "
                            "to satisfy the user's goal. If the goal is satisfied, say DESIGN_COMPLETE."
                        )
                    else:
                        self._emit_message(f"✅ DRC passed, but workflow is incomplete: {reason}")
                        current_message = (
                            f"DRC is clean but not complete: {reason}. "
                            "Continue with the next required design phase."
                        )
                        if self._phase.get("component_set_verified") is True and int(self._phase.get("nets_assigned", 0) or 0) <= 0:
                            current_message = (
                                f"DRC is clean but not complete: {reason}. "
                                "Next step: assign nets (DEFINE_NET/ASSIGN_NETS)."
                            )
                    self._last_feedback = current_message
                    continue

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
                    # Unlimited retries during deterministic overlap resolution.
                    if not self._deterministic_grid_phase_active:
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

    def _run_benchmark_spec_bom_only(self, goal: str, context: Dict[str, Any]) -> None:
        """Benchmark mode: one SPEC/BOM planning pass with no executable actions."""
        if not (self._use_subagents and self._orchestrator is not None):
            self._emit_message(
                "❌ Sub-agents are unavailable/failed to initialize, and monolithic fallback is disabled."
            )
            self._set_state(AgentState.ERROR)
            return
        try:
            self._iteration += 1
            self._set_state(AgentState.PLANNING)
            phase_name = self._orchestrator.phase.name
            board_snapshot = self._build_board_snapshot(context)
            self._inject_quality_constraints(context=context, board_snapshot=board_snapshot)
            result = self._orchestrator.step(
                goal=goal,
                context=context,
                board_snapshot=board_snapshot,
                feedback=None,
            )
            if isinstance(getattr(result, "artifacts", None), dict) and result.artifacts:
                self._artifacts.update(result.artifacts)
                for k in ("design_spec_draft", "design_spec", "coverage_checklist", "spec_debug"):
                    if k in result.artifacts and isinstance(result.artifacts.get(k), dict):
                        context[k] = result.artifacts.get(k)
            if result is not None and result.phase_complete:
                if str(phase_name or "").upper() == "PLACE":
                    self._phase["component_set_verified"] = True
                self._emit_phase_complete(phase_name, result)
            if result and result.message:
                self._emit_message(result.message)
            dropped = len(result.actions or []) if result else 0
            if dropped > 0:
                self._emit_thinking(
                    f"⏭️ Benchmark BOM-only mode dropped {dropped} planned action(s)."
                )
            self._set_state(AgentState.DONE)
        except Exception as e:
            logger.exception("Benchmark BOM-only planning failed")
            self._emit_message(f"❌ Benchmark BOM-only planning failed: {e}")
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
        snap["placement_drc_passed"] = self._phase.get("placement_drc_passed", None)
        snap["connectivity_drc_passed"] = self._phase.get("connectivity_drc_passed", None)
        snap["drc_last"] = self._phase.get("drc_last", {}) if isinstance(self._phase.get("drc_last", {}), dict) else {}

        # Try to extract component list from pcb_data in context.
        pcb_data = context.get("pcb_data")
        components: List[Dict[str, Any]] = []
        if pcb_data is not None:
            try:
                # Light stats used by runtime heuristics.
                try:
                    snap["tracks_count"] = len(getattr(pcb_data, "tracks", []) or [])
                    snap["vias_count"] = len(getattr(pcb_data, "vias", []) or [])
                    snap["zones_count"] = len(getattr(pcb_data, "zones", []) or [])
                except Exception:
                    pass

                for fp in getattr(pcb_data, "footprints", []) or []:
                    ref = str(getattr(fp, "reference", "") or "")
                    val = str(getattr(fp, "value", "") or "")
                    # PCBParser.Footprint stores position in fp.at (Point in mm).
                    x = 0.0
                    y = 0.0
                    try:
                        at = getattr(fp, "at", None)
                        if at is not None:
                            x = float(getattr(at, "x", 0.0))
                            y = float(getattr(at, "y", 0.0))
                        else:
                            x = float(getattr(fp, "x", 0.0))
                            y = float(getattr(fp, "y", 0.0))
                    except Exception:
                        x = float(getattr(fp, "x", 0.0))
                        y = float(getattr(fp, "y", 0.0))
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

        # Board dimensions from context (may be missing/stale).
        snap["board_width"] = context.get("board_width")
        snap["board_height"] = context.get("board_height")

        # If available, prefer live pcbnew board state (pcb_data can be stale
        # during the same agent run, which causes the planner to re-add parts).
        try:
            import pcbnew  # type: ignore

            board = context.get("board")
            if board is None:
                try:
                    board = pcbnew.GetBoard()
                except Exception:
                    board = None
            if board is not None:
                to_mm = getattr(pcbnew, "ToMM", None)
                def _to_mm(v) -> float:
                    try:
                        if callable(to_mm):
                            return float(to_mm(v))
                    except Exception:
                        pass
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                live: List[Dict[str, Any]] = []
                for fp in list(board.GetFootprints() or []):
                    try:
                        ref = str(fp.GetReference() or "")
                    except Exception:
                        ref = ""
                    try:
                        val = str(fp.GetValue() or "")
                    except Exception:
                        val = ""
                    lib_id = ""
                    try:
                        fpid = getattr(fp, "GetFPID", None)
                        if callable(fpid):
                            fid = fpid()
                            # KiCad 7/8/9: LIB_ID or FPID has several string forms.
                            for attr in ("GetUniStringLibId", "Format", "AsString"):
                                fn = getattr(fid, attr, None)
                                if callable(fn):
                                    lib_id = str(fn() or "")
                                    if lib_id:
                                        break
                            if not lib_id:
                                lib_id = str(fid) if fid is not None else ""
                    except Exception:
                        lib_id = ""
                    # Anchor position (used by MOVE_COMPONENT).
                    x = y = 0.0
                    try:
                        pos = fp.GetPosition()
                        x = _to_mm(getattr(pos, "x", 0.0))
                        y = _to_mm(getattr(pos, "y", 0.0))
                    except Exception:
                        pass

                    # Footprint bounding box for spatial reasoning.
                    # Prefer the footprint's own rect/bbox if available.
                    bbox_w = bbox_h = 10.0
                    bbox_cx = x
                    bbox_cy = y
                    bbox_l = bbox_r = bbox_t = bbox_b = None
                    rect = None
                    for fn_name in ("GetFootprintRect", "GetBoundingBox"):
                        try:
                            fn = getattr(fp, fn_name, None)
                            if callable(fn):
                                rect = fn()
                                if rect is not None:
                                    break
                        except Exception:
                            rect = None
                    if rect is not None:
                        try:
                            rx = _to_mm(rect.GetX())
                            ry = _to_mm(rect.GetY())
                            rw = _to_mm(rect.GetWidth())
                            rh = _to_mm(rect.GetHeight())
                            if rw > 0 and rh > 0:
                                bbox_w, bbox_h = float(rw), float(rh)
                                bbox_l = float(rx)
                                bbox_t = float(ry)
                                bbox_r = float(rx + rw)
                                bbox_b = float(ry + rh)
                                bbox_cx = float(rx + (rw / 2.0))
                                bbox_cy = float(ry + (rh / 2.0))
                        except Exception:
                            pass
                    # Pad metadata (used for phase gating + net assignment sanity).
                    pads: List[str] = []
                    pad_nets: Dict[str, str] = {}
                    pin_name_to_pad: Dict[str, str] = {}
                    try:
                        for p in list(fp.Pads() or []):
                            name = ""
                            # Usually GetNumber() is the numeric identifier '30'
                            pad_number = ""
                            try:
                                if hasattr(p, "GetNumber"):
                                    pad_number = str(p.GetNumber() or "")
                                elif hasattr(p, "GetName"):
                                    pad_number = str(p.GetName() or "")
                            except Exception:
                                pass

                            for attr in ("GetPadName", "GetName", "GetNumber"):
                                fn = getattr(p, attr, None)
                                if callable(fn):
                                    try:
                                        name = str(fn() or "")
                                    except Exception:
                                        name = ""
                                    if name:
                                        break
                                        
                            if name:
                                pads.append(name)
                                
                                # Harvest alternate names/functions for LLM lookup
                                alt_names = []
                                for func_attr in ("GetPinFunction", "GetPinType"):
                                    try:
                                        fn2 = getattr(p, func_attr, None)
                                        if callable(fn2):
                                            func_val = str(fn2() or "").strip()
                                            if func_val:
                                                alt_names.append(func_val)
                                    except Exception:
                                        pass
                                        
                                if pad_number and pad_number != name:
                                    alt_names.append(pad_number)
                                    
                                for alt in alt_names:
                                    if alt:
                                        pin_name_to_pad[alt] = name
                                net_name = ""
                                for attr in ("GetNetname", "GetNetName"):
                                    fn = getattr(p, attr, None)
                                    if callable(fn):
                                        try:
                                            net_name = str(fn() or "")
                                        except Exception:
                                            net_name = ""
                                        if net_name:
                                            break
                                if not net_name:
                                    try:
                                        net_obj = p.GetNet()
                                    except Exception:
                                        net_obj = None
                                    if net_obj is not None:
                                        for attr in ("GetNetname", "GetNetName"):
                                            fn = getattr(net_obj, attr, None)
                                            if callable(fn):
                                                try:
                                                    net_name = str(fn() or "")
                                                except Exception:
                                                    net_name = ""
                                                if net_name:
                                                    break
                                if net_name:
                                    pad_nets[str(name)] = str(net_name)
                            if len(pads) >= 64:
                                break
                    except Exception:
                        pads = []
                        pad_nets = {}
                        pin_name_to_pad = {}
                    live.append(
                        {
                            "reference": ref,
                            "value": val,
                            "footprint": lib_id,
                            "x": x,
                            "y": y,
                            # Anchor position (KiCad footprint origin).
                            "anchor_x": x,
                            "anchor_y": y,
                            "pads": pads,
                            "pad_nets": pad_nets,
                            "pin_name_to_pad": pin_name_to_pad,
                            "pads_count": len(pads),
                            # Bounding box in mm (best effort).
                            "bbox_center_x": bbox_cx,
                            "bbox_center_y": bbox_cy,
                            "width": bbox_w,
                            "height": bbox_h,
                            "bbox_left": bbox_l,
                            "bbox_right": bbox_r,
                            "bbox_top": bbox_t,
                            "bbox_bottom": bbox_b,
                        }
                    )
                if live:
                    snap["components"] = live

                # Board dimensions from board edges bounding box (more reliable than context keys).
                try:
                    bb = board.GetBoardEdgesBoundingBox()
                    if bb is not None and callable(to_mm):
                        w = float(to_mm(bb.GetWidth()))
                        h = float(to_mm(bb.GetHeight()))
                        ox = float(to_mm(bb.GetX()))
                        oy = float(to_mm(bb.GetY()))
                        if w > 0 and h > 0:
                            if not snap.get("board_width") or not snap.get("board_height"):
                                snap["board_width"] = round(w, 3)
                                snap["board_height"] = round(h, 3)
                            # Always capture origin/center for coordinate-aware planning.
                            snap["board_origin_x"] = round(ox, 3)
                            snap["board_origin_y"] = round(oy, 3)
                            snap["board_center_x"] = round(ox + (w / 2.0), 3)
                            snap["board_center_y"] = round(oy + (h / 2.0), 3)
                except Exception:
                    pass
        except Exception:
            pass

        # Derived spatial diagnostics for placement: overlaps + edge proximity.
        try:
            comps = snap.get("components") or []
            if isinstance(comps, list) and comps and snap.get("board_origin_x") is not None and snap.get("board_width") is not None:
                ox = float(snap.get("board_origin_x") or 0.0)
                oy = float(snap.get("board_origin_y") or 0.0)
                bw = float(snap.get("board_width") or 0.0)
                bh = float(snap.get("board_height") or 0.0)
                bx2 = ox + bw
                by2 = oy + bh

                # Build bboxes from best-available data.
                boxes: List[Dict[str, Any]] = []
                for c in comps:
                    if not isinstance(c, dict):
                        continue
                    ref = str(c.get("reference", "") or "")
                    cx = c.get("bbox_center_x")
                    cy = c.get("bbox_center_y")
                    w = c.get("width")
                    h = c.get("height")
                    if cx is None or cy is None:
                        # Fallback: treat anchor as center.
                        cx = c.get("x")
                        cy = c.get("y")
                    try:
                        cx = float(cx)
                        cy = float(cy)
                        w = float(w) if w is not None else 10.0
                        h = float(h) if h is not None else 10.0
                    except Exception:
                        continue
                    left = float(cx - (w / 2.0))
                    right = float(cx + (w / 2.0))
                    top = float(cy - (h / 2.0))
                    bottom = float(cy + (h / 2.0))
                    boxes.append(
                        {
                            "ref": ref,
                            "cx": cx,
                            "cy": cy,
                            "w": w,
                            "h": h,
                            "l": left,
                            "r": right,
                            "t": top,
                            "b": bottom,
                        }
                    )

                overlaps: List[Dict[str, Any]] = []
                for i in range(len(boxes)):
                    a = boxes[i]
                    for j in range(i + 1, len(boxes)):
                        b = boxes[j]
                        # Positive penetration => overlap on that axis.
                        pen_x = min(a["r"], b["r"]) - max(a["l"], b["l"])
                        pen_y = min(a["b"], b["b"]) - max(a["t"], b["t"])
                        if pen_x > 0 and pen_y > 0:
                            overlaps.append(
                                {
                                    "a": a["ref"],
                                    "b": b["ref"],
                                    "pen_x_mm": round(float(pen_x), 3),
                                    "pen_y_mm": round(float(pen_y), 3),
                                }
                            )
                overlaps.sort(key=lambda o: (o["pen_x_mm"] * o["pen_y_mm"]), reverse=True)

                edge: List[Dict[str, Any]] = []
                for a in boxes:
                    d_left = a["l"] - ox
                    d_right = bx2 - a["r"]
                    d_top = a["t"] - oy
                    d_bottom = by2 - a["b"]
                    min_d = min(d_left, d_right, d_top, d_bottom)
                    edge.append({"ref": a["ref"], "min_edge_mm": round(float(min_d), 3)})
                edge.sort(key=lambda e: e["min_edge_mm"])

                snap["spatial"] = {
                    "overlaps": overlaps[:30],
                    "edge_clearance": edge[:30],
                }
        except Exception:
            pass

        # If the caller didn't provide board dimensions, attempt to compute them
        # from pcb_data Edge.Cuts geometry so the Orchestrator can reliably
        # detect when an outline exists.
        if (not snap.get("board_width") or not snap.get("board_height")) and pcb_data is not None:
            try:
                xs: List[float] = []
                ys: List[float] = []

                def _add_pt(pt) -> None:
                    if pt is None:
                        return
                    try:
                        xs.append(float(getattr(pt, "x")))
                        ys.append(float(getattr(pt, "y")))
                    except Exception:
                        pass

                for ln in getattr(pcb_data, "board_outline_lines", []) or []:
                    _add_pt(getattr(ln, "start", None))
                    _add_pt(getattr(ln, "end", None))
                for arc in getattr(pcb_data, "board_outline_arcs", []) or []:
                    _add_pt(getattr(arc, "start", None))
                    _add_pt(getattr(arc, "mid", None))
                    _add_pt(getattr(arc, "end", None))
                for rect in getattr(pcb_data, "board_outline_rects", []) or []:
                    _add_pt(getattr(rect, "start", None))
                    _add_pt(getattr(rect, "end", None))
                for poly in getattr(pcb_data, "board_outline_polygons", []) or []:
                    for pt in getattr(poly, "points", []) or []:
                        _add_pt(pt)
                for circ in getattr(pcb_data, "board_outline_circles", []) or []:
                    c = getattr(circ, "center", None)
                    r = float(getattr(circ, "radius", 0.0) or 0.0)
                    if c is not None and r > 0:
                        try:
                            cx = float(getattr(c, "x", 0.0))
                            cy = float(getattr(c, "y", 0.0))
                            xs.extend([cx - r, cx + r])
                            ys.extend([cy - r, cy + r])
                        except Exception:
                            pass

                if xs and ys:
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    if w > 0 and h > 0:
                        snap["board_width"] = round(w, 3)
                        snap["board_height"] = round(h, 3)
                        snap["board_origin_x"] = round(min(xs), 3)
                        snap["board_origin_y"] = round(min(ys), 3)
            except Exception:
                pass
        raw_search = context.get("search_part_results", {})
        if isinstance(raw_search, dict):
            compact_search: Dict[str, Any] = {}
            for query, items in list(raw_search.items())[-12:]:
                if isinstance(items, list):
                    compact_search[str(query)] = items[:6]
                else:
                    compact_search[str(query)] = items
            snap["search_part_results"] = compact_search
        else:
            snap["search_part_results"] = {}
        raw_web_search = context.get("search_web_results", {})
        if isinstance(raw_web_search, dict):
            compact_web: Dict[str, Any] = {}
            for query, items in list(raw_web_search.items())[-12:]:
                if isinstance(items, list):
                    compact_web[str(query)] = items[:5]
                else:
                    compact_web[str(query)] = items
            snap["search_web_results"] = compact_web
        else:
            snap["search_web_results"] = {}

        return snap

    def _build_quality_constraints(
        self,
        *,
        context: Dict[str, Any],
        board_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build compact, generalizable self-correction constraints for subagents."""
        constraints: Dict[str, Any] = {
            "version": 1,
            "placement": {
                "max_plan_drift_mm": 0.25,
                "max_rotation_drift_deg": 1.0,
                "clock_distance_target_mm": 20.0,
                "companion_distance_target_mm": 16.0,
            },
            "net_quality": {
                "prefer_existing_canonical_names": True,
                "avoid_unconnected_required_pads": True,
                "flag_bridge_orphans": True,
            },
            "power_integrity": {
                "regulator_caps_radius_mm": 15.0,
                "decoupling_radius_mm": 12.0,
                "usb_protection_max_distance_mm": 20.0,
            },
        }

        artifacts = self._artifacts if isinstance(self._artifacts, dict) else {}
        placement_plan = artifacts.get("placement_plan") if isinstance(artifacts.get("placement_plan"), dict) else {}
        if placement_plan:
            constraints["placement"]["planned_component_count"] = int(len(placement_plan))

        net_plan = artifacts.get("net_plan") if isinstance(artifacts.get("net_plan"), dict) else {}
        if net_plan:
            coverage = net_plan.get("coverage") if isinstance(net_plan.get("coverage"), dict) else {}
            bridge = net_plan.get("bridge_lint") if isinstance(net_plan.get("bridge_lint"), dict) else {}
            constraints["net_quality"]["coverage"] = {
                "partial_refs_count": int(coverage.get("partial_refs_count", 0) or 0),
                "refs_without_nets_count": int(coverage.get("refs_without_nets_count", 0) or 0),
                "unassigned_required_pads_count": int(coverage.get("unassigned_required_pads_count", 0) or 0),
                "bridge_issue_count": int(bridge.get("issue_count", 0) or 0),
            }

        drc_errors = str(context.get("drc_errors", "") or "").strip()
        if drc_errors:
            lines = [line.strip() for line in drc_errors.splitlines() if line.strip()]
            constraints["drc_focus"] = {
                "active": True,
                "message_sample": lines[:8],
            }

        components = board_snapshot.get("components") if isinstance(board_snapshot.get("components"), list) else []
        if components:
            constraints["board_state"] = {
                "components_visible": int(len(components)),
                "outline_defined": bool(board_snapshot.get("outline_defined", False)),
                "routing_attempted": bool(board_snapshot.get("routing_attempted", False)),
            }

        return constraints

    def _inject_quality_constraints(
        self,
        *,
        context: Dict[str, Any],
        board_snapshot: Dict[str, Any],
    ) -> None:
        constraints = self._build_quality_constraints(context=context, board_snapshot=board_snapshot)
        context["quality_constraints"] = constraints
        self._artifacts["quality_constraints"] = constraints

    def _run_placement_optimization(
        self,
        board_snapshot: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[DesignAction]:
        """Use PlacementAgent to resolve overlaps and produce MOVE_COMPONENT actions."""
        if not _SUBAGENTS_AVAILABLE:
            return []
        try:
            from .sub_agents.placement import PlacementAgent
        except Exception:
            return []

        components = board_snapshot.get("components", [])
        bw = board_snapshot.get("board_width")
        bh = board_snapshot.get("board_height")
        ox = board_snapshot.get("board_origin_x", 0.0)
        oy = board_snapshot.get("board_origin_y", 0.0)
        if not components or not bw or not bh:
            return []

        try:
            # board_snapshot components use "x/y" as the footprint anchor.
            # For area-based overlap resolution we want bbox centres; and then we
            # must convert back to anchors for MOVE_COMPONENT.
            prepared: List[Dict[str, Any]] = []
            offsets: Dict[str, Dict[str, float]] = {}
            for c in components:
                if not isinstance(c, dict):
                    continue
                ref = str(c.get("reference", "") or c.get("ref", "") or "")
                if not ref:
                    continue
                anchor_x = float(c.get("anchor_x", c.get("x", 0.0)) or 0.0)
                anchor_y = float(c.get("anchor_y", c.get("y", 0.0)) or 0.0)
                cx = c.get("bbox_center_x")
                cy = c.get("bbox_center_y")
                if cx is None or cy is None:
                    cx = anchor_x
                    cy = anchor_y
                try:
                    cx_f = float(cx)
                    cy_f = float(cy)
                except Exception:
                    cx_f = anchor_x
                    cy_f = anchor_y
                offsets[ref] = {
                    "dx": anchor_x - cx_f,
                    "dy": anchor_y - cy_f,
                }
                prepared.append({
                    "ref": ref,
                    "x": cx_f,
                    "y": cy_f,
                    "width": float(c.get("width", 10.0) or 10.0),
                    "height": float(c.get("height", 10.0) or 10.0),
                    "value": str(c.get("value", "") or ""),
                })

            optimized_centres = PlacementAgent.resolve_overlaps(prepared, clearance_mm=2.0)

            # Compare old vs new positions, generate MOVE_COMPONENT actions.
            move_actions: List[DesignAction] = []
            old_by_ref: Dict[str, Dict[str, Any]] = {}
            for c in components:
                if isinstance(c, dict):
                    r = str(c.get("reference", "") or c.get("ref", "") or "")
                    if r:
                        old_by_ref[r] = c

            bx_min = float(ox or 0.0)
            by_min = float(oy or 0.0)
            bx_max = bx_min + float(bw)
            by_max = by_min + float(bh)

            for new in optimized_centres:
                ref = str(new.get("ref", "") or "")
                if not ref:
                    continue
                old = old_by_ref.get(ref, {})
                old_ax = float(old.get("anchor_x", old.get("x", 0.0)) or 0.0)
                old_ay = float(old.get("anchor_y", old.get("y", 0.0)) or 0.0)

                dx_off = float(offsets.get(ref, {}).get("dx", 0.0))
                dy_off = float(offsets.get(ref, {}).get("dy", 0.0))
                new_ax = float(new.get("x", 0.0)) + dx_off
                new_ay = float(new.get("y", 0.0)) + dy_off

                # Keep the whole bbox inside the board outline (best-effort).
                try:
                    w = float(old.get("width", new.get("width", 10.0)) or 10.0)
                    h = float(old.get("height", new.get("height", 10.0)) or 10.0)
                except Exception:
                    w, h = 10.0, 10.0
                hw = w / 2.0
                hh = h / 2.0
                # Convert anchor clamp to centre clamp using existing offsets.
                new_cx = new_ax - dx_off
                new_cy = new_ay - dy_off
                new_cx = max(bx_min + hw + 1.0, min(bx_max - hw - 1.0, new_cx))
                new_cy = max(by_min + hh + 1.0, min(by_max - hh - 1.0, new_cy))
                new_ax = new_cx + dx_off
                new_ay = new_cy + dy_off

                dx = abs(old_ax - new_ax)
                dy = abs(old_ay - new_ay)
                if dx > 0.5 or dy > 0.5:  # moved more than 0.5 mm
                    new_x = round(float(new_ax), 2)
                    new_y = round(float(new_ay), 2)
                    move_actions.append(DesignAction(
                        action_type=DesignActionType.MOVE_COMPONENT,
                        description=f"Move {ref} to ({new_x}, {new_y}) to resolve overlap",
                        parameters={"ref": ref, "location": {"x": new_x, "y": new_y}},
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

    # ── Atomic batch execution (board-mutating) ─────────────────────────

    def _prevalidate_mutating_batch(
        self,
        actions: List[DesignAction],
        context: Dict[str, Any],
        *,
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Deterministically validate a mutating action batch before executing any of it."""
        snap = board_snapshot or self._build_board_snapshot(context)
        comps = snap.get("components") if isinstance(snap, dict) else None
        comps = comps if isinstance(comps, list) else []
        by_ref: Dict[str, Dict[str, Any]] = {}
        for c in comps:
            if isinstance(c, dict):
                ref = str(c.get("reference", "") or "").strip().upper()
                if ref:
                    by_ref[ref] = c
        known_refs = set(by_ref.keys())

        def _has_ref(ref_in: str) -> bool:
            r = str(ref_in or "").strip().upper()
            return bool(r and r in known_refs)

        def _has_pad(ref_in: str, pad_in: str) -> bool:
            r = str(ref_in or "").strip().upper()
            p = str(pad_in or "").strip()
            if not (r and p):
                return False
            c = by_ref.get(r) or {}
            pads = c.get("pads")
            if isinstance(pads, list):
                return p in {str(x) for x in pads}
            try:
                if int(c.get("pads_count") or 0) > 0:
                    # If we only know the pad count, allow numeric pad references
                    # within range (best-effort, deterministic).
                    if p.isdigit():
                        return 1 <= int(p) <= int(c.get("pads_count") or 0)
                    return False
            except Exception:
                return False
            return False

        # Validate each action.
        for a in actions:
            at = a.action_type
            params = a.parameters or {}

            # Batch-aware ref tracking: allow later actions to reference refs
            # introduced earlier in the same batch (ADD_COMPONENT with explicit ref).
            if at == DesignActionType.ADD_COMPONENT:
                ref_new = str(params.get("ref") or "").strip().upper()
                if ref_new:
                    known_refs.add(ref_new)

            if at in {DesignActionType.MOVE_COMPONENT, DesignActionType.ROTATE_COMPONENT, DesignActionType.DELETE_COMPONENT}:
                ref = str(params.get("ref") or params.get("reference") or "").strip().upper()
                if not ref or not _has_ref(ref):
                    return False, f"{at.name}: unknown reference {ref!r}"

            if at == DesignActionType.ADD_COMPONENT:
                fp_path = params.get("local_footprint_path")
                if isinstance(fp_path, str) and fp_path.strip():
                    try:
                        from pathlib import Path
                        if not Path(fp_path.strip()).exists():
                            return False, f"ADD_COMPONENT: local_footprint_path does not exist: {fp_path}"
                    except Exception:
                        return False, f"ADD_COMPONENT: invalid local_footprint_path: {fp_path}"

            if at == DesignActionType.DEFINE_BOARD_OUTLINE:
                try:
                    w = float(params.get("width"))
                    h = float(params.get("height"))
                    if w <= 0 or h <= 0:
                        return False, "DEFINE_BOARD_OUTLINE: width/height must be > 0"
                except Exception:
                    return False, "DEFINE_BOARD_OUTLINE: width/height required"

            if at == DesignActionType.ASSIGN_NETS:
                assigns = params.get("assignments")
                if not isinstance(assigns, list) or not assigns:
                    return False, "ASSIGN_NETS: missing assignments[]"
                for it in assigns:
                    if not isinstance(it, dict):
                        return False, "ASSIGN_NETS: assignment must be object"
                    ref = str(it.get("ref") or it.get("reference") or "").strip().upper()
                    pad = str(it.get("pad") or "").strip()
                    if not ref or not _has_ref(ref):
                        return False, f"ASSIGN_NETS: unknown reference {ref!r}"
                    if pad and not _has_pad(ref, pad):
                        return False, f"ASSIGN_NETS: pad not found {ref}/{pad}"

            if at == DesignActionType.DEFINE_NET:
                net = str(params.get("net") or "").strip()
                if not net:
                    return False, "DEFINE_NET: missing net name"
                pads = params.get("pads")
                if pads is None:
                    continue
                if not isinstance(pads, list):
                    pads = [pads]
                for p in pads:
                    if isinstance(p, str):
                        import re as _re
                        m = _re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", p.strip())
                        if not m:
                            return False, f"DEFINE_NET: invalid pad spec {p!r}"
                        ref, pad = m.group(1).upper(), m.group(2).strip()
                    elif isinstance(p, dict):
                        ref = str(p.get("ref") or "").strip().upper()
                        pad = str(p.get("pad") or "").strip()
                    else:
                        return False, f"DEFINE_NET: invalid pad spec {p!r}"
                    if not ref or not _has_ref(ref):
                        return False, f"DEFINE_NET: unknown reference {ref!r}"
                    if pad and not _has_pad(ref, pad):
                        return False, f"DEFINE_NET: pad not found {ref}/{pad}"

        return True, "ok"

    def _execute_mutating_batch_atomic(
        self,
        actions: List[DesignAction],
        context: Dict[str, Any],
        action_results: List[str],
        *,
        batch_label: str = "Atomic batch",
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Execute a batch of mutating actions with best-effort undo-on-failure."""
        ok, msg = self._prevalidate_mutating_batch(actions, context, board_snapshot=board_snapshot)
        if not ok:
            self._emit_message(f"⛔ {batch_label} prevalidation failed: {msg}")
            for a in actions:
                action_results.append(f"[{a.action_type.name}] BLOCKED: {msg}")
            return False

        commit = None
        board = None
        try:
            import pcbnew  # type: ignore
            board = context.get("board")
            if board is None:
                board = pcbnew.GetBoard()
            commit_cls = getattr(pcbnew, "BOARD_COMMIT", None)
            if board is not None and commit_cls is not None:
                try:
                    commit = commit_cls(board)
                except Exception:
                    commit = None
        except Exception:
            commit = None

        executed: List[DesignAction] = []
        batch_failed: Optional[str] = None
        for a in actions:
            if self._check_stop():
                batch_failed = "stopped"
                break
            a.approved = True
            self._emit_thinking(f"⚙️ {a.description}...")
            res = self._execute_action(a, context)
            executed.append(res)
            if not bool(getattr(res, "success", False)):
                batch_failed = res.result_message or "action failed"
                break

        if batch_failed:
            # Undo best-effort.
            if commit is not None:
                for fn_name in ("Revert", "Rollback", "Undo", "Cancel"):
                    fn = getattr(commit, fn_name, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:
                            continue
            self._emit_message(f"❌ {batch_label} failed; changes undone: {batch_failed}")
            for a in executed:
                a.success = False
                a.result_message = (a.result_message or "").strip()
                if a.result_message:
                    a.result_message += " (UNDONE)"
                else:
                    a.result_message = "UNDONE due to batch failure"
                a.executed = True
                action_results.append(f"[{a.action_type.name}] FAILED: {a.result_message}")
            # Mark unexecuted actions as blocked.
            for a in actions[len(executed):]:
                a.executed = False
                a.success = False
                a.result_message = "Not executed (batch aborted)"
                action_results.append(f"[{a.action_type.name}] BLOCKED: {a.result_message}")
            return False

        # Commit success for undo support.
        if commit is not None:
            try:
                push = getattr(commit, "Push", None)
                if callable(push):
                    push(f"VibeCAD: {batch_label}")
            except Exception:
                pass

        # Now that the batch succeeded, update phase state and record results.
        for a in executed:
            self._update_phase_state(a)
            self._update_placement_plan_from_action(a)
            self._record_ledger_result(a)
            status = "✅" if a.success else "❌"
            result_text = a.result_message or "Done"
            rt = str(result_text or "").strip()
            if rt.lstrip().startswith(("✅", "❌")):
                self._emit_message(rt)
            else:
                self._emit_message(f"{status} {rt}")
            tag = "OK" if a.success else "FAILED"
            action_results.append(f"[{a.action_type.name}] {tag}: {result_text}")
        return True

    @staticmethod
    def _action_rotation_degrees(action: DesignAction) -> Optional[float]:
        params = action.parameters if isinstance(action.parameters, dict) else {}
        if not isinstance(params, dict):
            return None
        for key in ("rotation", "rot", "angle"):
            value = params.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def _update_placement_plan_from_action(self, action: DesignAction) -> None:
        """Keep placement_plan aligned with successful board mutating actions."""
        if not bool(getattr(action, "success", False)):
            return
        placement_plan = self._artifacts.get("placement_plan")
        if not isinstance(placement_plan, dict):
            return

        params = action.parameters if isinstance(action.parameters, dict) else {}
        if not isinstance(params, dict):
            return
        ref = str(
            params.get("ref")
            or params.get("reference")
            or params.get("designator")
            or ""
        ).strip()
        if not ref:
            return

        at = action.action_type
        ref_key = ref.upper()

        if at == DesignActionType.DELETE_COMPONENT:
            placement_plan.pop(ref_key, None)
            placement_plan.pop(ref, None)
            return

        if at not in {
            DesignActionType.ADD_COMPONENT,
            DesignActionType.MOVE_COMPONENT,
            DesignActionType.ROTATE_COMPONENT,
        }:
            return

        existing = placement_plan.get(ref_key)
        if not isinstance(existing, dict):
            existing = placement_plan.get(ref) if isinstance(placement_plan.get(ref), dict) else {}
        row = dict(existing) if isinstance(existing, dict) else {}

        location = params.get("location") if isinstance(params.get("location"), dict) else {}
        if not location and ("x" in params) and ("y" in params):
            location = {"x": params.get("x"), "y": params.get("y")}
        if isinstance(location, dict):
            try:
                row["x"] = float(location.get("x", row.get("x", 0.0)) or 0.0)
            except Exception:
                pass
            try:
                row["y"] = float(location.get("y", row.get("y", 0.0)) or 0.0)
            except Exception:
                pass

        rot = self._action_rotation_degrees(action)
        if rot is not None:
            row["rot"] = float(rot)

        # Preserve previous orientation if a MOVE action omitted rotation.
        if "rot" not in row and isinstance(existing, dict) and ("rot" in existing):
            try:
                row["rot"] = float(existing.get("rot", 0.0) or 0.0)
            except Exception:
                row["rot"] = 0.0

        placement_plan[ref_key] = row
        if ref != ref_key and ref in placement_plan:
            # Normalize to one key form to avoid split state for same component.
            placement_plan.pop(ref, None)

    def _execute_and_record(self, action: DesignAction, context: Dict[str, Any],
                            action_results: List[str], auto: bool = False,
                            emoji: str = "🔍") -> None:
        """Execute an action, record the result, and emit status to the UI."""
        # Canonicalize parameter keys so executors see a consistent schema.
        try:
            action.parameters = normalize_action_parameters(action.action_type, action.parameters or {})
        except Exception:
            pass

        # Hard de-dup: even if the planner keeps proposing repeats, never
        # re-execute idempotent actions that already succeeded earlier in this run.
        #
        # This intentionally happens at execution time (not just during planning)
        # because some integrations bypass/override planned action lists.
        if action.action_type in {
            DesignActionType.SEARCH_PART,
            DesignActionType.SEARCH_WEB,
            DesignActionType.LOOKUP_DATASHEET,
            DesignActionType.DOWNLOAD_SYMBOL,
            DesignActionType.DOWNLOAD_FOOTPRINT,
        }:
            try:
                key = self._action_ledger_key(action)
            except Exception:
                key = None
            if key and self._ledger_succeeded(key):
                msg = f"Skipping already-completed {action.action_type.name}: {key}"
                try:
                    logger.info(msg)
                except Exception:
                    pass
                action.executed = True
                action.success = True
                action.result_message = msg
                self._emit_message(f"⏭️ {msg}")
                action_results.append(f"[{action.action_type.name}] SKIPPED: {msg}")
                return

        # Pre-routing DRC: run once just-in-time before any routing attempt in this iteration.
        # - If the user asked for a full workflow, include unconnected-items checks.
        # - Otherwise, keep it as a physical check so partial tasks aren't blocked by unconnected items.
        if action.action_type in {
            DesignActionType.AUTOROUTE_BOARD,
            DesignActionType.ROUTE_NET,
            DesignActionType.DRAW_TRACK,
            DesignActionType.ADD_VIA,
        }:
            if not self._ran_routing_drc_this_iteration:
                focus = "connectivity" if self._require_full_workflow else "placement"
                drc_action = DesignAction(
                    action_type=DesignActionType.RUN_DRC,
                    description=f"Run DRC ({focus}) before routing",
                    parameters={"focus": focus},
                    requires_approval=False,
                )
                self._ran_routing_drc_this_iteration = True
                # Avoid re-entering this branch during the DRC itself.
                self._nets_dirty_for_connectivity_drc = False
                self._execute_and_record(drc_action, context, action_results, auto=True, emoji="🔍")
                if not drc_action.success:
                    if action.action_type == DesignActionType.AUTOROUTE_BOARD:
                        self._emit_message(
                            "⚠️ Pre-route DRC failed, but continuing with AUTOROUTE_BOARD."
                        )
                    else:
                        self._emit_message("⛔ DRC failed — fix issues before routing.")
                        return
        # De-duplicate / throttle SEARCH_PART loops that flood logs/context.
        if action.action_type == DesignActionType.SEARCH_PART:
            query = ""
            try:
                query = str(self._agent._extract_search_query(action)).strip()  # type: ignore[attr-defined]
            except Exception:
                params = action.parameters or {}
                if isinstance(params, dict):
                    query = str(params.get("query", "") or "").strip()
            if not query:
                # Treat as a schema error from the planner: executing this action
                # will always fail and can create tight retry loops.
                msg = (
                    "SEARCH_PART is missing a usable query "
                    "(expected parameters.query to be a non-empty string)."
                )
                action.executed = True
                action.success = False
                action.result_message = msg
                self._emit_message(f"❌ {msg}")
                action_results.append(f"[SEARCH_PART] FAILED: {msg}")
                self._abort_remaining_actions_this_iteration = True
                return
            query = self._normalize_search_query_key(query)
            if query:
                searches_this_step = sum(1 for line in action_results if line.startswith("[SEARCH_PART]"))
                if searches_this_step >= self._max_search_actions_per_step:
                    msg = (
                        f"Skipping SEARCH_PART '{query}': reached per-step cap "
                        f"({self._max_search_actions_per_step})"
                    )
                    self._emit_message(f"⏭️ {msg}")
                    action_results.append(f"[SEARCH_PART] SKIPPED: {msg}")
                    return

                last_iter = self._search_query_last_iteration.get(query)
                if last_iter is not None and (self._iteration - last_iter) <= self._search_query_cooldown_steps:
                    msg = f"Skipping duplicate SEARCH_PART query: {query}"
                    self._emit_message(f"⏭️ {msg}")
                    action_results.append(f"[SEARCH_PART] SKIPPED: {msg}")
                    return
                self._search_query_last_iteration[query] = self._iteration

        if action.action_type in {DesignActionType.DOWNLOAD_SYMBOL, DesignActionType.DOWNLOAD_FOOTPRINT}:
            params = action.parameters or {}
            part_name = ""
            if isinstance(params, dict):
                part_name = str(params.get("part_name", "") or params.get("query", "") or "").strip()
            if not part_name:
                msg = (
                    f"{action.action_type.name} is missing a part name "
                    "(expected parameters.part_name to be a non-empty string)."
                )
                action.executed = True
                action.success = False
                action.result_message = msg
                self._emit_message(f"❌ {msg}")
                action_results.append(f"[{action.action_type.name}] FAILED: {msg}")
                self._abort_remaining_actions_this_iteration = True
                return

        # Phase gate: check prerequisites before executing
        gate_msg = self._check_phase_gate(action, context=context)
        if gate_msg:
            self._emit_message(f"⛔ {gate_msg}")
            action_results.append(f"[{action.action_type.name}] BLOCKED: {gate_msg}")
            # Treat STOP gates as fatal: pause the loop and require user input
            # rather than automatically continuing into other phases.
            if str(gate_msg).lstrip().upper().startswith("STOP:"):
                self._emit_message("⏸️ Pausing due to STOP condition.")
                self.pause()
            return

        action.approved = True
        self._emit_thinking(f"{emoji} {action.description}...")
        self._execute_action(action, context)
        drc_condensed_text: Optional[str] = None
        if action.action_type == DesignActionType.RUN_DRC:
            try:
                focus_key = self._normalize_drc_focus_key(action.parameters if isinstance(action.parameters, dict) else {})
                raw_drc = str(action.result_message or "")
                drc_condensed_text = self._condense_drc_result_with_diff(
                    raw_text=raw_drc,
                    focus_key=focus_key,
                    passed=bool(action.success),
                )
            except Exception:
                drc_condensed_text = None

        # Update phase tracking after execution
        self._update_phase_state(action)
        self._update_placement_plan_from_action(action)

        # Any placement-affecting action should trigger a single post-placement DRC
        # later in the iteration (to avoid thrashing DRC per action).
        if action.action_type in {
            DesignActionType.ADD_COMPONENT,
            DesignActionType.MOVE_COMPONENT,
            DesignActionType.ROTATE_COMPONENT,
            DesignActionType.ALIGN_COMPONENTS,
            DesignActionType.DEFINE_BOARD_OUTLINE,
            DesignActionType.ADD_MOUNTING_HOLE,
            DesignActionType.DELETE_COMPONENT,
        }:
            self._needs_post_placement_drc = True

        # Persist verification output into context so planning
        # can propose fixes rather than re-running checks blindly.
        try:
            if action.action_type == DesignActionType.RUN_DRC:
                if action.success:
                    context.pop("drc_errors", None)
                else:
                    # Do not pass DRC warnings into LLM context.
                    context["drc_errors"] = self._strip_drc_warnings_text(
                        drc_condensed_text or action.result_message or ""
                    )
            # RUN_ERC intentionally disabled (we focus on DRC-driven iteration).
        except Exception:
            pass

        # Persist idempotent results so later planning can avoid repeating them.
        self._record_ledger_result(action)

        self._history.append(LoopStep(
            iteration=self._iteration,
            action=action,
            result_success=action.success,
            result_message=action.result_message,
            was_auto_executed=auto,
        ))
        result_text = (drc_condensed_text if action.action_type == DesignActionType.RUN_DRC else None) or action.result_message or "Done"
        if action.action_type == DesignActionType.SEARCH_PART:
            result_text = self._truncate_search_output(result_text)
        status = "✅" if action.success else "❌"
        # Throttle DRC output during deterministic grid phase to avoid flooding
        # the UI with repeated long DRC dumps.
        emit_ui = True
        if action.action_type == DesignActionType.RUN_DRC and self._deterministic_grid_phase_active:
            if action.success:
                self._det_grid_drc_ui_counter = 0
                result_text = self._strip_drc_warnings_text(result_text)
            else:
                self._det_grid_drc_ui_counter += 1
                # Emit every 3rd failure (and the first) to keep UI readable.
                if self._det_grid_drc_ui_counter % 3 != 1:
                    emit_ui = False
                result_text = self._summarize_drc_result(result_text)
        if emit_ui:
            rt = str(result_text or "").strip()
            if rt.lstrip().startswith(("✅", "❌")):
                self._emit_message(rt)
            else:
                self._emit_message(f"{status} {rt}")
        tag = 'OK' if action.success else 'FAILED'
        action_tag = action.action_type.name
        if action.action_type == DesignActionType.RUN_DRC:
            try:
                focus = (action.parameters or {}).get("focus") if isinstance(action.parameters, dict) else None
            except Exception:
                focus = None
            focus = str(focus or "").strip().lower()
            if focus:
                action_tag = f"{action_tag}:{focus}"
        action_results.append(f"[{action_tag}] {tag}: {result_text}")
        if action.action_type == DesignActionType.ADD_COMPONENT and action.success:
            self._needs_post_placement_drc = True
            try:
                self._placements_executed_this_iteration += 1
            except Exception:
                self._placements_executed_this_iteration = 1
        if action.action_type == DesignActionType.RUN_DRC:
            try:
                focus = (action.parameters or {}).get("focus") if isinstance(action.parameters, dict) else None
            except Exception:
                focus = None
            if str(focus or "").strip().lower() == "placement":
                self._ran_placement_drc_this_iteration = True
                if action.success is False:
                    # Placement DRC failure should stop execution of any further actions in this plan;
                    # the agent should replan to resolve overlaps/edge issues first.
                    self._abort_remaining_actions_this_iteration = True

    def _should_replan_after_action(self, action: DesignAction) -> bool:
        """Return True when we should stop executing queued actions and replan."""
        if action.action_type != DesignActionType.ADD_COMPONENT:
            return False
        if not bool(getattr(action, "success", False)):
            return False

        if bool(getattr(self._config, "component_by_component_placement", False)):
            return True

        batch_size = int(getattr(self._config, "placement_batch_size", 0) or 0)
        if batch_size <= 0:
            return False

        try:
            return int(getattr(self, "_placements_executed_this_iteration", 0) or 0) >= batch_size
        except Exception:
            return False

    @staticmethod
    def _truncate_search_output(text: str, *, max_lines: int = 22, tail_lines: int = 8, max_chars: int = 4500) -> str:
        """Truncate verbose SEARCH_PART output for chat/UI readability."""
        s = str(text or "").strip()
        if not s:
            return s
        if len(s) <= max_chars and s.count("\n") + 1 <= max_lines:
            return s

        lines = s.splitlines()
        head_n = max(0, int(max_lines))
        tail_n = max(0, int(tail_lines))
        if len(lines) <= head_n + tail_n:
            # Char-based truncation only.
            head = s[: max_chars]
            omitted = max(0, len(s) - len(head))
            return head.rstrip() + f"\n...[truncated {omitted} chars]..."

        head = lines[:head_n]
        tail = lines[-tail_n:] if tail_n else []
        omitted_lines = max(0, len(lines) - len(head) - len(tail))
        out = "\n".join(head + [f"... ({omitted_lines} line(s) truncated) ..."] + tail).strip()
        if len(out) > max_chars:
            out = out[:max_chars].rstrip() + "\n...[truncated]..."
        return out

    @staticmethod
    def _strip_drc_warnings_text(text: str) -> str:
        """Remove DRC warnings section to keep LLM feedback compact."""
        s = str(text or "")
        if not s:
            return s
        # Remove the warnings count to save tokens.
        s = re.sub(
            r"(DRC Results:\s*\d+\s+error\(s\)),\s*\d+\s+warning\(s\)",
            r"\1",
            s,
            flags=re.IGNORECASE,
        )
        # Drop the warnings section entirely.
        s = re.sub(r"\nWARNINGS\s*\(acceptable\):.*\Z", "", s, flags=re.IGNORECASE | re.DOTALL)
        return s.strip()

    @staticmethod
    def _summarize_drc_result(text: str) -> str:
        """Return a short, UI-friendly DRC summary line."""
        s = str(text or "").strip()
        if not s:
            return "DRC failed."
        m = re.search(r"DRC Results:\s*(\d+)\s+error\\(s\\)", s, flags=re.IGNORECASE)
        if m:
            try:
                n = int(m.group(1))
            except Exception:
                n = None
            if n is not None:
                return f"DRC_STATUS: FAIL (errors={n})"
        for line in s.splitlines():
            if "DRC_STATUS" in line:
                return line.strip()
        return s.splitlines()[0].strip()

    @staticmethod
    def _normalize_drc_focus_key(params: Dict[str, Any]) -> str:
        focus = ""
        try:
            focus = str((params or {}).get("focus", "") or (params or {}).get("mode", "") or "").strip().lower()
        except Exception:
            focus = ""
        if focus in {"overlap"}:
            return "placement"
        if focus in {"net", "nets"}:
            return "connectivity"
        return focus or "default"

    @staticmethod
    def _parse_drc_report(raw_text: str) -> Dict[str, Any]:
        text = str(raw_text or "")
        status = ""
        m_status = re.search(r"DRC_STATUS:\s*(PASS|FAIL)", text, flags=re.IGNORECASE)
        if m_status:
            status = str(m_status.group(1) or "").upper()
        errors_count: Optional[int] = None
        warnings_count: Optional[int] = None
        m_counts = re.search(
            r"DRC Results:\s*(\d+)\s+error\(s\)(?:,\s*(\d+)\s+warning\(s\))?",
            text,
            flags=re.IGNORECASE,
        )
        if m_counts:
            try:
                errors_count = int(m_counts.group(1))
            except Exception:
                errors_count = None
            try:
                warnings_count = int(m_counts.group(2)) if m_counts.group(2) is not None else 0
            except Exception:
                warnings_count = None

        section = ""
        errors_items: List[str] = []
        warnings_items: List[str] = []
        for raw_line in text.splitlines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("ERRORS"):
                section = "errors"
                continue
            if upper.startswith("WARNINGS"):
                section = "warnings"
                continue
            m_item = re.match(r"^\d+\.\s*(.+)$", line)
            if not m_item:
                continue
            item = re.sub(r"\s+", " ", str(m_item.group(1) or "").strip())
            if not item:
                continue
            if section == "errors":
                errors_items.append(item)
            elif section == "warnings":
                warnings_items.append(item)

        if errors_count is None:
            errors_count = len(errors_items)
        if warnings_count is None:
            warnings_count = len(warnings_items)
        if not status:
            status = "PASS" if int(errors_count or 0) <= 0 else "FAIL"

        return {
            "status": status,
            "errors_count": int(errors_count or 0),
            "warnings_count": int(warnings_count or 0),
            "errors_items": sorted(set(errors_items)),
            "warnings_items": sorted(set(warnings_items)),
        }

    @staticmethod
    def _drc_fingerprint(summary: Dict[str, Any]) -> str:
        payload = {
            "status": str(summary.get("status", "") or ""),
            "errors_count": int(summary.get("errors_count", 0) or 0),
            "warnings_count": int(summary.get("warnings_count", 0) or 0),
            "errors_items": list(summary.get("errors_items") or []),
            "warnings_items": list(summary.get("warnings_items") or []),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _drc_item_sample(items: List[str], max_items: int = 2, max_len: int = 140) -> str:
        out: List[str] = []
        for item in list(items or [])[:max_items]:
            s = str(item or "").strip()
            if not s:
                continue
            if len(s) > max_len:
                s = s[:max_len].rstrip() + "..."
            out.append(s)
        return " | ".join(out)

    def _condense_drc_result_with_diff(
        self,
        *,
        raw_text: str,
        focus_key: str,
        passed: bool,
    ) -> str:
        current = self._parse_drc_report(raw_text)
        current_fp = self._drc_fingerprint(current)
        previous = self._drc_diff_state.get(focus_key) if isinstance(self._drc_diff_state.get(focus_key), dict) else {}

        prev_errors = set(str(x) for x in list(previous.get("errors_items") or []))
        prev_warnings = set(str(x) for x in list(previous.get("warnings_items") or []))
        cur_errors = set(str(x) for x in list(current.get("errors_items") or []))
        cur_warnings = set(str(x) for x in list(current.get("warnings_items") or []))
        unchanged = bool(previous) and str(previous.get("fingerprint", "")) == current_fp

        if passed:
            repeat_count = 0
            stagnated = False
        else:
            prev_epoch_raw = previous.get("last_mutation_epoch", -1)
            try:
                prev_epoch = int(prev_epoch_raw)
            except Exception:
                prev_epoch = -1
            if unchanged and prev_epoch == int(self._board_mutation_epoch):
                repeat_count = int(previous.get("repeat_count", 0) or 0) + 1
            else:
                repeat_count = 1
            # Initial fail + one retry is allowed. Third identical attempt is blocked.
            stagnated = repeat_count >= 2

        self._drc_diff_state[focus_key] = {
            "fingerprint": current_fp,
            "repeat_count": int(repeat_count),
            "stagnated": bool(stagnated),
            "last_mutation_epoch": int(self._board_mutation_epoch),
            "status": str(current.get("status", "") or ""),
            "errors_count": int(current.get("errors_count", 0) or 0),
            "warnings_count": int(current.get("warnings_count", 0) or 0),
            "errors_items": sorted(cur_errors),
            "warnings_items": sorted(cur_warnings),
        }

        new_errors = sorted(cur_errors - prev_errors)
        resolved_errors = sorted(prev_errors - cur_errors)
        new_warnings = sorted(cur_warnings - prev_warnings)
        resolved_warnings = sorted(prev_warnings - cur_warnings)

        status = str(current.get("status", "PASS") or "PASS").upper()
        err_n = int(current.get("errors_count", 0) or 0)
        warn_n = int(current.get("warnings_count", 0) or 0)
        lines: List[str] = [
            f"DRC_STATUS: {status}",
            f"DRC Results: {err_n} error(s), {warn_n} warning(s)",
        ]
        if not previous:
            lines.append("DRC_DIFF: baseline (first run for this focus)")
            if err_n > 0:
                sample = self._drc_item_sample(list(current.get("errors_items") or []))
                if sample:
                    lines.append(f"ERRORS_SAMPLE: {sample}")
        elif unchanged:
            lines.append(f"DRC_DIFF: unchanged vs previous (fp={current_fp})")
        else:
            lines.append(
                "DRC_DIFF: "
                f"errors +{len(new_errors)}/-{len(resolved_errors)}, "
                f"warnings +{len(new_warnings)}/-{len(resolved_warnings)}"
            )
            if new_errors:
                lines.append(f"NEW_ERRORS: {self._drc_item_sample(new_errors)}")
            if resolved_errors:
                lines.append(f"RESOLVED_ERRORS: {self._drc_item_sample(resolved_errors)}")

        if (not passed) and stagnated:
            lines.append(
                "DRC_STAGNATION: unchanged after 1 retry; further RUN_DRC for this focus "
                "is blocked until a board-mutating action occurs."
            )
        return "\n".join(lines).strip()

    @staticmethod
    def _compact_action_results(action_results: List[str], max_lines: int = 30) -> str:
        """Keep feedback compact so iterative prompts don't grow unbounded."""
        if not action_results:
            return "No actions executed."
        # Strip verbose DRC warnings so we don't waste LLM context.
        filtered: List[str] = []
        for line in action_results:
            if "RUN_DRC" in (line or ""):
                filtered.append(AgentLoop._strip_drc_warnings_text(line))
            else:
                filtered.append(line)
        action_results = filtered
        if len(action_results) <= max_lines:
            return "\n".join(action_results)

        head = action_results[: max_lines // 2]
        tail = action_results[-(max_lines - len(head)) :]
        omitted = len(action_results) - len(head) - len(tail)
        return "\n".join(head + [f"... ({omitted} result line(s) omitted) ..."] + tail)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _extract_json_object_loose(text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _translate_clarification_with_llm(self, user_answer: str, creq: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use the LLM to translate free-text clarification into schema JSON.

        Retries exactly once on invalid output/error, then fails with no fallback.
        """
        if not isinstance(creq, dict):
            return None
        llm_client = getattr(self._agent, "_llm_client", None)
        if llm_client is None or not getattr(llm_client, "is_available", False):
            return None

        try:
            from vibecad.llm.client import LLMMessage
        except Exception:
            return None

        schema_hint = json.dumps(creq, ensure_ascii=True, sort_keys=True)
        system_prompt = (
            "You translate a user's clarification reply into a machine-readable JSON payload for a PCB design pipeline. "
            "Return ONLY a JSON object with exactly this shape: "
            "{\"payload\": <object>} "
            "or {\"payload\": null} if the reply cannot be translated confidently. "
            "Do not add markdown or explanation."
        )
        user_prompt = (
            "Clarification request schema/meta:\n"
            f"{schema_hint}\n\n"
            "User reply:\n"
            f"{str(user_answer or '').strip()}\n\n"
            "Translate the reply semantics into the payload object expected by this clarification request."
        )

        for attempt in range(2):
            try:
                resp = llm_client.chat(
                    [LLMMessage(role="user", content=user_prompt)],
                    system_prompt=system_prompt,
                    response_format={"type": "json_object"},
                )
                parsed = self._extract_json_object_loose(getattr(resp, "content", "") or "")
                if not isinstance(parsed, dict):
                    continue
                payload = parsed.get("payload")
                if payload is None:
                    return None
                if isinstance(payload, dict):
                    return payload
            except Exception as e:
                try:
                    logger.warning(
                        "Clarification translation LLM attempt %d failed: %s (schema=%s answer=%r)",
                        attempt + 1,
                        e,
                        json.dumps(creq, ensure_ascii=True, sort_keys=True),
                        str(user_answer or "")[:500],
                    )
                except Exception:
                    logger.warning("Clarification translation LLM attempt %d failed: %s", attempt + 1, e)
                continue
        return None

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
            r'\bcan you\b.*\bspecify\b',
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

    @staticmethod
    def _is_board_mutating_action_type(action_type: DesignActionType) -> bool:
        return action_type in {
            DesignActionType.ADD_COMPONENT,
            DesignActionType.DRAW_TRACK,
            DesignActionType.DRAW_WIRE,
            DesignActionType.ROUTE_NET,
            DesignActionType.ADD_VIA,
            DesignActionType.ASSIGN_NETS,
            DesignActionType.DEFINE_NET,
            DesignActionType.MOVE_COMPONENT,
            DesignActionType.ROTATE_COMPONENT,
            DesignActionType.ALIGN_COMPONENTS,
            DesignActionType.ADD_POLYGON,
            DesignActionType.ADD_TEXT,
            DesignActionType.ADD_MOUNTING_HOLE,
            DesignActionType.DEFINE_BOARD_OUTLINE,
            DesignActionType.AUTOROUTE_BOARD,
            DesignActionType.SET_LAYER_COUNT,
            DesignActionType.DELETE_TRACKS,
            DesignActionType.DELETE_COMPONENT,
        }


    def _completion_requirements_met(self) -> Tuple[bool, str]:
        if self._require_full_workflow:
            if self._last_drc_passed is not True:
                return False, "DRC has not passed yet"
            if not self._phase.get("outline_defined"):
                return False, "board outline is not defined"
            if int(self._phase.get("nets_assigned", 0)) <= 0:
                return False, "nets have not been assigned"
            if not self._phase.get("routing_attempted"):
                return False, "routing has not been attempted"
            return True, ""

        if self._last_drc_passed is False:
            return False, "DRC has not passed yet"
        return True, ""

    @staticmethod
    def _normalize_query_key(value: str) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*:\s*", ":", text)
        return text

    # ── Phase-gate logic ─────────────────────────────────────────

    def _check_phase_gate(self, action: DesignAction, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Check if an action's prerequisites are met.

        Returns None if the action is allowed, or a feedback string to inject
        back to the LLM instead of executing the action.
        """
        at = action.action_type
        params = getattr(action, 'parameters', None) or {}
        context = context or {}

        # If placement DRC is clean, freeze placement and proceed to net assignment.
        # This prevents "layout churn" (moving parts after a clean placement check)
        # that re-introduces overlaps/edge violations.
        if (
            self._require_full_workflow
            and self._phase.get("placement_drc_passed") is True
            and self._phase.get("component_set_verified") is True
            and int(self._phase.get("nets_assigned", 0) or 0) <= 0
            and at in {
                DesignActionType.ADD_COMPONENT,
                DesignActionType.MOVE_COMPONENT,
                DesignActionType.ROTATE_COMPONENT,
                DesignActionType.ALIGN_COMPONENTS,
                DesignActionType.DELETE_COMPONENT,
                DesignActionType.DEFINE_BOARD_OUTLINE,
                DesignActionType.ADD_MOUNTING_HOLE,
            }
        ):
            return (
                "Placement DRC is clean. Do not move/add/delete components now. "
                "Proceed to net assignment using DEFINE_NET/ASSIGN_NETS."
            )

        if at == DesignActionType.DEFINE_BOARD_OUTLINE:
            if self._phase.get("outline_defined"):
                return (
                    "Board outline is already defined. Do not DEFINE_BOARD_OUTLINE again "
                    "unless the user explicitly asked to change dimensions."
                )

        # In full workflow mode, do not begin connectivity work until the component
        # set has been explicitly verified as complete by the PLACE-phase audit.
        #
        # This prevents the loop from "graduating" to net assignment just because
        # placement DRC is clean on a mostly-empty board (e.g. only mounting holes).
        if (
            self._require_full_workflow
            and self._phase.get("component_set_verified") is not True
            and at in {
                DesignActionType.DEFINE_NET,
                DesignActionType.ASSIGN_NETS,
                DesignActionType.ROUTE_NET,
                DesignActionType.DRAW_TRACK,
                DesignActionType.ADD_VIA,
            }
        ):
            return (
                "Cannot proceed to net assignment/routing yet: the component set has not been verified as complete. "
                "Stay in PLACE: add/search/download the missing components first, then re-run placement DRC."
            )

        # In full workflow mode, do not begin connectivity work until placement DRC
        # has explicitly passed. This prevents net assignment on a physically
        # invalid board (e.g. overlapping footprints, edge violations).
        if (
            self._require_full_workflow
            and self._phase.get("placement_drc_passed") is not True
            and at in {
                DesignActionType.DEFINE_NET,
                DesignActionType.ASSIGN_NETS,
                DesignActionType.ROUTE_NET,
                DesignActionType.DRAW_TRACK,
                DesignActionType.ADD_VIA,
            }
        ):
            return (
                "Cannot proceed to net assignment/routing yet: placement DRC has not passed. "
                "Fix overlaps/edge clearance (MOVE_COMPONENT/ROTATE_COMPONENT), then RUN_DRC with focus='placement'."
            )

        # ── Death-loop detection: block repeated add/delete cycles ──
        if at == DesignActionType.ADD_COMPONENT:
            query = ''
            for key in ('query', 'part_name', 'mpn', 'part', 'name'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    query = self._normalize_query_key(v)
                    break
            if query:
                count = self._component_add_fail_count.get(query, 0)
                if count >= 3:
                    return (
                        f"STOP: '{query}' failed to add {count} times. "
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

            # Duplicate placement guard: if the exact footprint is already present,
            # block re-adding it (except for common multi-instance footprints).
            try:
                qn = self._normalize_query_key(query)
            except Exception:
                qn = ''
            multi_ok = False
            if qn:
                # Allow many instances of passives and mechanicals.
                if re.match(r"^(r_|c_|l_)", qn):
                    multi_ok = True
                if any(k in qn for k in ("mountinghole", "fiducial", "testpoint")):
                    multi_ok = True
            if qn and not multi_ok:
                existing: List[str] = []
                # Prefer live pcbnew board.
                try:
                    import pcbnew  # type: ignore

                    board = context.get("board")
                    if board is None:
                        board = pcbnew.GetBoard()
                    for fp in list(board.GetFootprints() or []):
                        try:
                            fpid = getattr(fp, "GetFPID", None)
                            lib_id = ""
                            if callable(fpid):
                                fid = fpid()
                                for attr in ("GetUniStringLibId", "Format", "AsString"):
                                    fn = getattr(fid, attr, None)
                                    if callable(fn):
                                        lib_id = str(fn() or "")
                                        if lib_id:
                                            break
                                if not lib_id:
                                    lib_id = str(fid) if fid is not None else ""
                            if lib_id:
                                existing.append(self._normalize_query_key(lib_id))
                        except Exception:
                            continue
                except Exception:
                    existing = []
                # Fallback to pcb_data if provided.
                if not existing:
                    try:
                        pcb_data = context.get("pcb_data")
                        for fp in getattr(pcb_data, "footprints", []) or []:
                            lib = str(getattr(fp, "library", "") or "").strip()
                            name = str(getattr(fp, "footprint_name", "") or "").strip()
                            if lib and name:
                                existing.append(self._normalize_query_key(f"{lib}:{name}"))
                    except Exception:
                        pass

                def _matches_existing(q: str, e: str) -> bool:
                    if not q or not e:
                        return False
                    if q == e:
                        return True
                    # If query omits library, match by suffix.
                    if ":" not in q and e.endswith(":" + q):
                        return True
                    # As a last resort, token containment (avoid tiny strings).
                    if len(q) >= 8 and q in e:
                        return True
                    return False

                if any(_matches_existing(qn, e) for e in existing):
                    return (
                        f"Duplicate ADD_COMPONENT blocked: footprint '{query}' already exists on the board. "
                        "Use MOVE_COMPONENT/ROTATE_COMPONENT to adjust placement, or specify explicitly if you need multiple instances."
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
            focus = str(params.get("focus", "") or params.get("mode", "") or "").strip().lower()
            is_placement_focus = focus in {"placement", "overlap"}
            if (not is_placement_focus) and self._require_full_workflow and self._phase.get('nets_assigned', 0) <= 0:
                return (
                    "Cannot RUN_DRC yet: no nets are assigned. "
                    "Run DEFINE_NET/ASSIGN_NETS first, then RUN_DRC."
                )
            focus_key = self._normalize_drc_focus_key(params if isinstance(params, dict) else {})
            drc_state = self._drc_diff_state.get(focus_key) if isinstance(self._drc_diff_state.get(focus_key), dict) else {}
            state_epoch_raw = drc_state.get("last_mutation_epoch", -1) if isinstance(drc_state, dict) else -1
            try:
                state_epoch = int(state_epoch_raw)
            except Exception:
                state_epoch = -1
            if (
                drc_state
                and bool(drc_state.get("stagnated", False))
                and state_epoch == int(self._board_mutation_epoch)
            ):
                fp = str(drc_state.get("fingerprint", "") or "")
                return (
                    "RUN_DRC is unchanged after 1 retry "
                    f"(focus='{focus_key}', fp={fp}). "
                    "Make at least one board change (move/route/net assignment/etc.) "
                    "before re-running DRC."
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
        if ok and self._is_board_mutating_action_type(at):
            self._board_mutation_epoch = int(self._board_mutation_epoch) + 1

        if at == DesignActionType.ADD_COMPONENT:
            # Track add failures per query (for death-loop detection)
            query = ''
            for key in ('query', 'part_name', 'mpn', 'part', 'name'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    query = self._normalize_query_key(v)
                    break
            # Track consecutive footprint failures
            if ok:
                self._phase['components_placed'] = self._phase.get('components_placed', 0) + 1
                self._consecutive_fp_failures = 0  # reset on success
                if query:
                    self._component_add_fail_count.pop(query, None)
            else:
                if any(s in msg for s in (
                    'failed to load footprint',
                    'no footprint found',
                    'cannot read footprint',
                    'footprint file does not exist',
                    'rejected footprint',
                )):
                    self._consecutive_fp_failures += 1
                    if query:
                        self._component_add_fail_count[query] = self._component_add_fail_count.get(query, 0) + 1

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
                self._nets_dirty_for_connectivity_drc = True
            if warn_m:
                self._phase['net_assign_warnings'] = (
                    self._phase.get('net_assign_warnings', 0) + int(warn_m.group(1))
                )

        elif at == DesignActionType.AUTOROUTE_BOARD:
            self._phase['routing_attempted'] = True

        elif at == DesignActionType.RUN_DRC:
            # Track placement DRC status so the Orchestrator can gate NET_ASSIGN.
            focus = ""
            try:
                focus = str(params.get("focus", "") or params.get("mode", "") or "").strip().lower()
            except Exception:
                focus = ""
            if focus in {"placement", "overlap"}:
                self._phase["placement_drc_passed"] = ok
            elif focus in {"connectivity", "net", "nets"}:
                self._phase["connectivity_drc_passed"] = ok
            # Parse DRC counts deterministically for monotonic badness checks.
            try:
                import re as _re

                text = str(getattr(action, "result_message", "") or "")
                errors = None
                warnings = None
                m = _re.search(r"DRC Results:\s*(\d+)\s+error\(s\)(?:,\s*(\d+)\s+warning\(s\))?", text, flags=_re.IGNORECASE)
                if m:
                    try:
                        errors = int(m.group(1))
                    except Exception:
                        errors = None
                    if m.group(2) is not None:
                        try:
                            warnings = int(m.group(2))
                        except Exception:
                            warnings = None
                if errors is None and ok:
                    errors = 0
                if warnings is None and ok:
                    warnings = 0

                key = focus or "default"
                if not isinstance(self._phase.get("drc_last"), dict):
                    self._phase["drc_last"] = {}
                self._phase["drc_last"][key] = {
                    "passed": bool(ok),
                    "errors": errors,
                    "warnings": warnings,
                }
            except Exception:
                pass

    # ── Deterministic post-placement DRC fix ────────────────────────────────

    def _parse_drc_violation_coords(self, drc_text: str) -> List[Dict[str, Any]]:
        """Extract individual clearance violations from DRC output.

        Returns a list of dicts with keys: x, y, required, actual.
        Only clearance-type violations (pad-to-pad or pad-to-courtyard) are
        returned; solder-mask bridges are excluded because they are caused by
        the same root-cause pads and would create duplicate nudges.
        """
        import re as _re
        violations: List[Dict[str, Any]] = []
        # Pattern: "clearance X mm; actual Y mm) at (A, B)mm"
        pattern = _re.compile(
            r"clearance\s+(\d+(?:\.\d+)?)\s*mm;\s*actual\s+(\d+(?:\.\d+)?)\s*mm"
            r"[^)]*\)\s*at\s+\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)\s*mm",
            _re.IGNORECASE,
        )
        for m in pattern.finditer(drc_text):
            violations.append({
                "required": float(m.group(1)),
                "actual":   float(m.group(2)),
                "x":        float(m.group(3)),
                "y":        float(m.group(4)),
            })
        # Courtyard overlaps don't carry clearance values - treat as zero-clearance
        cy_pattern = _re.compile(
            r"[Cc]ourtyard[^(]*\(\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)\s*mm"
        )
        for m in cy_pattern.finditer(drc_text):
            violations.append({
                "required": 0.2,
                "actual":   0.0,
                "x":        float(m.group(1)),
                "y":        float(m.group(2)),
            })
        return violations

    def _cluster_drc_violations(
        self, violations: List[Dict[str, Any]], radius_mm: float = 2.5
    ) -> List[Dict[str, Any]]:
        """Cluster nearby violations and return one representative per cluster.

        Each cluster dict has: x, y (centroid), required, actual (worst-case).
        """
        import math as _math
        clusters: List[Dict[str, Any]] = []
        used = [False] * len(violations)
        for i, v in enumerate(violations):
            if used[i]:
                continue
            members = [v]
            used[i] = True
            for j in range(i + 1, len(violations)):
                if used[j]:
                    continue
                dx = violations[j]["x"] - v["x"]
                dy = violations[j]["y"] - v["y"]
                if _math.hypot(dx, dy) <= radius_mm:
                    members.append(violations[j])
                    used[j] = True
            cx = sum(m["x"] for m in members) / len(members)
            cy = sum(m["y"] for m in members) / len(members)
            worst_actual   = min(m["actual"]   for m in members)
            worst_required = max(m["required"] for m in members)
            clusters.append({
                "x":        cx,
                "y":        cy,
                "required": worst_required,
                "actual":   worst_actual,
            })
        return clusters

    def _apply_targeted_drc_fix(
        self,
        drc_action: "DesignAction",
        context: Dict[str, Any],
        action_results: List[str],
        max_attempts: int = 3,
    ) -> bool:
        """Deterministically nudge violating components to resolve DRC clearance errors.

        Parses the DRC error text for violation coordinates, finds the two closest
        components in the placement plan, and moves the farther one away by the
        minimum required distance.  No LLM is involved.

        Returns True if DRC passes after the fixes, False otherwise.
        """
        import math as _math

        drc_text = str(getattr(drc_action, "result_message", "") or "")
        placement: Dict[str, Any] = dict(self._artifacts.get("placement_plan") or {})
        if not placement:
            return False

        for attempt in range(max_attempts):
            violations = self._parse_drc_violation_coords(drc_text)
            if not violations:
                return False

            clusters = self._cluster_drc_violations(violations)
            moves_made = 0

            for cluster in clusters:
                cx, cy   = cluster["x"], cluster["y"]
                required = float(cluster.get("required", 0.2) or 0.2)
                actual   = float(cluster.get("actual",   0.0) or 0.0)

                # Find the two closest component centres to this violation.
                nearby: List[Tuple[float, str, Dict[str, Any]]] = []
                for ref, pos in placement.items():
                    if not isinstance(pos, dict):
                        continue
                    dx = float(pos.get("x", 0.0) or 0.0) - cx
                    dy = float(pos.get("y", 0.0) or 0.0) - cy
                    dist = _math.hypot(dx, dy)
                    if dist < 8.0:  # 8 mm search radius
                        nearby.append((dist, ref, pos))

                nearby.sort(key=lambda t: t[0])
                if len(nearby) < 2:
                    continue  # Can't determine pair; skip this cluster.

                _, ref1, pos1 = nearby[0]
                _, ref2, pos2 = nearby[1]

                # Nudge the second component away from the first.
                if actual <= 1e-6:
                    nudge_mm = 1.5  # Full pad overlap — move 1.5 mm
                else:
                    nudge_mm = max(0.15, (required - actual) + 0.15)

                dx = float(pos2.get("x", 0.0) or 0.0) - float(pos1.get("x", 0.0) or 0.0)
                dy = float(pos2.get("y", 0.0) or 0.0) - float(pos1.get("y", 0.0) or 0.0)
                dist12 = _math.hypot(dx, dy) or 1.0
                nx = (dx / dist12) * nudge_mm
                ny = (dy / dist12) * nudge_mm

                new_x = round(float(pos2.get("x", 0.0) or 0.0) + nx, 3)
                new_y = round(float(pos2.get("y", 0.0) or 0.0) + ny, 3)

                move = DesignAction(
                    action_type=DesignActionType.MOVE_COMPONENT,
                    description=(
                        f"DRC fix (attempt {attempt + 1}): nudge {ref2} by "
                        f"{nudge_mm:.2f} mm to clear {ref1}"
                    ),
                    parameters={"ref": ref2, "location": {"x": new_x, "y": new_y}},
                    requires_approval=False,
                )
                self._execute_and_record(move, context, action_results, auto=True, emoji="🔧")
                if getattr(move, "success", False):
                    placement[ref2] = {
                        "x": new_x,
                        "y": new_y,
                        "rot": float(pos2.get("rot", 0.0) or 0.0),
                    }
                    moves_made += 1

            if moves_made == 0:
                return False  # Nothing could be moved; give up.

            if "placement_plan" in self._artifacts:
                self._artifacts["placement_plan"].update(placement)

            # Re-run DRC to check.
            check = DesignAction(
                action_type=DesignActionType.RUN_DRC,
                description=f"Re-check DRC after targeted fix (attempt {attempt + 1})",
                parameters={"focus": "placement"},
                requires_approval=False,
            )
            self._execute_and_record(check, context, action_results, auto=True, emoji="🔍")
            if getattr(check, "success", True) and not getattr(check, "success", True) is False:
                # success attribute may not be set; check result text
                pass
            if getattr(check, "success", False):
                return True

            # DRC still failing — update error text and try again.
            drc_text = str(getattr(check, "result_message", "") or drc_text)

        return False

    # ── DRC hint builder ─────────────────────────────────────────────────────

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
