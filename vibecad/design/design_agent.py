# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UNIVERSAL PLUGIN — NO BOARD-SPECIFIC HARDCODING IN THIS FILE      ║
# ║  Prompts must use only goal_str / context variables.               ║
# ║  Never embed specific MPNs, part names, board names, or            ║
# ║  design-specific quantities in prompt strings or system prompts.   ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
Design Agent for interpreting user design requests.

This is the "brain" of VibeCAD's design assistance - it interprets
natural language requests and translates them into actionable design
operations with visual previews.

Works like GitHub Copilot: suggests, previews, user approves.
"""

import logging
import re
import ast
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


from .design_actions import (
    DesignActionType,
    DesignAction,
    DesignRequest,
    sanitize_llm_json_text,
    normalize_action_parameters,
)
from .design_agent_handlers import DesignAgentHandlersMixin


class DesignAgent(DesignAgentHandlersMixin):
    """Interprets user requests and generates design actions.
    
    The agent works in phases:
    1. Parse: Understand what the user wants
    2. Plan: Create a sequence of actions
    3. Preview: Show user what will happen
    4. Execute: Apply changes (with undo support)
    
    All modifications require explicit user approval.
    """

    # IMPORTANT: LLMClient has a default system prompt geared toward
    # *explanations only* (no modifications). For the design assistant we
    # provide a separate system prompt that allows proposing tool actions,
    # while still keeping execution gated by explicit user approval.
    DESIGN_SYSTEM_PROMPT = """You are VibeCAD Design Assistant for KiCad.

Core behavior:
- Propose actions only; never execute.
- Do only what the user asked.
- If request is high-level, use this order: ADD_COMPONENT -> DEFINE_BOARD_OUTLINE -> MOVE/ROTATE/ALIGN -> DEFINE_NET/ASSIGN_NETS -> ROUTE -> RUN_ERC -> RUN_DRC.
- Ask a clarifying question and return no actions when required inputs are missing.

Hard rules:
- Return strict JSON matching the caller schema.
- Do not invent coordinates; prefer refs/pads/nets. Omit location when unknown.
- You may propose ADD_COMPONENT directly with a descriptive query or explicit package hint. SEARCH_PART is optional and only needed if you are unsure of the footprint.
- ADD_COMPONENT.parameters.query can be a descriptive part name, an explicit footprint identifier ("LibName:FootprintName"), or a package hint (e.g. "ATmega328 DIP-28"). The system will automatically search for the best match.
- If the footprint is not available locally, propose DOWNLOAD_FOOTPRINT (and DOWNLOAD_SYMBOL if needed) before ADD_COMPONENT.
- For from-scratch board goals, avoid prebuilt module/shield footprints unless the user explicitly asks for a module/shield.
- DEFINE_BOARD_OUTLINE accepts width/height and optional shape controls:
  shape=("rectangle"|"rounded_rectangle"|"circle"), optional corner_radius for rounded_rectangle.
  Do not mention origin or 0,0.
- Do not route before nets are assigned.
- If net assignment says pad not found, replace the footprint (DELETE_COMPONENT + ADD_COMPONENT with correct package) before routing.
- Report DESIGN_COMPLETE only when ERC and DRC both have 0 errors.
"""

    # Only advertise tools that the plugin can actually execute today.
    # (Prevents the model from proposing actions like ADD_COMPONENT that are
    # not yet implemented.)
    SUPPORTED_LLM_TOOLS: List[DesignActionType] = [
        DesignActionType.SEARCH_PART,
        DesignActionType.DOWNLOAD_SYMBOL,
        DesignActionType.DOWNLOAD_FOOTPRINT,
        DesignActionType.ADD_COMPONENT,
        DesignActionType.EXPORT_BOM,
        DesignActionType.DRAW_TRACK,
        DesignActionType.ASSIGN_NETS,
        DesignActionType.DEFINE_NET,
        DesignActionType.MOVE_COMPONENT,
        DesignActionType.ROTATE_COMPONENT,
        DesignActionType.RUN_DRC,
        DesignActionType.RUN_ERC,
        DesignActionType.ADD_VIA,
        DesignActionType.DEFINE_BOARD_OUTLINE,
        DesignActionType.ADD_MOUNTING_HOLE,
        DesignActionType.ALIGN_COMPONENTS,
        DesignActionType.ADD_TEXT,
        DesignActionType.ADD_POLYGON,
        DesignActionType.AUTOROUTE_BOARD,
        DesignActionType.SET_LAYER_COUNT,
        DesignActionType.DELETE_TRACKS,
        DesignActionType.DELETE_COMPONENT,
        DesignActionType.SEARCH_WEB,
        DesignActionType.LOOKUP_DATASHEET,
    ]
    
    def __init__(self, llm_client=None):
        """Initialize the design agent.
        
        Args:
            llm_client: Optional LLM client for advanced interpretation.
                       If None, uses local pattern matching only.
        """
        self._llm_client = llm_client
        self._progress_callback: Optional[Callable] = None
        
        # Component managers (set by plugin)
        self._library_manager = None
        self._connection_manager = None
        self._bom_exporter = None

        # Session-level placement tracker: list of (x_iu, y_iu) tuples for every
        # component placed during this agent's lifetime.  This is the ONLY
        # reliable source of truth for anti-overlap because pcbnew's
        # GetFootprints() can silently fail or return stale data on KiCad 9.
        self._session_placed_positions: List[Tuple[int, int]] = []

        # Conversation history for multi-turn support.
        # Each entry is {"role": "user"|"assistant", "content": str}.
        self._conversation_history: List[Dict[str, str]] = []
        self._max_history_turns = 20  # keep last N messages

        # Per-run LLM usage ledger (normalized across providers).
        self._llm_usage_totals: Dict[str, int] = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self._llm_usage_events: List[Dict[str, Any]] = []

    def clear_history(self) -> None:
        """Reset conversation history (e.g., when opening a new board)."""
        self._conversation_history.clear()

    def record_history_turn(self, role: str, content: str) -> None:
        """Record an external turn (e.g., Q&A path outside agent planning)."""
        role_norm = str(role or "").strip().lower()
        if role_norm not in {"user", "assistant"}:
            return
        self._append_history(role_norm, content)

    def recent_history(
        self,
        max_turns: int = 8,
        include_roles: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Return recent conversation turns in chronological order."""
        try:
            limit = max(0, int(max_turns))
        except Exception:
            limit = 0
        if limit <= 0:
            return []

        roles = {str(r or "").strip().lower() for r in (include_roles or ["user", "assistant"])}
        if not roles:
            roles = {"user", "assistant"}

        filtered = [
            {"role": str(entry.get("role", "") or ""), "content": str(entry.get("content", "") or "")}
            for entry in self._conversation_history
            if str(entry.get("role", "") or "").strip().lower() in roles
        ]
        if len(filtered) <= limit:
            return filtered
        return filtered[-limit:]

    def _append_history(self, role: str, content: str) -> None:
        # Keep per-turn payload bounded to avoid prompt bloat from verbose
        # action result messages (especially long SEARCH_PART listings).
        content = str(content or "")
        if len(content) > 3000:
            content = content[:3000] + "\n...[truncated]"
        self._conversation_history.append({"role": role, "content": content})
        # Trim to keep memory bounded.
        if len(self._conversation_history) > self._max_history_turns:
            self._conversation_history = self._conversation_history[-self._max_history_turns:]

    def _assistant_history_text(self, assistant_message: str, actions: List[DesignAction]) -> str:
        """Store a compact assistant turn in history.

        We include a short recap of proposed actions so later confirmations like
        "yes do it" have enough context without requiring brittle slot parsing.
        """
        assistant_message = (assistant_message or "").strip()
        if not actions:
            return assistant_message

        action_lines: List[str] = []
        for action in actions:
            params = action.parameters or {}
            param_bits = []
            for key in ("part_name", "query", "ref", "net_name", "location", "angle"):
                if key in params and params.get(key) not in (None, ""):
                    param_bits.append(f"{key}={params.get(key)}")
            suffix = f" ({', '.join(param_bits)})" if param_bits else ""
            action_lines.append(f"- {action.action_type.name}{suffix}: {action.description}")

        recap = "\n\nPROPOSED_ACTIONS:\n" + "\n".join(action_lines)
        return (assistant_message + recap).strip()

    def set_llm_client(self, llm_client) -> None:
        """Update the LLM client (e.g., after settings changes)."""
        self._llm_client = llm_client

    @staticmethod
    def _coerce_usage_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                iv = int(value)
                return max(0, iv)
            s = str(value).strip()
            if not s:
                return None
            if s.isdigit():
                return int(s)
            # Best-effort for providers returning numeric-like strings.
            return max(0, int(float(s)))
        except Exception:
            return None

    @classmethod
    def _normalize_usage_payload(cls, usage: Any) -> Dict[str, int]:
        if not isinstance(usage, dict):
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            }

        def _pick(*keys: str) -> Optional[int]:
            for key in keys:
                if key in usage:
                    iv = cls._coerce_usage_int(usage.get(key))
                    if iv is not None:
                        return iv
            return None

        input_tokens = _pick("input_tokens", "prompt_tokens")
        output_tokens = _pick("output_tokens", "completion_tokens")
        reasoning_tokens = _pick("reasoning_tokens")

        if reasoning_tokens is None:
            # OpenAI-compatible APIs may nest reasoning stats under details keys.
            for details_key in ("completion_tokens_details", "output_tokens_details", "token_details", "details"):
                details = usage.get(details_key)
                if isinstance(details, dict):
                    reasoning_tokens = cls._coerce_usage_int(
                        details.get("reasoning_tokens", details.get("reasoning"))
                    )
                    if reasoning_tokens is not None:
                        break

        total_tokens = _pick("total_tokens")
        if total_tokens is None:
            total_tokens = max(0, int((input_tokens or 0) + (output_tokens or 0)))

        return {
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "total_tokens": int(total_tokens or 0),
        }

    def _record_llm_usage(self, response_obj: Any) -> None:
        try:
            usage_payload = {}
            if response_obj is not None:
                usage_payload = getattr(response_obj, "usage", {}) or {}
            normalized = self._normalize_usage_payload(usage_payload)

            self._llm_usage_totals["llm_calls"] = int(self._llm_usage_totals.get("llm_calls", 0) or 0) + 1
            for key in ("input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"):
                self._llm_usage_totals[key] = int(self._llm_usage_totals.get(key, 0) or 0) + int(
                    normalized.get(key, 0) or 0
                )

            event = {
                "t_s": time.time(),
                "model": str(getattr(response_obj, "model", "") or ""),
                "usage": dict(normalized),
            }
            self._llm_usage_events.append(event)
            if len(self._llm_usage_events) > 200:
                self._llm_usage_events = self._llm_usage_events[-200:]
        except Exception:
            logger.debug("Failed recording LLM usage event", exc_info=True)

    def reset_llm_usage_stats(self) -> None:
        self._llm_usage_totals = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        self._llm_usage_events = []

    def get_llm_usage_totals(self) -> Dict[str, int]:
        return {
            "llm_calls": int(self._llm_usage_totals.get("llm_calls", 0) or 0),
            "input_tokens": int(self._llm_usage_totals.get("input_tokens", 0) or 0),
            "output_tokens": int(self._llm_usage_totals.get("output_tokens", 0) or 0),
            "reasoning_tokens": int(self._llm_usage_totals.get("reasoning_tokens", 0) or 0),
            "total_tokens": int(self._llm_usage_totals.get("total_tokens", 0) or 0),
        }
    
    def set_library_manager(self, manager):
        self._library_manager = manager
    
    def set_connection_manager(self, manager):
        self._connection_manager = manager
    
    def set_bom_exporter(self, exporter):
        self._bom_exporter = exporter
    
    def set_progress_callback(self, callback: Callable):
        self._progress_callback = callback
    
    def interpret_request(self, 
                          user_input: str,
                          context: Dict[str, Any] = None) -> DesignRequest:
        """Interpret a user's natural language request.
        
        Args:
            user_input: The user's request text
            context: Current design context (selected items, active layer, etc.)
        
        Returns:
            DesignRequest with interpreted actions
        """
        context = context or {}
        
        request = DesignRequest(
            original_text=user_input,
            active_editor=context.get('active_editor', 'pcb'),
            current_file=context.get('current_file'),
        )

        from ..llm.client import LLMError

        if not self._llm_client or not getattr(self._llm_client, "is_available", False):
            raise LLMError("LLM is required for request interpretation but is not available/configured.")

        actions = self._interpret_with_llm(user_input, context)
        for a in actions:
            request.add_action(a)
        request.confidence = 0.9 if actions else 0.2
        
        return request

    def chat(self, user_input: str, context: Dict[str, Any] = None) -> Tuple[str, DesignRequest]:
        """LLM-first conversational entrypoint.

        Returns an assistant message plus a structured DesignRequest containing
        tool-like actions that can be previewed and approved.
        """
        context = context or {}

        from ..llm.client import LLMError
        if not self._llm_client or not getattr(self._llm_client, "is_available", False):
            raise LLMError("LLM is required for the design agent but is not available/configured.")

        # Record the user's message before calling the LLM. `_chat_with_llm`
        # will include prior turns to preserve multi-turn context.
        self._append_history("user", user_input)
        assistant_message, actions, confidence = self._chat_with_llm(user_input, context)

        # Preflight: do not propose downloads we can't actually execute.
        assistant_message, actions = self._preflight_actions(assistant_message, actions, context)
        request = DesignRequest(
            original_text=user_input,
            active_editor=context.get('active_editor', 'pcb'),
            current_file=context.get('current_file'),
            confidence=confidence,
        )
        for action in actions:
            request.add_action(action)
        if not assistant_message and actions:
            assistant_message = "Got it. Here’s what I propose."

        # Record assistant response for next-turn references.
        self._append_history("assistant", self._assistant_history_text(assistant_message, actions))
        return assistant_message, request

    def _preflight_actions(self,
                           assistant_message: str,
                           actions: List['DesignAction'],
                           context: Dict[str, Any]) -> Tuple[str, List['DesignAction']]:
        """Validate / adjust LLM-proposed actions before previewing them.

        Key rule: do not propose DOWNLOAD_* actions unless we can locate a
        corresponding symbol/footprint (otherwise users approve an action that
        will immediately fail).
        """
        if not actions:
            return assistant_message, actions
        if not self._library_manager:
            return assistant_message, actions

        safe_actions: List[DesignAction] = []
        missing: List[str] = []
        malformed: List[str] = []

        def _clean_component_query(text: str) -> str:
            q = str(text or "").strip()
            if not q:
                return ""
            q = re.sub(r"^[\"']|[\"']$", "", q).strip()
            q = re.sub(
                r"^(add|place|insert|use|put|mount)\s+",
                "",
                q,
                flags=re.IGNORECASE,
            ).strip()
            q = re.sub(r"\b(component|footprint)\b", "", q, flags=re.IGNORECASE).strip()
            q = re.sub(r"\s+", " ", q).strip(" .,:;")
            return q

        def _tokenize_for_match(text: str) -> List[str]:
            s = str(text or "").lower()
            s = s.replace("pinheader", "pin header")
            s = re.sub(r"(\d+)\s*-\s*pin", r"\1 pin", s)
            s = re.sub(r"\b1x0?(\d+)\b", r"\1 pin", s)
            s = s.replace("-", " ")
            return re.findall(r"[a-z0-9]+(?:mhz|khz|ghz)?", s)

        def _best_search_query_hint(text: str) -> str:
            store = context.get("search_part_results", {})
            if not isinstance(store, dict) or not store:
                return ""
            target = set(_tokenize_for_match(text))
            if not target:
                return ""
            best_query = ""
            best_score = 0.0
            for query in store.keys():
                q_tokens = set(_tokenize_for_match(str(query)))
                if not q_tokens:
                    continue
                overlap = len(target & q_tokens)
                score = overlap / max(len(target), 1)
                if score > best_score:
                    best_score = score
                    best_query = str(query).strip()
            return best_query if best_score >= 0.45 else ""

        def _looks_instruction_query(query: str) -> bool:
            q = str(query or "").strip().lower()
            if not q:
                return True
            return bool(re.match(r"^(add|place|insert|use|put|mount)\b", q))

        for a in actions:
            # Normalize malformed component/net actions before execution.
            if a.action_type == DesignActionType.ADD_COMPONENT:
                params = dict(a.parameters or {})
                raw_query = ""
                for key in ("query", "part_name", "mpn", "part", "name"):
                    value = params.get(key)
                    if isinstance(value, str) and value.strip():
                        raw_query = value.strip()
                        break
                # Some models send an explicit KiCad footprint identifier separately.
                fp_hint = ""
                for key in ("footprint", "footprint_id", "footprintId", "kicad_footprint", "kicadFootprint"):
                    v = params.get(key)
                    if isinstance(v, str) and v.strip():
                        fp_hint = v.strip()
                        break
                desc = str(a.description or "").strip()
                query = _clean_component_query(raw_query) if raw_query else ""
                if not query:
                    query = _clean_component_query(desc)
                # If the model provided a concrete footprint hint, prefer it over a bare MPN.
                if fp_hint:
                    cleaned_fp = _clean_component_query(fp_hint)
                    if ":" in cleaned_fp or re.search(r"\b(?:DIP|PDIP|QFN|DFN|TQFP|LQFP|SOIC|SOT|SSOP|TSSOP)\b", cleaned_fp, re.IGNORECASE):
                        query = cleaned_fp
                if _looks_instruction_query(query):
                    inferred = _best_search_query_hint(query or desc)
                    if inferred:
                        query = inferred
                if not query:
                    malformed.append("ADD_COMPONENT missing usable query")
                    logger.error(
                        "Dropping malformed ADD_COMPONENT action: desc=%r params=%r",
                        a.description, a.parameters
                    )
                    continue
                if query != raw_query:
                    logger.warning(
                        "Normalized ADD_COMPONENT query from %r to %r",
                        raw_query or desc, query
                    )
                params["query"] = query
                a.parameters = params
                safe_actions.append(a)
                continue

            if a.action_type == DesignActionType.DEFINE_NET:
                params = dict(a.parameters or {})

                # Accept common key variants.
                if not str(params.get("net", "") or "").strip():
                    for k in ("net_name", "netName", "name"):
                        v = params.get(k)
                        if isinstance(v, str) and v.strip():
                            params["net"] = v.strip()
                            break

                # Support bulk net definitions: {"net_names":[...]}.
                net_names = params.get("net_names") or params.get("netNames")
                if isinstance(net_names, list) and net_names:
                    for n in net_names:
                        nn = str(n or "").strip()
                        if not nn:
                            continue
                        safe_actions.append(DesignAction(
                            action_type=DesignActionType.DEFINE_NET,
                            description=(a.description or f"Define net {nn}").strip(),
                            parameters={"net": nn, "pads": params.get("pads") or []},
                            requires_approval=a.requires_approval,
                        ))
                    continue

                # Pads are optional: DEFINE_NET can just create the net object.
                pads = params.get("pads")
                if pads is None:
                    for k in ("pin_refs", "pin_references", "pinReferences", "pins"):
                        if k in params:
                            pads = params.get(k)
                            break
                if pads is not None and not isinstance(pads, list):
                    # Keep the action; handler will return a helpful error.
                    pads = [pads]
                params["pads"] = pads if isinstance(pads, list) else []

                if not str(params.get("net", "") or "").strip():
                    malformed.append("DEFINE_NET missing net")
                    logger.error(
                        "Dropping malformed DEFINE_NET action: desc=%r params=%r",
                        a.description, a.parameters
                    )
                    continue

                a.parameters = params
                safe_actions.append(a)
                continue

            if a.action_type == DesignActionType.ASSIGN_NETS:
                params = dict(a.parameters or {})
                assigns = params.get("assignments")

                # Common alternate container key.
                if not isinstance(assigns, list) or not assigns:
                    assigns = params.get("net_assignments") or params.get("netAssignments")

                normalized: List[Dict[str, Any]] = []

                def _split_pin_ref(text: str) -> Optional[Tuple[str, str]]:
                    s = str(text or "").strip()
                    if not s:
                        return None
                    m = re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", s)
                    if m:
                        return m.group(1).upper(), m.group(2)
                    return None

                if isinstance(assigns, list) and assigns:
                    for it in assigns:
                        if not isinstance(it, dict):
                            continue
                        net = (
                            it.get("net")
                            or it.get("net_name")
                            or it.get("netName")
                            or it.get("name")
                        )
                        net_s = str(net or "").strip()
                        if not net_s:
                            continue

                        # Either explicit {ref,pad}, or a "pin_ref"/"pin_references" style.
                        ref = it.get("ref") or it.get("reference") or it.get("designator")
                        pad = it.get("pad") or it.get("pin") or it.get("pad_num")

                        if isinstance(ref, str) and isinstance(pad, (str, int, float)) and str(pad).strip():
                            normalized.append(
                                {"net": net_s, "ref": str(ref).strip().upper(), "pad": str(pad).strip()}
                            )
                            continue

                        pin_ref = it.get("pin_ref") or it.get("pinRef")
                        if isinstance(pin_ref, str):
                            sp = _split_pin_ref(pin_ref)
                            if sp:
                                normalized.append({"net": net_s, "ref": sp[0], "pad": sp[1]})
                                continue

                        pin_refs = it.get("pin_references") or it.get("pinReferences") or it.get("pin_refs")
                        if isinstance(pin_refs, list):
                            for pr in pin_refs:
                                sp = _split_pin_ref(str(pr))
                                if sp:
                                    normalized.append({"net": net_s, "ref": sp[0], "pad": sp[1]})
                            continue

                        # Common grouped encoding:
                        # {"net_name":"GND","pads":["U3/8","U3/22","C1/2", ...]}
                        pads = it.get("pads") or it.get("pad_refs") or it.get("padRefs")
                        if isinstance(pads, list):
                            for p in pads:
                                if isinstance(p, dict):
                                    r = p.get("ref") or p.get("reference") or p.get("designator")
                                    pn = p.get("pad") or p.get("pin") or p.get("pad_num")
                                    if isinstance(r, str) and str(pn or "").strip():
                                        normalized.append(
                                            {"net": net_s, "ref": str(r).strip().upper(), "pad": str(pn).strip()}
                                        )
                                    continue
                                sp = _split_pin_ref(str(p))
                                if sp:
                                    normalized.append({"net": net_s, "ref": sp[0], "pad": sp[1]})
                            continue

                        # Another common shape: {net_name, pin_ref: "U1-1"}.
                        pin_ref_alt = it.get("pin_ref") or it.get("pin_ref")
                        if isinstance(pin_ref_alt, str):
                            sp = _split_pin_ref(pin_ref_alt)
                            if sp:
                                normalized.append({"net": net_s, "ref": sp[0], "pad": sp[1]})

                if normalized:
                    params["assignments"] = normalized
                    a.parameters = params
                    safe_actions.append(a)
                    continue

                malformed.append("ASSIGN_NETS missing assignments")
                logger.error(
                    "Dropping malformed ASSIGN_NETS action: desc=%r params=%r",
                    a.description, a.parameters
                )
                continue

            if a.action_type not in {DesignActionType.DOWNLOAD_SYMBOL, DesignActionType.DOWNLOAD_FOOTPRINT}:
                safe_actions.append(a)
                continue

            params = a.parameters or {}
            part_name = ''
            for key in ('part_name', 'query', 'mpn', 'part', 'name'):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    part_name = v.strip()
                    break
            if not part_name:
                missing.append('part number')
                continue

            # For footprints, package is often required because footprint names are typically package-based.
            package_hint = ''
            if a.action_type == DesignActionType.DOWNLOAD_FOOTPRINT:
                v = params.get('package')
                if isinstance(v, str) and v.strip():
                    package_hint = v.strip()

            try:
                if a.action_type == DesignActionType.DOWNLOAD_SYMBOL:
                    results = self._library_manager.search_parts_sync(part_name, limit=10)
                    # If we can't find anything at all, don't propose a download.
                    if not results:
                        missing.append(f"symbol for {part_name}")
                        continue
                    safe_actions.append(a)
                    continue

                # DOWNLOAD_FOOTPRINT
                resolver = getattr(self._library_manager, 'resolve_best_footprint_path', None)
                resolved = None
                if callable(resolver):
                    resolved = resolver(part_name, package_hint=package_hint or None)
                    if resolved is None and package_hint:
                        # Try resolving from package alone.
                        resolved = resolver(package_hint)

                if resolved is None:
                    # No footprint could be located. Ask for package if missing.
                    if not package_hint:
                        missing.append(f"package for {part_name}")
                    else:
                        missing.append(f"footprint for {part_name} ({package_hint})")
                    continue

                # We found something placeable; downloads may still be useful, but
                # avoid proposing a download as the *only* next step.
                safe_actions.append(a)

            except Exception:
                # If preflight fails unexpectedly, keep the action rather than dropping user intent.
                safe_actions.append(a)

        # If we dropped download actions and nothing remains, turn it into a clear clarifying question.
        if not safe_actions and actions:
            if malformed:
                return (
                    "I skipped malformed action(s) from the model output. "
                    "I need concrete parameters (for example ADD_COMPONENT.query "
                    "or DEFINE_NET.net/pads) before executing."
                ), []
            # Prefer asking for package (most common missing detail).
            if any(m.startswith('package for ') for m in missing):
                part = missing[0].replace('package for ', '')
                msg = (
                    f"I couldn’t locate a specific footprint for '{part}'. Footprints are usually package-based, "
                    f"so I need the package (e.g., SSOP-28 or TSSOP-28) to pick the right one."
                )
                return msg, []

            if missing:
                msg = f"I couldn’t locate: {', '.join(missing)}. I won’t propose a download until I can find it."
                return msg, []
        return assistant_message, safe_actions
    

    def _interpret_with_llm(self, 
                            text: str,
                            context: Dict[str, Any]) -> List[DesignAction]:
        """Use LLM for advanced interpretation."""
        from ..llm.client import LLMError, LLMMessage
        if not self._llm_client or not getattr(self._llm_client, "is_available", False):
            raise LLMError("LLM is required for interpretation but is not available/configured.")

        # Build prompt with context
        prompt = self._build_interpretation_prompt(text, context)

        # Call LLM with a design-tool system prompt (NOT the explanation-only prompt)
        response_obj = self._llm_client.chat(
            [LLMMessage(role='user', content=prompt)],
            system_prompt=self.DESIGN_SYSTEM_PROMPT,
        )
        self._record_llm_usage(response_obj)
        response = response_obj.content

        # Parse response into actions
        return self._parse_llm_response(response)
    
    def _build_interpretation_prompt(self, text: str, context: Dict[str, Any]) -> str:
        """Build prompt for LLM interpretation."""
        tools = [a.name for a in self.SUPPORTED_LLM_TOOLS]

        prompt = f"""You are VibeCAD, a KiCad design assistant.

Interpret the user's request into one or more TOOL calls. The tools are the action types listed below.

TOOLS (action_type values):
{', '.join(tools)}

USER:
"{text}"

CONTEXT:
- Active editor: {context.get('active_editor', 'pcb')}
- Selected items: {context.get('selected_items', [])}

Return ONLY a JSON array of actions with this shape:
[
    {{
        "action_type": "TOOL_NAME",
        "description": "short human summary",
        "parameters": {{ "key": "value" }},
        "requires_approval": true,
        "preview_text": "optional short preview for user"
    }}
]

Parameter schema rules (use exactly these keys; do not use aliases like "id"):
- MOVE_COMPONENT.parameters = {{ "ref": "U1", "location": {{ "x": 10.0, "y": 20.0 }} }}  (mm)
- ROTATE_COMPONENT.parameters = {{ "ref": "U1", "angle": 90 }}  (degrees)
- ADD_COMPONENT.parameters = {{ "query": "Library:Footprint_Name" }}; optional "location": {{ "x": ..., "y": ... }}
- DELETE_COMPONENT.parameters = {{ "ref": "U1" }}
- ALIGN_COMPONENTS.parameters = {{ "refs": ["U1","J1"], "direction": "horizontal" }}
- DEFINE_BOARD_OUTLINE.parameters = {{ "width": 80.0, "height": 50.0, "shape": "rectangle" }}
  optional: "corner_radius" for rounded_rectangle, "diameter" for circle

Rules:
- If the request is purely informational, return []
- Prefer the minimal number of actions
- Do not invent board coordinates; use references/nets/pins when possible
"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> List[DesignAction]:
        """Parse LLM response into actions."""
        import json
        from ..llm.client import LLMError
        if not response:
            raise LLMError("LLM returned empty response content.")
        start = response.find('[')
        end = response.rfind(']') + 1
        if start < 0 or end <= start:
            raise LLMError("Invalid LLM response: expected a JSON array.")
        try:
            data = json.loads(response[start:end])
        except Exception as e:
            raise LLMError(f"Invalid LLM response: failed to parse JSON array: {e}") from e
        if not isinstance(data, list):
            raise LLMError("Invalid LLM response: expected a JSON array.")
        actions: List[DesignAction] = []
        for idx, item in enumerate(data):
            a = self._parse_action_item(item)
            if a is None:
                raise LLMError(f"Invalid LLM action item at index {idx}.")
            actions.append(a)
        return actions

    # ------------------------------------------------------------------
    # Shared helpers for action JSON parsing (used by _chat_with_llm and
    # _parse_llm_response).
    # ------------------------------------------------------------------

    _APPROVAL_REQUIRED_TYPES = frozenset({
        DesignActionType.ADD_COMPONENT,
        DesignActionType.MOVE_COMPONENT,
        DesignActionType.ROTATE_COMPONENT,
        DesignActionType.DRAW_TRACK,
        DesignActionType.DRAW_WIRE,
        DesignActionType.ROUTE_NET,
        DesignActionType.ADD_VIA,
        DesignActionType.ASSIGN_NETS,
        DesignActionType.ADD_POLYGON,
        DesignActionType.ADD_TEXT,
        DesignActionType.ADD_MOUNTING_HOLE,
        DesignActionType.DEFINE_BOARD_OUTLINE,
        DesignActionType.UPDATE_BOM_FIELDS,
        DesignActionType.AUTOROUTE_BOARD,
        DesignActionType.SET_LAYER_COUNT,
        DesignActionType.DELETE_TRACKS,
        DesignActionType.DELETE_COMPONENT,
    })

    @staticmethod
    def _coerce_bool(value: Any, default: bool = True) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
        return default

    @staticmethod
    def _normalize_action_type(value: Any) -> 'DesignActionType':
        if isinstance(value, DesignActionType):
            return value
        if value in (None, ""):
            return DesignActionType.UNKNOWN
        raw_name = str(value).strip()
        if not raw_name:
            return DesignActionType.UNKNOWN
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw_name).strip("_").upper()
        if cleaned:
            try:
                return DesignActionType[cleaned]
            except KeyError:
                pass
        return DesignActionType.UNKNOWN

    @staticmethod
    def _get_finish_reason(response_obj: Any) -> Optional[str]:
        try:
            rr = getattr(response_obj, 'raw_response', None) or {}
            choices = rr.get('choices') or []
            if choices and isinstance(choices, list) and isinstance(choices[0], dict):
                fr = choices[0].get('finish_reason')
                return str(fr) if fr is not None else None
        except Exception:
            pass
        return None

    @classmethod
    def _parse_action_item(cls, item: Any) -> Optional['DesignAction']:
        """Parse a dict into a DesignAction with fuzzy key/type matching."""
        if not isinstance(item, dict):
            return None
        # Action type from various key names
        action_type_val = None
        for k in ('action_type', 'actiontype', 'actionType', 'type', 'tool', 'tool_name', 'toolName'):
            if k in item and item.get(k) not in (None, ""):
                action_type_val = item.get(k)
                break
        action_type = cls._normalize_action_type(action_type_val)
        if action_type == DesignActionType.UNKNOWN:
            return None

        # Parameters (canonical: object; for parameterless actions allow missing/null)
        params = item.get('parameters') or item.get('params') or item.get('arguments')
        if params is None:
            params = {}
        if not isinstance(params, dict):
            # Allow a missing/blank parameters field for actions that take no parameters.
            if action_type in (DesignActionType.RUN_DRC, DesignActionType.RUN_ERC, DesignActionType.AUTOROUTE_BOARD, DesignActionType.DELETE_TRACKS):
                params = {}
            else:
                return None
        params = normalize_action_parameters(action_type, params)

        # Approval
        requires_val = None
        for rk in ('requires_approval', 'requiresApproval', 'requiresapproval',
                    'approval_required', 'approvalRequired'):
            if rk in item:
                requires_val = item.get(rk)
                break
        requires_approval = cls._coerce_bool(requires_val, default=True)
        if action_type == DesignActionType.SEARCH_PART:
            requires_approval = False
        elif action_type in cls._APPROVAL_REQUIRED_TYPES:
            requires_approval = True

        return DesignAction(
            action_type=action_type,
            description=str(item.get('description', '')).strip(),
            parameters=params,
            requires_approval=requires_approval,
            preview_text=item.get('preview_text') or item.get('previewText'),
        )

    def _chat_with_llm(self, text: str, context: Dict[str, Any]) -> Tuple[str, List[DesignAction], float]:
        """Use the LLM to produce both an assistant response and tool-like actions.

        Sends the full conversation history so multi-turn interactions work.
        """
        import json
        try:
            from ..llm.client import LLMError, LLMMessage
        except Exception:  # pragma: no cover
            LLMError = Exception

        def _looks_truncated_json(raw_text: str, err: Optional[Exception] = None) -> bool:
            s = (raw_text or "").strip()
            if not s:
                return False
            # Heuristics: partial object, missing closing braces/brackets, or common truncation errors.
            if '"actions"' in s or '"assistant_message"' in s:
                if s.count("{") > s.count("}"):
                    return True
                if s.count("[") > s.count("]"):
                    return True
                if not s.endswith("}"):
                    # Valid JSON can end with whitespace; we already stripped.
                    return True
            if err is not None:
                msg = str(err)
                if "Unterminated string" in msg or "Expecting value" in msg or "EOF" in msg:
                    return True
            return False

        try:
            tools = [a.name for a in self.SUPPORTED_LLM_TOOLS]

            # Build the message list: history + current user turn with
            # structured instructions so the model always produces JSON.
            messages: list = []

            # Keep chat memory for follow-up imperative commands ("place ...",
            # "move ..."), but bound it so stale earlier attempts do not dominate.
            q_clean = text.strip().lower()
            is_new_command = bool(re.match(r'^(add|place|search|find|download|create|get)\b', q_clean))
            if is_new_command:
                history_subset = self._conversation_history[-6:-1]
            else:
                history_subset = self._conversation_history[:-1]

            for entry in history_subset:  # exclude the latest user turn (we add it below)
                messages.append(LLMMessage(
                    role=entry["role"],
                    content=entry["content"],
                ))

            # The final user message includes the structured instructions.
            user_prompt = f"""USER REQUEST:
{text}

CONTEXT:
- Active editor: {context.get('active_editor', 'pcb')}
- Available action_type values: {', '.join(tools)}
- Board outline already defined: {bool(context.get('outline_defined', False))}
- Board outline (mm): width={context.get('board_width','?')} height={context.get('board_height','?')} origin=({context.get('board_origin_x','?')},{context.get('board_origin_y','?')}) center=({context.get('board_center_x','?')},{context.get('board_center_y','?')})

Return JSON only:
{{
  "assistant_message": "short user-facing reply",
  "actions": [
    {{
      "action_type": "TOOL_NAME",
      "description": "short summary",
      "parameters": {{ }},
      "requires_approval": true,
      "preview_text": "short preview"
    }}
  ]
}}

Rules:
- If inputs are missing, ask one clarifying question and return actions: [].
- Reuse conversation history; do not re-ask answered questions.
- For corner-relative requests (e.g. "each corner"), infer dimensions from current board outline context/history when available.
- Treat part-like tokens as part numbers; do not reinterpret suffixes.
- Omit ADD_COMPONENT location when user did not provide one.
- If board outline is already defined, do not propose DEFINE_BOARD_OUTLINE unless user asked to change it.
- DEFINE_BOARD_OUTLINE may use width/height plus optional shape controls (shape, corner_radius, diameter).
- Do not propose DOWNLOAD_* actions unless a matching asset is already resolved.
- Use exactly one key per parameter (no aliases): component reference key is "ref" (never "id"); location is an object {{ "x": ..., "y": ... }} (never a 2-item array).
- Output valid JSON only."""

            messages.append(LLMMessage(role='user', content=user_prompt))

            # --- Call LLM and parse response ---------------------------
            did_retry_for_length = False
            did_retry_for_schema = False
            old_max_tokens: Optional[int] = None
            try:
                while True:
                    response_obj = self._llm_client.chat(
                        messages,
                        system_prompt=self.DESIGN_SYSTEM_PROMPT,
                    )
                    self._record_llm_usage(response_obj)
                    raw = (response_obj.content or "").strip()

                    if not raw:
                        raise LLMError("LLM returned empty content.")

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("DesignAgent raw LLM response: %s", raw[:500])

                    # Try to parse as JSON.
                    obj: Any = None
                    try:
                        obj = json.loads(raw)
                    except Exception as e:
                        finish_reason = self._get_finish_reason(response_obj)
                        truncated = _looks_truncated_json(raw, err=e) or (finish_reason == 'length')
                        if not did_retry_for_length and truncated:
                            # Ask the model to resend the full JSON object. This is not a
                            # deterministic fallback; it keeps the LLM in the loop.
                            old_max_tokens = int(getattr(self._llm_client.config, 'max_tokens', 1024))
                            self._llm_client.config.max_tokens = min(8192, max(2048, old_max_tokens * 4))
                            did_retry_for_length = True
                            messages.append(LLMMessage(
                                role="user",
                                content=(
                                    "Your previous response was truncated/invalid JSON. "
                                    "Resend the COMPLETE JSON object only (no markdown, no commentary)."
                                ),
                            ))
                            continue
                        raise LLMError(f"Invalid LLM response: expected JSON object: {e}") from e

                    # If we couldn't get a dict, retry once for truncation.
                    if not isinstance(obj, dict):
                        has_partial = '"actions"' in raw or '"assistant_message"' in raw
                        finish_reason = self._get_finish_reason(response_obj)
                        truncated = raw.count('{') > raw.count('}')
                        if (
                            not did_retry_for_length
                            and has_partial
                            and (finish_reason == 'length' or truncated)
                        ):
                            old_max_tokens = int(getattr(self._llm_client.config, 'max_tokens', 1024))
                            self._llm_client.config.max_tokens = min(8192, max(2048, old_max_tokens * 4))
                            did_retry_for_length = True
                            continue

                        raise LLMError("Invalid LLM response: expected a JSON object.")

                    # --- Successful JSON parse ---
                    assistant_message: Optional[str] = None
                    for key in ("assistant_message", "assistantMessage", "message", "assistant", "response"):
                        if key in obj and obj.get(key) not in (None, ""):
                            assistant_message = str(obj.get(key)).strip()
                            break
                    if assistant_message is None:
                        raise LLMError("Invalid LLM response: missing 'assistant_message'.")

                    actions_key = None
                    for k in ('actions', 'tool_actions', 'toolActions'):
                        if k in obj:
                            actions_key = k
                            break
                    if actions_key is None:
                        raise LLMError("Invalid LLM response: missing 'actions' array.")
                    actions_json = obj.get(actions_key) or []
                    if not isinstance(actions_json, list):
                        raise LLMError("Invalid LLM response: 'actions' must be a JSON array.")

                    actions: List[DesignAction] = []
                    for idx, item in enumerate(actions_json):
                        a = self._parse_action_item(item)
                        if a is None:
                            if not did_retry_for_schema:
                                did_retry_for_schema = True
                                logger.error(
                                    "Invalid LLM action item at index %d. item=%r raw_head=%r",
                                    idx,
                                    item,
                                    raw[:500],
                                )
                                messages.append(LLMMessage(
                                    role="user",
                                    content=(
                                        f"Your previous JSON had an invalid action item at index {idx}. "
                                        "Resend the COMPLETE JSON object only. "
                                        "Every action must be an object with keys: action_type, description, parameters (object), requires_approval."
                                    ),
                                ))
                                continue
                            raise LLMError(f"Invalid LLM action item at index {idx}.")
                        actions.append(a)
                    confidence = 0.9 if actions else 0.5
                    return assistant_message, actions, confidence
            finally:
                if old_max_tokens is not None:
                    try:
                        self._llm_client.config.max_tokens = old_max_tokens
                    except Exception:
                        pass

        except LLMError:
            raise
        except Exception as e:
            logger.exception("LLM chat failed: %s", e)
            raise LLMError(f"LLM chat failed: {e}") from e
    
    def _generate_description(self, 
                              action_type: DesignActionType,
                              params: Dict[str, Any]) -> str:
        """Generate human-readable description for an action."""
        descriptions = {
            DesignActionType.SEARCH_PART: 
                lambda p: f"Search for '{p.get('query', 'component')}'",
            DesignActionType.DOWNLOAD_SYMBOL:
                lambda p: f"Download symbol for {p.get('part_name', 'component')}",
            DesignActionType.DOWNLOAD_FOOTPRINT:
                lambda p: f"Download footprint for {p.get('part_name', 'component')}",
            DesignActionType.ADD_COMPONENT:
                lambda p: f"Place {p.get('query', p.get('part_name', 'component'))} on the PCB",
            DesignActionType.DRAW_TRACK:
                lambda p: f"Draw track from {p.get('from_point', '?')} to {p.get('to_point', '?')}",
            DesignActionType.ROUTE_NET:
                lambda p: f"Route the '{p.get('net_name', '?')}' net",
            DesignActionType.EXPORT_BOM:
                lambda p: f"Export BOM{' for ' + p.get('format', '') if p.get('format') else ''}",
            DesignActionType.MOVE_COMPONENT:
                lambda p: f"Move {p.get('ref', '?')} to {p.get('location', '?')}",
            DesignActionType.ROTATE_COMPONENT:
                lambda p: f"Rotate {p.get('ref', '?')} by {p.get('angle', '90')}°",
            DesignActionType.ALIGN_COMPONENTS:
                lambda p: f"Align {p.get('refs', '?')} {p.get('direction', '?')}",
            DesignActionType.ADD_MOUNTING_HOLE:
                lambda p: f"Add {p.get('size', '3.2')}mm mounting hole at {p.get('location', '?')}",
            DesignActionType.DEFINE_BOARD_OUTLINE:
                lambda p: (
                    "Define board outline"
                    + (f" {p.get('width')}x{p.get('height')}mm" if (p.get('width') is not None and p.get('height') is not None) else "")
                    + (f" shape={p.get('shape')}" if p.get('shape') else "")
                    + (f" r={p.get('corner_radius')}mm" if p.get('corner_radius') not in (None, "") else "")
                ),
            DesignActionType.ADD_TEXT:
                lambda p: f"Add text '{p.get('text', '?')}'",
            DesignActionType.RUN_DRC:
                lambda p: "Run Design Rule Check",
            DesignActionType.RUN_ERC:
                lambda p: "Run Electrical Rule Check",
            DesignActionType.ADD_VIA:
                lambda p: f"Add via at {p.get('location', '?')}",
            DesignActionType.ASSIGN_NETS:
                lambda p: f"Assign nets to pads" + (f" ({len(p.get('assignments', []) or [])} assignments)" if isinstance(p.get('assignments', None), list) else ""),
            DesignActionType.DEFINE_NET:
                lambda p: f"Define net {p.get('net', '') or '?'}" + (f" ({len(p.get('pads', []) or [])} pads)" if isinstance(p.get('pads', None), list) else ""),
            DesignActionType.ADD_POLYGON:
                lambda p: f"Add copper zone on {p.get('layer', 'F.Cu')} for {p.get('net', 'GND')}",
            DesignActionType.AUTOROUTE_BOARD:
                lambda p: "Autoroute all unconnected nets",
            DesignActionType.SET_LAYER_COUNT:
                lambda p: f"Set copper layer count to {p.get('count', 2)}",
        }
        
        generator = descriptions.get(action_type, lambda p: action_type.name)
        return generator(params)
    
    def create_preview(self, action: DesignAction, context: Dict[str, Any] = None) -> str:
        """Create a preview description for an action.
        
        Args:
            action: The action to preview
            context: Design context
        
        Returns:
            Human-readable preview string
        """
        context = context or {}
        
        preview_lines = [
            f"🎯 Action: {action.description}",
            "",
        ]
        
        if action.action_type == DesignActionType.SEARCH_PART:
            query = self._extract_search_query(action) or '?'
            preview_lines.extend([
                f"Search query: {query}",
                "",
                "This will search KiCad's built-in libraries first,",
                "then fall back to SnapEDA if needed.",
            ])

        elif action.action_type == DesignActionType.DOWNLOAD_SYMBOL:
            part = action.parameters.get('part_name', action.parameters.get('query', '?'))
            dest = action.parameters.get('destination', 'project library')
            preview_lines.extend([
                f"Part: {part}",
                f"Asset: KiCad schematic symbol (.kicad_sym)",
                f"Install to: {dest}",
                "",
                "Will check KiCad's built-in libraries first.",
                "If the part isn't built-in, it will be downloaded from SnapEDA.",
            ])

        elif action.action_type == DesignActionType.DOWNLOAD_FOOTPRINT:
            part = action.parameters.get('part_name', action.parameters.get('query', '?'))
            dest = action.parameters.get('destination', 'project library')
            package = action.parameters.get('package', '')
            preview_lines.extend([
                f"Part: {part}",
                f"Asset: KiCad PCB footprint (.kicad_mod)",
            ])
            if package:
                preview_lines.append(f"Package: {package}")
            preview_lines.extend([
                f"Install to: {dest}",
                "",
                "Will check KiCad's built-in libraries first.",
                "If the part isn't built-in, it will be downloaded from SnapEDA.",
            ])

        elif action.action_type == DesignActionType.ADD_COMPONENT:
            q = action.parameters.get('query', action.parameters.get('part_name', '?'))
            loc = action.parameters.get('location', '').strip() if isinstance(action.parameters.get('location', ''), str) else ''
            preview_lines.extend([
                f"Component/footprint: {q}",
                "",
                "This will search your local KiCad footprint libraries and place the best match on the current PCB.",
            ])
            if loc:
                preview_lines.append(f"Requested location: {loc} mm")
            else:
                preview_lines.append("Placement: board center (default)")
            preview_lines.extend([
                "",
                "Note: Reference designator will be auto-assigned if possible (e.g., BT1).",
            ])

        elif action.action_type == DesignActionType.DRAW_TRACK:
            from_p = action.parameters.get('from_point', '?')
            to_p = action.parameters.get('to_point', '?')
            preview_lines.extend([
                f"From: {from_p}",
                f"To: {to_p}",
                "",
                "A copper track will be drawn between these points.",
                "The route will use the current layer and track width.",
            ])

        elif action.action_type == DesignActionType.ASSIGN_NETS:
            assigns = []
            try:
                assigns = action.parameters.get('assignments', []) or []
            except Exception:
                assigns = []
            preview_lines.append("This will assign net names to specific pads:")
            shown = 0
            for it in assigns:
                if not isinstance(it, dict):
                    continue
                ref = it.get('ref')
                pad = it.get('pad')
                net = it.get('net')
                if ref and pad and net:
                    preview_lines.append(f"- {str(ref).upper()} pad {pad} → {net}")
                    shown += 1
                if shown >= 8:
                    break
            if isinstance(assigns, list) and len(assigns) > shown:
                preview_lines.append(f"…and {len(assigns) - shown} more")
            preview_lines.extend([
                "",
                "Note: Preferred workflow is KiCad → Tools → Update PCB from Schematic.",
            ])

        elif action.action_type == DesignActionType.DEFINE_NET:
            net_name = ''
            try:
                net_name = str(action.parameters.get('net', '') or '').strip()
            except Exception:
                net_name = ''
            pads = []
            try:
                pads = action.parameters.get('pads', []) or []
            except Exception:
                pads = []
            preview_lines.append(f"This will assign net '{net_name or '?'}' to these pads:")
            shown = 0
            for p in pads:
                if isinstance(p, str) and p.strip():
                    preview_lines.append(f"- {p.strip()}")
                    shown += 1
                elif isinstance(p, dict):
                    ref = p.get('ref')
                    pad = p.get('pad')
                    if ref and pad:
                        preview_lines.append(f"- {str(ref).upper()}/{pad}")
                        shown += 1
                if shown >= 10:
                    break
            if isinstance(pads, list) and len(pads) > shown:
                preview_lines.append(f"…and {len(pads) - shown} more")
            preview_lines.extend([
                "",
                "Tip: Use this to name critical nets (GND, 5V, 3V3, SPI, I2C) before routing.",
            ])
        
        elif action.action_type == DesignActionType.EXPORT_BOM:
            fmt = action.parameters.get('format', 'generic CSV')
            preview_lines.extend([
                f"Format: {fmt}",
                "",
                "The Bill of Materials will include all placed components.",
                "You'll be prompted to choose the output location.",
            ])
        
        elif action.action_type == DesignActionType.MOVE_COMPONENT:
            ref = action.parameters.get('ref', '?')
            loc = action.parameters.get('location', '?')
            preview_lines.extend([
                f"Component: {ref}",
                f"New position: {loc}",
                "",
                "The component will be moved to the new position.",
                "This action is undoable via Edit → Undo.",
            ])
        
        else:
            preview_lines.append(f"Parameters: {action.parameters}")
        
        if action.undoable:
            preview_lines.extend(["", "✓ This action can be undone"])
        
        return "\n".join(preview_lines)
    
    async def execute_action(self, 
                              action: DesignAction,
                              context: Dict[str, Any] = None) -> DesignAction:
        """Execute an approved action.
        
        Args:
            action: The action to execute (must be approved)
            context: Design context including board, schematic data
        
        Returns:
            The action with updated execution status
        """
        if action.requires_approval and not action.approved:
            action.success = False
            action.result_message = "Action requires approval before execution"
            return action
        
        context = context or {}
        
        try:
            # Dispatch to appropriate handler
            handler = self._get_action_handler(action.action_type)
            if handler:
                success, message = await handler(action, context)
                action.success = success
                action.result_message = message
                if not success:
                    logger.warning(
                        "Action failed: type=%s desc=%r params=%r message=%r",
                        action.action_type.name,
                        action.description,
                        action.parameters,
                        action.result_message,
                    )
            else:
                action.success = False
                action.result_message = f"No handler for action type: {action.action_type.name}"
                logger.error(
                    "Missing action handler: type=%s desc=%r params=%r",
                    action.action_type.name,
                    action.description,
                    action.parameters,
                )
            
            action.executed = True
            
        except Exception as e:
            logger.exception(f"Action execution failed: {e}")
            action.success = False
            action.result_message = f"Execution failed: {e}"
            action.executed = True
        
        return action
    
    def _get_action_handler(self, action_type: DesignActionType):
        """Get the handler function for an action type."""
        handlers = {
            DesignActionType.SEARCH_PART: self._handle_search_part,
            DesignActionType.DOWNLOAD_SYMBOL: self._handle_download_symbol_or_footprint,
            DesignActionType.DOWNLOAD_FOOTPRINT: self._handle_download_symbol_or_footprint,
            DesignActionType.ADD_COMPONENT: self._handle_add_component,
            DesignActionType.EXPORT_BOM: self._handle_export_bom,
            DesignActionType.DRAW_TRACK: self._handle_draw_track,
            DesignActionType.ASSIGN_NETS: self._handle_assign_nets,
            DesignActionType.DEFINE_NET: self._handle_define_net,
            DesignActionType.MOVE_COMPONENT: self._handle_move_component,
            DesignActionType.ROTATE_COMPONENT: self._handle_rotate_component,
            DesignActionType.RUN_DRC: self._handle_run_drc,
            DesignActionType.RUN_ERC: self._handle_run_erc,
            DesignActionType.ADD_VIA: self._handle_add_via,
            DesignActionType.DEFINE_BOARD_OUTLINE: self._handle_define_board_outline,
            DesignActionType.ADD_MOUNTING_HOLE: self._handle_add_mounting_hole,
            DesignActionType.ALIGN_COMPONENTS: self._handle_align_components,
            DesignActionType.ADD_TEXT: self._handle_add_text,
            DesignActionType.ADD_POLYGON: self._handle_add_polygon,
            DesignActionType.AUTOROUTE_BOARD: self._handle_autoroute,
            DesignActionType.SET_LAYER_COUNT: self._handle_set_layer_count,
            DesignActionType.DELETE_TRACKS: self._handle_delete_tracks,
            DesignActionType.DELETE_COMPONENT: self._handle_delete_component,
            DesignActionType.SEARCH_WEB: self._handle_search_web,
            DesignActionType.LOOKUP_DATASHEET: self._handle_lookup_datasheet,
        }
        return handlers.get(action_type)


    

    @staticmethod
    def _extract_search_query(action: DesignAction) -> str:
        """Best-effort extraction of a SEARCH_PART query from an action.

        The LLM may use any parameter key, so after checking well-known keys
        we fall back to *any* non-empty string value in the parameters dict,
        then try to pull a part-number-like token from the description.
        """
        params = action.parameters or {}
        if isinstance(params, dict):
            # 1. Well-known keys first.
            for key in ("query", "part_name", "mpn", "part", "name",
                        "search_query", "keyword", "q", "component",
                        "search", "term", "value"):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()

            # 2. Any non-empty string value (LLM used an unexpected key).
            for v in params.values():
                if isinstance(v, str) and v.strip():
                    logger.debug(
                        "SEARCH_PART: fell back to arbitrary param value %r "
                        "(keys were %s)", v, list(params.keys()),
                    )
                    return v.strip()

        def _clean_free_text(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return ""
            # Strip leading command-y verbs to avoid polluting local search tokens.
            s = re.sub(r"^(search|find|lookup|look\s*up|web\s*search)\s*(for\s+)?", "", s, flags=re.IGNORECASE).strip()
            return s

        # 3. Extract from description text.
        desc = (action.description or "").strip()
        desc_clean = _clean_free_text(desc)
        # "Search for 'ATmega328P'"
        m = re.search(r"['\"]([A-Za-z0-9][A-Za-z0-9_+\-./ ]{1,80}?)['\"]", desc)
        if m and m.group(1).strip():
            return _clean_free_text(m.group(1))
        # "Search for ATmega328P-PU component"
        m = re.search(r"(?:search|find|look)\s+(?:for\s+)?([A-Za-z0-9][A-Za-z0-9_+\-./ ]{1,80})", desc, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            return _clean_free_text(m.group(1))
        # "… for ATmega328P-PU"
        m = re.search(r"\bfor\s+([A-Za-z0-9][A-Za-z0-9_+\-./]{2,80})", desc, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            return _clean_free_text(m.group(1))
        # Last resort: anything that looks like a part number in description
        m = re.search(r'\b([A-Z][A-Za-z]*\d{2,}[A-Za-z0-9\-]*)', desc)
        if m and m.group(1).strip():
            return m.group(1).strip()

        # If description is non-empty but doesn't match our strict patterns
        # (e.g. "USB-C connector"), use it directly.
        if desc_clean:
            return desc_clean

        # 4. Preview text.
        prev = (action.preview_text or "").strip()
        m = re.search(r"(?m)^Search\s+query:\s*(.+?)\s*$", prev)
        if m and m.group(1).strip():
            return _clean_free_text(m.group(1))
        # Part number in preview
        m = re.search(r'\b([A-Z][A-Za-z]*\d{2,}[A-Za-z0-9\-]*)', prev)
        if m and m.group(1).strip():
            return m.group(1).strip()

        logger.warning(
            "SEARCH_PART: could not extract query. params=%r desc=%r preview=%r",
            params, desc, prev,
        )
        return ""

    @staticmethod
    def _parse_location_mm(location: Any) -> Optional[Tuple[float, float]]:
        """Parse common location formats into (x_mm, y_mm)."""
        if isinstance(location, dict):
            if "x" in location and "y" in location:
                try:
                    return float(location["x"]), float(location["y"])
                except Exception:
                    return None
            return None

        if isinstance(location, (list, tuple)) and len(location) == 2:
            try:
                return float(location[0]), float(location[1])
            except Exception:
                return None

        if isinstance(location, str):
            loc_s = location.strip()
            if not loc_s:
                return None

            # Handle dict-like strings from LLM/tool output, e.g. "{'x': 10, 'y': -5}".
            if loc_s.startswith("{") and loc_s.endswith("}"):
                try:
                    parsed = ast.literal_eval(loc_s)
                    return DesignAgent._parse_location_mm(parsed)
                except Exception:
                    # Fallback regex for loosely formatted dict-like values.
                    mx = re.search(r"['\"]?x['\"]?\s*:\s*(-?\d+(?:\.\d+)?)", loc_s, re.IGNORECASE)
                    my = re.search(r"['\"]?y['\"]?\s*:\s*(-?\d+(?:\.\d+)?)", loc_s, re.IGNORECASE)
                    if mx and my:
                        return float(mx.group(1)), float(my.group(1))
                    return None

            loc_s = loc_s.lower().replace("mm", "").strip()
            if loc_s.startswith("(") and loc_s.endswith(")"):
                loc_s = loc_s[1:-1].strip()

            m = re.match(
                r"^\s*(-?\d+(?:\.\d+)?)\s*(?:,|\s+)\s*(-?\d+(?:\.\d+)?)\s*$",
                loc_s,
            )
            if m:
                return float(m.group(1)), float(m.group(2))

        return None
    
    
    

    def _try_auto_resolve_overlap(
        self, footprint, ref: str, board, overlap_refs: List[str],
        pcbnew_mod, from_mm
    ) -> Optional[str]:
        """Attempt to nudge the footprint to a nearby clear position.

        Uses the PlacementAgent's force-directed resolver when available,
        otherwise does a simple spiral search.
        """
        try:
            to_mm = getattr(pcbnew_mod, 'ToMM', None)
            if not callable(to_mm) or not callable(from_mm):
                return None

            # Gather all footprints' bounding data.
            components: List[Dict[str, Any]] = []
            target_idx = -1
            for fp in board.GetFootprints():
                fp_ref = str(fp.GetReference()).upper()
                pos = fp.GetPosition()
                px, py = to_mm(pos.x), to_mm(pos.y)
                # Get bounding box size.
                w_mm, h_mm = 10.0, 10.0
                for bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                    fn = getattr(fp, bbm, None)
                    if callable(fn):
                        try:
                            bb = fn() if bbm == 'GetCourtyardBoundingBox' else fn(False, False)
                            if bb and bb.GetWidth() > 0:
                                w_mm = to_mm(bb.GetWidth())
                                h_mm = to_mm(bb.GetHeight())
                                break
                        except Exception:
                            pass
                entry = {
                    "ref": fp_ref, "x": px, "y": py,
                    "width": w_mm, "height": h_mm,
                }
                if fp_ref == ref:
                    target_idx = len(components)
                components.append(entry)

            if target_idx < 0 or len(components) < 2:
                return None

            # Use PlacementAgent if available.
            try:
                from .sub_agents.placement import PlacementAgent
                resolved = PlacementAgent.resolve_overlaps(components, clearance_mm=2.0)
                new_pos = resolved[target_idx]
                new_x = round(float(new_pos["x"]), 2)
                new_y = round(float(new_pos["y"]), 2)
                old_x = round(components[target_idx]["x"], 2)
                old_y = round(components[target_idx]["y"], 2)

                if abs(new_x - old_x) > 0.3 or abs(new_y - old_y) > 0.3:
                    target = pcbnew_mod.VECTOR2I(int(from_mm(new_x)), int(from_mm(new_y)))
                    footprint.SetPosition(target)
                    return (
                        f"Moved {ref} to ({new_x}, {new_y}) mm "
                        f"(auto-adjusted from ({old_x}, {old_y}) to clear overlaps "
                        f"with {', '.join(overlap_refs[:3])})"
                    )
            except ImportError:
                pass

            return None
        except Exception:
            logger.exception("Auto-resolve overlap failed")
            return None
    
    

    def _format_drc_results(self, errors: List[str], warnings: List[str]) -> str:
        """Format DRC results into a human-readable summary."""
        if not errors and not warnings:
            return "DRC_STATUS: PASS\nDRC passed: 0 errors, 0 warnings. Design is clean!"

        lines = []
        lines.append("DRC_STATUS: FAIL" if errors else "DRC_STATUS: PASS")
        lines.append(f"DRC Results: {len(errors)} error(s), {len(warnings)} warning(s)")
        lines.append("")

        if errors:
            lines.append("ERRORS (must fix):")
            for i, e in enumerate(errors[:20], 1):  # Limit to 20
                lines.append(f"  {i}. {e}")
            if len(errors) > 20:
                lines.append(f"  ... and {len(errors) - 20} more errors")

        if warnings:
            lines.append("")
            lines.append("WARNINGS (acceptable):")
            for i, w in enumerate(warnings[:10], 1):
                lines.append(f"  {i}. {w}")
            if len(warnings) > 10:
                lines.append(f"  ... and {len(warnings) - 10} more warnings")

        return "\n".join(lines)

    @staticmethod
    def _parse_drc_text_report(text: str) -> Tuple[List[str], List[str]]:
        """Parse a text-format DRC report into errors and warnings."""
        errors = []
        warnings = []
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('**') or line.startswith('--'):
                continue
            if 'error' in line.lower() or 'violation' in line.lower():
                errors.append(line)
            elif 'warning' in line.lower():
                warnings.append(line)
            elif line.startswith('[') and ']' in line:
                # [rule_id]: description format
                errors.append(line)
        return errors, warnings


    def _format_erc_results(self, errors: List[str], warnings: List[str]) -> str:
        """Format ERC results into a human-readable summary."""
        if not errors and not warnings:
            return "ERC_STATUS: PASS\nERC passed: 0 errors, 0 warnings. Electrical rules are clean!"

        lines = []
        lines.append("ERC_STATUS: FAIL" if errors else "ERC_STATUS: PASS")
        lines.append(f"ERC Results: {len(errors)} error(s), {len(warnings)} warning(s)")
        lines.append("")

        if errors:
            lines.append("ERRORS (must fix):")
            for i, e in enumerate(errors[:20], 1):
                lines.append(f"  {i}. {e}")
            if len(errors) > 20:
                lines.append(f"  ... and {len(errors) - 20} more errors")

        if warnings:
            lines.append("")
            lines.append("WARNINGS (acceptable):")
            for i, w in enumerate(warnings[:10], 1):
                lines.append(f"  {i}. {w}")
            if len(warnings) > 10:
                lines.append(f"  ... and {len(warnings) - 10} more warnings")

        return "\n".join(lines)

    def _run_pcb_electrical_check(self, board, pcbnew):
        """PCB-native electrical sanity checks.

        This is not schematic ERC. It flags pads with no net assigned (netcode 0).
        That provides a real signal in PCB-only sessions.
        """

        def is_mechanical_footprint(fp) -> bool:
            try:
                ref = str(fp.GetReference() or '').upper()
            except Exception:
                ref = ''
            if ref.startswith('H'):
                return True
            try:
                fpid = getattr(fp, 'GetFPID', None)
                name = str(fpid()) if callable(fpid) else ''
                name = name.lower()
                if 'mountinghole' in name or 'mounting_hole' in name:
                    return True
            except Exception:
                pass
            return False

        netless: List[str] = []
        total_pads = 0
        connected_pads = 0

        for fp in list(board.GetFootprints() or []):
            try:
                if is_mechanical_footprint(fp):
                    continue
            except Exception:
                pass

            try:
                ref = str(fp.GetReference() or '').upper()
            except Exception:
                ref = ''

            try:
                pads_iter = fp.Pads()
            except Exception:
                pads_iter = []

            for pad in pads_iter:
                # Exclude NPTH where possible
                try:
                    is_npth = getattr(pad, 'IsNPTH', None)
                    if callable(is_npth) and bool(is_npth()):
                        continue
                except Exception:
                    pass

                total_pads += 1
                try:
                    netcode = int(pad.GetNetCode())
                except Exception:
                    netcode = 0

                if netcode > 0:
                    connected_pads += 1
                    continue

                try:
                    pad_num = str(pad.GetNumber())
                except Exception:
                    pad_num = '?'
                netless.append(f"{ref}/{pad_num}")

        errors: List[str] = []
        warnings: List[str] = []

        if netless:
            errors.append(
                f"{len(netless)} pad(s) have no net assigned (examples: {', '.join(netless[:12])})"
            )

        if total_pads > 0:
            cov = connected_pads / total_pads
            if cov < 0.50:
                warnings.append(
                    f"Only {connected_pads}/{total_pads} pads ({cov:.0%}) have nets; routing/DRC may be meaningless until nets are assigned."
                )

        ok = len(errors) == 0
        lines = ["ERC_STATUS: PASS" if ok else "ERC_STATUS: FAIL"]
        lines.append("PCB Electrical Check (no schematic found)")
        lines.append(f"Pads with nets: {connected_pads}/{total_pads}")
        if errors:
            lines.append("ERRORS (must fix):")
            for i, e in enumerate(errors, 1):
                lines.append(f"  {i}. {e}")
        if warnings:
            lines.append("WARNINGS:")
            for i, w in enumerate(warnings, 1):
                lines.append(f"  {i}. {w}")
        return ok, "\n".join(lines)

    def _footprint_has_colocated_drills(self, fp, pcbnew) -> Tuple[bool, str]:
        """Detect footprints with co-located drilled holes.

        This prevents a class of footprints (often large 'module' footprints) that
        contain multiple drilled pads at the same XY, which triggers persistent
        DRC errors like 'Drilled holes co-located' even after moving the footprint.
        """

        try:
            from_mm = getattr(pcbnew, 'FromMM', None)
            eps = int(from_mm(0.01)) if callable(from_mm) else int(0.01 * 1e6)
        except Exception:
            eps = int(0.01 * 1e6)

        def _pos_key(pad) -> Tuple[int, int]:
            try:
                pos = pad.GetPosition()
                try:
                    x = int(getattr(pos, 'x'))
                    y = int(getattr(pos, 'y'))
                except Exception:
                    x = int(pos.GetX()) if hasattr(pos, 'GetX') else 0
                    y = int(pos.GetY()) if hasattr(pos, 'GetY') else 0
            except Exception:
                x, y = 0, 0
            return (int(round(x / eps)), int(round(y / eps)))

        seen: Dict[Tuple[int, int], int] = {}
        duplicates = 0

        try:
            pads_iter = fp.Pads()
        except Exception:
            pads_iter = []

        for pad in pads_iter:
            # Only consider pads that have an actual drill.
            try:
                ds = getattr(pad, 'GetDrillSize', None)
                if callable(ds):
                    drill = ds()
                    try:
                        dx = int(getattr(drill, 'x'))
                        dy = int(getattr(drill, 'y'))
                    except Exception:
                        dx = int(drill.GetX()) if hasattr(drill, 'GetX') else 0
                        dy = int(drill.GetY()) if hasattr(drill, 'GetY') else 0
                    if dx <= 0 and dy <= 0:
                        continue
                else:
                    dx = int(getattr(pad, 'GetDrillSizeX', lambda: 0)())
                    dy = int(getattr(pad, 'GetDrillSizeY', lambda: 0)())
                    if dx <= 0 and dy <= 0:
                        continue
            except Exception:
                continue

            k = _pos_key(pad)
            if k in seen:
                duplicates += 1
            else:
                seen[k] = 1

        if duplicates > 0:
            try:
                fpid = getattr(fp, 'GetFPID', None)
                name = str(fpid()) if callable(fpid) else ''
            except Exception:
                name = ''
            return True, f"co-located drilled holes detected (duplicates={duplicates}, fpid={name})"

        return False, ""


    
    def get_suggestions(self, context: Dict[str, Any] = None) -> List[str]:
        """Get contextual suggestions based on current state.
        
        Args:
            context: Current design context
        
        Returns:
            List of suggested commands the user might want
        """
        context = context or {}
        suggestions = []
        
        # Common suggestions
        suggestions.append("Export BOM for JLCPCB")
        suggestions.append("Run DRC")
        
        # Context-specific suggestions
        if context.get('has_unrouted_nets'):
            suggestions.insert(0, "Route unconnected nets")
        
        if context.get('missing_components'):
            suggestions.insert(0, "Search for missing footprints")
        
        if not context.get('has_board_outline'):
            suggestions.insert(0, "Define board outline")
        
        if context.get('selected_ref'):
            ref = context['selected_ref']
            suggestions.insert(0, f"Rotate {ref} 90°")
            suggestions.insert(1, f"Move {ref}")
        
        return suggestions[:5]  # Return top 5

    # ── Component Arrangement ───────────────────────────────────

    @staticmethod
    def _arrange_overlapping_components(board, pcbnew_mod) -> int:
        """Detect and spread overlapping components on a simple grid.

        This is a best-effort pre-routing pass.  It uses courtyard or
        bounding-box rectangles and moves any components that overlap so
        the autorouter has clear space.  Returns the number of footprints
        moved (or rotated).

        The algorithm:
        1. Compute an axis-aligned bounding box for every footprint.
        2. Check all pairs for overlap (with 2mm clearance).
        3. Re-place overlapping footprints on a grid inside the board
           outline, ensuring each placed part clears all others.
        """
        from_mm = getattr(pcbnew_mod, 'FromMM', None)
        to_mm = getattr(pcbnew_mod, 'ToMM', None)
        if not callable(from_mm) or not callable(to_mm):
            return 0

        footprints = list(board.GetFootprints())
        if len(footprints) < 2:
            return 0

        CLEARANCE_NM = int(from_mm(2.0))  # 2mm gap between courtyards

        # ── helpers ─────────────────────────────────────────────
        class _Rect:
            __slots__ = ('left', 'top', 'right', 'bottom', 'w', 'h')
            def __init__(self, l, t, r, b):
                self.left, self.top, self.right, self.bottom = l, t, r, b
                self.w = r - l
                self.h = b - t

        def _get_bb(fp):
            for attr in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                fn = getattr(fp, attr, None)
                if callable(fn):
                    try:
                        bb = fn() if attr == 'GetCourtyardBoundingBox' else fn(False, False)
                        if bb is not None and bb.GetWidth() > 0:
                            return _Rect(
                                int(bb.GetLeft()), int(bb.GetTop()),
                                int(bb.GetRight()), int(bb.GetBottom()),
                            )
                    except Exception:
                        try:
                            bb = fn()
                            if bb is not None and bb.GetWidth() > 0:
                                return _Rect(
                                    int(bb.GetLeft()), int(bb.GetTop()),
                                    int(bb.GetRight()), int(bb.GetBottom()),
                                )
                        except Exception:
                            pass
            return None

        def _overlaps(a, b):
            if a is None or b is None:
                return False
            return not (
                a.right + CLEARANCE_NM <= b.left or
                b.right + CLEARANCE_NM <= a.left or
                a.bottom + CLEARANCE_NM <= b.top or
                b.bottom + CLEARANCE_NM <= a.top
            )

        # ── 1. Gather initial bounding boxes ────────────────────
        fp_rects = [_get_bb(fp) for fp in footprints]

        # ── 2. Find which components overlap ────────────────────
        # Keep the largest footprint fixed; move smaller ones.
        fp_areas = []
        for r in fp_rects:
            fp_areas.append(r.w * r.h if r else 0)

        needs_move: set = set()
        for i in range(len(footprints)):
            if i in needs_move:
                continue
            for j in range(i + 1, len(footprints)):
                if j in needs_move:
                    continue
                if _overlaps(fp_rects[i], fp_rects[j]):
                    # Move the smaller one
                    if fp_areas[j] <= fp_areas[i]:
                        needs_move.add(j)
                    else:
                        needs_move.add(i)

        if not needs_move:
            return 0

        # ── 3. Determine board area for placement ───────────────
        get_edges_bb = getattr(board, 'GetBoardEdgesBoundingBox', None)
        board_bb = None
        if callable(get_edges_bb):
            try:
                board_bb = get_edges_bb()
                if board_bb.GetWidth() <= 0:
                    board_bb = None
            except Exception:
                board_bb = None
        if board_bb is None:
            get_bb_fn = getattr(board, 'ComputeBoundingBox', None)
            if callable(get_bb_fn):
                try:
                    board_bb = get_bb_fn(False)
                except Exception:
                    pass

        if board_bb is None:
            origin_x, origin_y = int(from_mm(115)), int(from_mm(73))
            area_w = int(from_mm(70))
            area_h = int(from_mm(54))
        else:
            margin = int(from_mm(3))
            origin_x = int(board_bb.GetLeft()) + margin
            origin_y = int(board_bb.GetTop()) + margin
            area_w = int(board_bb.GetWidth()) - 2 * margin
            area_h = int(board_bb.GetHeight()) - 2 * margin

        # ── 4. Place each overlapping FP at the first grid cell that
        #       doesn't overlap any already-placed FP ────────────
        # Build list of "fixed" rects (the ones not being moved)
        fixed_rects = [fp_rects[i] for i in range(len(footprints)) if i not in needs_move and fp_rects[i] is not None]

        grid_step = int(from_mm(8.0))  # 8mm grid step

        moved = 0
        for idx in sorted(needs_move):
            fp = footprints[idx]
            r = fp_rects[idx]
            fp_hw = r.w // 2 if r else int(from_mm(5))
            fp_hh = r.h // 2 if r else int(from_mm(5))

            # Try positions on a grid, spiraling from board center
            center_x = origin_x + area_w // 2
            center_y = origin_y + area_h // 2
            placed = False
            for ring in range(0, 30):
                if placed:
                    break
                for dx in range(-ring, ring + 1):
                    if placed:
                        break
                    for dy in range(-ring, ring + 1):
                        if ring > 0 and abs(dx) != ring and abs(dy) != ring:
                            continue
                        cx = center_x + dx * grid_step
                        cy = center_y + dy * grid_step
                        # Bounds check
                        if (cx - fp_hw < origin_x or cx + fp_hw > origin_x + area_w or
                                cy - fp_hh < origin_y or cy + fp_hh > origin_y + area_h):
                            continue
                        # Build candidate rect
                        cand = _Rect(cx - fp_hw, cy - fp_hh, cx + fp_hw, cy + fp_hh)
                        # Check against all fixed rects
                        overlap = False
                        for fr in fixed_rects:
                            if _overlaps(cand, fr):
                                overlap = True
                                break
                        if not overlap:
                            try:
                                fp.SetPosition(
                                    pcbnew_mod.VECTOR2I(int(cx), int(cy))
                                )
                                fixed_rects.append(cand)
                                fp_rects[idx] = cand
                                moved += 1
                                placed = True
                            except Exception:
                                pass
                            break

        return moved

    # ── Web Search / Datasheet Lookup Handlers ──────────────────
