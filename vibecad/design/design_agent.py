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
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


class DesignActionType(Enum):
    """Types of design actions the agent can perform."""
    ASSIGN_NETS = auto()
    DEFINE_NET = auto()
    SEARCH_PART = auto()
    DOWNLOAD_SYMBOL = auto()
    DOWNLOAD_FOOTPRINT = auto()
    ADD_COMPONENT = auto()
    
    # Connection operations
    DRAW_TRACK = auto()
    DRAW_WIRE = auto()
    ROUTE_NET = auto()
    ADD_VIA = auto()
    
    # BOM operations
    EXPORT_BOM = auto()
    UPDATE_BOM_FIELDS = auto()
    
    # Layout operations
    MOVE_COMPONENT = auto()
    ROTATE_COMPONENT = auto()
    ALIGN_COMPONENTS = auto()
    
    # Design operations
    ADD_POLYGON = auto()
    ADD_TEXT = auto()
    ADD_MOUNTING_HOLE = auto()
    DEFINE_BOARD_OUTLINE = auto()
    
    # Checks
    RUN_DRC = auto()
    RUN_ERC = auto()
    
    # Routing
    AUTOROUTE_BOARD = auto()
    
    # Layer management
    SET_LAYER_COUNT = auto()

    # Deletion
    DELETE_TRACKS = auto()
    DELETE_COMPONENT = auto()

    # Web search / component lookup
    SEARCH_WEB = auto()
    LOOKUP_DATASHEET = auto()
    
    # Unknown
    UNKNOWN = auto()


@dataclass
class DesignAction:
    """A single design action to be performed."""
    action_type: DesignActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Preview information
    preview_text: Optional[str] = None
    preview_visual: Optional[Any] = None  # Could be image data, SVG, etc.
    
    # Execution state
    requires_approval: bool = True
    approved: bool = False
    executed: bool = False
    success: bool = False
    result_message: Optional[str] = None
    
    # Undo information
    undoable: bool = True
    undo_data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type.name,
            'description': self.description,
            'parameters': self.parameters,
            'requires_approval': self.requires_approval,
            'approved': self.approved,
            'executed': self.executed,
            'success': self.success,
        }


@dataclass
class DesignRequest:
    """A user's design request, potentially containing multiple actions."""
    original_text: str
    interpreted_actions: List[DesignAction] = field(default_factory=list)
    
    # Context
    active_editor: str = "pcb"  # "pcb" or "schematic"
    current_file: Optional[str] = None
    
    # Request metadata
    timestamp: Optional[str] = None
    confidence: float = 0.0  # 0-1 confidence in interpretation
    
    def add_action(self, action: DesignAction):
        self.interpreted_actions.append(action)
    
    def pending_actions(self) -> List[DesignAction]:
        return [a for a in self.interpreted_actions if not a.executed]
    
    def requires_approval(self) -> bool:
        return any(a.requires_approval and not a.approved for a in self.pending_actions())


class IntentPattern:
    """Pattern for recognizing user intent."""
    
    def __init__(self, 
                 action_type: DesignActionType,
                 patterns: List[str],
                 param_extractors: Dict[str, str] = None):
        self.action_type = action_type
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.param_extractors = param_extractors or {}
    
    def match(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if text matches this intent pattern."""
        for pattern in self.patterns:
            match = pattern.search(text)
            if match:
                params = {}
                # Extract named groups from regex
                params.update(match.groupdict())
                return True, params
        return False, {}


class DesignAgent:
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
    DESIGN_SYSTEM_PROMPT = """You are VibeCAD Design Assistant — an autonomous PCB design agent for KiCad.

Your job:
- Help the user design PCBs from scratch or modify existing designs.
- ONLY do exactly what the user asks. Do NOT assume additional steps.
- If the user says "draw a board outline", ONLY propose DEFINE_BOARD_OUTLINE. Do NOT add components, mounting holes, routing, or any other actions unless explicitly requested.
- When given an explicit high-level goal (e.g., "design an Arduino UNO" or "design a full board"), THEN break it down into phases:
  1. Component selection & search (SEARCH_PART for each component)
  2. Component placement (ADD_COMPONENT for each — ALWAYS specify the package parameter, e.g. package:"DIP-28", package:"TQFP-32", package:"SOT-223")
  3. Board outline (DEFINE_BOARD_OUTLINE — size it generously for the components)
  4. Layout arrangement: use MOVE_COMPONENT and ROTATE_COMPONENT to position all components INSIDE the board outline. No component should extend beyond the board edges. Space them logically (e.g., connectors at edges, ICs centered, decoupling caps near their ICs).
  5. Net assignment (DEFINE_NET / ASSIGN_NETS for ALL critical nets — GND, VCC, signal nets, data buses, crystal, reset, etc.)
  6. Routing (AUTOROUTE_BOARD)
  7. ERC verification (RUN_ERC) — fix any electrical rule violations
  8. DRC verification (RUN_DRC) — fix any design rule violations

PHASE COMPLETION CHECKPOINTS (mandatory before advancing):
- After Phase 2: Verify ALL components from Phase 1 have been placed. If any are missing, place them before continuing.
- For "design an Arduino UNO": Do NOT place footprints/modules named like Arduino_UNO_R2/Arduino_UNO_R3. Those represent a pre-made board/module footprint and can cause severe DRC issues (including co-located drills). Instead, place discrete headers (PinHeader footprints) and the actual ICs/connectors.
- After Phase 3: Verify the board outline is large enough for all components. A good rule: at least 20% margin beyond the component area.
- After Phase 4: Verify ALL component centers are within the board outline. If any are outside, move them inside.
- After Phase 5: Verify net assignment success. If DEFINE_NET/ASSIGN_NETS reports "pad not found" warnings, the footprint likely has wrong pin count — consider removing and re-adding the component with an explicit package (e.g., "ATmega328P-PU" with package:"DIP-28"). Do NOT proceed to routing with >30% net assignment failures.
- After Phase 6: If AUTOROUTE_BOARD reports "Could not route N net(s)", do NOT proceed to DRC. Instead, fix the unroutable nets by checking net assignments and re-routing.

- IMPORTANT: ALL components MUST be moved within the board outline BEFORE routing. Do NOT route while components are outside or overlapping the board edge.
- Execute one phase at a time. After each phase completes, verify the checkpoint, then propose the next.
- Ask clarifying questions when required inputs are ambiguous.
- Prefix internal reasoning with [THINKING] so the user sees your progress.

Safety rules:
- Never execute actions yourself — propose them and the host app will preview/execute.
- The host app requires explicit user approval before executing any board-modifying action.
- Do not invent PCB coordinates. Prefer references (e.g., U1), pins/pads, nets, and standard package names.
- When placing components, use relative positions (e.g., "next to U1") or let the host auto-place at board center.
- For DEFINE_BOARD_OUTLINE: The host app automatically centers the outline on the A4 page (~150,100mm).
  You MUST NOT mention origin, (0,0), starting point, or top-left corner in your reasoning or response.
  The ONLY parameters are width and height. Do NOT say "I'll set the origin" — there is no origin parameter.

Layer guidance:
- For 2-layer boards: use F.Cu (front copper) and B.Cu (back copper).
- For 4-layer boards: use In1.Cu for ground plane and In2.Cu for power plane.
- Use SET_LAYER_COUNT to change the number of copper layers.
- Use ADD_POLYGON on inner layers to create copper pour zones (ground/power planes).

Routing rules:
- All tracks must be routed along cardinal directions (horizontal/vertical) or at 45-degree diagonals.
- Never route arbitrary-angle traces. The host enforces this automatically.

Connectivity prerequisite:
- Do NOT attempt routing (DRAW_TRACK/AUTOROUTE_BOARD) until pads have nets assigned.
- If pads are unconnected (autorouter reports "No routable nets found"), you MUST propose either:
    1) DEFINE_NET to assign a net to multiple pads at once, OR
    2) ASSIGN_NETS with explicit ref/pad → net mappings, OR
    3) instruct the user to run KiCad's "Update PCB from Schematic" to import the netlist,
    then retry AUTOROUTE_BOARD.
- CRITICAL: If DEFINE_NET or ASSIGN_NETS reports "pad not found" with available pad list, the footprint has fewer pins than expected. This means the wrong footprint was placed. You MUST:
    1) DELETE_COMPONENT to remove the wrong-footprint component
    2) Re-add it with ADD_COMPONENT and specify the correct package explicitly (e.g. package:"DIP-28")
    3) Re-try net assignments
    Do NOT proceed to routing if pad assignments are failing.

Large / complex boards:
- Work in batches. Propose at most ~10 ADD_COMPONENT actions per iteration.
- After each batch, verify progress (components placed count, nets assigned ratio) and continue.
- Use SEARCH_WEB and LOOKUP_DATASHEET to get pinout info, datasheets, and reference designs from LCSC/Mouser/Octopart. This is especially important for ICs where you need exact pin mappings.
- If the schematic has few/no net labels, use SEARCH_WEB / LOOKUP_DATASHEET to pull pinout hints for key ICs/connectors, then use DEFINE_NET to name critical nets (GND, 5V, 3V3, USB_D+/D-, SPI, I2C) so routing can proceed.

Placement rules:
- The placement engine automatically avoids overlaps using bounding-box collision detection, but the LLM's MOVE_COMPONENT commands can still create overlaps. Always check the response for overlap warnings after MOVE_COMPONENT.
- If MOVE_COMPONENT reports "WARNING: ... overlaps with ...", immediately move the overlapping component(s) to clear positions before proceeding.
- After Phase 4 (Layout Arrangement), verify no overlap warnings exist before proceeding to net assignment.
- For Arduino-style boards: place the DIP MCU in the center, connectors (USB, DC jack) on the left edge, and headers on the top/bottom edges. Space them by at least 15mm center-to-center.
- NEVER place a regulator as ref J* — voltage regulators should be U* (e.g. U2, U3). Only connectors use J*.

DRC iteration:
- After running DRC, analyze each error and propose specific fixes.
- For shorts / crossings (e.g., "Items shorting two nets" or "Tracks crossing"): use DELETE_TRACKS to remove existing routing, then re-route the correct nets (DRAW_TRACK/AUTOROUTE_BOARD). Do NOT move components for these errors.
- For wrong connectivity (e.g., LED cathode accidentally on +5V): fix by changing which pads are connected and/or using ASSIGN_NETS, then re-route. Do NOT move components for connectivity mistakes.
- For clearance / courtyard / solder-mask aperture bridge issues caused by proximity: MOVE_COMPONENT to increase spacing, then re-route, then re-run DRC.
- For courtyard overlaps: use MOVE_COMPONENT to separate the overlapping components before re-routing.
- Always re-run DRC after fixes. Repeat until no errors remain (warnings are acceptable).

ERC iteration:
- Run ERC (RUN_ERC) BEFORE DRC to catch electrical issues early (unconnected pins, missing power flags, pin-type conflicts).
- Fix ERC errors using DEFINE_NET, ASSIGN_NETS, or by adding missing power symbols.
- Re-run ERC after fixes. Proceed to DRC only when ERC passes.

Completion:
- Do NOT say DESIGN_COMPLETE until BOTH ERC and DRC report 0 errors.
- When both pass with 0 errors, respond with DESIGN_COMPLETE.

When you propose actions, you must follow the exact JSON schema requested by the user message.
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
    
    # Intent patterns for local interpretation
    INTENT_PATTERNS = [
        # Placement / add to PCB
        IntentPattern(
            DesignActionType.ADD_COMPONENT,
            [
                r"(?:add|place|insert)\s+(?:a\s+)?(?P<query>.+?)\s+(?:to|on|onto)\s+(?:the\s+)?pcb$",
                r"(?:add|place|insert)\s+(?:a\s+)?(?P<query>.+?)\s+(?:footprint|component)\s+(?:to|on|onto)\s+(?:the\s+)?pcb$",
                r"(?:add|place)\s+(?:a\s+)?(?P<query>.+?)\s+to\s+board$",
            ]
        ),
        # Library operations
        IntentPattern(
            DesignActionType.SEARCH_PART,
            [
                r"(?:find|search|look for|get)\s+(?:a\s+)?(?:part|component|symbol|footprint)?\s*(?:for\s+)?(?P<query>.+)",
                r"(?:i need|add)\s+(?:a\s+)?(?P<query>.+?)(?:\s+component|\s+part)?$",
            ]
        ),
        IntentPattern(
            DesignActionType.DOWNLOAD_SYMBOL,
            [
                r"download\s+(?:the\s+)?symbol\s+(?:for\s+)?(?P<part_name>.+)",
                r"get\s+(?:the\s+)?(?P<part_name>.+?)\s+symbol",
            ]
        ),
        IntentPattern(
            DesignActionType.DOWNLOAD_FOOTPRINT,
            [
                r"download\s+(?:the\s+)?footprint\s+(?:for\s+)?(?P<part_name>.+)",
                r"get\s+(?:the\s+)?(?P<part_name>.+?)\s+footprint",
            ]
        ),
        
        # Connection operations
        IntentPattern(
            DesignActionType.DRAW_TRACK,
            [
                r"(?:draw|add|create|route)\s+(?:a\s+)?track\s+(?:from\s+)?(?P<from_point>.+?)\s+to\s+(?P<to_point>.+)",
                r"connect\s+(?P<from_point>.+?)\s+to\s+(?P<to_point>.+)",
                r"wire\s+(?P<from_point>.+?)\s+to\s+(?P<to_point>.+)",
            ]
        ),
        IntentPattern(
            DesignActionType.ROUTE_NET,
            [
                r"route\s+(?:the\s+)?(?P<net_name>\w+)\s+net",
                r"autoroute\s+(?P<net_name>\w+)",
            ]
        ),
        IntentPattern(
            DesignActionType.ADD_VIA,
            [
                r"add\s+(?:a\s+)?via\s+(?:at\s+)?(?P<location>.+)",
            ]
        ),
        
        # BOM operations
        IntentPattern(
            DesignActionType.EXPORT_BOM,
            [
                r"(?:export|generate|create)\s+(?:a\s+)?(?:the\s+)?bom",
                r"(?:export|generate|create)\s+(?:a\s+)?bill\s+of\s+materials",
                r"(?:make|get)\s+(?:a\s+)?bom\s+(?:for\s+)?(?P<format>jlcpcb|lcsc|mouser|digikey)?",
            ]
        ),
        
        # Layout operations
        IntentPattern(
            DesignActionType.MOVE_COMPONENT,
            [
                r"move\s+(?P<ref>\w+)\s+to\s+(?P<location>.+)",
                r"place\s+(?P<ref>\w+)\s+at\s+(?P<location>.+)",
            ]
        ),
        IntentPattern(
            DesignActionType.ROTATE_COMPONENT,
            [
                r"rotate\s+(?P<ref>\w+)(?:\s+(?:by\s+)?(?P<angle>\d+)\s*(?:degrees?|°)?)?",
                r"turn\s+(?P<ref>\w+)(?:\s+(?P<angle>\d+))?",
            ]
        ),
        IntentPattern(
            DesignActionType.ALIGN_COMPONENTS,
            [
                r"align\s+(?P<refs>.+?)\s+(?P<direction>horizontally|vertically|left|right|top|bottom|center)",
            ]
        ),
        
        # Design operations
        IntentPattern(
            DesignActionType.ADD_MOUNTING_HOLE,
            [
                r"add\s+(?:a\s+)?(?:(?P<size>\d+(?:\.\d+)?)\s*mm\s+)?mounting\s+hole\s+(?:at\s+)?(?P<location>.+)",
            ]
        ),
        IntentPattern(
            DesignActionType.DEFINE_BOARD_OUTLINE,
            [
                r"(?:create|draw|define|set)\s+(?:a\s+)?(?:the\s+)?board\s+(?:outline|edge|shape)",
                r"make\s+(?:the\s+)?board\s+(?P<width>\d+(?:\.\d+)?)\s*(?:mm|x)\s*(?:by\s*)?(?P<height>\d+(?:\.\d+)?)\s*mm",
            ]
        ),
        IntentPattern(
            DesignActionType.ADD_TEXT,
            [
                r"add\s+(?:the\s+)?text\s+['\"](?P<text>.+?)['\"]\s*(?:at\s+(?P<location>.+))?",
                r"write\s+['\"](?P<text>.+?)['\"]\s*(?:at\s+(?P<location>.+))?",
            ]
        ),
        
        # DRC/ERC
        IntentPattern(
            DesignActionType.RUN_DRC,
            [
                r"(?:run|check|perform)\s+(?:a\s+)?drc",
                r"design\s+rule\s+check",
            ]
        ),
        IntentPattern(
            DesignActionType.RUN_ERC,
            [
                r"(?:run|check|perform)\s+(?:a\s+)?erc",
                r"electrical\s+rule\s+check",
            ]
        ),
        
        # Deletion
        IntentPattern(
            DesignActionType.DELETE_TRACKS,
            [
                r"(?:delete|remove|clear|erase)\s+(?:all\s+|old\s+|existing\s+)?(?:the\s+)?tracks?",
                r"(?:unroute|ripup)\s+(?:the\s+)?(?:board|pcb|everything)",
                r"clear\s+routing",
            ]
        ),
        IntentPattern(
            DesignActionType.DELETE_COMPONENT,
            [
                r"(?:delete|remove|erase)\s+(?:component\s+)?(?P<ref>\w+)",
            ]
        ),
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

    def clear_history(self) -> None:
        """Reset conversation history (e.g., when opening a new board)."""
        self._conversation_history.clear()

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

        # Prefer LLM for interpretation when configured. Local patterns remain
        # as an offline fallback.
        if self._llm_client:
            actions = self._interpret_with_llm(user_input, context)
            for a in actions:
                request.add_action(a)
            request.confidence = 0.9 if actions else 0.2
        else:
            action = self._interpret_locally(user_input)
            if action and action.action_type != DesignActionType.UNKNOWN:
                request.add_action(action)
                request.confidence = 0.8
            else:
                request.add_action(DesignAction(
                    action_type=DesignActionType.UNKNOWN,
                    description=f"Could not understand: {user_input}",
                    requires_approval=False,
                ))
                request.confidence = 0.0
        
        return request

    def chat(self, user_input: str, context: Dict[str, Any] = None) -> Tuple[str, DesignRequest]:
        """LLM-first conversational entrypoint.

        Returns an assistant message plus a structured DesignRequest containing
        tool-like actions that can be previewed and approved.
        """
        context = context or {}

        # Deterministic shortcut: if the user asks to connect "pad N on each LED" and
        # there are exactly two LEDs on the board, propose a concrete DRAW_TRACK action
        # without asking for references.
        m_each_led = re.search(
            r"pads?\s+(?:numbered\s+)?(?P<pad>\w+)\s+on\s+each\s+led\b",
            (user_input or ""),
            flags=re.IGNORECASE,
        )
        if m_each_led:
            pad = str(m_each_led.group('pad')).strip()
            pcb_data = context.get('pcb_data')
            led_refs: List[str] = []
            try:
                if pcb_data:
                    for fp in getattr(pcb_data, 'footprints', []) or []:
                        ref = str(getattr(fp, 'reference', '') or '').strip()
                        val = str(getattr(fp, 'value', '') or '').strip().lower()
                        if not ref:
                            continue
                        if ref.upper().startswith('D') or 'led' in val:
                            led_refs.append(ref)
            except Exception:
                led_refs = []

            if len(led_refs) == 2:
                ref_a, ref_b = led_refs[0], led_refs[1]
                action = DesignAction(
                    action_type=DesignActionType.DRAW_TRACK,
                    description=f"Draw track from {ref_a} pad {pad} to {ref_b} pad {pad}",
                    parameters={
                        'from_point': f"{ref_a}/{pad}",
                        'to_point': f"{ref_b}/{pad}",
                    },
                    requires_approval=True,
                )
                request = DesignRequest(
                    original_text=user_input,
                    active_editor=context.get('active_editor', 'pcb'),
                    current_file=context.get('current_file'),
                    confidence=0.85,
                )
                request.add_action(action)
                return (
                    f"I can draw a track between pad {pad} on {ref_a} and {ref_b}.",
                    request,
                )

        if self._llm_client:
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
            # Only show a hard failure message if we truly received nothing.
            if not assistant_message and not actions:
                # This typically happens when the LLM client returned an empty
                # string (provider compatibility/config issue). Give the user a
                # concrete next step rather than asking them to rephrase.
                if bool((context or {}).get('verbose')):
                    assistant_message = (
                        "LLM returned an empty response (no assistant text and no actions). "
                        "This usually indicates an API base/model compatibility issue. "
                        "Try switching to a different model, or verify your `api_base` is an OpenAI-chat-compatible `/v1` endpoint."
                    )
                else:
                    assistant_message = (
                        "I didn’t get a response back from the LLM. "
                        "Please check your API/model settings and try again."
                    )
            # If we did receive actions but the model omitted the assistant text,
            # provide a friendly default.
            if not assistant_message and actions:
                assistant_message = "Got it. Here’s what I propose."

            # Record assistant response for next-turn references.
            self._append_history("assistant", self._assistant_history_text(assistant_message, actions))
            return assistant_message, request

        # Offline fallback: local intent patterns + template response.
        request = self.interpret_request(user_input, context)
        if request.interpreted_actions and request.interpreted_actions[0].action_type != DesignActionType.UNKNOWN:
            assistant_message = "Here’s a proposed action for your approval."
        else:
            assistant_message = (
                "I couldn’t understand that request. Try something like: "
                "'find a USB-C connector', 'export BOM for JLCPCB', or 'connect U1 pin 1 to R1 pin 2'."
            )
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

        for a in actions:
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
    
    def _interpret_locally(self, text: str) -> Optional[DesignAction]:
        """Use pattern matching for local interpretation."""
        text = text.strip()
        
        for pattern in self.INTENT_PATTERNS:
            matched, params = pattern.match(text)
            if matched:
                requires_approval = True
                if pattern.action_type == DesignActionType.SEARCH_PART:
                    # Safe informational action: run immediately.
                    requires_approval = False
                return DesignAction(
                    action_type=pattern.action_type,
                    description=self._generate_description(pattern.action_type, params),
                    parameters=params,
                    requires_approval=requires_approval,
                )
        
        return None
    
    def _interpret_with_llm(self, 
                            text: str,
                            context: Dict[str, Any]) -> List[DesignAction]:
        """Use LLM for advanced interpretation."""
        if not self._llm_client:
            return []
        
        try:
            # Build prompt with context
            prompt = self._build_interpretation_prompt(text, context)
            
            # Call LLM with a design-tool system prompt (NOT the explanation-only prompt)
            from ..llm.client import LLMMessage
            response_obj = self._llm_client.chat(
                [LLMMessage(role='user', content=prompt)],
                system_prompt=self.DESIGN_SYSTEM_PROMPT,
            )
            response = response_obj.content
            
            # Parse response into actions
            actions = self._parse_llm_response(response)
            return actions
            
        except Exception as e:
            logger.exception(f"LLM interpretation failed: {e}")
            return []
    
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

Rules:
- If the request is purely informational, return []
- Prefer the minimal number of actions
- Do not invent board coordinates; use references/nets/pins when possible
"""

        return prompt
    
    def _parse_llm_response(self, response: str) -> List[DesignAction]:
        """Parse LLM response into actions."""
        import json
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return [a for a in (self._parse_action_item(item) for item in data) if a is not None]
        except Exception as e:
            logger.warning("Failed to parse LLM response: %s", e)
        return []

    # ------------------------------------------------------------------
    # Shared helpers for action JSON parsing (used by _chat_with_llm,
    # _parse_llm_response, and _extract_actions_from_malformed).
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
        squeezed = re.sub(r"[^A-Za-z0-9]", "", raw_name).upper()
        if squeezed:
            for t in DesignActionType:
                if re.sub(r"_", "", t.name).upper() == squeezed:
                    return t
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

        # Parameters
        params = item.get('parameters') or item.get('params') or item.get('arguments')
        if not isinstance(params, dict):
            params = {}

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

    def _extract_actions_from_malformed(self, raw_text: str) -> Tuple[str, List['DesignAction']]:
        """Best-effort extraction when the model emits malformed/truncated JSON."""
        import json
        assistant_msg = ""
        actions: List[DesignAction] = []

        # Assistant message (best-effort)
        m = re.search(
            r'"(?:assistant_message|assistantMessage|message|assistant|response)"\s*:\s*"(?P<msg>[^"\\]*(?:\\.[^"\\]*)*)"',
            raw_text,
        )
        if m:
            try:
                assistant_msg = bytes(m.group('msg'), 'utf-8').decode('unicode_escape').strip()
            except Exception:
                assistant_msg = m.group('msg').strip()

        # Scan for JSON objects inside the "actions" array char-by-char.
        idx = raw_text.find('"actions"')
        if idx != -1:
            lb = raw_text.find('[', idx)
            if lb != -1:
                i, in_str, esc, depth, obj_start = lb + 1, False, False, 0, None
                while i < len(raw_text):
                    ch = raw_text[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == '\\':
                            esc = True
                        elif ch == '"':
                            in_str = False
                        i += 1
                        continue
                    if ch == '"':
                        in_str = True
                        i += 1
                        continue
                    if ch == '{':
                        if depth == 0:
                            obj_start = i
                        depth += 1
                    elif ch == '}':
                        if depth > 0:
                            depth -= 1
                            if depth == 0 and obj_start is not None:
                                try:
                                    item_obj = json.loads(raw_text[obj_start:i + 1])
                                except Exception:
                                    item_obj = None
                                if isinstance(item_obj, dict):
                                    a = self._parse_action_item(item_obj)
                                    if a is not None:
                                        actions.append(a)
                                obj_start = None
                    elif ch == ']':
                        break
                    i += 1

        if actions:
            return assistant_msg, actions

        # Regex fallback for the most common case.
        m_action = re.search(
            r'"(?:action_type|actiontype|actionType)"\s*:\s*"(?P<atype>[A-Za-z0-9_\- ]+)"',
            raw_text,
        )
        if m_action:
            params: Dict[str, Any] = {}
            for key, pat in [
                ('query', r'"query"\s*:\s*"(?P<v>[^"]+)"'),
                ('part_name', r'"part_name"\s*:\s*"(?P<v>[^"]+)"'),
                ('package', r'"package"\s*:\s*"(?P<v>[^"]+)"'),
            ]:
                m = re.search(pat, raw_text)
                if m:
                    params[key] = m.group('v').strip()
            atype = self._normalize_action_type(m_action.group('atype'))
            actions.append(DesignAction(
                action_type=atype,
                description="",
                parameters=params,
                requires_approval=atype != DesignActionType.SEARCH_PART,
            ))

        return assistant_msg, actions

    def _chat_with_llm(self, text: str, context: Dict[str, Any]) -> Tuple[str, List[DesignAction], float]:
        """Use the LLM to produce both an assistant response and tool-like actions.

        Sends the full conversation history so multi-turn interactions work.
        """
        import json
        try:
            from ..llm.client import LLMError, LLMMessage
        except Exception:  # pragma: no cover
            LLMError = Exception

        try:
            tools = [a.name for a in self.SUPPORTED_LLM_TOOLS]

            # Build the message list: history + current user turn with
            # structured instructions so the model always produces JSON.
            messages: list = []

            # Inject a compact history recap so the model has context.
            # If the current User Request is a standalone command (starts with strong verb),
            # we skip previous history to avoid sticking to failed/superseded attempts.
            # "Add ADS1256..." should not be polluted by previous "Add ADS1256..." failures.
            is_new_command = False
            q_clean = text.strip().lower()
            if re.match(r'^(add|place|search|find|download|create|get)\b', q_clean):
                is_new_command = True
            
            history_subset = []
            if is_new_command:
                # Only include the very last Assistant message if it exists (for continuity?),
                # or just reset. User requested "clean", so let's reset.
                history_subset = []
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

AVAILABLE TOOLS (action_type values):
{', '.join(tools)}

PARAMETER KEYS for common tools:
- SEARCH_PART: {{ "query": "<part number or description>" }}
- DOWNLOAD_SYMBOL: {{ "part_name": "<MPN>" }}
- DOWNLOAD_FOOTPRINT: {{ "part_name": "<MPN>", "package": "<optional package>" }}
- ADD_COMPONENT: {{ "query": "<footprint name/manufacturer+mpn/description>", "location": "<optional x,y in mm>", "ref_prefix": "<optional ref prefix>" }}
- EXPORT_BOM: {{ "format": "<jlcpcb|lcsc|mouser|digikey>" }}
- DRAW_TRACK: {{ "from_point": "<ref/pin>", "to_point": "<ref/pin>", "layer": "<optional: F.Cu, B.Cu, In1.Cu, In2.Cu>" }}
- ASSIGN_NETS: {{ "assignments": [{{"ref":"U1","pad":"1","net":"GND"}}, {{"ref":"U1","pad":"2","net":"VCC"}}] }}
- DEFINE_NET: {{ "net": "<net name>", "pads": ["U1/1", "J1/2", {{"ref":"U2","pad":"14"}}] }}
- MOVE_COMPONENT: {{ "ref": "<designator>", "location": "<x,y>" }}
- ROTATE_COMPONENT: {{ "ref": "<designator>", "angle": "<degrees>" }}
- ADD_VIA: {{ "location": "<x,y in mm>", "net": "<net name>", "size_mm": 0.8, "drill_mm": 0.4 }}
- DEFINE_BOARD_OUTLINE: {{ "width": <mm>, "height": <mm> }}  (ONLY width+height; host auto-centers on page — NEVER mention origin/0,0)
- ADD_MOUNTING_HOLE: {{ "size": "3.2", "location": "<x,y in mm>" }}
- ALIGN_COMPONENTS: {{ "refs": "R1 R2 R3", "direction": "horizontally|vertically" }}
- ADD_TEXT: {{ "text": "<content>", "location": "<x,y>", "layer": "F.SilkS" }}
- ADD_POLYGON: {{ "layer": "In1.Cu", "net": "GND", "x": 0, "y": 0, "width": <mm>, "height": <mm> }}
- AUTOROUTE_BOARD: {{ }}
- SET_LAYER_COUNT: {{ "count": 2 }}
- DELETE_TRACKS: {{ }}  (deletes ALL tracks and vias on the board — use before re-routing)
- DELETE_COMPONENT: {{ "ref": "<designator>" }}
- RUN_DRC: {{ }}
- RUN_ERC: {{ }}  (run Electrical Rule Check — checks netlist connectivity, missing connections, pin conflicts)
- SEARCH_WEB: {{ "query": "<component MPN or description>" }}  (search LCSC/Mouser/Octopart for pricing, specs, datasheets)
- LOOKUP_DATASHEET: {{ "mpn": "<manufacturer part number>" }}  (find and return datasheet URL)

INSTRUCTIONS:
Respond with valid JSON matching this schema:
{{
  "assistant_message": "your conversational response to the user",
  "actions": [
    {{
      "action_type": "TOOL_NAME",
      "description": "short summary of what this does",
      "parameters": {{ "key": "value" }},
      "requires_approval": true,
      "preview_text": "user-facing preview text"
    }}
  ]
}}

Rules:
- If the user is confirming a previous proposal, you MUST include the corresponding action(s).
- Use the full conversation history above; do not re-ask questions the user already answered.
- If inputs are missing, ask a clarifying question in assistant_message and set actions to [].
- Treat strings that look like part numbers (e.g., ATmega328P-PU, STM32F103C8T6) as part numbers.
- Do NOT reinterpret part-number suffixes like "-PU" as "pull-up".
- If the user wants the component placed in the center (or gives no location), omit "location" — the host will place it at the board center by default.
- Do NOT propose DOWNLOAD_SYMBOL / DOWNLOAD_FOOTPRINT unless you have successfully located a matching part/asset (otherwise ask a question and return actions: []).
- You MUST always output valid JSON. No markdown, no extra text outside the JSON object."""

            messages.append(LLMMessage(role='user', content=user_prompt))

            # --- Call LLM and parse response ---------------------------
            did_retry_for_length = False
            old_max_tokens: Optional[int] = None
            try:
                while True:
                    response_obj = self._llm_client.chat(
                        messages,
                        system_prompt=self.DESIGN_SYSTEM_PROMPT,
                    )
                    raw = (response_obj.content or "").strip()

                    if not raw:
                        verbose = bool((context or {}).get('verbose'))
                        if verbose:
                            rr = getattr(response_obj, 'raw_response', None)
                            rr_keys = list(rr.keys()) if isinstance(rr, dict) else []
                            return (
                                f"LLM returned empty content. Raw response keys: {rr_keys}",
                                [], 0.1,
                            )
                        return "LLM returned an empty response. Check your API base/model settings.", [], 0.1

                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("DesignAgent raw LLM response: %s", raw[:500])

                    # Try to parse as JSON.
                    obj: Any = None
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        start = raw.find('{')
                        end = raw.rfind('}') + 1
                        if start >= 0 and end > start:
                            try:
                                obj = json.loads(raw[start:end])
                            except Exception:
                                pass

                    # If we couldn't get a dict, try retry-for-length or salvage.
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

                        salvage_msg, salvage_actions = self._extract_actions_from_malformed(raw)
                        if salvage_actions:
                            return salvage_msg or raw, salvage_actions, 0.7
                        return raw, [], 0.3

                    # --- Successful JSON parse ---
                    assistant_message = ""
                    for key in ("assistant_message", "assistantMessage", "message", "assistant", "response"):
                        if key in obj and obj.get(key) not in (None, ""):
                            assistant_message = str(obj.get(key)).strip()
                            break

                    actions_json = obj.get('actions') or obj.get('tool_actions') or obj.get('toolActions') or []
                    if not isinstance(actions_json, list):
                        actions_json = []

                    actions = [a for a in (self._parse_action_item(item) for item in actions_json) if a is not None]
                    confidence = 0.9 if actions else 0.5
                    return assistant_message, actions, confidence
            finally:
                if old_max_tokens is not None:
                    try:
                        self._llm_client.config.max_tokens = old_max_tokens
                    except Exception:
                        pass

        except LLMError as e:
            # Provider/network/auth errors should be shown even when not verbose,
            # otherwise users get stuck in a generic loop.
            logger.warning("LLM request failed: %s", e)
            verbose = bool((context or {}).get('verbose'))
            msg = str(e).strip()
            lower_msg = msg.lower()
            if len(msg) > 600:
                msg = msg[:600] + "…"
            extra_tip = "Tip: enable Verbose for more diagnostics."
            if "finish_reason" in msg and "length" in msg:
                extra_tip = (
                    "Tip: this looks like the completion was cut off by a token limit. "
                    "Increase `Max tokens` in ⚙ Settings (try 1024-2048) and retry."
                )
            elif "http 401" in lower_msg:
                extra_tip = (
                    "Tip: 401 usually means your API key does not match the API base/provider. "
                    "For OpenAI use an OpenAI key with `https://api.openai.com/v1`. "
                    "For GitHub Models use `GITHUB_TOKEN` (or API key field) with "
                    "`https://models.github.ai/inference` and model `openai/gpt-5`."
                )
            if verbose:
                return f"LLM error: {msg}", [], 0.1
            return (
                f"LLM request failed: {msg} \n\n{extra_tip}",
                [],
                0.1,
            )
        except Exception as e:
            logger.exception(f"LLM chat failed: {e}")
            verbose = bool((context or {}).get('verbose'))
            if verbose:
                return f"LLM error: {e}", [], 0.1
            return "I hit an error while contacting the LLM. Tip: enable Verbose for details.", [], 0.1
    
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
                lambda p: f"Define board outline" + (f" {p.get('width')}x{p.get('height')}mm" if p.get('width') else ""),
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
            else:
                action.success = False
                action.result_message = f"No handler for action type: {action.action_type.name}"
            
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

    async def _handle_assign_nets(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Assign nets to pads (ref/pad → net).

        This is primarily a recovery step when a PCB exists but the netlist
        hasn't been imported from the schematic.

        Parameters:
            assignments: list of objects with keys: ref, pad, net
        """
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (this action must run inside KiCad)"

        params = action.parameters or {}
        assignments = params.get('assignments')
        if not isinstance(assignments, list) or not assignments:
            return False, "Missing 'assignments' list (e.g., {assignments:[{ref,pad,net}, ...]})"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        def _find_footprint(ref: str):
            ref_u = (ref or '').strip().upper()
            if not ref_u:
                return None
            try:
                fn = getattr(board, 'FindFootprintByReference', None)
                if callable(fn):
                    fp = fn(ref_u)
                    if fp is not None:
                        return fp
            except Exception:
                pass
            try:
                for fp in board.GetFootprints():
                    try:
                        if str(fp.GetReference()).upper() == ref_u:
                            return fp
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        def _find_pad(fp, pad_number: str):
            if fp is None:
                return None
            pn = str(pad_number)
            try:
                f = getattr(fp, 'FindPadByNumber', None)
                if callable(f):
                    p = f(pn)
                    if p is not None:
                        return p
            except Exception:
                pass
            try:
                for p in fp.Pads():
                    try:
                        if str(p.GetNumber()) == pn:
                            return p
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        def _find_or_create_net(net_name: str):
            name = (net_name or '').strip()
            if not name:
                return None
            try:
                n = board.FindNet(name)
                if n is not None:
                    return n
            except Exception:
                pass
            try:
                net_item = pcbnew.NETINFO_ITEM(board, name)
                add = getattr(board, 'Add', None)
                if callable(add):
                    board.Add(net_item)
                else:
                    an = getattr(board, 'AddNet', None)
                    if callable(an):
                        an(net_item)
                return net_item
            except Exception:
                return None

        assigned = 0
        errors: List[str] = []

        for item in assignments:
            if not isinstance(item, dict):
                continue
            ref = str(item.get('ref', '') or '').strip().upper()
            pad_num = str(item.get('pad', '') or '').strip()
            net_name = str(item.get('net', '') or '').strip()
            if not ref or not pad_num or not net_name:
                continue

            fp = _find_footprint(ref)
            if fp is None:
                errors.append(f"{ref}: footprint not found on board")
                continue
            pad = _find_pad(fp, pad_num)
            if pad is None:
                # List available pads so the LLM can self-correct
                avail = []
                try:
                    for p in fp.Pads():
                        try:
                            avail.append(str(p.GetNumber()))
                        except Exception:
                            pass
                except Exception:
                    pass
                avail_str = ', '.join(sorted(set(avail), key=lambda x: (len(x), x))[:20]) if avail else 'none'
                errors.append(f"{ref}/{pad_num}: pad not found. {ref} has {len(avail)} pads: [{avail_str}]")
                continue
            net_obj = _find_or_create_net(net_name)
            if net_obj is None:
                errors.append(f"{net_name}: net create/find failed")
                continue

            try:
                if hasattr(pad, 'SetNet'):
                    pad.SetNet(net_obj)
                else:
                    # Very old APIs: best-effort via net code.
                    try:
                        pad.SetNetCode(int(net_obj.GetNet()))
                    except Exception:
                        pass
                assigned += 1
            except Exception as e:
                errors.append(f"{ref}/{pad_num}: set net failed ({e})")

        # Rebuild connectivity so subsequent routing sees updated net codes.
        try:
            if hasattr(board, 'BuildListOfNets'):
                board.BuildListOfNets()
        except Exception:
            pass
        try:
            conn = getattr(board, 'GetConnectivity', None)
            if callable(conn):
                c = conn()
                for m in ('RecalculateRatsnest', 'Recalculate', 'Rebuild', 'Build'):
                    fn = getattr(c, m, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:
                            continue
        except Exception:
            pass

        if assigned <= 0:
            suffix = (" Errors: " + "; ".join(errors[:8])) if errors else ""
            return False, "No nets assigned." + suffix

        msg = f"Assigned nets to {assigned} pad(s)."
        if errors:
            msg += f" Warnings: {len(errors)} item(s) could not be assigned."
            msg += " Details: " + "; ".join(errors[:5])
        return True, msg

    async def _handle_define_net(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Assign a single net name to multiple pads.

        Parameters:
            net: net name (e.g., "GND")
            pads: list of pad specs, either:
                - strings like "U1/1" (ref/pad)
                - objects like {"ref": "U1", "pad": "1"}
        """
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (this action must run inside KiCad)"

        params = action.parameters or {}
        net_name = str(params.get('net', '') or '').strip()
        pads = params.get('pads')
        if not net_name:
            return False, "Missing 'net' (e.g., {net:'GND', pads:[...]})"
        if not isinstance(pads, list) or not pads:
            return False, "Missing 'pads' list (e.g., {net:'GND', pads:['U1/1','J1/2']})"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        def _find_footprint(ref: str):
            ref_u = (ref or '').strip().upper()
            if not ref_u:
                return None
            try:
                fn = getattr(board, 'FindFootprintByReference', None)
                if callable(fn):
                    fp = fn(ref_u)
                    if fp is not None:
                        return fp
            except Exception:
                pass
            try:
                for fp in board.GetFootprints():
                    try:
                        if str(fp.GetReference()).upper() == ref_u:
                            return fp
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        def _find_pad(fp, pad_number: str):
            if fp is None:
                return None
            pn = str(pad_number)
            try:
                f = getattr(fp, 'FindPadByNumber', None)
                if callable(f):
                    p = f(pn)
                    if p is not None:
                        return p
            except Exception:
                pass
            try:
                for p in fp.Pads():
                    try:
                        if str(p.GetNumber()) == pn:
                            return p
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        def _find_or_create_net(name_in: str):
            name = (name_in or '').strip()
            if not name:
                return None
            try:
                n = board.FindNet(name)
                if n is not None:
                    return n
            except Exception:
                pass
            try:
                net_item = pcbnew.NETINFO_ITEM(board, name)
                add = getattr(board, 'Add', None)
                if callable(add):
                    board.Add(net_item)
                else:
                    an = getattr(board, 'AddNet', None)
                    if callable(an):
                        an(net_item)
                return net_item
            except Exception:
                return None

        net_obj = _find_or_create_net(net_name)
        if net_obj is None:
            return False, f"{net_name}: net create/find failed"

        assigned = 0
        errors: List[str] = []
        invalid: List[str] = []

        for item in pads:
            ref = ''
            pad_num = ''
            if isinstance(item, str):
                text = item.strip()
                if '/' not in text:
                    invalid.append(text)
                    continue
                ref, pad_num = text.split('/', 1)
            elif isinstance(item, dict):
                ref = str(item.get('ref', '') or '')
                pad_num = str(item.get('pad', '') or '')
            else:
                invalid.append(str(item))
                continue

            ref = ref.strip().upper()
            pad_num = pad_num.strip()
            if not ref or not pad_num:
                invalid.append(str(item))
                continue

            fp = _find_footprint(ref)
            if fp is None:
                errors.append(f"{ref}: footprint not found on board")
                continue
            pad = _find_pad(fp, pad_num)
            if pad is None:
                # List available pads so the LLM can self-correct
                avail = []
                try:
                    for p in fp.Pads():
                        try:
                            avail.append(str(p.GetNumber()))
                        except Exception:
                            pass
                except Exception:
                    pass
                avail_str = ', '.join(sorted(set(avail), key=lambda x: (len(x), x))[:20]) if avail else 'none'
                errors.append(f"{ref}/{pad_num}: pad not found. {ref} has {len(avail)} pads: [{avail_str}]")
                continue

            try:
                if hasattr(pad, 'SetNet'):
                    pad.SetNet(net_obj)
                else:
                    try:
                        pad.SetNetCode(int(net_obj.GetNet()))
                    except Exception:
                        pass
                assigned += 1
            except Exception as e:
                errors.append(f"{ref}/{pad_num}: set net failed ({e})")

        # Rebuild connectivity so subsequent routing sees updated net codes.
        try:
            if hasattr(board, 'BuildListOfNets'):
                board.BuildListOfNets()
        except Exception:
            pass
        try:
            conn = getattr(board, 'GetConnectivity', None)
            if callable(conn):
                c = conn()
                for m in ('RecalculateRatsnest', 'Recalculate', 'Rebuild', 'Build'):
                    fn = getattr(c, m, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:
                            continue
        except Exception:
            pass

        if assigned <= 0:
            suffix = (" Errors: " + "; ".join(errors[:8])) if errors else ""
            if invalid:
                suffix += (" Invalid: " + "; ".join(invalid[:5]))
            return False, f"No pads were assigned to net '{net_name}'." + suffix

        msg = f"Defined net '{net_name}' on {assigned} pad(s)."
        if errors:
            msg += f" Warnings: {len(errors)} item(s) could not be assigned."
            # Include the first few error details so the LLM can diagnose
            msg += " Details: " + "; ".join(errors[:5])
        if invalid:
            msg += f" Ignored {len(invalid)} invalid pad spec(s)."
        return True, msg

    async def _handle_add_component(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Place a footprint on the current PCB.

        Radical rewrite — simple, linear, no LLM verification, no nested closures.
        KiCad 7/8/9 compatible.
        """
        if not self._library_manager:
            return False, "Library manager not configured"
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (must run inside KiCad)"

        # ── helpers (defined once, used throughout) ──────────────────────────
        def mm2iu(mm_val: float) -> int:
            """Convert mm to KiCad internal units. Works on KiCad 7/8/9."""
            try:
                return int(pcbnew.FromMM(float(mm_val)))
            except Exception:
                return int(float(mm_val) * 1e6)  # nanometer fallback

        def make_pt(x_iu: int, y_iu: int):
            """Create a position object that SetPosition accepts."""
            for ctor_name in ('VECTOR2I', 'wxPoint'):
                ctor = getattr(pcbnew, ctor_name, None)
                if ctor is not None:
                    try:
                        return ctor(int(x_iu), int(y_iu))
                    except Exception:
                        continue
            return None

        def set_pos(footprint_obj, x_iu: int, y_iu: int) -> bool:
            pt = make_pt(x_iu, y_iu)
            if pt is not None:
                try:
                    footprint_obj.SetPosition(pt)
                    return True
                except Exception:
                    pass
            return False

        def read_xy(pos_obj) -> Tuple[int, int]:
            """Extract (x, y) ints from any KiCad point type."""
            if pos_obj is None:
                return 0, 0
            # .x / .y attributes (KiCad 8/9 VECTOR2I)
            for ax, ay in [('x', 'y'), ('X', 'Y')]:
                try:
                    xv, yv = getattr(pos_obj, ax, None), getattr(pos_obj, ay, None)
                    if isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
                        return int(xv), int(yv)
                except Exception:
                    pass
            # .GetX() / .GetY() (KiCad 7)
            try:
                gx, gy = getattr(pos_obj, 'GetX', None), getattr(pos_obj, 'GetY', None)
                if callable(gx) and callable(gy):
                    return int(gx()), int(gy())
            except Exception:
                pass
            return 0, 0

        # ── extract query ────────────────────────────────────────────────────
        params = action.parameters or {}
        query = ''
        for key in ('query', 'part_name', 'mpn', 'part', 'name'):
            v = params.get(key)
            if isinstance(v, str) and v.strip():
                query = v.strip()
                break
        if not query:
            query = (action.description or '').strip()
        if not query:
            return False, "No component/footprint query provided"

        # ── resolve board ────────────────────────────────────────────────────
        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                pass
        if board is None:
            return False, "No active board found"

        # ── resolve footprint path ───────────────────────────────────────────
        fp_path = None
        resolved_mpn = query

        # Fast path: caller already resolved path (plugin's _resolve_footprint_for_action)
        pre_path = params.get('local_footprint_path')
        if isinstance(pre_path, str) and pre_path.strip():
            fp_path = pre_path.strip()
            resolved_mpn = str(params.get('resolved_mpn', query)) or query
        else:
            # Search libraries
            if self._progress_callback:
                self._progress_callback(f"🔍 Searching for {query}...")

            package_hint = ''
            pkg_val = params.get('package')
            if isinstance(pkg_val, str) and pkg_val.strip():
                package_hint = pkg_val.strip()

            # Try resolve_best_footprint_item (fastest)
            for attempt_query in [query, package_hint, f"{query} {package_hint}"]:
                if not attempt_query:
                    continue
                try:
                    resolver = getattr(self._library_manager, 'resolve_best_footprint_item', None)
                    if callable(resolver):
                        item = resolver(attempt_query, package_hint=package_hint or None)
                        if item and getattr(item, 'local_footprint_path', None):
                            fp_path = item.local_footprint_path
                            resolved_mpn = getattr(item, 'mpn', query) or query
                            break
                except Exception:
                    pass

            # Try resolve_best_footprint_path
            if not fp_path:
                for attempt_query in [query, package_hint]:
                    if not attempt_query:
                        continue
                    try:
                        resolver = getattr(self._library_manager, 'resolve_best_footprint_path', None)
                        if callable(resolver):
                            result = resolver(attempt_query, package_hint=package_hint or None)
                            if result:
                                resolved_mpn, fp_path = result
                                break
                    except Exception:
                        pass

            # Full search fallback
            if not fp_path:
                search_queries = [q for q in [query, package_hint, f"{query} {package_hint}"] if q]
                for sq in search_queries:
                    try:
                        if hasattr(self._library_manager, 'search_parts'):
                            results = await self._library_manager.search_parts(sq)
                        else:
                            results = self._library_manager.search_parts_sync(sq)
                        for item in (results or []):
                            if getattr(item, 'local_footprint_path', None):
                                fp_path = item.local_footprint_path
                                resolved_mpn = getattr(item, 'mpn', query) or query
                                break
                        if fp_path:
                            break
                        # Try downloading online candidates
                        if not fp_path and results:
                            for item in results:
                                if getattr(item, 'footprint_url', None):
                                    try:
                                        project_dir = context.get('project_dir') if isinstance(context, dict) else None
                                        dl = self._library_manager.download_item(item, install=True, project_dir=project_dir)
                                        if dl.success and dl.footprint_path:
                                            fp_path = dl.footprint_path
                                            resolved_mpn = getattr(item, 'mpn', query) or query
                                            break
                                    except Exception:
                                        pass
                            if fp_path:
                                break
                    except Exception:
                        continue

        if not fp_path:
            return False, f"No footprint found for '{query}'. Try a more specific part name."

        # ── validate footprint file ──────────────────────────────────────────
        fp_file = Path(str(fp_path))
        if not fp_file.is_file():
            return False, f"Footprint file does not exist: {fp_file}"
        if fp_file.suffix.lower() != '.kicad_mod':
            return False, f"Not a .kicad_mod file: {fp_file}"
        try:
            head = fp_file.read_bytes()[:2048]
            if b'(footprint' not in head and b'(module' not in head:
                return False, f"File doesn't look like a KiCad footprint: {fp_file}"
        except Exception as e:
            return False, f"Cannot read footprint: {e}"

        # ── load footprint ───────────────────────────────────────────────────
        pretty_dir = fp_file.parent
        fp_name = fp_file.stem
        fp = None
        # Approach 1: Direct pcbnew top-level loaders
        for loader_name in ('FootprintLoad', 'LoadFootprint'):
            loader = getattr(pcbnew, loader_name, None)
            if callable(loader):
                try:
                    fp = loader(str(pretty_dir), fp_name)
                    if fp is not None:
                        break
                except Exception:
                    continue
        # Approach 2: IO_MGR plugin-based loading (KiCad 9 fallback)
        if fp is None:
            for io_cls_name in ('PCB_IO_KICAD_SEXPR', 'PCB_IO'):
                io_cls = getattr(pcbnew, io_cls_name, None)
                if io_cls is None:
                    continue
                try:
                    io = io_cls()
                    fp = io.FootprintLoad(str(pretty_dir), fp_name)
                    if fp is not None:
                        break
                except Exception:
                    continue
        # Approach 3: IO_MGR.PluginFind (alternative KiCad 9 API)
        if fp is None:
            io_mgr = getattr(pcbnew, 'IO_MGR', None)
            if io_mgr:
                for fmt_name in ('KICAD_SEXP', 'KICAD'):
                    fmt = getattr(io_mgr, fmt_name, None)
                    if fmt is None:
                        continue
                    try:
                        plugin_find = getattr(io_mgr, 'PluginFind', None)
                        if callable(plugin_find):
                            plugin = plugin_find(fmt)
                            if plugin:
                                fp = plugin.FootprintLoad(str(pretty_dir), fp_name)
                                if fp is not None:
                                    break
                    except Exception:
                        continue
        if fp is None:
            return False, f"Failed to load footprint '{fp_name}' from {pretty_dir}"

        # Reject footprints with obviously broken drill geometry.
        # This avoids persistent DRC failures that don't go away when you move parts.
        try:
            bad, why = self._footprint_has_colocated_drills(fp, pcbnew)
            if bad:
                return False, f"Rejected footprint '{fp_name}': {why}. Try a different footprint (e.g., discrete headers instead of board modules)."
        except Exception:
            pass

        # ── compute placement position ───────────────────────────────────────
        # Step 1: Get board center (where components SHOULD go)
        board_center_x = mm2iu(148.5)  # A4 page center default
        board_center_y = mm2iu(105.0)

        # Try board outline center first (most reliable on KiCad 9)
        for bb_method in ('GetBoardEdgesBoundingBox', 'ComputeBoundingBox'):
            try:
                fn = getattr(board, bb_method, None)
                if not callable(fn):
                    continue
                bbox = fn() if bb_method == 'GetBoardEdgesBoundingBox' else fn(False)
                if bbox is None:
                    continue
                bw = int(bbox.GetWidth())
                bh = int(bbox.GetHeight())
                if bw > mm2iu(5) and bh > mm2iu(5):
                    board_center_x = int(bbox.GetX()) + bw // 2
                    board_center_y = int(bbox.GetY()) + bh // 2
                    break
            except Exception:
                continue

        # Step 2: Parse explicit location (if provided by LLM/user)
        target_x = board_center_x
        target_y = board_center_y

        loc = params.get('location')
        parsed_loc = self._parse_location_mm(loc)
        if parsed_loc is not None:
            target_x = mm2iu(parsed_loc[0])
            target_y = mm2iu(parsed_loc[1])

        # Step 3: ANTI-OVERLAP — uses real bounding boxes, not just center points
        # Collect occupied rectangles (left, top, right, bottom) in IU for all existing FPs
        class _OccRect:
            __slots__ = ('l', 't', 'r', 'b')
            def __init__(self, l, t, r, b):
                self.l, self.t, self.r, self.b = l, t, r, b

        occ_rects: List[_OccRect] = []
        PLACE_MARGIN = mm2iu(2.0)  # 2mm clearance between parts
        try:
            for efp in board.GetFootprints():
                try:
                    ebb = None
                    for bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                        fn = getattr(efp, bbm, None)
                        if callable(fn):
                            try:
                                ebb = fn() if bbm == 'GetCourtyardBoundingBox' else fn(False, False)
                                if ebb and ebb.GetWidth() > 0:
                                    break
                                ebb = None
                            except Exception:
                                ebb = None
                    if ebb is not None:
                        occ_rects.append(_OccRect(
                            int(ebb.GetLeft()) - PLACE_MARGIN,
                            int(ebb.GetTop()) - PLACE_MARGIN,
                            int(ebb.GetRight()) + PLACE_MARGIN,
                            int(ebb.GetBottom()) + PLACE_MARGIN,
                        ))
                    else:
                        # Fallback: use position + conservative 10mm radius
                        pos = read_xy(efp.GetPosition())
                        if pos[0] != 0 or pos[1] != 0:
                            half = mm2iu(5.0)
                            occ_rects.append(_OccRect(
                                pos[0] - half - PLACE_MARGIN,
                                pos[1] - half - PLACE_MARGIN,
                                pos[0] + half + PLACE_MARGIN,
                                pos[1] + half + PLACE_MARGIN,
                            ))
                except Exception:
                    pass
        except Exception:
            pass

        # Estimate incoming footprint size for proper collision detection
        new_fp_hw = mm2iu(5.0)  # default half-width
        new_fp_hh = mm2iu(5.0)  # default half-height
        try:
            new_fp_bb = None
            for bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                fn = getattr(fp, bbm, None)
                if callable(fn):
                    try:
                        new_fp_bb = fn() if bbm == 'GetCourtyardBoundingBox' else fn(False, False)
                        if new_fp_bb and new_fp_bb.GetWidth() > 0:
                            new_fp_hw = int(new_fp_bb.GetWidth()) // 2
                            new_fp_hh = int(new_fp_bb.GetHeight()) // 2
                            break
                    except Exception:
                        pass
        except Exception:
            pass

        def would_overlap(cx: int, cy: int) -> bool:
            """Check if placing the new FP centered at (cx, cy) overlaps any existing FP."""
            nl = cx - new_fp_hw
            nt = cy - new_fp_hh
            nr = cx + new_fp_hw
            nb = cy + new_fp_hh
            for o in occ_rects:
                if not (nr + PLACE_MARGIN <= o.l or nl - PLACE_MARGIN >= o.r or
                        nb + PLACE_MARGIN <= o.t or nt - PLACE_MARGIN >= o.b):
                    return True
            return False

        grid = mm2iu(10.0)  # 10mm placement grid step
        if would_overlap(target_x, target_y):
            # Spiral outward to find a free slot
            placed = False
            for r in range(1, 20):
                if placed:
                    break
                for dx in range(-r, r + 1):
                    if placed:
                        break
                    for dy in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            continue
                        nx = target_x + dx * grid
                        ny = target_y + dy * grid
                        if not would_overlap(nx, ny):
                            target_x, target_y = nx, ny
                            placed = True
                            break

        # ── place on board ───────────────────────────────────────────────────
        try:
            board.Add(fp)
        except Exception as e:
            return False, f"board.Add() failed: {e}"

        # ── clamp to board outline if one exists ─────────────────────────────
        try:
            bb = board.GetBoardEdgesBoundingBox()
            bw, bh = int(bb.GetWidth()), int(bb.GetHeight())
            if bw > mm2iu(5) and bh > mm2iu(5):
                margin = mm2iu(1.0)  # 1mm inset from board edge
                bx_min = int(bb.GetX()) + margin
                bx_max = int(bb.GetX()) + bw - margin
                by_min = int(bb.GetY()) + margin
                by_max = int(bb.GetY()) + bh - margin
                # Also account for footprint physical size via courtyard/bbox
                fp_bb = None
                for bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                    fn = getattr(fp, bbm, None)
                    if callable(fn):
                        try:
                            fp_bb = fn() if bbm == 'GetCourtyardBoundingBox' else fn(False, False)
                            if fp_bb and fp_bb.GetWidth() > 0:
                                break
                            fp_bb = None
                        except Exception:
                            fp_bb = None
                if fp_bb is not None:
                    hw = int(fp_bb.GetWidth()) // 2
                    hh = int(fp_bb.GetHeight()) // 2
                    bx_min += hw
                    bx_max -= hw
                    by_min += hh
                    by_max -= hh
                clamped = False
                if bx_max > bx_min and by_max > by_min:
                    new_x = max(bx_min, min(bx_max, target_x))
                    new_y = max(by_min, min(by_max, target_y))
                    if new_x != target_x or new_y != target_y:
                        target_x, target_y = new_x, new_y
                        clamped = True
        except Exception:
            pass

        if not set_pos(fp, target_x, target_y):
            return False, "SetPosition failed (KiCad API incompatible)"

        # Record for future anti-overlap
        self._session_placed_positions.append((target_x, target_y))

        # ── assign reference designator ──────────────────────────────────────
        try:
            def _infer_ref_prefix(q: str, fp_lib: str, fp_name: str) -> str:
                t = f"{q} {fp_lib} {fp_name}".lower()
                # Mechanical
                if "mounting" in t or "mountinghole" in t or fp_lib.lower().startswith("mount"):
                    return "H"
                # Passives
                if fp_lib.lower().startswith("resistor") or "resistor" in t or t.startswith("r_"):
                    return "R"
                if fp_lib.lower().startswith("capacitor") or "capacitor" in t or t.startswith("c_"):
                    return "C"
                if fp_lib.lower().startswith("inductor") or "inductor" in t or t.startswith("l_"):
                    return "L"
                if fp_lib.lower().startswith("crystal") or "crystal" in t or "oscillator" in t or "xtal" in t:
                    return "Y"
                # Diodes / LEDs
                if fp_lib.lower().startswith("led") or "led" in t or "diode" in t:
                    return "D"
                # Switches
                if fp_lib.lower().startswith(("button_switch", "switch")) or "switch" in t or "button" in t:
                    return "SW"
                # Connectors
                if fp_lib.lower().startswith("connector") or "connector" in t or "header" in t or "usb" in t or "jack" in t:
                    return "J"
                # Fuses / batteries
                if "fuse" in t:
                    return "F"
                if "battery" in t:
                    return "BT"
                # Packages / ICs default
                if fp_lib.lower().startswith("package") or "qfn" in t or "tqfp" in t or "soic" in t or "dip" in t:
                    return "U"
                return "U"

            user_prefix = params.get('ref_prefix', '')
            if isinstance(user_prefix, str):
                user_prefix = user_prefix.strip()
            else:
                user_prefix = ""

            fp_lib = ""
            fp_name = ""
            try:
                fp_lib = fp_file.parent.stem  # "*.pretty" dir name
                fp_name = fp_file.stem
            except Exception:
                fp_lib = ""
                fp_name = ""

            inferred = _infer_ref_prefix(query, fp_lib, fp_name)
            # Treat LLM-provided ref_prefix as a hint; override when it disagrees
            # with footprint/library-based inference (prevents e.g. crystals as J*).
            ref_prefix = inferred if inferred else (user_prefix or "U")

            max_n = 0
            try:
                for efp in board.GetFootprints():
                    try:
                        ref = efp.GetReference() or ''
                        if ref.startswith(ref_prefix):
                            m2 = re.match(re.escape(ref_prefix) + r'(\d+)', ref)
                            if m2:
                                max_n = max(max_n, int(m2.group(1)))
                    except Exception:
                        continue
            except Exception:
                pass
            new_ref = f"{ref_prefix}{max_n + 1}"
            fp.SetReference(new_ref)
        except Exception:
            new_ref = 'U?'

        # ── pad-count sanity check ───────────────────────────────────────────
        pad_warning = ''
        try:
            pad_count = sum(1 for _ in fp.Pads())
            if pad_count <= 4 and ref_prefix == 'U':
                pad_warning = (
                    f" ⚠️ WARNING: footprint has only {pad_count} pad(s) — this "
                    f"may be wrong for an IC. Consider removing and re-adding "
                    f"with an explicit package (e.g. 'DIP-28', 'TQFP-32')."
                )
        except Exception:
            pass

        placed_name = Path(str(fp_path)).stem if fp_path else resolved_mpn
        return True, f"Placed {placed_name} as {new_ref}.{pad_warning}"

    async def _handle_download_symbol_or_footprint(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Handle DOWNLOAD_SYMBOL / DOWNLOAD_FOOTPRINT by searching + downloading."""
        if not self._library_manager:
            return False, "Library manager not configured"

        def _extract_part_name() -> str:
            params = action.parameters or {}
            for key in ("part_name", "query", "mpn", "part", "name"):
                v = params.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # Last-resort: try to extract a likely part string from the description.
            desc = (action.description or "").strip()
            # e.g. "Download DIP-28 footprint for ATmega328P-PU"
            m = re.search(r"for\s+([A-Za-z0-9][A-Za-z0-9_+\-./]{2,80})", desc)
            if m:
                return m.group(1).strip()
            return ""

        part_name = _extract_part_name()
        if not part_name:
            return False, "No part name provided"

        # Extract package hint from action parameters (e.g. "TSSOP-28").
        package_hint = ''
        params = action.parameters or {}
        v = params.get('package')
        if isinstance(v, str) and v.strip():
            package_hint = v.strip()

        is_footprint_action = action.action_type == DesignActionType.DOWNLOAD_FOOTPRINT

        # For footprint downloads, try package-aware resolution first.
        if is_footprint_action and package_hint:
            resolver = getattr(self._library_manager, 'resolve_best_footprint_path', None)
            if callable(resolver):
                resolved = resolver(part_name, package_hint=package_hint)
                if resolved is None:
                    resolved = resolver(package_hint)
                if resolved is not None:
                    resolved_mpn, fp_path = resolved
                    return True, (
                        f"✅ Found footprint for {part_name} ({package_hint})!\n\n"
                        f"Footprint: {resolved_mpn}\n"
                        f"Path: {fp_path}\n\n"
                        f"No download needed — the footprint is already available locally."
                    )

        if self._progress_callback:
            self._progress_callback(f"Searching for {part_name}...")

        try:
            # Search with part name first; if that fails and we have a package
            # hint, retry with the package (footprints are often named by package,
            # not by the IC part number).
            results = None
            search_queries = [part_name]
            if package_hint and package_hint.upper() not in part_name.upper():
                search_queries.append(package_hint)
                search_queries.append(f"{part_name} {package_hint}")

            for sq in search_queries:
                if hasattr(self._library_manager, 'search_parts'):
                    results = await self._library_manager.search_parts(sq)
                elif hasattr(self._library_manager, 'search_parts_sync'):
                    results = self._library_manager.search_parts_sync(sq)
                else:
                    return False, "Library manager does not support search"
                if results:
                    break

            if not results:
                return False, (
                    f"No results found for '{part_name}'"
                    + (f" ({package_hint})" if package_hint else "") + ".\n"
                    f"Tip: try using the exact manufacturer part number "
                    f"(e.g. 'XYZ123' instead of 'XYZ123-PU')."
                )

            # Pick the first (best) match and process.
            item = results[0]

            # KiCad built-in parts need no download
            from .library_manager import LibrarySource
            if item.source == LibrarySource.KICAD_BUILTIN:
                is_footprint_action = action.action_type == DesignActionType.DOWNLOAD_FOOTPRINT
                chooser = "Footprint Chooser" if is_footprint_action else "symbol chooser"
                editor = "PCB editor" if is_footprint_action else "schematic editor"
                footprint_hint = ""
                try:
                    pkg = (item.package or "").strip().upper()
                except Exception:
                    pkg = ""
                if pkg.startswith("DIP-") or pkg.startswith("PDIP-"):
                    # Common through-hole footprint name in official KiCad libs.
                    footprint_hint = (
                        "\n\nFootprint hint: in the PCB editor, open the Footprint Chooser and search for "
                        "\"DIP-28\" (e.g. `Package_DIP:DIP-28_W7.62mm`)."
                    )
                return True, (
                    f"✅ Found {item.mpn} in KiCad's built-in library!\n\n"
                    f"Library: {item.name}\n"
                    f"Package: {item.package}\n"
                    f"Description: {item.description}\n\n"
                    f"No download needed — open the {chooser} in the "
                    f"{editor} and search for \"{item.mpn}\"."
                    f"{footprint_hint}"
                )

            if self._progress_callback:
                self._progress_callback(f"Downloading {item.mpn or part_name}...")

            project_dir = None
            try:
                project_dir = context.get('project_dir') if isinstance(context, dict) else None
            except Exception:
                project_dir = None

            download_result = self._library_manager.download_item(item, install=True, project_dir=project_dir)
            if download_result.success:
                paths = []
                if download_result.symbol_path:
                    paths.append(f"Symbol: {download_result.symbol_path}")
                if download_result.footprint_path:
                    paths.append(f"Footprint: {download_result.footprint_path}")
                detail = "\n".join(paths) if paths else "Files downloaded."
                return True, f"Installed {item.mpn or part_name}.\n{detail}"
            else:
                return False, f"Download failed: {download_result.message}"

        except Exception as e:
            return False, f"Download/install failed: {e}"
    
    async def _handle_search_part(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Handle part search action."""
        if not self._library_manager:
            return False, "Library manager not configured"

        query = self._extract_search_query(action)
        if not query:
            return False, (
                "No search query provided. "
                "Try: 'search USB-C connector' or 'find part XYZ123-PU'."
            )
        
        try:
            # LibraryManager may expose either async or sync APIs depending on runtime.
            if hasattr(self._library_manager, 'search_parts'):
                maybe_coro = self._library_manager.search_parts(query)
                results = await maybe_coro
            elif hasattr(self._library_manager, 'search_parts_sync'):
                results = self._library_manager.search_parts_sync(query)
            else:
                return False, "Library manager does not support search"
            if results:
                max_show = min(len(results), 20)
                lines = [f"Found {len(results)} matching parts:"]
                serialized_results = []
                for item in results[:max_show]:
                    label = (getattr(item, 'name', '') or getattr(item, 'mpn', '')).strip()
                    pkg = (getattr(item, 'package', '') or '').strip()
                    src = ''
                    try:
                        src = getattr(getattr(item, 'source', None), 'value', '')
                    except Exception:
                        src = ''
                    suffix = ""
                    if pkg:
                        suffix += f" ({pkg})"
                    if src:
                        suffix += f" [{src}]"
                    lines.append(f"- {label}{suffix}")
                    serialized_results.append({
                        "name": label,
                        "package": pkg,
                        "source": src,
                    })
                if len(results) > max_show:
                    lines.append(f"…and {len(results) - max_show} more")
                # Persist structured search output for downstream phases.
                store = context.setdefault("search_part_results", {})
                if isinstance(store, dict):
                    store[query] = serialized_results
                return True, "\n".join(lines)
            else:
                return True, "No parts found matching the query. Tip: try a shorter part number, a description, or a manufacturer name."
        except Exception as e:
            return False, f"Search failed: {e}"

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
    
    async def _handle_export_bom(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Handle BOM export action."""
        if not self._bom_exporter:
            return False, "BOM exporter not configured"
        
        pcb_data = context.get('pcb_data')
        if not pcb_data:
            return False, "No PCB data available"
        
        try:
            entries = self._bom_exporter.extract_from_pcb(pcb_data)
            if not entries:
                return False, "No components found in PCB"
            
            from .bom_exporter import BOMExportRequest, BOMFormat
            
            # Determine format
            format_str = action.parameters.get('format', '').lower()
            format_map = {
                'jlcpcb': BOMFormat.CSV_JLCPCB,
                'lcsc': BOMFormat.CSV_LCSC,
                'mouser': BOMFormat.CSV_MOUSER,
                'digikey': BOMFormat.CSV_DIGIKEY,
            }
            bom_format = format_map.get(format_str, BOMFormat.CSV_GENERIC)
            
            request = BOMExportRequest(
                entries=entries,
                format=bom_format,
                project_name=context.get('project_name'),
            )
            
            preview = self._bom_exporter.create_preview(request)
            action.preview_text = preview
            
            return True, f"BOM ready with {len(entries)} unique parts"
            
        except Exception as e:
            return False, f"BOM generation failed: {e}"
    
    async def _handle_draw_track(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Handle PCB track drawing action.

        Note: This must execute on KiCad's GUI thread (pcbnew is not thread-safe).
        """
        from_point = str(action.parameters.get('from_point', '') or '').strip()
        to_point = str(action.parameters.get('to_point', '') or '').strip()

        if not from_point or not to_point:
            return False, "Missing from_point or to_point"

        # Import pcbnew lazily so unit tests/outside-KiCad runs still work.
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (this action must run inside KiCad)"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        def _parse_ref_pad(text: str) -> Optional[Tuple[str, str]]:
            s = (text or '').strip()
            if not s:
                return None
            # Normalize common variants
            s = re.sub(r"\bpad\b", " ", s, flags=re.IGNORECASE)
            s = re.sub(r"\bpin\b", " ", s, flags=re.IGNORECASE)
            s = re.sub(r"\s+of\s+", " ", s, flags=re.IGNORECASE)
            # Accept "-" as a separator (e.g. "J1-1", "R1-2")
            s = s.replace(':', '/').replace('.', '/').replace('#', '/').replace('-', ' ').strip()
            s = re.sub(r"\s+", " ", s)
            # Accept: D1/2, D1 / 2, "2 D1", "D1 2", "J1 1"
            m = re.match(r"^(?P<ref>[A-Za-z]+\d+)\s*(?:/|\s)\s*(?P<pad>[A-Za-z0-9_+-]+)$", s)
            if m:
                return m.group('ref').upper(), m.group('pad')
            m = re.match(r"^(?P<pad>[A-Za-z0-9_+-]+)\s+(?P<ref>[A-Za-z]+\d+)$", s)
            if m:
                return m.group('ref').upper(), m.group('pad')
            return None

        def _find_footprint(ref: str):
            # KiCad API varies by version; try best-effort.
            for fp in board.GetFootprints():
                try:
                    if str(fp.GetReference()) == ref:
                        return fp
                except Exception:
                    continue
            return None

        def _find_pad(footprint, pad_number: str):
            try:
                fn = getattr(footprint, 'FindPadByNumber', None)
                if callable(fn):
                    p = fn(str(pad_number))
                    if p is not None:
                        return p
            except Exception:
                pass
            try:
                for p in footprint.Pads():
                    try:
                        if str(p.GetNumber()) == str(pad_number):
                            return p
                    except Exception:
                        continue
            except Exception:
                pass
            return None

        parsed_a = _parse_ref_pad(from_point)
        parsed_b = _parse_ref_pad(to_point)

        # If the endpoints aren't explicit, try the "each LED" inference at execution time too.
        if not parsed_a or not parsed_b:
            pcb_data = context.get('pcb_data')
            led_refs: List[str] = []
            try:
                if pcb_data:
                    for fp in getattr(pcb_data, 'footprints', []) or []:
                        ref = str(getattr(fp, 'reference', '') or '').strip()
                        val = str(getattr(fp, 'value', '') or '').strip().lower()
                        if not ref:
                            continue
                        if ref.upper().startswith('D') or 'led' in val:
                            led_refs.append(ref)
            except Exception:
                led_refs = []
            # Only use this inference when there are exactly two LED-like refs and the user
            # provided a shared pad number via 'from_point'/'to_point' being 'pad N'.
            if len(led_refs) == 2:
                # If user used 'pad 2' in either endpoint, parse_ref_pad won't capture it; attempt to grab a lone pad token.
                pad_m = re.search(r"\b(\w+)\b", from_point) if from_point else None
                pad = pad_m.group(1) if pad_m else None
                if not pad:
                    pad_m = re.search(r"\b(\w+)\b", to_point) if to_point else None
                    pad = pad_m.group(1) if pad_m else None
                if pad:
                    parsed_a = (led_refs[0].upper(), str(pad))
                    parsed_b = (led_refs[1].upper(), str(pad))

        if not parsed_a or not parsed_b:
            # Fall back to the connection manager's resolver when available.
            if not self._connection_manager:
                return False, f"Could not resolve endpoints: {from_point}, {to_point}"
            try:
                pcb_data = context.get('pcb_data')
                request = self._connection_manager.parse_connection_request(
                    f"connect {from_point} to {to_point}",
                    pcb_data=pcb_data,
                )
                if not request:
                    return False, f"Could not resolve endpoints: {from_point}, {to_point}"
                # Draw a straight track between resolved points.
                from_mm = getattr(pcbnew, 'FromMM', None)
                if not callable(from_mm):
                    return False, "KiCad pcbnew API missing FromMM(); cannot draw track."
                start = pcbnew.VECTOR2I(int(from_mm(request.from_point.x)), int(from_mm(request.from_point.y)))
                end = pcbnew.VECTOR2I(int(from_mm(request.to_point.x)), int(from_mm(request.to_point.y)))
                a_pad = None
                b_pad = None
            except Exception as e:
                return False, f"Track planning failed: {e}"
        else:
            ref_a, pad_a = parsed_a
            ref_b, pad_b = parsed_b
            fp_a = _find_footprint(ref_a)
            fp_b = _find_footprint(ref_b)
            if fp_a is None or fp_b is None:
                return False, f"Could not find footprints: {ref_a if fp_a is None else ''} {ref_b if fp_b is None else ''}".strip()

            a_pad = _find_pad(fp_a, pad_a)
            b_pad = _find_pad(fp_b, pad_b)
            if a_pad is None or b_pad is None:
                return False, f"Could not find pads: {ref_a}/{pad_a if a_pad is None else ''} {ref_b}/{pad_b if b_pad is None else ''}".strip()

            try:
                start = a_pad.GetPosition()
                end = b_pad.GetPosition()
            except Exception as e:
                return False, f"Failed to read pad positions: {e}"

        # Create tracks using 45-degree routing (horizontal/vertical + diagonal).
        try:
            # Width: use current board setting when possible.
            width = None
            try:
                ds = board.GetDesignSettings()
                width = int(ds.GetCurrentTrackWidth())
            except Exception:
                width = None
            if not width:
                from_mm = getattr(pcbnew, 'FromMM', None)
                width = int(from_mm(0.25)) if callable(from_mm) else 250000

            # Layer: active layer if possible, else F.Cu
            layer = None
            try:
                layer = int(board.GetActiveLayer())
            except Exception:
                layer = getattr(pcbnew, 'F_Cu', None)
            if layer is None:
                layer = 0

            # Net info for all segments
            def _pad_net_name(pad) -> str:
                if pad is None:
                    return ""
                try:
                    fn = getattr(pad, 'GetNetname', None)
                    if callable(fn):
                        return str(fn() or '')
                except Exception:
                    pass
                try:
                    net = pad.GetNet()
                    if net is not None:
                        # KiCad versions vary; try common getters.
                        for attr in ('GetNetname', 'GetNetName', 'GetName'):
                            g = getattr(net, attr, None)
                            if callable(g):
                                v = g()
                                if v:
                                    return str(v)
                except Exception:
                    pass
                return ""

            a_nc = 0
            b_nc = 0
            a_nn = ""
            b_nn = ""
            try:
                if a_pad is not None:
                    a_nc = int(a_pad.GetNetCode())
                    a_nn = _pad_net_name(a_pad)
            except Exception:
                a_nc = 0
            try:
                if b_pad is not None:
                    b_nc = int(b_pad.GetNetCode())
                    b_nn = _pad_net_name(b_pad)
            except Exception:
                b_nc = 0

            # Safety: never create an intentional short between two assigned nets.
            # If both pads are already assigned and different, refuse.
            if a_nc > 0 and b_nc > 0 and a_nc != b_nc:
                a_label = a_nn or str(a_nc)
                b_label = b_nn or str(b_nc)
                return False, f"Refusing to draw track: pads are on different nets ({a_label} vs {b_label}). Pick pads on the same net or fix net assignments (ASSIGN_NETS / Update PCB from Schematic)."

            # Choose a net for the new track: prefer any non-zero endpoint net.
            net_obj = None
            net_code = 0
            prefer_pad = None
            if a_nc > 0:
                prefer_pad = a_pad
            elif b_nc > 0:
                prefer_pad = b_pad
            else:
                prefer_pad = a_pad
            if prefer_pad is not None:
                try:
                    net_obj = prefer_pad.GetNet()
                except Exception:
                    net_obj = None
                try:
                    net_code = int(prefer_pad.GetNetCode())
                except Exception:
                    net_code = 0

            def _add_track_segment(seg_start, seg_end):
                """Helper: create a single PCB_TRACK segment and add to board."""
                t = None
                try:
                    t = pcbnew.PCB_TRACK(board)
                except Exception:
                    t = pcbnew.PCB_TRACK()
                t.SetStart(seg_start)
                t.SetEnd(seg_end)
                t.SetWidth(width)
                t.SetLayer(layer)
                if net_obj is not None and hasattr(t, 'SetNet'):
                    try:
                        t.SetNet(net_obj)
                    except Exception:
                        pass
                elif net_code:
                    try:
                        t.SetNetCode(net_code)
                    except Exception:
                        pass
                board.Add(t)

            # Route using 45-degree segments (cardinal + diagonal).
            # Strategy: go diagonal as far as possible, then finish
            # with a horizontal or vertical segment.
            sx = start.x if hasattr(start, 'x') else start.GetX() if hasattr(start, 'GetX') else start[0]
            sy = start.y if hasattr(start, 'y') else start.GetY() if hasattr(start, 'GetY') else start[1]
            ex = end.x if hasattr(end, 'x') else end.GetX() if hasattr(end, 'GetX') else end[0]
            ey = end.y if hasattr(end, 'y') else end.GetY() if hasattr(end, 'GetY') else end[1]

            dx = ex - sx
            dy = ey - sy

            # If already aligned (horizontal, vertical, or 45°), single segment
            if dx == 0 or dy == 0 or abs(dx) == abs(dy):
                _add_track_segment(start, end)
            else:
                # Two-segment 45° route: diagonal + cardinal
                diag = min(abs(dx), abs(dy))
                diag_dx = diag if dx > 0 else -diag
                diag_dy = diag if dy > 0 else -diag

                mid_x = sx + diag_dx
                mid_y = sy + diag_dy
                mid_pt = pcbnew.VECTOR2I(int(mid_x), int(mid_y))

                _add_track_segment(start, mid_pt)
                _add_track_segment(mid_pt, end)

            return True, "Track drawn"

        except Exception as e:
            return False, f"Failed to draw track: {e}"
    
    async def _handle_move_component(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Move a footprint to an (x,y) location in mm.

        This must be executed on KiCad's GUI thread (pcbnew is not thread-safe).
        """
        params = action.parameters or {}
        ref = str(params.get('ref', '') or '').strip().upper()
        location = params.get('location')
        if not ref or location in (None, ""):
            return False, "Missing component reference or location"

        # Import pcbnew lazily so unit tests/outside-KiCad runs still work.
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (this action must run inside KiCad)"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        parsed = self._parse_location_mm(location)
        if parsed is None:
            return False, f"Could not parse location '{location}'. Use 'x,y' in mm (e.g., '50,25')."

        x_mm, y_mm = parsed

        from_mm = getattr(pcbnew, 'FromMM', None)
        if not callable(from_mm):
            return False, "KiCad pcbnew API missing FromMM(); cannot move by coordinates."

        target = pcbnew.VECTOR2I(int(from_mm(x_mm)), int(from_mm(y_mm)))

        # Find the footprint
        footprint = None
        try:
            for fp in board.GetFootprints():
                try:
                    if str(fp.GetReference()).upper() == ref:
                        footprint = fp
                        break
                except Exception:
                    continue
        except Exception:
            footprint = None

        if footprint is None:
            return False, f"Could not find footprint {ref} on the board"

        try:
            footprint.SetPosition(target)
        except Exception as e:
            return False, f"Failed to move {ref}: {e}"

        # Check for overlaps with other footprints after moving
        overlap_warnings: List[str] = []
        try:
            moved_bb = None
            for bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                fn = getattr(footprint, bbm, None)
                if callable(fn):
                    try:
                        moved_bb = fn() if bbm == 'GetCourtyardBoundingBox' else fn(False, False)
                        if moved_bb and moved_bb.GetWidth() > 0:
                            break
                        moved_bb = None
                    except Exception:
                        moved_bb = None
            if moved_bb is not None:
                margin = int(from_mm(0.5))
                ml = int(moved_bb.GetLeft()) - margin
                mt = int(moved_bb.GetTop()) - margin
                mr = int(moved_bb.GetRight()) + margin
                mb = int(moved_bb.GetBottom()) + margin
                for other_fp in board.GetFootprints():
                    try:
                        other_ref = str(other_fp.GetReference()).upper()
                        if other_ref == ref:
                            continue
                        other_bb = None
                        for bbm2 in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                            fn2 = getattr(other_fp, bbm2, None)
                            if callable(fn2):
                                try:
                                    other_bb = fn2() if bbm2 == 'GetCourtyardBoundingBox' else fn2(False, False)
                                    if other_bb and other_bb.GetWidth() > 0:
                                        break
                                    other_bb = None
                                except Exception:
                                    other_bb = None
                        if other_bb is None:
                            continue
                        ol = int(other_bb.GetLeft())
                        ot = int(other_bb.GetTop())
                        orr = int(other_bb.GetRight())
                        ob = int(other_bb.GetBottom())
                        if not (mr <= ol or ml >= orr or mb <= ot or mt >= ob):
                            overlap_warnings.append(other_ref)
                    except Exception:
                        continue
        except Exception:
            pass

        msg = f"Moved {ref} to ({x_mm}, {y_mm}) mm"
        if overlap_warnings:
            # Try auto-resolving the overlap using PlacementAgent's spatial optimizer.
            resolved = self._try_auto_resolve_overlap(
                footprint, ref, board, overlap_warnings, pcbnew, from_mm
            )
            if resolved:
                msg = resolved
            else:
                msg += (
                    f" ⚠️ WARNING: {ref} now overlaps with "
                    f"{', '.join(overlap_warnings[:5])}. "
                    "Use MOVE_COMPONENT to separate them before routing."
                )
        return True, msg

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
    
    async def _handle_rotate_component(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Rotate a footprint by an angle in degrees.

        This must be executed on KiCad's GUI thread (pcbnew is not thread-safe).
        """
        params = action.parameters or {}
        ref = str(params.get('ref', '') or '').strip().upper()
        angle = params.get('angle', '90')
        if not ref:
            return False, "Missing component reference"

        try:
            angle_deg = float(str(angle).strip())
        except Exception:
            return False, f"Invalid angle: {angle}"

        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (this action must run inside KiCad)"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        footprint = None
        try:
            for fp in board.GetFootprints():
                try:
                    if str(fp.GetReference()).upper() == ref:
                        footprint = fp
                        break
                except Exception:
                    continue
        except Exception:
            footprint = None

        if footprint is None:
            return False, f"Could not find footprint {ref} on the board"

        # Best-effort across KiCad versions
        try:
            if hasattr(footprint, 'SetOrientationDegrees'):
                footprint.SetOrientationDegrees(float(angle_deg))
            else:
                # Older APIs may use SetOrientation(EDA_ANGLE)
                eda_angle = getattr(pcbnew, 'EDA_ANGLE', None)
                if eda_angle is not None:
                    footprint.SetOrientation(eda_angle(angle_deg, pcbnew.DEGREES_T))
                else:
                    return False, "KiCad pcbnew API missing SetOrientationDegrees/EDA_ANGLE"
        except Exception as e:
            return False, f"Failed to rotate {ref}: {e}"

        return True, f"Rotated {ref} by {angle_deg}°"
    
    async def _handle_run_drc(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Run KiCad Design Rule Check and return structured results.

        Strategy:
          1. Try kicad-cli pcb drc (KiCad 8+) for JSON output
          2. Fallback: pcbnew.WriteDRCReport()
          3. Fallback: parse board markers after manual DRC
        """
        import subprocess
        import tempfile
        import json as _json

        # Get the board file path
        board_path = context.get('board_path', '')
        if not board_path:
            try:
                import pcbnew as _pcbnew
                board = _pcbnew.GetBoard()
                if board:
                    board_path = board.GetFileName() or ''
            except Exception:
                pass

        if not board_path:
            # Try to save current board to temp file
            try:
                import pcbnew as _pcbnew
                board = _pcbnew.GetBoard()
                if board:
                    fd, tmp = tempfile.mkstemp(suffix='.kicad_pcb')
                    import os
                    os.close(fd)
                    _pcbnew.SaveBoard(tmp, board)
                    board_path = tmp
            except Exception:
                return False, "No board available for DRC"

        errors: List[str] = []
        warnings: List[str] = []
        drc_ran = False

        # Strategy 1: kicad-cli (KiCad 7/8/9+)
        # We assume the board might have in-memory changes (YOLO mode), so we ALWAYS save 
        # a temporary copy to run the CLI against. This ensures truth.
        try:
            fd, temp_board_path = tempfile.mkstemp(suffix='.kicad_pcb')
            import os
            os.close(fd)
            
            import pcbnew as _pcbnew
            # Save current board state to temp file
            current_board = context.get('board') or _pcbnew.GetBoard()
            if current_board:
                _pcbnew.SaveBoard(temp_board_path, current_board)
            else:
                 # Fallback if no board object found (unlikely)
                 if board_path and os.path.exists(board_path):
                     import shutil
                     shutil.copy2(board_path, temp_board_path)

            fd_rep, report_path = tempfile.mkstemp(suffix='.json')
            os.close(fd_rep)

            # Locate kicad-cli
            cli_cmd = 'kicad-cli'
            import platform
            if platform.system() == 'Darwin':
                # Common macOS locations
                candidates = [
                    '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',
                    '/Applications/KiCad/kicad.app/Contents/MacOS/kicad-cli',
                    'kicad-cli' # fallback to PATH
                ]
                for c in candidates:
                    if os.path.exists(c) and os.access(c, os.X_OK):
                        cli_cmd = c
                        break
            
            # Run DRC
            result = subprocess.run(
                [cli_cmd, 'pcb', 'drc', '--output', report_path,
                 '--format', 'json', '--severity-all', temp_board_path],
                capture_output=True, text=True, timeout=60,
            )

            # If the CLI reports failure, do not treat the absence of output as a PASS.
            if getattr(result, 'returncode', 0) != 0:
                raise RuntimeError(f"kicad-cli DRC failed (code {result.returncode}): {result.stderr.strip() or result.stdout.strip()}")

            if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
                try:
                    with open(report_path, 'r') as f:
                        drc_data = _json.load(f)
                except _json.JSONDecodeError:
                    # sometimes empty or malformed
                    drc_data = {}

                drc_ran = True

                # Cope with schema differences: violations may be a list or a dict with 'items'
                violations = drc_data.get('violations', [])
                if isinstance(violations, dict):
                    violations = violations.get('items', [])

                unconn = drc_data.get('unconnected_items', [])
                if isinstance(unconn, dict):
                    unconn = unconn.get('items', [])

                for violation in (violations or []):
                    sev = violation.get('severity', 'error')
                    desc = violation.get('description', 'Unknown violation')
                    items = violation.get('items', [])
                    pos_str = ''
                    for item in items:
                        pos = item.get('pos', {})
                        if pos:
                            pos_str = f" at ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f})mm"
                            break
                    entry = f"{desc}{pos_str}"
                    if sev == 'warning':
                        warnings.append(entry)
                    else:
                        errors.append(entry)

                for violation in (unconn or []):
                    if isinstance(violation, dict):
                        desc = violation.get('description', 'Unconnected item')
                    else:
                        desc = str(violation)
                    errors.append(desc)

                # Cleanup
                for f_del in [report_path, temp_board_path]:
                    try:
                        if os.path.exists(f_del): os.unlink(f_del)
                    except Exception:
                        pass

                return (len(errors) == 0), self._format_drc_results(errors, warnings)
            
            # Cleanup if failed
            for f_del in [report_path, temp_board_path]:
                try:
                    if os.path.exists(f_del): os.unlink(f_del)
                except Exception:
                    pass

        except FileNotFoundError:
            pass  # kicad-cli not available
        except Exception as e:
            logger.debug(f"kicad-cli DRC failed: {e}")

        # Strategy 2: pcbnew.WriteDRCReport
        try:
            import pcbnew as _pcbnew
            write_drc = getattr(_pcbnew, 'WriteDRCReport', None)
            if callable(write_drc):
                fd, report_path = tempfile.mkstemp(suffix='.txt')
                import os
                os.close(fd)

                board = _pcbnew.GetBoard()
                rules_path = ''
                try:
                    rules_path = str(Path(board_path).with_suffix('.kicad_dru'))
                    if not os.path.exists(rules_path):
                        rules_path = ''
                except Exception:
                    rules_path = ''

                write_drc(board, rules_path, report_path)

                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        report_text = f.read()
                    errors, warnings = self._parse_drc_text_report(report_text)
                    try:
                        os.unlink(report_path)
                    except Exception:
                        pass
                    drc_ran = True
                    return (len(errors) == 0), self._format_drc_results(errors, warnings)

        except Exception as e:
            logger.debug(f"WriteDRCReport failed: {e}")

        # Strategy 3: Read board markers
        try:
            import pcbnew as _pcbnew
            board = _pcbnew.GetBoard()
            if board:
                markers = []
                try:
                    for m in board.Markers():
                        rc = m.GetRCItem()
                        if rc:
                            desc = str(rc.GetErrorMessage()) if hasattr(rc, 'GetErrorMessage') else str(rc)
                            markers.append(desc)
                except Exception:
                    pass

                if markers:
                    for m in markers:
                        if 'warning' in m.lower():
                            warnings.append(m)
                        else:
                            errors.append(m)
                    drc_ran = True
                    return (len(errors) == 0), self._format_drc_results(errors, warnings)

        except Exception as e:
            logger.debug(f"Board markers read failed: {e}")

        if not drc_ran:
            return False, "DRC could not be run (no kicad-cli output, no WriteDRCReport, and no board markers)."

        return (len(errors) == 0), self._format_drc_results(errors, warnings)

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

    async def _handle_run_erc(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Run a meaningful electrical verification step.

        If a schematic exists, run KiCad schematic ERC via `kicad-cli sch erc`.
        If no schematic exists (PCB-only sessions), run a PCB-native connectivity
        sanity check: report pads that have no net assigned.

        Note: schematic ERC can pass even when PCB DRC fails (shorts/clearance are
        physical layout problems, not schematic ERC violations).
        """
        import subprocess
        import tempfile
        import json as _json
        import os

        # ERC needs a schematic file.  Try to find one from the project.
        sch_path = context.get('schematic_path', '')
        if not sch_path:
            try:
                import pcbnew as _pcbnew
                board = _pcbnew.GetBoard()
                if board:
                    pcb_fn = board.GetFileName() or ''
                    if pcb_fn:
                        from pathlib import Path as _P
                        sch_candidate = _P(pcb_fn).with_suffix('.kicad_sch')
                        if sch_candidate.exists():
                            sch_path = str(sch_candidate)
            except Exception:
                pass

        if not sch_path:
            # PCB-only fallback: surface a real electrical signal instead of a hard fail.
            try:
                import pcbnew as _pcbnew  # type: ignore
                board = context.get('board')
                if board is None:
                    board = _pcbnew.GetBoard()
                if board is None:
                    return False, "No active board found (cannot run electrical checks)"
                return self._run_pcb_electrical_check(board, _pcbnew)
            except Exception as e:
                return False, (
                    "No schematic file found for ERC (.kicad_sch). "
                    "Also failed to run PCB electrical check. "
                    f"Details: {e}"
                )

        errors: List[str] = []
        warnings: List[str] = []

        # Locate kicad-cli
        cli_cmd = 'kicad-cli'
        import platform
        if platform.system() == 'Darwin':
            candidates = [
                '/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli',
                '/Applications/KiCad/kicad.app/Contents/MacOS/kicad-cli',
                'kicad-cli',
            ]
            for c in candidates:
                if os.path.exists(c) and os.access(c, os.X_OK):
                    cli_cmd = c
                    break

        try:
            fd_rep, report_path = tempfile.mkstemp(suffix='.json')
            os.close(fd_rep)

            result = subprocess.run(
                [
                    cli_cmd, 'sch', 'erc',
                    '--output', report_path,
                    '--format', 'json',
                    '--severity-all',
                    '--exit-code-violations',
                    sch_path,
                ],
                capture_output=True, text=True, timeout=60,
            )

            if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
                try:
                    with open(report_path, 'r') as f:
                        erc_data = _json.load(f)
                except _json.JSONDecodeError:
                    erc_data = {}

                sheets = erc_data.get('sheets', [])
                for sheet in sheets:
                    violations = sheet.get('violations', [])
                    if isinstance(violations, dict):
                        violations = violations.get('items', [])
                    for v in (violations or []):
                        sev = v.get('severity', 'error')
                        desc = v.get('description', 'Unknown ERC violation')
                        items = v.get('items', [])
                        pos_str = ''
                        for item in items:
                            pos = item.get('pos', {})
                            if pos:
                                pos_str = f" at ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f})"
                                break
                        entry = f"{desc}{pos_str}"
                        if sev == 'warning':
                            warnings.append(entry)
                        else:
                            errors.append(entry)

                try:
                    os.unlink(report_path)
                except Exception:
                    pass

                sch_result_text = self._format_erc_results(errors, warnings)
                sch_ok = len(errors) == 0

                # ALWAYS also run PCB connectivity check and append it
                pcb_suffix = ""
                try:
                    import pcbnew as _pcbnew2
                    _board2 = context.get('board')
                    if _board2 is None:
                        _board2 = _pcbnew2.GetBoard()
                    if _board2 is not None:
                        pcb_ok, pcb_text = self._run_pcb_electrical_check(_board2, _pcbnew2)
                        pcb_suffix = "\n\n--- PCB Connectivity Check ---\n" + pcb_text
                        if not pcb_ok:
                            sch_ok = False
                except Exception:
                    pass

                return sch_ok, sch_result_text + pcb_suffix

            try:
                os.unlink(report_path)
            except Exception:
                pass

            # If no report was produced, be explicit: avoid the appearance of a fake always-0 check.
            if result.returncode == 0:
                extra = result.stderr.strip()
                if extra:
                    extra = "\n(kicad-cli stderr: " + extra + ")"
                return True, (
                    "ERC_STATUS: PASS\n"
                    "ERC found no violations, but KiCad emitted no JSON report output. "
                    "(This can happen on clean designs.)" + extra
                )

            return False, (
                "ERC_STATUS: FAIL\n"
                "ERC failed to produce a JSON report. "
                f"kicad-cli exit={result.returncode}. "
                f"stderr: {result.stderr.strip()}"
            )

        except FileNotFoundError:
            return False, ("kicad-cli not found. ERC requires KiCad 7+ with kicad-cli. "
                           "On macOS, check /Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli")
        except subprocess.TimeoutExpired:
            return False, "ERC timed out after 60 seconds."
        except Exception as e:
            logger.exception("ERC failed")
            return False, f"ERC failed: {e}"

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

    async def _handle_add_via(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Place a via at specified coordinates."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}
        location = params.get('location')

        if location in (None, ""):
            return False, "No location specified for via"

        parsed = self._parse_location_mm(location)
        if parsed is None:
            return False, f"Could not parse location '{location}'. Use 'x,y' in mm."

        x_mm, y_mm = parsed

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        from_mm = getattr(pcbnew, 'FromMM', None)
        if not callable(from_mm):
            return False, "pcbnew.FromMM not available"

        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(int(from_mm(x_mm)), int(from_mm(y_mm))))

        # Size
        via_size = float(params.get('size_mm', 0.8))
        via_drill = float(params.get('drill_mm', 0.4))
        via.SetWidth(int(from_mm(via_size)))
        via.SetDrill(int(from_mm(via_drill)))

        # Net
        net_name = str(params.get('net', '') or '').strip()
        if net_name:
            net = board.FindNet(net_name)
            if net:
                via.SetNet(net)

        # Via type
        via_type = getattr(pcbnew, 'VIATYPE_THROUGH', None)
        if via_type is not None:
            via.SetViaType(via_type)

        board.Add(via)
        return True, f"Via placed at ({x_mm}, {y_mm}) mm"

    async def _handle_define_board_outline(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Draw a rectangular Edge.Cuts outline, ALWAYS centered on the page."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}
        width = float(params.get('width', 0) or 0) or 100.0
        height = float(params.get('height', 0) or 0) or 80.0

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                return False, "No active board found"
        if board is None:
            return False, "No active board found"

        def mm2iu(v: float) -> int:
            try:
                return int(pcbnew.FromMM(float(v)))
            except Exception:
                return int(float(v) * 1e6)

        # Page size (A4 = 297x210 default)
        pw_mm, ph_mm = 297.0, 210.0
        try:
            page = board.GetPageSettings()
            gw = getattr(page, 'GetWidthMM', None)
            gh = getattr(page, 'GetHeightMM', None)
            if callable(gw) and callable(gh):
                w, h = float(gw()), float(gh())
                if w > 0 and h > 0:
                    pw_mm, ph_mm = w, h
        except Exception:
            pass

        # CENTER on the page — ignore any caller-supplied origin.
        # Special-case A4 to match user expectation of ~ (150,100) mm.
        cx_mm, cy_mm = pw_mm / 2.0, ph_mm / 2.0
        try:
            if abs(pw_mm - 297.0) < 0.25 and abs(ph_mm - 210.0) < 0.25:
                cx_mm, cy_mm = 150.0, 100.0
        except Exception:
            pass

        ox_mm = cx_mm - (width / 2.0)
        oy_mm = cy_mm - (height / 2.0)

        # Force logging without except block to ensure we see values
        logger.info(
            "DEFINE_BOARD_OUTLINE: page=(%.2f,%.2f)mm size=(%.2f,%.2f)mm center=(%.2f,%.2f)mm origin=(%.2f,%.2f)mm",
            float(pw_mm), float(ph_mm), float(width), float(height),
            float(cx_mm), float(cy_mm), float(ox_mm), float(oy_mm),
        )

        edge_cuts = getattr(pcbnew, 'Edge_Cuts', 44)
        try:
            edge_cuts = board.GetLayerID('Edge.Cuts')
        except Exception:
            pass

        # Replace existing Edge.Cuts outline instead of duplicating.
        # This avoids the "move outline" workflow leaving the old rectangle behind.
        removed = 0
        try:
            drawings = getattr(board, 'GetDrawings', None)
            if callable(drawings):
                for d in list(drawings()):
                    try:
                        gl = getattr(d, 'GetLayer', None)
                        if callable(gl) and int(gl()) == int(edge_cuts):
                            try:
                                if hasattr(pcbnew, 'BOARD_COMMIT'):
                                    # Even if commit isn't used elsewhere here, board.Remove is fine.
                                    pass
                            except Exception:
                                pass
                            try:
                                board.Remove(d)
                                removed += 1
                            except Exception:
                                try:
                                    board.Delete(d)
                                    removed += 1
                                except Exception:
                                    pass
                    except Exception:
                        continue
        except Exception:
            pass

        try:
            if removed:
                logger.info("DEFINE_BOARD_OUTLINE: removed %d existing Edge.Cuts drawing(s)", removed)
        except Exception:
            pass

        def mk(x_mm, y_mm):
            ix, iy = mm2iu(x_mm), mm2iu(y_mm)
            for ctor in ('VECTOR2I', 'wxPoint'):
                try:
                    return getattr(pcbnew, ctor)(ix, iy)
                except Exception:
                    continue
            return None

        corners = [
            (ox_mm, oy_mm),
            (ox_mm + width, oy_mm),
            (ox_mm + width, oy_mm + height),
            (ox_mm, oy_mm + height),
        ]

        for i in range(4):
            seg = pcbnew.PCB_SHAPE(board)
            # KiCad 9: SHAPE_T_SEGMENT; KiCad 7/8: S_SEGMENT or raw 0
            for enum_name in ('SHAPE_T_SEGMENT', 'S_SEGMENT'):
                try:
                    seg.SetShape(getattr(pcbnew, enum_name))
                    break
                except Exception:
                    continue
            else:
                try:
                    seg.SetShape(0)
                except Exception:
                    pass
            p1 = mk(*corners[i])
            p2 = mk(*corners[(i + 1) % 4])
            if p1:
                seg.SetStart(p1)
            if p2:
                seg.SetEnd(p2)
            seg.SetLayer(edge_cuts)
            try:
                sd = getattr(seg, 'SetDescription', None)
                if callable(sd):
                    sd('VibeCAD board outline')
            except Exception:
                pass
            try:
                seg.SetWidth(mm2iu(0.1))
            except Exception:
                try:
                    seg.SetStroke(pcbnew.STROKE_PARAMS(mm2iu(0.1)))
                except Exception:
                    pass
            try:
                board.Add(seg)
            except Exception:
                pass

        # Aggressively refresh KiCad's PCB view using every known method.
        for refresh_fn in (
            lambda: pcbnew.Refresh(),
            lambda: board.GetDesignSettings(),  # force internal refresh
            lambda: pcbnew.UpdateUserInterface(),
        ):
            try:
                refresh_fn()
            except Exception:
                pass

        logger.info(
            "DEFINE_BOARD_OUTLINE DONE: drew 4 segments, center=(%.1f,%.1f) origin=(%.1f,%.1f)",
            cx_mm, cy_mm, ox_mm, oy_mm,
        )

        return True, f"Board outline: {width}x{height} mm centered at ({cx_mm:.1f}, {cy_mm:.1f}) mm"

    async def _handle_add_mounting_hole(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Add a mounting hole by searching for MountingHole footprint and placing it."""
        params = action.parameters or {}
        size = str(params.get('size', '3.2')).strip()
        location = str(params.get('location', '') or '').strip()

        # Delegate to ADD_COMPONENT with a mounting hole query
        mount_action = DesignAction(
            action_type=DesignActionType.ADD_COMPONENT,
            description=f"Place {size}mm mounting hole",
            parameters={
                'query': f'MountingHole {size}mm',
                'location': location,
                'ref_prefix': 'H',
            },
            requires_approval=False,
        )
        mount_action.approved = True
        return await self._handle_add_component(mount_action, context)

    async def _handle_align_components(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Align multiple components by X or Y coordinate."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}
        refs_str = str(params.get('refs', '') or '').strip()
        direction = str(params.get('direction', 'horizontally') or '').strip().lower()

        if not refs_str:
            return False, "No component references specified"

        refs = [r.strip().upper() for r in re.split(r'[,;\s]+', refs_str) if r.strip()]
        if len(refs) < 2:
            return False, "Need at least 2 components to align"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        footprints = {}
        for fp in board.GetFootprints():
            ref = str(fp.GetReference()).upper()
            if ref in refs:
                footprints[ref] = fp

        if len(footprints) < 2:
            return False, f"Found only {len(footprints)} of {len(refs)} components"

        positions = [(ref, fp.GetPosition()) for ref, fp in footprints.items()]

        if direction in ('horizontally', 'left', 'right', 'center'):
            avg_y = sum(p.y for _, p in positions) // len(positions)
            for ref, fp in footprints.items():
                pos = fp.GetPosition()
                fp.SetPosition(pcbnew.VECTOR2I(pos.x, avg_y))
        else:  # vertically, top, bottom
            avg_x = sum(p.x for _, p in positions) // len(positions)
            for ref, fp in footprints.items():
                pos = fp.GetPosition()
                fp.SetPosition(pcbnew.VECTOR2I(avg_x, pos.y))

        return True, f"Aligned {len(footprints)} components {direction}"

    async def _handle_add_text(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Add text to the PCB (silkscreen by default)."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}
        text_content = str(params.get('text', '') or '').strip()
        if not text_content:
            return False, "No text content specified"

        location = str(params.get('location', '') or '').strip()
        layer_name = str(params.get('layer', 'F.SilkS') or 'F.SilkS').strip()

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        from_mm = getattr(pcbnew, 'FromMM', None)
        if not callable(from_mm):
            return False, "pcbnew.FromMM not available"

        pcb_text = pcbnew.PCB_TEXT(board)
        pcb_text.SetText(text_content)

        # Position
        x_mm, y_mm = 50.0, 50.0
        if location:
            loc_s = location.lower().replace('mm', '').strip()
            if loc_s.startswith('(') and loc_s.endswith(')'):
                loc_s = loc_s[1:-1].strip()
            m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*(?:,|\s+)\s*(\d+(?:\.\d+)?)\s*$', loc_s)
            if m:
                x_mm, y_mm = float(m.group(1)), float(m.group(2))

        pcb_text.SetPosition(pcbnew.VECTOR2I(int(from_mm(x_mm)), int(from_mm(y_mm))))

        # Layer
        try:
            layer_id = board.GetLayerID(layer_name)
            pcb_text.SetLayer(layer_id)
        except Exception:
            pcb_text.SetLayer(getattr(pcbnew, 'F_SilkS', 37))

        # Font size
        try:
            size = int(from_mm(1.0))
            pcb_text.SetTextSize(pcbnew.VECTOR2I(size, size))
        except Exception:
            pass

        board.Add(pcb_text)
        return True, f"Text '{text_content}' added at ({x_mm}, {y_mm}) mm on {layer_name}"

    async def _handle_add_polygon(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Add a filled copper zone (polygon pour) on a specified layer."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}
        layer_name = str(params.get('layer', 'F.Cu') or 'F.Cu').strip()
        net_name = str(params.get('net', 'GND') or 'GND').strip()
        x = float(params.get('x', 0) or 0)
        y = float(params.get('y', 0) or 0)
        width = float(params.get('width', 0) or 0)
        height = float(params.get('height', 0) or 0)

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        from_mm = getattr(pcbnew, 'FromMM', None)
        if not callable(from_mm):
            return False, "pcbnew.FromMM not available"

        # Auto-size to board outline if not specified
        if width <= 0 or height <= 0:
            try:
                bbox = board.GetBoardEdgesBoundingBox()
                to_mm = getattr(pcbnew, 'ToMM', None)
                if callable(to_mm) and bbox.GetWidth() > 0:
                    x = to_mm(bbox.GetX())
                    y = to_mm(bbox.GetY())
                    width = to_mm(bbox.GetWidth())
                    height = to_mm(bbox.GetHeight())
            except Exception:
                pass
            if width <= 0 or height <= 0:
                return False, "No zone dimensions specified and no board outline to auto-size from"

        try:
            layer_id = board.GetLayerID(layer_name)
        except Exception:
            layer_id = getattr(pcbnew, 'F_Cu', 0)

        zone = pcbnew.ZONE(board)
        zone.SetLayer(layer_id)

        net = board.FindNet(net_name)
        if net:
            zone.SetNet(net)

        # Define zone outline
        outline = zone.Outline()
        outline.NewOutline()
        corners = [
            (x, y), (x + width, y),
            (x + width, y + height), (x, y + height),
        ]
        for cx, cy in corners:
            outline.Append(int(from_mm(cx)), int(from_mm(cy)))

        zone.SetIsFilled(True)
        board.Add(zone)

        return True, f"Copper zone added on {layer_name} for net {net_name} ({width}x{height} mm)"

    async def _handle_autoroute(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Autoroute the board via Freerouting or manhattan fallback.

        Before routing, automatically arrange overlapping components so the
        autorouter has a realistic chance of completing successfully.
        """
        try:
            from .autorouter import autoroute
        except ImportError:
            return False, "Autorouter module not available"

        # ── Comprehensive pre-routing readiness checks ──────────────────
        try:
            import pcbnew as _pcbnew  # type: ignore
            b = context.get('board')
            if b is None:
                b = _pcbnew.GetBoard()
            if b is not None:
                # 1. Board outline must exist
                has_outline = False
                outline_bb = None
                try:
                    outline_bb = b.GetBoardEdgesBoundingBox()
                    ow, oh = int(outline_bb.GetWidth()), int(outline_bb.GetHeight())
                    if ow > 5000000 and oh > 5000000:  # > 5mm in nm
                        has_outline = True
                except Exception:
                    pass
                if not has_outline:
                    return False, (
                        "❌ Cannot route: no board outline defined. "
                        "Use DEFINE_BOARD_OUTLINE first, then ensure all components "
                        "are placed inside it before routing."
                    )

                # 2. Check net assignment coverage
                total_pads = 0
                pads_with_net = 0
                for fp in b.GetFootprints():
                    try:
                        pads_iter = fp.Pads()
                    except Exception:
                        pads_iter = []
                    for pad in pads_iter:
                        total_pads += 1
                        try:
                            if int(pad.GetNetCode()) > 0:
                                pads_with_net += 1
                        except Exception:
                            continue

                if pads_with_net == 0:
                    return False, (
                        "❌ No routable nets found because pads have no net assignments. "
                        "Run KiCad: Tools → Update PCB from Schematic (imports netlist), "
                        "or use ASSIGN_NETS to explicitly assign nets to pads, then retry AUTOROUTE_BOARD."
                    )

                net_coverage = pads_with_net / max(total_pads, 1)
                if net_coverage < 0.15:
                    return False, (
                        f"❌ Only {pads_with_net}/{total_pads} pads ({net_coverage:.0%}) have net assignments. "
                        "This is too few for meaningful routing. Use DEFINE_NET / ASSIGN_NETS "
                        "to assign nets to at least the critical connections (power, ground, signals) "
                        "before routing."
                    )

                # 3. Check how many components are outside the board outline
                if has_outline and outline_bb is not None:
                    ox_min = int(outline_bb.GetX())
                    oy_min = int(outline_bb.GetY())
                    ox_max = ox_min + int(outline_bb.GetWidth())
                    oy_max = oy_min + int(outline_bb.GetHeight())
                    fp_list = list(b.GetFootprints())
                    outside_count = 0
                    for fp in fp_list:
                        try:
                            pos = fp.GetPosition()
                            px, py = int(pos.x), int(pos.y)
                            if px < ox_min or px > ox_max or py < oy_min or py > oy_max:
                                outside_count += 1
                        except Exception:
                            continue
                    if fp_list and outside_count > len(fp_list) * 0.3:
                        return False, (
                            f"❌ {outside_count}/{len(fp_list)} components are outside the board outline. "
                            "Use MOVE_COMPONENT to place all components inside the board outline "
                            "before routing. Components outside the outline cannot be routed."
                        )
        except Exception:
            # Non-fatal; autoroute() will still run and produce its own message.
            pass

        # ── Pre-routing arrangement: spread overlapping components ──────
        arrange_msg = ""
        try:
            import pcbnew as _pcbnew
            b = context.get('board')
            if b is None:
                b = _pcbnew.GetBoard()
            if b is not None:
                moved = self._arrange_overlapping_components(b, _pcbnew)
                if moved:
                    arrange_msg = f"Arranged {moved} overlapping component(s) before routing. "
                    try:
                        _pcbnew.Refresh()
                    except Exception:
                        pass
        except Exception:
            pass

        board_path = context.get('board_path', '')
        if not board_path:
            try:
                import pcbnew as _pcbnew
                board = _pcbnew.GetBoard()
                if board:
                    board_path = board.GetFileName() or ''
            except Exception:
                pass

        if not board_path:
            # Save to temp file
            try:
                import pcbnew as _pcbnew
                import tempfile
                import os
                board = _pcbnew.GetBoard()
                if board:
                    fd, tmp = tempfile.mkstemp(suffix='.kicad_pcb')
                    os.close(fd)
                    _pcbnew.SaveBoard(tmp, board)
                    board_path = tmp
            except Exception:
                return False, "No board available for autorouting"

        result = autoroute(board_path)
        if result.success:
            msg = arrange_msg + f"Autorouting complete ({result.method}): {result.tracks_added} tracks"
            if result.vias_added:
                msg += f", {result.vias_added} vias"
            return True, msg
        return False, arrange_msg + result.message

    async def _handle_set_layer_count(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Set the number of copper layers on the board."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}
        count = int(params.get('count', 2) or 2)
        if count not in (1, 2, 4, 6, 8):
            return False, f"Invalid layer count {count}. Use 1, 2, 4, 6, or 8."

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        old_count = board.GetCopperLayerCount()
        board.SetCopperLayerCount(count)

        return True, f"Copper layer count changed from {old_count} to {count}"

    async def _handle_delete_tracks(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Delete all tracks and vias from the board."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        tracks = list(board.GetTracks())
        count = len(tracks)
        
        for track in tracks:
            board.Remove(track)
            
        return True, f"Deleted {count} tracks/vias"

    async def _handle_delete_component(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Delete a component (footprint) from the board."""
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"
            
        params = action.parameters or {}
        ref = str(params.get('ref', '') or '').strip().upper()
        
        if not ref:
            return False, "Missing component reference"

        board = context.get('board')
        if board is None:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found"

        full_ref = ref
        # Check if ref has empty annotation '?' and try to infer
        # This is a bit risky but standard KiCad behavior is usually unique refs.
        
        footprint = None
        # Try exact match first
        footprint = board.FindFootprintByReference(full_ref)
        
        # Fallback search if exact match fails or FindFootprintByReference is unavailable
        if footprint is None:
            for fp in board.GetFootprints():
                if str(fp.GetReference()).upper() == full_ref:
                    footprint = fp
                    break
        
        if footprint is None:
            return False, f"Component {full_ref} not found"
            
        board.Remove(footprint)
        return True, f"Deleted component {full_ref}"
    
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

    async def _handle_search_web(self, action: 'DesignAction', context: Dict) -> Tuple[bool, str]:
        """Search online component databases for specs, pricing, datasheets.

        Parameters:
            query: search string (MPN, description, etc.)
        """
        params = action.parameters or {}
        query = (params.get('query') or params.get('mpn') or params.get('part_name') or '').strip()
        if not query:
            # Fall back to the same extraction we use for SEARCH_PART.
            query = self._extract_search_query(action)
        if not query:
            return False, "Missing 'query' parameter for web search"

        try:
            from .component_search import ComponentWebSearch
            searcher = ComponentWebSearch()
            results = searcher.search(query, limit=5)
            if not results:
                return True, f"No results found for '{query}'. Try a different search term or MPN."
            text_parts = [f"## Web Search Results for '{query}'\n"]
            for r in results:
                text_parts.append(r.to_text())
                text_parts.append("")
            return True, "\n".join(text_parts)
        except Exception as e:
            logger.exception("Web search failed")
            return False, f"Web search failed: {e}"

    async def _handle_lookup_datasheet(self, action: 'DesignAction', context: Dict) -> Tuple[bool, str]:
        """Look up a datasheet URL for a given MPN.

        Parameters:
            mpn: manufacturer part number
        """
        params = action.parameters or {}
        mpn = (params.get('mpn') or params.get('query') or params.get('part_name') or '').strip()
        if not mpn:
            mpn = self._extract_search_query(action)
        if not mpn:
            return False, "Missing 'mpn' parameter for datasheet lookup"

        try:
            from .component_search import ComponentWebSearch
            searcher = ComponentWebSearch()
            url = searcher.get_datasheet_url(mpn)
            if url:
                return True, f"Datasheet for **{mpn}**: {url}"
            # Fallback: do a full search and see if any result has a datasheet
            results = searcher.search(mpn, limit=3)
            for r in results:
                if r.datasheet_url:
                    return True, f"Datasheet for **{r.mpn}**: {r.datasheet_url}"
            return True, f"No datasheet found for '{mpn}'. Try searching LCSC or Mouser directly."
        except Exception as e:
            logger.exception("Datasheet lookup failed")
            return False, f"Datasheet lookup failed: {e}"
