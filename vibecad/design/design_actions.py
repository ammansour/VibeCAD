"""Shared action/data model types for VibeCAD design workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple

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


def sanitize_llm_json_text(text: str) -> str:
    """Best-effort sanitation for JSON emitted by LLMs."""
    if not text:
        return ""
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "\u2212": "-",
        "\u2014": "-",
        "\u2013": "-",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _extract_json_object(raw_text: str) -> str:
    """Extract the outermost JSON object from *raw_text* (best effort)."""
    s = (raw_text or "").strip()
    if not s:
        return ""
    start = s.find("{")
    end = s.rfind("}") + 1
    if start < 0 or end <= start:
        return ""
    return s[start:end]


def normalize_action_parameters(action_type: 'DesignActionType', parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize action parameter keys to the canonical schema.

    Goal: every action has one canonical key per concept (e.g. component reference is
    always under 'ref', never 'id' / 'reference' / etc).
    """
    params: Dict[str, Any] = dict(parameters or {})

    def _pop_first(keys: tuple) -> Any:
        for k in keys:
            if k in params and params.get(k) not in (None, ""):
                val = params.get(k)
                if k != keys[0]:
                    params.pop(k, None)
                return val
        return None

    def _normalize_ref() -> None:
        ref_val = _pop_first(("ref", "id", "reference", "designator", "component"))
        if ref_val is not None:
            ref = str(ref_val).strip()
            if ref:
                params["ref"] = ref
        # Ensure aliases are removed if present.
        for k in ("id", "reference", "designator", "component"):
            params.pop(k, None)

    def _normalize_location() -> None:
        # Promote x/y into a single location object.
        if params.get("location") in (None, ""):
            if "x" in params and "y" in params:
                params["location"] = {"x": params.pop("x"), "y": params.pop("y")}
            elif "x_mm" in params and "y_mm" in params:
                params["location"] = {"x": params.pop("x_mm"), "y": params.pop("y_mm")}
            elif "pos" in params:
                params["location"] = params.pop("pos")
            elif "at" in params:
                params["location"] = params.pop("at")

        loc = params.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) == 2:
            params["location"] = {"x": loc[0], "y": loc[1]}
        elif isinstance(loc, str):
            m = re.match(r"^\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?\s*$", loc.strip())
            if m:
                params["location"] = {"x": float(m.group(1)), "y": float(m.group(2))}

        # If we now have a dict, try to coerce numeric x/y.
        loc2 = params.get("location")
        if isinstance(loc2, dict) and "x" in loc2 and "y" in loc2:
            try:
                params["location"] = {"x": float(loc2["x"]), "y": float(loc2["y"])}
            except Exception:
                pass

    def _normalize_refs_list() -> None:
        # Canonical: "refs" is a list of reference strings (["U1","J1",...]).
        refs = params.get("refs")
        if isinstance(refs, str):
            items = [r.strip().upper() for r in re.split(r"[,;\s]+", refs) if r.strip()]
            params["refs"] = items
            return
        if isinstance(refs, list):
            out: List[str] = []
            for r in refs:
                s = str(r or "").strip()
                if not s:
                    continue
                out.append(s.upper())
            params["refs"] = out
            return

    def _normalize_assign_nets() -> None:
        # Canonical: {"assignments":[{"net":"GND","ref":"U1","pad":"3"}, ...]}
        assigns = params.get("assignments")
        if isinstance(assigns, list) and assigns:
            return

        # Common grouped encoding:
        # {"net_name":"GND","pads":["U1:3","U1:5", ...]}
        net = params.get("net") or params.get("net_name") or params.get("netName") or params.get("name")
        pads = params.get("pads") or params.get("pad_refs") or params.get("padRefs")
        if not (isinstance(net, str) and net.strip() and isinstance(pads, list) and pads):
            return

        net_s = str(net).strip()
        normalized: List[Dict[str, Any]] = []
        for p in pads:
            s = str(p or "").strip()
            if not s:
                continue
            m = re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", s)
            if not m:
                continue
            normalized.append({"net": net_s, "ref": m.group(1).upper(), "pad": m.group(2)})

        if normalized:
            params.pop("net_name", None)
            params.pop("netName", None)
            params.pop("pads", None)
            params["assignments"] = normalized

    def _normalize_board_outline() -> None:
        width_val = _pop_first(("width", "width_mm", "w"))
        if width_val is not None:
            params["width"] = width_val
        height_val = _pop_first(("height", "height_mm", "h"))
        if height_val is not None:
            params["height"] = height_val
        shape_val = _pop_first(("shape", "outline_shape", "shape_type"))
        if shape_val is not None:
            shape_s = str(shape_val).strip()
            if shape_s:
                params["shape"] = shape_s
        corner_val = _pop_first(("corner_radius", "corner_radius_mm", "radius", "fillet_radius", "r"))
        if corner_val is not None:
            params["corner_radius"] = corner_val
        diameter_val = _pop_first(("diameter", "diameter_mm"))
        if diameter_val is not None:
            params["diameter"] = diameter_val

    if action_type in (
        DesignActionType.MOVE_COMPONENT,
        DesignActionType.ROTATE_COMPONENT,
        DesignActionType.DELETE_COMPONENT,
    ):
        _normalize_ref()

    if action_type in (DesignActionType.MOVE_COMPONENT, DesignActionType.ADD_COMPONENT):
        _normalize_location()

    if action_type == DesignActionType.ALIGN_COMPONENTS:
        _normalize_refs_list()

    if action_type == DesignActionType.ASSIGN_NETS:
        _normalize_assign_nets()

    if action_type == DesignActionType.DEFINE_BOARD_OUTLINE:
        _normalize_board_outline()

    return params


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
