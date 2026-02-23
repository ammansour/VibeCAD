"""
PlacementAgent — smart component placement with spatial optimization.

Handles ADD_COMPONENT, MOVE_COMPONENT, ROTATE_COMPONENT, ALIGN_COMPONENTS,
and DEFINE_BOARD_OUTLINE.

Key improvement over the monolithic agent: after proposing placements the
``optimize_layout()`` method runs a lightweight force-directed pass that
pushes overlapping parts apart while keeping related components close
(e.g. decoupling caps near their ICs).  This eliminates the ongoing
"WARNING: … overlaps with …" churn that wastes agent loop iterations.

Spatial strategies:
  1. **Functional grouping** — ICs in centre, connectors at edges, passives
     near their parent IC, mounting holes in corners.
  2. **Force-directed de-overlap** — repulsive forces between all pairs that
     overlap (or violate clearance), attractive forces between net-connected
     pairs, resolved iteratively.
  3. **Board-outline clamping** — every position is kept ≥ margin inside the
     outline at all times.
  4. **Grid snap** — final positions are snapped to a configurable grid so the
     layout looks intentional, not random.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .base import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)

# ── Spatial helpers ─────────────────────────────────────────────

@dataclass
class _BBox:
    """Axis-aligned bounding box in mm."""
    cx: float  # centre x
    cy: float  # centre y
    hw: float  # half-width
    hh: float  # half-height

    @property
    def left(self) -> float:
        return self.cx - self.hw

    @property
    def right(self) -> float:
        return self.cx + self.hw

    @property
    def top(self) -> float:
        return self.cy - self.hh

    @property
    def bottom(self) -> float:
        return self.cy + self.hh

    def overlaps(self, other: "_BBox", clearance: float = 2.0) -> bool:
        """True if this bbox overlaps *other* with *clearance* mm margin."""
        return not (
            self.right + clearance <= other.left
            or other.right + clearance <= self.left
            or self.bottom + clearance <= other.top
            or other.bottom + clearance <= self.top
        )

    def overlap_vector(self, other: "_BBox", clearance: float = 2.0) -> Tuple[float, float]:
        """Return (dx, dy) to push *self* away from *other* to resolve overlap.

        Returns (0, 0) if no overlap.
        """
        if not self.overlaps(other, clearance):
            return (0.0, 0.0)

        # Minimum translation vector (MTV) — axis-aligned.
        dx_right = (other.right + clearance) - self.left
        dx_left = self.right - (other.left - clearance)
        dy_down = (other.bottom + clearance) - self.top
        dy_up = self.bottom - (other.top - clearance)

        # Pick the axis with the *smallest* penetration.
        pen_x = min(abs(dx_right), abs(dx_left))
        pen_y = min(abs(dy_down), abs(dy_up))

        if pen_x <= pen_y:
            return (-dx_left if dx_left < dx_right else dx_right, 0.0)
        else:
            return (0.0, -dy_up if dy_up < dy_down else dy_down)


@dataclass
class _PlacedComponent:
    """Lightweight record used during layout optimization."""
    ref: str
    bbox: _BBox
    category: str = "generic"  # "ic", "passive", "connector", "mounting", "generic"
    nets: Set[str] = field(default_factory=set)
    fixed: bool = False  # if True, don't move (e.g. user-locked)

    # Velocity for force-directed iterations.
    vx: float = 0.0
    vy: float = 0.0


# ── Constants ───────────────────────────────────────────────────

# Minimum clearance in mm between courtyard bounding boxes.
MIN_CLEARANCE_MM = 2.0

# Force-directed tuning.
REPULSION_STRENGTH = 8.0
ATTRACTION_STRENGTH = 0.3
DAMPING = 0.6
MAX_FORCE = 15.0   # mm per iteration cap
ITERATIONS = 60

# Grid snap (mm).
GRID_SNAP_MM = 1.27  # 50 mil — standard KiCad grid


def _snap(val: float, grid: float = GRID_SNAP_MM) -> float:
    return round(val / grid) * grid


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _classify_ref(ref: str, value: str = "") -> str:
    """Heuristic category from reference designator / value."""
    r = ref.upper()
    v = value.lower()
    if r.startswith("U") or "mcu" in v or "ic" in v or "stm32" in v or "atmega" in v:
        return "ic"
    if r.startswith(("R", "C", "L")) and len(r) <= 4:
        return "passive"
    if r.startswith("J") or "connector" in v or "usb" in v or "header" in v:
        return "connector"
    if r.startswith("H") or "mount" in v:
        return "mounting"
    if r.startswith("D") or "led" in v or "diode" in v:
        return "passive"
    if r.startswith("Q") or "mosfet" in v or "transistor" in v:
        return "passive"
    if r.startswith("Y") or "crystal" in v or "oscillator" in v:
        return "passive"
    return "generic"


# ── PlacementAgent ──────────────────────────────────────────────

class PlacementAgent(SubAgent):
    NAME = "placement"

    SYSTEM_PROMPT = (
        "You are the Placement sub-agent.\n"
        "Only propose: ADD_COMPONENT, MOVE_COMPONENT, ROTATE_COMPONENT, ALIGN_COMPONENTS, DEFINE_BOARD_OUTLINE, ADD_MOUNTING_HOLE.\n"
        "Never propose routing/net-assignment/verification actions.\n"
        "ADD_COMPONENT parameters.query must be a concrete part/footprint identifier, not an instruction.\n"
        "For from-scratch goals, avoid prebuilt module/shield footprints unless the user explicitly requests a module/shield.\n"
        "Keep parts inside board bounds with at least ~2 mm clearance.\n"
        "All coordinates are absolute mm in KiCad's board coordinate system. Do not assume the outline starts at (0,0);\n"
        "use the provided board origin/center and existing component coordinates.\n"
        "Parameter schema (use exactly these keys; do not use aliases like 'id'):\n"
        "- MOVE_COMPONENT.parameters = {\"ref\":\"U1\",\"location\":{\"x\":10.0,\"y\":20.0}} (mm)\n"
        "- ROTATE_COMPONENT.parameters = {\"ref\":\"U1\",\"angle\":90} (degrees)\n"
        "- ALIGN_COMPONENTS.parameters = {\"refs\":[\"U1\",\"J1\"],\"direction\":\"horizontal\"}\n"
        "- DELETE_COMPONENT.parameters = {\"ref\":\"U1\"}\n"
        "Return JSON array only.\n"
    )

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        from ..design_agent import DesignActionType
        self.HANDLED_ACTION_TYPES = frozenset({
            DesignActionType.ADD_COMPONENT,
            DesignActionType.MOVE_COMPONENT,
            DesignActionType.ROTATE_COMPONENT,
            DesignActionType.ALIGN_COMPONENTS,
            DesignActionType.DEFINE_BOARD_OUTLINE,
            DesignActionType.ADD_MOUNTING_HOLE,
            DesignActionType.DELETE_COMPONENT,
        })

    # ── plan ────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        raw = self._llm_chat(self._build_prompt(goal, context, board_snapshot))
        actions = self._parse_actions(raw, board_snapshot=board_snapshot)
        if actions:
            return SubAgentResult(
                message=f"Proposing {len(actions)} placement action(s).",
                actions=actions,
                confidence=0.85,
                thinking=f"LLM proposed {len(actions)} placement actions",
            )
        return SubAgentResult(
            message="No placement actions proposed.",
            confidence=0.3,
            phase_complete=False,
            thinking="LLM proposed no placement actions",
        )

    # ── Smart layout optimization ───────────────────────────────

    @staticmethod
    def optimize_layout(
        components: List[Dict[str, Any]],
        board_width_mm: float,
        board_height_mm: float,
        board_origin_x_mm: float = 0.0,
        board_origin_y_mm: float = 0.0,
        net_connections: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run force-directed layout optimization and return new positions.

        Parameters
        ----------
        components : list of dicts
            Each dict must have:
              ``ref``, ``x`` (centre, mm), ``y`` (centre, mm),
              ``width`` (mm), ``height`` (mm).
            Optional: ``value``, ``nets`` (set of net names), ``fixed``.
        board_width_mm, board_height_mm : float
            Board outline dimensions.
        board_origin_x_mm, board_origin_y_mm : float
            Top-left corner of the board outline.
        net_connections : dict  (net_name → [ref, ref, ...])
            Which components share each net.  Used for attraction force.

        Returns
        -------
        list of dicts  — same components with updated ``x``, ``y`` fields.
        """
        if not components:
            return components

        net_connections = net_connections or {}

        # ── 1. Build internal representation ────────────────────
        placed: List[_PlacedComponent] = []
        for c in components:
            ref = str(c.get("ref", ""))
            w = float(c.get("width", 10.0))
            h = float(c.get("height", 10.0))
            bbox = _BBox(
                cx=float(c.get("x", 0.0)),
                cy=float(c.get("y", 0.0)),
                hw=w / 2.0,
                hh=h / 2.0,
            )
            cat = _classify_ref(ref, str(c.get("value", "")))
            nets: Set[str] = set(c.get("nets", []))
            placed.append(_PlacedComponent(
                ref=ref,
                bbox=bbox,
                category=cat,
                nets=nets,
                fixed=bool(c.get("fixed", False)),
            ))

        # ── 2. Build adjacency from net_connections ─────────────
        # ref → index
        ref_idx = {p.ref: i for i, p in enumerate(placed)}
        # pairs (i, j) that share a net → should attract
        attracted_pairs: Set[Tuple[int, int]] = set()
        for net_name, refs in net_connections.items():
            idxs = [ref_idx[r] for r in refs if r in ref_idx]
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    attracted_pairs.add((min(idxs[a], idxs[b]), max(idxs[a], idxs[b])))

        # Board bounds (with margin for component half-size).
        bx_min = board_origin_x_mm
        by_min = board_origin_y_mm
        bx_max = board_origin_x_mm + board_width_mm
        by_max = board_origin_y_mm + board_height_mm

        # ── 3. Initial category-based seeding ───────────────────
        _seed_positions(placed, bx_min, by_min, bx_max, by_max)

        # ── 4. Force-directed iterations ────────────────────────
        n = len(placed)
        for iteration in range(ITERATIONS):
            forces = [[0.0, 0.0] for _ in range(n)]

            # 4a. Repulsion — push apart overlapping (or too-close) pairs.
            for i in range(n):
                if placed[i].fixed:
                    continue
                for j in range(i + 1, n):
                    if placed[j].fixed and placed[i].fixed:
                        continue
                    bi = placed[i].bbox
                    bj = placed[j].bbox
                    if not bi.overlaps(bj, MIN_CLEARANCE_MM):
                        continue
                    dx, dy = bi.overlap_vector(bj, MIN_CLEARANCE_MM)
                    mag = math.hypot(dx, dy) or 1.0
                    scale = min(REPULSION_STRENGTH, mag) / mag
                    fx, fy = dx * scale, dy * scale
                    if not placed[i].fixed:
                        forces[i][0] += fx
                        forces[i][1] += fy
                    if not placed[j].fixed:
                        forces[j][0] -= fx
                        forces[j][1] -= fy

            # 4b. Attraction — pull net-connected pairs toward each other.
            for i, j in attracted_pairs:
                bi = placed[i].bbox
                bj = placed[j].bbox
                dx = bj.cx - bi.cx
                dy = bj.cy - bi.cy
                dist = math.hypot(dx, dy) or 1.0
                # Only attract if they are far apart (> 20 mm) — otherwise
                # the repulsion is enough.
                ideal = 12.0  # target centre-to-centre mm
                if dist > ideal:
                    pull = ATTRACTION_STRENGTH * (dist - ideal)
                    pull = min(pull, MAX_FORCE)
                    fx = pull * dx / dist
                    fy = pull * dy / dist
                    if not placed[i].fixed:
                        forces[i][0] += fx
                        forces[i][1] += fy
                    if not placed[j].fixed:
                        forces[j][0] -= fx
                        forces[j][1] -= fy

            # 4c. Apply forces with damping.
            any_moved = False
            for i in range(n):
                if placed[i].fixed:
                    continue
                fx = _clamp(forces[i][0], -MAX_FORCE, MAX_FORCE)
                fy = _clamp(forces[i][1], -MAX_FORCE, MAX_FORCE)
                placed[i].vx = (placed[i].vx + fx) * DAMPING
                placed[i].vy = (placed[i].vy + fy) * DAMPING
                new_cx = placed[i].bbox.cx + placed[i].vx
                new_cy = placed[i].bbox.cy + placed[i].vy

                # Clamp to board outline.
                new_cx = _clamp(new_cx, bx_min + placed[i].bbox.hw + 1.0,
                                bx_max - placed[i].bbox.hw - 1.0)
                new_cy = _clamp(new_cy, by_min + placed[i].bbox.hh + 1.0,
                                by_max - placed[i].bbox.hh - 1.0)

                if abs(new_cx - placed[i].bbox.cx) > 0.01 or abs(new_cy - placed[i].bbox.cy) > 0.01:
                    any_moved = True
                placed[i].bbox.cx = new_cx
                placed[i].bbox.cy = new_cy

            if not any_moved:
                logger.debug("Layout converged at iteration %d", iteration)
                break

        # ── 5. Grid-snap final positions ────────────────────────
        for p in placed:
            if not p.fixed:
                p.bbox.cx = _snap(p.bbox.cx)
                p.bbox.cy = _snap(p.bbox.cy)
                # Re-clamp after snap.
                p.bbox.cx = _clamp(p.bbox.cx, bx_min + p.bbox.hw + 1.0,
                                   bx_max - p.bbox.hw - 1.0)
                p.bbox.cy = _clamp(p.bbox.cy, by_min + p.bbox.hh + 1.0,
                                   by_max - p.bbox.hh - 1.0)

        # ── 6. Build result ─────────────────────────────────────
        result = []
        for idx, c in enumerate(components):
            out = dict(c)
            out["x"] = placed[idx].bbox.cx
            out["y"] = placed[idx].bbox.cy
            result.append(out)
        return result

    @staticmethod
    def resolve_overlaps(
        components: List[Dict[str, Any]],
        clearance_mm: float = MIN_CLEARANCE_MM,
    ) -> List[Dict[str, Any]]:
        """Quick pass that resolves pair-wise overlaps without full force sim.

        This is useful as a post-MOVE_COMPONENT fixup.  Each overlapping pair
        is pushed apart along the minimum-translation axis, repeated until
        convergence or 100 iterations.

        Returns the list with updated ``x``, ``y``.
        """
        if len(components) < 2:
            return components

        placed = []
        for c in components:
            w = float(c.get("width", 10.0))
            h = float(c.get("height", 10.0))
            placed.append(_BBox(
                cx=float(c.get("x", 0.0)),
                cy=float(c.get("y", 0.0)),
                hw=w / 2.0,
                hh=h / 2.0,
            ))

        for _it in range(100):
            moved = False
            for i in range(len(placed)):
                for j in range(i + 1, len(placed)):
                    if not placed[i].overlaps(placed[j], clearance_mm):
                        continue
                    dx, dy = placed[i].overlap_vector(placed[j], clearance_mm)
                    # Split the displacement 50/50.
                    placed[i].cx += dx * 0.5
                    placed[i].cy += dy * 0.5
                    placed[j].cx -= dx * 0.5
                    placed[j].cy -= dy * 0.5
                    moved = True
            if not moved:
                break

        # Snap to grid.
        for b in placed:
            b.cx = _snap(b.cx)
            b.cy = _snap(b.cy)

        result = []
        for idx, c in enumerate(components):
            out = dict(c)
            out["x"] = placed[idx].cx
            out["y"] = placed[idx].cy
            result.append(out)
        return result

    @staticmethod
    def estimate_board_size(
        components: List[Dict[str, Any]],
        margin_pct: float = 0.25,
        min_size_mm: float = 30.0,
    ) -> Tuple[float, float]:
        """Estimate a board outline size from the component bounding boxes.

        Returns (width_mm, height_mm) with *margin_pct* extra space.
        """
        if not components:
            return (min_size_mm, min_size_mm)

        total_area = 0.0
        max_w = 0.0
        max_h = 0.0
        for c in components:
            w = float(c.get("width", 10.0))
            h = float(c.get("height", 10.0))
            total_area += w * h
            max_w = max(max_w, w)
            max_h = max(max_h, h)

        # Heuristic: sqrt of total area with aspect ratio influenced by the
        # largest component.
        side = math.sqrt(total_area) * (1.0 + margin_pct)
        # Ensure we fit the largest single component.
        width = max(side, max_w + 10.0, min_size_mm)
        height = max(side * 0.8, max_h + 10.0, min_size_mm)
        # Round up to nearest 5 mm.
        width = math.ceil(width / 5.0) * 5.0
        height = math.ceil(height / 5.0) * 5.0
        return (width, height)

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _clean_component_query(text: str) -> str:
        q = str(text or "").strip()
        if not q:
            return ""
        q = re.sub(r"^[\"']|[\"']$", "", q).strip()
        q = re.sub(r"^(add|place|insert|use|put|mount)\s+", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\b(component|footprint)\b", "", q, flags=re.IGNORECASE).strip()
        q = re.sub(r"\s+", " ", q).strip(" .,:;")
        return q

    @staticmethod
    def _looks_instruction_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return True
        return bool(re.match(r"^(add|place|insert|use|put|mount)\b", q))

    def _build_prompt(self, goal, context, board_snapshot) -> str:
        existing = ""
        if board_snapshot and board_snapshot.get("components"):
            parts = []
            for c in board_snapshot["components"][:40]:
                ref = c.get("reference", "?")
                val = c.get("value", "")
                x = c.get("x", "?")
                y = c.get("y", "?")
                parts.append(f"{ref} ({val}) @ ({x},{y})")
            existing = "\nComponents already on the board:\n" + "\n".join(parts)

        search_context = ""
        if board_snapshot and isinstance(board_snapshot.get("search_part_results"), dict):
            summary_lines: List[str] = []
            for query, items in list(board_snapshot["search_part_results"].items())[-12:]:
                if not isinstance(items, list):
                    continue
                picked: List[str] = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "")).strip()
                    pkg = str(item.get("package", "")).strip()
                    is_footprint = bool(item.get("is_footprint_candidate", False))
                    if not name:
                        continue
                    if not is_footprint and ":" not in name:
                        continue
                    label = f"{name} ({pkg})" if pkg else name
                    if label not in picked:
                        picked.append(label)
                    if len(picked) >= 2:
                        break
                if picked:
                    summary_lines.append(f"- {query}: {', '.join(picked)}")
            if summary_lines:
                search_context = (
                    "\nPhase-1 searched candidate parts (ensure required ones are placed):\n"
                    + "\n".join(summary_lines[:20])
                )

        outline = ""
        if board_snapshot and board_snapshot.get("board_width"):
            ox = board_snapshot.get("board_origin_x", 0.0)
            oy = board_snapshot.get("board_origin_y", 0.0)
            cx = board_snapshot.get("board_center_x")
            cy = board_snapshot.get("board_center_y")
            outline = (
                f"\nBoard outline: {board_snapshot['board_width']}×"
                f"{board_snapshot['board_height']} mm"
                f" (origin approx: ({ox},{oy}) mm)"
            )
            if cx is not None and cy is not None:
                outline += f" (center approx: ({cx},{cy}) mm)"

        return (
            f"USER GOAL:\n{goal}\n{existing}{search_context}{outline}\n\n"
            "Propose placement actions.  Follow the spatial strategy in your prompt.  "
            "Return a JSON array of actions.  Return [] if nothing to place."
        )

    def _parse_actions(self, raw: str, board_snapshot: Optional[Dict[str, Any]] = None) -> list:
        from ..design_agent import DesignAction, DesignActionType, normalize_action_parameters
        from ...llm.client import LLMError
        if not raw:
            raise LLMError("placement: empty LLM response.")
        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start < 0 or end <= start:
                raise LLMError("placement: expected a JSON array.")
            items = json.loads(raw[start:end])
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"placement: failed to parse JSON array: {e}") from e

        type_map = {
            "ADD_COMPONENT": DesignActionType.ADD_COMPONENT,
            "MOVE_COMPONENT": DesignActionType.MOVE_COMPONENT,
            "ROTATE_COMPONENT": DesignActionType.ROTATE_COMPONENT,
            "ALIGN_COMPONENTS": DesignActionType.ALIGN_COMPONENTS,
            "DEFINE_BOARD_OUTLINE": DesignActionType.DEFINE_BOARD_OUTLINE,
            "ADD_MOUNTING_HOLE": DesignActionType.ADD_MOUNTING_HOLE,
            "DELETE_COMPONENT": DesignActionType.DELETE_COMPONENT,
        }
        actions: list = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise LLMError("placement: each action must be an object.")
            raw_atype = item.get("action_type", None)
            if raw_atype is None or (isinstance(raw_atype, str) and not raw_atype.strip()):
                # Common model variants.
                for alt_key in ("actionType", "type", "action"):
                    raw_atype = item.get(alt_key, raw_atype)
                    if isinstance(raw_atype, str) and raw_atype.strip():
                        break
            atype_str = str(raw_atype or "").strip().upper()
            if not atype_str:
                raise LLMError(f"placement: missing required action_type for action[{idx}].")
            atype = type_map.get(atype_str)
            if atype is None:
                raise LLMError(f"placement: unknown action_type: {atype_str!r}")
            params = item.get("parameters") or {}
            if not isinstance(params, dict):
                raise LLMError(f"placement: parameters must be an object for {atype.name}.")
            description = str(item.get("description", "") or "").strip()

            if atype == DesignActionType.ADD_COMPONENT:
                raw_query = str(params.get("query", "") or "").strip()
                query = self._clean_component_query(raw_query)
                if self._looks_instruction_query(query):
                    raise LLMError("placement: ADD_COMPONENT.query must be a concrete part/footprint identifier, not an instruction.")
                if not query:
                    raise LLMError("placement: missing required ADD_COMPONENT.parameters.query.")
                params["query"] = query
            params = normalize_action_parameters(atype, params)

            actions.append(DesignAction(
                action_type=atype,
                description=description,
                parameters=params,
                requires_approval=True,
            ))
        return actions


# ── Category-based initial seeding ──────────────────────────────

def _seed_positions(
    placed: List[_PlacedComponent],
    bx_min: float,
    by_min: float,
    bx_max: float,
    by_max: float,
) -> None:
    """Set initial positions based on component category.

    ICs → board centre.
    Connectors → left/top/right edge.
    Passives → distributed around their nearest IC.
    Mounting holes → corners.
    """
    cx = (bx_min + bx_max) / 2.0
    cy = (by_min + by_max) / 2.0
    w = bx_max - bx_min
    h = by_max - by_min

    # Collect ICs for passive clustering later.
    ic_positions: List[Tuple[float, float]] = []
    connector_count = 0

    for p in placed:
        if p.fixed:
            continue
        cat = p.category
        if cat == "ic":
            # Already centred by default; offset slightly per IC.
            offset = len(ic_positions) * 15.0
            p.bbox.cx = _clamp(cx + offset, bx_min + p.bbox.hw + 2, bx_max - p.bbox.hw - 2)
            p.bbox.cy = cy
            ic_positions.append((p.bbox.cx, p.bbox.cy))
        elif cat == "connector":
            # Place connectors at the edges.
            edge = connector_count % 3
            if edge == 0:  # left edge
                p.bbox.cx = bx_min + p.bbox.hw + 3
                p.bbox.cy = cy + connector_count * 12.0
            elif edge == 1:  # right edge
                p.bbox.cx = bx_max - p.bbox.hw - 3
                p.bbox.cy = cy + (connector_count - 1) * 12.0
            else:  # top edge
                p.bbox.cx = cx + (connector_count - 2) * 12.0
                p.bbox.cy = by_min + p.bbox.hh + 3
            connector_count += 1
            # Clamp.
            p.bbox.cx = _clamp(p.bbox.cx, bx_min + p.bbox.hw + 1, bx_max - p.bbox.hw - 1)
            p.bbox.cy = _clamp(p.bbox.cy, by_min + p.bbox.hh + 1, by_max - p.bbox.hh - 1)
        elif cat == "mounting":
            # Corners (inset 3 mm).
            corners = [
                (bx_min + 3 + p.bbox.hw, by_min + 3 + p.bbox.hh),
                (bx_max - 3 - p.bbox.hw, by_min + 3 + p.bbox.hh),
                (bx_min + 3 + p.bbox.hw, by_max - 3 - p.bbox.hh),
                (bx_max - 3 - p.bbox.hw, by_max - 3 - p.bbox.hh),
            ]
            idx = sum(1 for pp in placed if pp.category == "mounting" and pp is not p)
            ci = idx % len(corners)
            p.bbox.cx, p.bbox.cy = corners[ci]

    # Passives: cluster near the closest IC (or centre if no ICs).
    passive_idx = 0
    for p in placed:
        if p.fixed or p.category not in ("passive", "generic"):
            continue
        if ic_positions:
            # Pick closest IC by shared nets, else by index.
            nearest_ic = ic_positions[passive_idx % len(ic_positions)]
            angle = (passive_idx * 45.0) * math.pi / 180.0
            radius = 8.0 + (passive_idx // 8) * 5.0
            p.bbox.cx = nearest_ic[0] + radius * math.cos(angle)
            p.bbox.cy = nearest_ic[1] + radius * math.sin(angle)
        else:
            # Grid around centre.
            cols = max(1, int(math.sqrt(passive_idx + 1)))
            row, col = divmod(passive_idx, cols)
            p.bbox.cx = cx - w * 0.3 + col * 12.0
            p.bbox.cy = cy - h * 0.3 + row * 12.0
        p.bbox.cx = _clamp(p.bbox.cx, bx_min + p.bbox.hw + 1, bx_max - p.bbox.hw - 1)
        p.bbox.cy = _clamp(p.bbox.cy, by_min + p.bbox.hh + 1, by_max - p.bbox.hh - 1)
        passive_idx += 1
