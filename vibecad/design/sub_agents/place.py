# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UNIVERSAL PLUGIN — NO BOARD-SPECIFIC HARDCODING IN THIS FILE      ║
# ║  Prompts must use only goal_str / context variables.               ║
# ║  Never embed specific MPNs, part names, board names, or            ║
# ║  design-specific quantities in prompt strings or system prompts.   ║
# ╚══════════════════════════════════════════════════════════════════════╝
import json
import logging
import math
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)

# ── Packer tuning knobs ────────────────────────────────────────────────────
_MAX_COLS = 4
_INNER_GAP_MM = 3.0
_EDGE_GAP_MM = 2.0
_PERIMETER_MARGIN_MM = 1.5
_BAND_CLEARANCE_MM = 3.0
_EDGE_END_MARGIN_MM = 2.0
_EDGE_TO_CORE_GAP_MM = 4.0
_PLACEMENT_CLEARANCE_MM = 1.6
_PACK_OUTLINE_MAX_PASSES = 4
_VERIFY_REPAIR_MAX_PASSES = 1
_CLOCK_TARGET_MAX_MM = 20.0
_COMPANION_TARGET_MAX_MM = 16.0
_COMPANION_MAX_BODY_MM = 8.0

# The _handle_define_board_outline handler centres the outline on the KiCad
# page.  For A4 (297×210 mm) it hard-codes centre=(150, 100).  Mirror that
# here so our component origins match the drawn outline.
_PAGE_CX_MM = 150.0
_PAGE_CY_MM = 100.0

_METRIC_SIZE_RE = re.compile(r"(?P<w>\d{2})(?P<h>\d{2})Metric", re.IGNORECASE)
_DIM_MM_RE = re.compile(r"(?P<w>\d+(?:\.\d+)?)x(?P<h>\d+(?:\.\d+)?)mm", re.IGNORECASE)
_EIA_SIZE_RE = re.compile(r"(?:^|[-_])(?P<w>\d{2})(?P<h>\d{2})(?:[-_]|$)", re.IGNORECASE)
_PINHEADER_RE = re.compile(r"Pin(Header|Socket)_(?P<rows>\d+)x(?P<pins>\d+)_P(?P<pitch>\d+(?:\.\d+)?)mm", re.IGNORECASE)
_RADIAL_DIAMETER_RE = re.compile(r"(?:^|[_-])D(?P<d>\d+(?:\.\d+)?)mm", re.IGNORECASE)
_CONTROL_TOKEN_RE = re.compile(r"\b(button|switch|pushbutton|tactile|tact|spst|spdt)\b", re.IGNORECASE)
_FOOTPRINT_GEOMETRY_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


class ComponentPlaceAgent(SubAgent):
    """
    SubAgent responsible for placing components on the PCB canvas.
    Uses a hybrid approach:
      1. Asks the LLM to output a semantic 'Zoning Plan' clustering components.
      2. Deterministically grid-packs the groups to assign exact (X, Y, Rot).
      3. Emits ADD_COMPONENT DesignActions so footprints actually land on the board.
    """

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        self.name = "PLACE"
        self.description = "Assigns X,Y coordinates and places all components from the manifest"

    def can_handle(self, goal: str, context: Dict[str, Any]) -> bool:
        manifest = context.get("manifest") or context.get("artifacts", {}).get("manifest", {})
        return bool(manifest and "placement_plan" not in context)

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        manifest = context.get("manifest") or context.get("artifacts", {}).get("manifest", {})
        parts = (manifest or {}).get("parts", [])

        if not parts:
            return SubAgentResult(
                message="No parts found in the manifest to place.",
                confidence=1.0,
                phase_complete=True,
            )

        # 1. LLM Concept Pass: Semantic Zoning Plan
        zoning_plan, thinking = self._get_llm_zoning_plan(parts)

        # 2. Deterministic board-agnostic placement:
        #    - infer rough footprint sizes from package strings
        #    - reserve edge bands for connectors/headers/controls
        #    - align long headers to edges and ports outward
        #    - anchor semantic interior zones near relevant edges / board center
        classified = self._classify_parts(parts, zoning_plan)
        outline_w, outline_h, placement_plan, placement_stats = self._pack_components(classified)

        origin_x = _PAGE_CX_MM - (outline_w / 2.0)
        origin_y = _PAGE_CY_MM - (outline_h / 2.0)

        logger.info(
            "PLACE: %d parts → outline %.1f×%.1f mm, origin (%.1f, %.1f)",
            len(parts), outline_w, outline_h, origin_x, origin_y,
        )

        # 3. Build ADD_COMPONENT actions so the board actually gets populated
        add_actions = self._build_add_actions(parts, placement_plan)

        # 4. Prepend DEFINE_BOARD_OUTLINE so Edge.Cuts is drawn first
        outline_action = self._build_outline_action(outline_w, outline_h)
        actions = ([outline_action] if outline_action else []) + add_actions

        return SubAgentResult(
            message=f"Placing {len(placement_plan)} components onto board.",
            confidence=0.9,
            phase_complete=True,
            thinking=thinking,
            actions=actions,
            artifacts={
                "placement_plan": placement_plan,
                "placement_stats": placement_stats,
            },
        )

    # ── Classification / geometry helpers ─────────────────────────────────

    def _combined_text(self, part: Dict[str, Any]) -> str:
        tokens = []
        for key in (
            "ref", "mpn", "footprint", "package", "value", "description",
            "role", "role_type", "role_class", "type", "kind", "blob",
        ):
            val = part.get(key)
            if val:
                tokens.append(str(val))
        return " ".join(tokens).lower()

    def _parse_metric_package_mm(self, footprint: str) -> Optional[Tuple[float, float]]:
        m = _METRIC_SIZE_RE.search(footprint or "")
        if not m:
            return None
        w = float(m.group("w")) / 10.0
        h = float(m.group("h")) / 10.0
        return (w, h)

    def _parse_explicit_dims_mm(self, footprint: str) -> Optional[Tuple[float, float]]:
        m = _DIM_MM_RE.search(footprint or "")
        if not m:
            return None
        return (float(m.group("w")), float(m.group("h")))

    def _parse_eia_dims_mm(self, footprint: str) -> Optional[Tuple[float, float]]:
        m = _EIA_SIZE_RE.search(footprint or "")
        if not m:
            return None
        return (float(m.group("w")) / 10.0, float(m.group("h")) / 10.0)

    def _normalize_footprint_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    def _footprint_similarity_score(self, target_name: str, candidate_name: str) -> float:
        target_norm = self._normalize_footprint_key(target_name)
        candidate_norm = self._normalize_footprint_key(candidate_name)
        if not target_norm or not candidate_norm:
            return 0.0
        if target_norm == candidate_norm:
            return 10.0

        prefix_len = len(os.path.commonprefix([target_norm, candidate_norm]))
        prefix_ratio = prefix_len / max(len(target_norm), len(candidate_norm), 1)
        contains_ratio = 1.0 if (target_norm in candidate_norm or candidate_norm in target_norm) else 0.0

        target_tokens = {tok for tok in re.split(r"[^a-z0-9]+", str(target_name or "").lower()) if tok}
        candidate_tokens = {tok for tok in re.split(r"[^a-z0-9]+", str(candidate_name or "").lower()) if tok}
        token_overlap = 0.0
        if target_tokens and candidate_tokens:
            token_overlap = len(target_tokens & candidate_tokens) / len(target_tokens | candidate_tokens)

        ratio = SequenceMatcher(None, target_norm, candidate_norm).ratio()
        return ratio + (0.8 * prefix_ratio) + (0.6 * contains_ratio) + (0.8 * token_overlap)

    def _resolve_footprint_file(self, footprint: str) -> Optional[Path]:
        fp_id = str(footprint or "").strip()
        if ":" not in fp_id:
            return None
        lib, name = fp_id.split(":", 1)
        lib = lib.strip()
        name = name.strip()
        if not lib or not name:
            return None

        search_roots: List[Path] = []
        for env_name, env_value in os.environ.items():
            if env_name.startswith("KICAD") and env_name.endswith("_FOOTPRINT_DIR") and env_value:
                p = Path(env_value).expanduser()
                if p.exists():
                    search_roots.append(p)

        default_root = Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/footprints")
        if default_root.exists():
            search_roots.append(default_root)

        best_match: Optional[Path] = None
        best_score = 0.0
        seen: set = set()
        for root in search_roots:
            root_key = str(root)
            if root_key in seen:
                continue
            seen.add(root_key)
            lib_dir = root / f"{lib}.pretty"
            candidate = lib_dir / f"{name}.kicad_mod"
            if candidate.is_file():
                return candidate
            if not lib_dir.is_dir():
                continue
            for fp_file in lib_dir.glob("*.kicad_mod"):
                score = self._footprint_similarity_score(name, fp_file.stem)
                if score > best_score:
                    best_match = fp_file
                    best_score = score
        if best_match is not None and best_score >= 1.85:
            return best_match
        return None

    def _footprint_geometry(self, footprint: str) -> Optional[Dict[str, Any]]:
        fp_id = str(footprint or "").strip()
        if not fp_id:
            return None
        if fp_id in _FOOTPRINT_GEOMETRY_CACHE:
            return _FOOTPRINT_GEOMETRY_CACHE[fp_id]

        fp_file = self._resolve_footprint_file(fp_id)
        if fp_file is None:
            _FOOTPRINT_GEOMETRY_CACHE[fp_id] = None
            return None

        try:
            text = fp_file.read_text()
        except Exception:
            _FOOTPRINT_GEOMETRY_CACHE[fp_id] = None
            return None

        def layer_bbox(layer_name: str) -> Optional[Tuple[float, float, float, float]]:
            pts: List[Tuple[float, float]] = []
            pattern = re.compile(
                r"\(fp_(?:line|rect)\b.*?\(start\s+([-0-9.]+)\s+([-0-9.]+)\).*?\(end\s+([-0-9.]+)\s+([-0-9.]+)\).*?\(layer\s+\""
                + re.escape(layer_name)
                + r"\"\)",
                re.S,
            )
            for match in pattern.finditer(text):
                pts.append((float(match.group(1)), float(match.group(2))))
                pts.append((float(match.group(3)), float(match.group(4))))
            if not pts:
                return None
            xs = [pt[0] for pt in pts]
            ys = [pt[1] for pt in pts]
            return (min(xs), min(ys), max(xs), max(ys))

        bbox = layer_bbox("F.CrtYd") or layer_bbox("F.Fab")
        if bbox is None:
            _FOOTPRINT_GEOMETRY_CACHE[fp_id] = None
            return None

        pad_points: List[Tuple[float, float]] = []
        for match in re.finditer(r"\(pad\s+\"[^\"]+\".*?\(at\s+([-0-9.]+)\s+([-0-9.]+)", text, re.S):
            pad_points.append((float(match.group(1)), float(match.group(2))))

        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y
        pad_cx = sum(pt[0] for pt in pad_points) / len(pad_points) if pad_points else (min_x + max_x) / 2.0
        pad_cy = sum(pt[1] for pt in pad_points) / len(pad_points) if pad_points else (min_y + max_y) / 2.0
        extents = {
            "px": max_x - pad_cx,
            "nx": pad_cx - min_x,
            "py": max_y - pad_cy,
            "ny": pad_cy - min_y,
        }
        body_dir = max(extents, key=lambda key: extents[key])
        geometry = {
            "width": round(width, 3),
            "height": round(height, 3),
            "body_dir": body_dir,
            "extents": {k: round(float(v), 3) for k, v in extents.items()},
            "bbox": (min_x, min_y, max_x, max_y),
            "pad_centroid": (round(pad_cx, 3), round(pad_cy, 3)),
            "source": str(fp_file),
            "resolved_name": fp_file.stem,
        }
        _FOOTPRINT_GEOMETRY_CACHE[fp_id] = geometry
        return geometry

    def _estimate_body_size_mm(self, part: Dict[str, Any]) -> Tuple[float, float]:
        footprint = str(part.get("footprint", "") or part.get("package", "") or "")
        text = self._combined_text(part)

        hdr = _PINHEADER_RE.search(footprint)
        if hdr:
            rows = max(1, int(hdr.group("rows")))
            pins = max(1, int(hdr.group("pins")))
            pitch = float(hdr.group("pitch"))
            long_span = max(rows, pins) * pitch
            short_span = min(rows, pins) * pitch
            return (short_span + 4.0, long_span + 4.0)

        geom = self._footprint_geometry(footprint)
        if geom:
            return (
                max(1.8, float(geom.get("width", 0.0) or 0.0)),
                max(1.8, float(geom.get("height", 0.0) or 0.0)),
            )

        dims = self._parse_explicit_dims_mm(footprint)
        if dims:
            return (dims[0] + 3.5, dims[1] + 3.5)

        dims = self._parse_metric_package_mm(footprint)
        if dims:
            return (max(1.8, dims[0] + 2.0), max(1.6, dims[1] + 2.0))

        dims = self._parse_eia_dims_mm(footprint)
        if dims:
            return (dims[0] + 2.4, dims[1] + 2.4)

        radial = _RADIAL_DIAMETER_RE.search(footprint or "")
        if radial:
            diameter = float(radial.group("d"))
            return (diameter + 2.4, diameter + 2.4)

        if "jack" in text and any(token in text for token in ("power", "dc", "vin", "pwr")):
            return (15.0, 12.0)
        if "usb" in text and "connector" in text:
            return (18.0, 15.0)
        if "qfn" in text:
            return (8.0, 8.0)
        if "tqfp" in text or "qfp" in text:
            return (11.5, 11.5)
        if "sot-223" in text:
            return (8.0, 9.0)
        if "sot-23-6" in text:
            return (5.5, 4.5)
        if "sot-23-5" in text or "sot-23" in text:
            return (5.0, 4.0)
        if "vssop" in text or "soic" in text:
            return (7.0, 6.0)
        if "crystal" in text or "resonator" in text:
            return (7.0, 5.0)
        if "switch" in text:
            return (8.0, 8.0)
        if "fuse" in text:
            return (7.0, 6.0)
        if "sma" in text:
            return (6.5, 4.5)
        if "minimelf" in text:
            return (5.5, 3.5)
        if "diode" in text:
            return (4.5, 3.0)
        if "led" in text:
            return (3.5, 2.5)
        if "electrolytic" in text or "bulk_cap" in text:
            # SMD electrolytic caps (10uF–100uF) have large courtyards (~9–10 mm)
            return (10.0, 9.0)
        if "capacitor" in text or "resistor" in text:
            return (3.2, 2.4)
        return (6.5, 4.5)

    def _part_edge_hint(self, part: Dict[str, Any]) -> Optional[str]:
        for key in ("preferred_edge", "edge", "side", "placement_edge"):
            value = str(part.get(key, "") or "").strip().lower()
            if value in {"left", "right", "top", "bottom"}:
                return value
        return None

    def _choose_common_port_edge(self, classified: List[Dict[str, Any]]) -> Optional[str]:
        edges = ["left", "right", "top", "bottom"]
        ports = [item for item in classified if str(item.get("category", "") or "") == "port"]
        if not ports:
            return None

        hint_counts = {edge: 0 for edge in edges}
        for item in ports:
            hinted = str(item.get("preferred_edge", "") or "").strip().lower()
            if hinted in hint_counts:
                hint_counts[hinted] += 1
        if max(hint_counts.values()) > 0:
            return min(edges, key=lambda edge: (-hint_counts[edge], edges.index(edge)))

        port_refs = {str(item.get("ref", "") or "") for item in ports}
        edge_counts = {edge: 0 for edge in edges}
        edge_loads = {edge: 0.0 for edge in edges}
        for item in classified:
            ref = str(item.get("ref", "") or "")
            if ref in port_refs:
                continue
            edge = str(item.get("preferred_edge", "") or "").strip().lower()
            if edge not in edge_counts:
                continue
            edge_counts[edge] += 1
            w, h = self._oriented_dims(item, edge)
            edge_loads[edge] += w if edge in {"top", "bottom"} else h
        return min(edges, key=lambda edge: (edge_counts[edge], edge_loads[edge], edges.index(edge)))

    def _categorize_part(self, part: Dict[str, Any]) -> Dict[str, Any]:
        ref = str(part.get("ref", "") or "")
        text = self._combined_text(part)
        width, height = self._estimate_body_size_mm(part)
        footprint = str(part.get("footprint", "") or part.get("package", "") or "")
        geom = self._footprint_geometry(footprint)

        category = "support"
        region = "interior"
        preferred_edge = None
        priority = 50

        normalized_text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = {tok for tok in normalized_text.split() if tok}
        is_header = "pinheader" in text or "pinsocket" in text or (ref.startswith("J") and "header" in text)
        is_jack_like = bool(re.search(r"\b\w*jack\w*\b", normalized_text))
        is_connector_like = bool(tokens & {"connector", "receptacle", "socket", "port"})
        is_external_port = (
            (not is_header)
            and (
                ("usb" in tokens)
                or is_jack_like
                or (is_connector_like and ref.startswith("J"))
            )
        )
        is_receptacle_like = bool(tokens & {"receptacle", "socket"})
        is_programming = is_header and ("2x03" in text or "icsp" in text)
        is_control = ref.startswith("SW") or bool(_CONTROL_TOKEN_RE.search(normalized_text))
        is_active = ref.startswith(("U", "Q")) or any(k in text for k in ("regulator", "op amp", "amplifier", "mcu", "mosfet"))
        is_clock = ref.startswith("Y") or "crystal" in text or "resonator" in text
        is_powerish_port = bool(tokens & {"power", "vin", "vcc", "pwr", "dc"}) or (
            is_jack_like and "usb" not in tokens
        )
        port_kind = "usb" if "usb" in tokens else ("power" if is_powerish_port else "io")
        port_rotation_preference: Optional[str] = None
        hinted_edge = self._part_edge_hint(part)

        if is_external_port:
            category = "port"
            region = "edge"
            preferred_edge = hinted_edge
            priority = 0
            port_rotation_preference = "extents_first"
            # Port receptacles are better oriented by inferred mating-face
            # direction than by body-depth extents.
            if is_jack_like:
                # Barrel-jack local face directions vary a lot across libraries.
                # Prioritizing inferred face direction avoids edge-axis lock-ins
                # that can make the connector point up/down on some variants.
                port_rotation_preference = "face_dir_first"
            elif is_receptacle_like:
                port_rotation_preference = "face_dir_first"
        elif is_header and not is_programming:
            category = "long_header"
            region = "edge"
            preferred_edge = "top"
            priority = 5
        elif is_programming:
            category = "small_header"
            region = "edge"
            preferred_edge = "right"
            priority = 10
        elif is_active:
            category = "active"
            priority = 20
        elif is_control:
            category = "control"
            region = "edge"
            preferred_edge = "top"
            priority = 15
        elif is_clock:
            category = "clock"
            priority = 25
        elif ref.startswith(("D", "F", "FB")):
            category = "interface_support"
            priority = 30
        elif ref.startswith(("C", "R", "RN", "LED")):
            category = "support"
            priority = 40

        # Fallback: any J-ref that wasn't explicitly classified as a port or
        # programming header goes to an edge. Prevents pin headers whose
        # footprint string lacks "pinheader" from landing in the interior grid.
        if ref.startswith("J") and region == "interior":
            category = "long_header"
            region = "edge"
            preferred_edge = "top"
            priority = 5

        port_face_dir = self._infer_port_face_dir(
            category,
            geom.get("body_dir") if isinstance(geom, dict) else None,
            width,
            height,
            text,
        )

        return {
            "ref": ref,
            "part": part,
            "text": text,
            "width": width,
            "height": height,
            "category": category,
            "region": region,
            "preferred_edge": preferred_edge,
            "priority": priority,
            "footprint_body_dir": geom.get("body_dir") if isinstance(geom, dict) else None,
            "footprint_extents": geom.get("extents") if isinstance(geom, dict) else None,
            "footprint_geom_source": geom.get("source") if isinstance(geom, dict) else None,
            "port_face_dir": port_face_dir,
            "port_face_dir_source": "footprint_body_dir" if port_face_dir else None,
            "port_kind": port_kind if category == "port" else None,
            "port_rotation_preference": port_rotation_preference if category == "port" else None,
        }

    def _classify_parts(self, parts: List[Dict[str, Any]], zoning_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        group_order: Dict[str, int] = {}
        for gi, group in enumerate(zoning_plan.get("groups") or []):
            for ref in group.get("refs") or []:
                if ref not in group_order:
                    group_order[str(ref)] = gi

        classified = []
        for idx, part in enumerate(parts):
            item = self._categorize_part(part)
            item["group_order"] = group_order.get(item["ref"], len(group_order) + idx)
            classified.append(item)

        common_port_edge = self._choose_common_port_edge(classified)
        if common_port_edge in {"left", "right", "top", "bottom"}:
            for item in classified:
                if str(item.get("category", "") or "") == "port":
                    item["preferred_edge"] = common_port_edge

        classified.sort(key=lambda item: (item["priority"], item["group_order"], item["ref"]))
        return classified

    def _opposite_dir(self, direction: str) -> str:
        return {
            "px": "nx",
            "nx": "px",
            "py": "ny",
            "ny": "py",
        }.get(direction, direction)

    def _rotation_from_dir(self, source_dir: str, target_dir: str) -> float:
        order = ["px", "py", "nx", "ny"]
        if source_dir not in order or target_dir not in order:
            return 0.0
        return float(((order.index(target_dir) - order.index(source_dir)) % 4) * 90)

    def _rotation_steps(self, rotation_deg: float) -> int:
        try:
            return int(round((float(rotation_deg) % 360.0) / 90.0)) % 4
        except Exception:
            return 0

    def _local_dir_for_board_dir(self, board_dir: str, rotation_deg: float) -> Optional[str]:
        order = ["px", "py", "nx", "ny"]
        if board_dir not in order:
            return None
        steps = self._rotation_steps(rotation_deg)
        return order[(order.index(board_dir) - steps) % 4]

    def _extent_for_board_dir(
        self,
        extents: Dict[str, Any],
        board_dir: str,
        rotation_deg: float,
    ) -> Optional[float]:
        if not isinstance(extents, dict):
            return None
        local_dir = self._local_dir_for_board_dir(board_dir, rotation_deg)
        if local_dir is None:
            return None
        try:
            return float(extents.get(local_dir, 0.0) or 0.0)
        except Exception:
            return None

    def _best_port_rotation_from_extents(self, item: Dict[str, Any], edge: str) -> Optional[float]:
        extents = item.get("footprint_extents")
        if not isinstance(extents, dict) or edge not in {"left", "right", "top", "bottom"}:
            return None

        outward = self._outward_dir_for_edge(edge)
        inward = self._inward_dir_for_edge(edge)
        best: Optional[Tuple[float, float, float]] = None
        for rot in (0.0, 90.0, 180.0, 270.0):
            outward_depth = self._extent_for_board_dir(extents, outward, rot)
            inward_depth = self._extent_for_board_dir(extents, inward, rot)
            if outward_depth is None or inward_depth is None:
                continue
            margin = outward_depth - inward_depth
            # Prefer rotations where body depth extends outward from pads.
            score = (margin * 10.0) + outward_depth
            if best is None or score > best[0]:
                best = (score, margin, rot)

        if best is None:
            return None

        _score, margin, rotation = best
        # Keep prior behavior for ambiguous/symmetric footprints.
        return float(rotation) if margin >= 0.75 else None

    def _best_axis_constrained_port_rotation_from_extents(self, item: Dict[str, Any], edge: str) -> Optional[float]:
        """Pick a rotation on the edge axis only (no up/down on left/right edges)."""
        extents = item.get("footprint_extents")
        if not isinstance(extents, dict) or edge not in {"left", "right", "top", "bottom"}:
            return None
        if edge in {"left", "right"}:
            candidates = (0.0, 180.0)
        else:
            candidates = (90.0, 270.0)

        outward = self._outward_dir_for_edge(edge)
        inward = self._inward_dir_for_edge(edge)
        best: Optional[Tuple[float, float, float]] = None
        for rot in candidates:
            outward_depth = self._extent_for_board_dir(extents, outward, rot)
            inward_depth = self._extent_for_board_dir(extents, inward, rot)
            if outward_depth is None or inward_depth is None:
                continue
            margin = outward_depth - inward_depth
            score = (margin * 10.0) + outward_depth
            if best is None or score > best[0]:
                best = (score, margin, rot)
        if best is None:
            return None
        return float(best[2])

    def _port_edge_depths(
        self,
        item: Dict[str, Any],
        edge: str,
        rotation_deg: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        extents = item.get("footprint_extents")
        if not isinstance(extents, dict) or edge not in {"left", "right", "top", "bottom"}:
            return (None, None)
        outward = self._outward_dir_for_edge(edge)
        inward = self._inward_dir_for_edge(edge)
        outward_depth = self._extent_for_board_dir(extents, outward, rotation_deg)
        inward_depth = self._extent_for_board_dir(extents, inward, rotation_deg)
        return (outward_depth, inward_depth)

    def _inward_dir_for_edge(self, edge: str) -> str:
        return {
            "left": "px",
            "right": "nx",
            "top": "py",
            "bottom": "ny",
        }.get(edge, "px")

    def _outward_dir_for_edge(self, edge: str) -> str:
        return {
            "left": "nx",
            "right": "px",
            "top": "ny",
            "bottom": "py",
        }.get(edge, "nx")

    def _infer_port_face_dir(
        self,
        category: str,
        body_dir: Optional[str],
        width: float,
        height: float,
        text: str,
    ) -> Optional[str]:
        direction = str(body_dir or "").strip()
        normalized_text = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
        is_jack_like = bool(re.search(r"\b\w*jack\w*\b", normalized_text) or ("barrel" in normalized_text))
        is_receptacle_like = bool(re.search(r"\b(receptacle|socket)\b", normalized_text))

        if direction not in {"px", "nx", "py", "ny"}:
            return None
        if category == "port":
            # Barrel-jack footprints in common KiCad libraries tend to model the
            # barrel opening along the dominant body direction (pads trail behind
            # that opening). Keep face_dir aligned with body_dir for stability.
            if is_jack_like:
                return direction
            # Receptacle/socket footprints are often modeled with the mating face
            # opposite the dominant body direction.
            if is_receptacle_like:
                return self._opposite_dir(direction)
            # For edge ports, map footprint body axis to a stable horizontal
            # mating-face estimate so left/right edge placement stays consistent.
            return {
                "px": "px",
                "nx": "nx",
                "py": "px",
                "ny": "nx",
            }.get(direction, direction)
        if is_receptacle_like:
            return self._opposite_dir(direction)
        if is_jack_like:
            return direction if width >= height else self._opposite_dir(direction)
        return None

    def _normalize_face_dir_for_edge(self, face_dir: str, edge: str) -> str:
        fd = str(face_dir or "").strip()
        if fd not in {"px", "nx", "py", "ny"}:
            return ""
        # Face directions are expressed in footprint-local coordinates.
        # Do not coerce them based on board edge; rotation mapping handles that.
        return fd

    def _edge_rotation(self, edge: str, item: Dict[str, Any]) -> float:
        category = str(item.get("category", ""))
        if category == "long_header":
            return 90.0 if edge in {"top", "bottom"} else (0.0 if edge == "left" else 180.0)
        if category == "small_header":
            return 0.0 if edge == "left" else (180.0 if edge == "right" else 90.0)
        if category == "port":
            preference = str(item.get("port_rotation_preference", "") or "extents_first").strip().lower()
            face_dir = self._normalize_face_dir_for_edge(str(item.get("port_face_dir", "") or ""), edge)
            if preference == "axis_extents_first":
                axis_rot = self._best_axis_constrained_port_rotation_from_extents(item, edge)
                if axis_rot is not None:
                    return axis_rot
                if face_dir and face_dir in {"px", "nx", "py", "ny"}:
                    return self._rotation_from_dir(face_dir, self._outward_dir_for_edge(edge))
                return {"left": 180.0, "right": 0.0, "top": 90.0, "bottom": 270.0}.get(edge, 0.0)
            if preference == "face_dir_first":
                if face_dir and face_dir in {"px", "nx", "py", "ny"}:
                    return self._rotation_from_dir(face_dir, self._outward_dir_for_edge(edge))
                extents_rot = self._best_port_rotation_from_extents(item, edge)
                if extents_rot is not None:
                    return extents_rot
                return {"left": 180.0, "right": 0.0, "top": 90.0, "bottom": 270.0}.get(edge, 0.0)

            extents_rot = self._best_port_rotation_from_extents(item, edge)
            if extents_rot is not None:
                return extents_rot
            if face_dir and face_dir in {"px", "nx", "py", "ny"}:
                return self._rotation_from_dir(face_dir, self._outward_dir_for_edge(edge))

            # No explicit face_dir: use deterministic edge rotation, independent
            # of footprint-native local axis conventions.
            return {"left": 180.0, "right": 0.0, "top": 90.0, "bottom": 270.0}.get(edge, 0.0)
        if category == "control":
            return 0.0
        return 0.0

    def _edge_inset(self, item: Dict[str, Any], edge: str) -> float:
        category = str(item.get("category", ""))
        if edge in {"left", "right"} and category == "port":
            return 2.5 if str(item.get("port_kind", "") or "") == "usb" else 2.0
        if edge in {"top", "bottom"} and category == "long_header":
            return 1.0
        return 0.0

    def _oriented_dims(self, item: Dict[str, Any], edge: Optional[str] = None) -> Tuple[float, float]:
        w = float(item.get("width", 6.0) or 6.0)
        h = float(item.get("height", 4.0) or 4.0)
        if edge in {"left", "right", "top", "bottom"}:
            rot = self._edge_rotation(edge, item)
            if abs((rot % 180.0) - 90.0) < 1e-3:
                return (h, w)
        return (w, h)

    def _stack_span(self, items: List[Dict[str, Any]], edge: str) -> float:
        if not items:
            return 0.0
        total = 0.0
        previous: Optional[Dict[str, Any]] = None
        for item in items:
            w, h = self._oriented_dims(item, edge)
            span = w if edge in {"top", "bottom"} else h
            if previous is not None:
                total += self._edge_item_gap(previous, item, edge)
            total += span
            previous = item
        return total

    def _edge_item_gap(self, previous: Dict[str, Any], current: Dict[str, Any], edge: str) -> float:
        prev_cat = str(previous.get("category", "") or "")
        curr_cat = str(current.get("category", "") or "")
        categories = {prev_cat, curr_cat}
        if edge in {"left", "right"} and "port" in categories:
            return _EDGE_GAP_MM + 4.0
        if categories == {"long_header"}:
            return _EDGE_GAP_MM + 0.5
        if categories & {"small_header", "control"}:
            return _EDGE_GAP_MM + 1.0
        return _EDGE_GAP_MM

    def _edge_clearance(self, item: Dict[str, Any]) -> float:
        category = str(item.get("category", ""))
        if category == "port":
            return _BAND_CLEARANCE_MM + 5.0
        if category == "long_header":
            return _BAND_CLEARANCE_MM + 2.5
        if category in {"small_header", "control"}:
            return _BAND_CLEARANCE_MM + 2.0
        return _BAND_CLEARANCE_MM

    def _max_depth(self, items: List[Dict[str, Any]], edge: str) -> float:
        if not items:
            return 0.0
        depths = []
        for item in items:
            w, h = self._oriented_dims(item, edge)
            depth = h if edge in {"top", "bottom"} else w
            depths.append(depth + self._edge_clearance(item))
        return max(depths)

    def _edge_to_core_gap(self, items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        categories = {str(item.get("category", "")) for item in items}
        if "long_header" in categories:
            return _EDGE_TO_CORE_GAP_MM + 4.0
        if categories & {"port", "long_header", "small_header"}:
            return _EDGE_TO_CORE_GAP_MM + 0.5
        if "control" in categories:
            return _EDGE_TO_CORE_GAP_MM
        return max(0.5, _EDGE_TO_CORE_GAP_MM - 0.5)

    def _header_edge_rank(self, item: Dict[str, Any]) -> Tuple[int, str]:
        text = str(item.get("text", "") or "")
        if "power header" in text or "power" in text:
            return (0, item["ref"])
        if "analog header" in text or "analog" in text:
            return (1, item["ref"])
        if "digital header low" in text or "digital" in text:
            return (2, item["ref"])
        if "digital header high" in text:
            return (3, item["ref"])
        return (4, item["ref"])

    def _grid_dims_for_cols(self, items: List[Dict[str, Any]], cols: int) -> Tuple[int, int, List[float], List[float], float, float]:
        rows = math.ceil(len(items) / cols)
        col_widths = [0.0 for _ in range(cols)]
        row_heights = [0.0 for _ in range(rows)]
        for idx, item in enumerate(items):
            col = idx % cols
            row = idx // cols
            col_widths[col] = max(col_widths[col], float(item.get("width", 6.0) or 6.0))
            row_heights[row] = max(row_heights[row], float(item.get("height", 4.0) or 4.0))
        width = sum(col_widths) + max(0, cols - 1) * _INNER_GAP_MM
        height = sum(row_heights) + max(0, rows - 1) * _INNER_GAP_MM
        return (cols, rows, col_widths, row_heights, width, height)

    def _interior_grid_dims(self, items: List[Dict[str, Any]]) -> Tuple[int, int, List[float], List[float], float, float]:
        if not items:
            return (0, 0, [], [], 0.0, 0.0)
        n = len(items)
        max_cols = max(1, min(_MAX_COLS, n))
        min_cols = 1 if n < 6 else 3
        best: Optional[Tuple[float, Tuple[int, int, List[float], List[float], float, float]]] = None
        for cols in range(min_cols, max_cols + 1):
            dims = self._grid_dims_for_cols(items, cols)
            _, _, _, _, width, height = dims
            aspect = width / max(height, 1.0)
            score = abs(aspect - 1.2) * 18.0 + width * 0.85 + height * 0.35
            if best is None or score < best[0]:
                best = (score, dims)
        return best[1] if best else self._grid_dims_for_cols(items, max_cols)

    def _interior_sort_key(self, item: Dict[str, Any]) -> Tuple[int, int, float, float, str]:
        category_rank = {
            "active": 0,
            "clock": 1,
            "interface_support": 2,
            "support": 3,
        }
        return (
            int(item.get("group_order", 0) or 0),
            category_rank.get(str(item.get("category", "") or ""), 4),
            -max(float(item.get("width", 0.0) or 0.0), float(item.get("height", 0.0) or 0.0)),
            -min(float(item.get("width", 0.0) or 0.0), float(item.get("height", 0.0) or 0.0)),
            str(item.get("ref", "") or ""),
        )

    def _layout_shelves(
        self,
        items: List[Dict[str, Any]],
        target_width: float,
    ) -> Tuple[float, float, List[Tuple[Dict[str, Any], float, float]]]:
        if not items:
            return 0.0, 0.0, []

        ordered = sorted(items, key=self._interior_sort_key)
        max_item_width = max(float(item.get("width", 6.0) or 6.0) for item in ordered)
        usable_width = max(target_width, max_item_width)
        placements: List[Tuple[Dict[str, Any], float, float]] = []
        x_cursor = 0.0
        y_cursor = 0.0
        shelf_height = 0.0
        used_width = 0.0
        prev_group = None

        for item in ordered:
            item_w = float(item.get("width", 6.0) or 6.0)
            item_h = float(item.get("height", 4.0) or 4.0)
            group_order = int(item.get("group_order", 0) or 0)
            gap = 0.0
            if placements:
                gap = _INNER_GAP_MM * (1.75 if prev_group is not None and group_order != prev_group else 1.0)
            if x_cursor > 0.0 and x_cursor + gap + item_w > usable_width:
                y_cursor += shelf_height + (_INNER_GAP_MM * 1.25)
                x_cursor = 0.0
                shelf_height = 0.0
                gap = 0.0
            x_cursor += gap
            placements.append((item, x_cursor + (item_w / 2.0), y_cursor + (item_h / 2.0)))
            x_cursor += item_w
            shelf_height = max(shelf_height, item_h)
            used_width = max(used_width, x_cursor)
            prev_group = group_order

        used_height = y_cursor + shelf_height
        return used_width, used_height, placements

    def _estimate_zone_layout(self, items: List[Dict[str, Any]]) -> Tuple[float, float]:
        if not items:
            return 0.0, 0.0

        _, _, _, _, grid_w, grid_h = self._interior_grid_dims(items)
        total_area = sum(
            (float(item.get("width", 6.0) or 6.0) + _INNER_GAP_MM)
            * (float(item.get("height", 4.0) or 4.0) + _INNER_GAP_MM)
            for item in items
        )
        max_item_w = max(float(item.get("width", 6.0) or 6.0) for item in items)
        span_w = sum(float(item.get("width", 6.0) or 6.0) for item in items) + _INNER_GAP_MM * max(0, len(items) - 1)
        base = max(max_item_w, math.sqrt(max(total_area, 1.0)) * 1.1)
        candidates = sorted(
            {
                round(max(max_item_w, candidate), 2)
                for candidate in (
                    grid_w * 0.9,
                    grid_w,
                    grid_w * 1.15,
                    base,
                    base * 1.2,
                    min(span_w, base * 1.5),
                )
                if candidate > 0.0
            }
        )

        best: Optional[Tuple[float, Tuple[float, float]]] = None
        for candidate_width in candidates:
            layout_w, layout_h, _ = self._layout_shelves(items, candidate_width)
            aspect = layout_w / max(layout_h, 1.0)
            score = abs(aspect - 1.15) * 18.0 + layout_w * 0.75 + layout_h * 0.45
            if best is None or score < best[0]:
                best = (score, (layout_w, layout_h))
        return best[1] if best else (grid_w, grid_h)

    def _split_long_edge_headers(
        self,
        headers: List[Dict[str, Any]],
        top_items: Optional[List[Dict[str, Any]]] = None,
        bottom_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not headers:
            return [], []
        ordered = sorted(headers, key=self._header_edge_rank)
        top: List[Dict[str, Any]] = []
        bottom: List[Dict[str, Any]] = []
        top_load = self._stack_span(top_items or [], "top")
        bottom_load = self._stack_span(bottom_items or [], "bottom")
        for item in ordered:
            span = self._oriented_dims(item, "top")[0]
            if len(top) > len(bottom):
                if bottom:
                    bottom_load += self._edge_item_gap(bottom[-1], item, "bottom")
                bottom.append(item)
                bottom_load += span
                continue
            if len(bottom) > len(top):
                if top:
                    top_load += self._edge_item_gap(top[-1], item, "top")
                top.append(item)
                top_load += span
                continue

            projected_top = top_load + span + (self._edge_item_gap(top[-1], item, "top") if top else 0.0)
            projected_bottom = bottom_load + span + (self._edge_item_gap(bottom[-1], item, "bottom") if bottom else 0.0)
            if projected_top <= projected_bottom:
                if top:
                    top_load += self._edge_item_gap(top[-1], item, "top")
                top.append(item)
                top_load += span
            else:
                if bottom:
                    bottom_load += self._edge_item_gap(bottom[-1], item, "bottom")
                bottom.append(item)
                bottom_load += span
        return top, bottom

    def _place_small_headers_inside(
        self,
        items: List[Dict[str, Any]],
        placement: Dict[str, Dict[str, float]],
        inner_right: float,
        outline_center_y: float,
    ) -> None:
        if not items:
            return
        ordered = sorted(items, key=lambda item: (item["group_order"], item["ref"]))
        spans = [max(self._oriented_dims(item)) for item in ordered]
        total_span = sum(spans) + _EDGE_GAP_MM * max(0, len(spans) - 1)
        cursor = outline_center_y - (total_span / 2.0)
        max_width = max(min(self._oriented_dims(item)) for item in ordered)
        x = inner_right - (max_width / 2.0) - 5.5
        for item, span in zip(ordered, spans):
            placement[item["ref"]] = {
                "x": round(x, 3),
                "y": round(cursor + (span / 2.0), 3),
                "rot": 0.0,
            }
            cursor += span + _EDGE_GAP_MM

    def _interior_zone(self, item: Dict[str, Any]) -> str:
        ref = str(item.get("ref", "") or "")
        text = str(item.get("text", "") or "")
        category = str(item.get("category", "") or "")
        net_names = self._item_net_names(item)
        has_power_net = any(self._net_name_is_power(net_name) for net_name in net_names)
        if category == "interface_support" or ref.startswith(("RV", "FB")) or any(
            token in text for token in ("polyfuse", "fuse", "esd", "tvs", "varistor", "clamp", "interface")
        ):
            return "usb"
        if category in {"active", "interface_support"} and (
            has_power_net
            or any(token in text for token in ("regulator", "vin", "3v3", "+3v3", "5v", "+5v", "mosfet", "power"))
        ):
            return "power"
        return "main"

    def _placement_dims(self, item: Dict[str, Any], pos: Dict[str, Any]) -> Tuple[float, float]:
        edge = str(item.get("placed_edge", "") or "")
        if edge in {"left", "right", "top", "bottom"}:
            return self._oriented_dims(item, edge)

        w = float(item.get("width", 6.0) or 6.0)
        h = float(item.get("height", 4.0) or 4.0)
        rot = abs(float(pos.get("rot", 0.0) or 0.0)) % 180.0
        if abs(rot - 90.0) < 1e-3:
            return (h, w)
        return (w, h)

    def _placement_rect(
        self,
        item: Dict[str, Any],
        pos: Dict[str, Any],
        clearance: float = _PLACEMENT_CLEARANCE_MM,
    ) -> Tuple[float, float, float, float]:
        w, h = self._placement_dims(item, pos)
        x = float(pos.get("x", 0.0) or 0.0)
        y = float(pos.get("y", 0.0) or 0.0)
        half_w = (w + clearance) / 2.0
        half_h = (h + clearance) / 2.0
        return (x - half_w, y - half_h, x + half_w, y + half_h)

    def _rects_overlap(
        self,
        rect_a: Tuple[float, float, float, float],
        rect_b: Tuple[float, float, float, float],
    ) -> bool:
        return not (
            rect_a[2] <= rect_b[0]
            or rect_a[0] >= rect_b[2]
            or rect_a[3] <= rect_b[1]
            or rect_a[1] >= rect_b[3]
        )

    def _candidate_positions(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        item_w: float,
        item_h: float,
        anchor_x: float,
        anchor_y: float,
    ) -> List[Tuple[float, float]]:
        min_x = left + (item_w / 2.0)
        max_x = right - (item_w / 2.0)
        min_y = top + (item_h / 2.0)
        max_y = bottom - (item_h / 2.0)
        if max_x < min_x or max_y < min_y:
            return []

        step = max(1.27, min(3.0, min(item_w, item_h, 6.0) / 1.8))
        xs: List[float] = []
        ys: List[float] = []

        cursor_x = min_x
        while cursor_x <= max_x + 1e-6:
            xs.append(round(cursor_x, 3))
            cursor_x += step
        cursor_y = min_y
        while cursor_y <= max_y + 1e-6:
            ys.append(round(cursor_y, 3))
            cursor_y += step

        anchor_x = round(min(max(anchor_x, min_x), max_x), 3)
        anchor_y = round(min(max(anchor_y, min_y), max_y), 3)
        if anchor_x not in xs:
            xs.append(anchor_x)
        if anchor_y not in ys:
            ys.append(anchor_y)

        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        candidates = [(x, y) for x in xs for y in ys]
        candidates.sort(
            key=lambda pt: (
                round(abs(pt[0] - anchor_x) + abs(pt[1] - anchor_y), 4),
                round(abs(pt[0] - center_x) + abs(pt[1] - center_y), 4),
                round(abs(pt[1] - anchor_y), 4),
                round(abs(pt[0] - anchor_x), 4),
                pt[1],
                pt[0],
            )
        )
        return candidates

    def _edge_anchor(
        self,
        ref: str,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        fallback: Tuple[float, float],
        inward: float = 12.0,
    ) -> Tuple[float, float]:
        item = part_map.get(ref)
        pos = placement.get(ref)
        if item is None or pos is None:
            return fallback

        edge = str(item.get("placed_edge", "") or "")
        w, h = self._placement_dims(item, pos)
        x = float(pos.get("x", 0.0) or 0.0)
        y = float(pos.get("y", 0.0) or 0.0)
        if edge == "left":
            return (x + (w / 2.0) + inward, y)
        if edge == "right":
            return (x - (w / 2.0) - inward, y)
        if edge == "top":
            return (x, y + (h / 2.0) + inward)
        if edge == "bottom":
            return (x, y - (h / 2.0) - inward)
        return fallback

    def _zone_anchor(
        self,
        zone: str,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        core_left: float,
        core_top: float,
        core_right: float,
        core_bottom: float,
    ) -> Tuple[float, float]:
        core_w = max(1.0, core_right - core_left)
        core_h = max(1.0, core_bottom - core_top)
        center_x = core_left + (core_w / 2.0)
        center_y = core_top + (core_h / 2.0)

        def pick_port(kind: Optional[str] = None) -> str:
            for ref, item in part_map.items():
                if item.get("category") != "port":
                    continue
                if kind and str(item.get("port_kind", "") or "") != kind:
                    continue
                return str(ref)
            return ""

        any_port = pick_port(None)
        usb_port = pick_port("usb") or any_port
        power_port = pick_port("power") or any_port

        if zone == "usb":
            fallback = (core_left + core_w * 0.28, core_top + core_h * 0.32)
            return self._edge_anchor(usb_port, placement, part_map, fallback, inward=max(12.0, core_w * 0.08))
        if zone == "power":
            fallback = (core_left + core_w * 0.30, core_top + core_h * 0.72)
            return self._edge_anchor(power_port, placement, part_map, fallback, inward=max(12.0, core_w * 0.08))
        return (center_x, center_y)

    def _zone_place_key(self, item: Dict[str, Any]) -> Tuple[int, int, float, float, str]:
        category_rank = {
            "active": 0,
            "clock": 1,
            "interface_support": 2,
            "support": 3,
        }
        return (
            category_rank.get(str(item.get("category", "") or ""), 4),
            int(item.get("group_order", 0) or 0),
            -max(float(item.get("width", 0.0) or 0.0), float(item.get("height", 0.0) or 0.0)),
            -min(float(item.get("width", 0.0) or 0.0), float(item.get("height", 0.0) or 0.0)),
            str(item.get("ref", "") or ""),
        )

    def _approx_overlap_pairs(
        self,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        refs: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        check_refs = refs if refs is not None else list(placement.keys())
        pairs: List[Tuple[str, str]] = []
        for idx, ref_a in enumerate(check_refs):
            item_a = part_map.get(ref_a)
            pos_a = placement.get(ref_a)
            if item_a is None or pos_a is None:
                continue
            rect_a = self._placement_rect(item_a, pos_a)
            for ref_b in check_refs[idx + 1:]:
                item_b = part_map.get(ref_b)
                pos_b = placement.get(ref_b)
                if item_b is None or pos_b is None:
                    continue
                rect_b = self._placement_rect(item_b, pos_b)
                if self._rects_overlap(rect_a, rect_b):
                    pairs.append((ref_a, ref_b))
        return pairs

    def _try_place_interior(
        self,
        zones: Dict[str, List[Dict[str, Any]]],
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        core_left: float,
        core_top: float,
        core_right: float,
        core_bottom: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        core_center_x = (core_left + core_right) / 2.0
        core_center_y = (core_top + core_bottom) / 2.0
        anchors = {
            zone: self._zone_anchor(zone, placement, part_map, core_left, core_top, core_right, core_bottom)
            for zone in zones
        }
        placed_by_zone: Dict[str, List[str]] = {zone: [] for zone in zones}
        zone_quadrants: Dict[str, Dict[str, int]] = {
            zone: {"tl": 0, "tr": 0, "bl": 0, "br": 0}
            for zone in zones
        }

        def quadrant_for(x: float, y: float) -> str:
            horizontal = "l" if x < core_center_x else "r"
            vertical = "t" if y < core_center_y else "b"
            return vertical + horizontal

        for zone in ("usb", "power", "main"):
            items = sorted(zones.get(zone, []), key=self._zone_place_key)
            for item in items:
                ref = item["ref"]
                item_w = float(item.get("width", 6.0) or 6.0)
                item_h = float(item.get("height", 4.0) or 4.0)
                anchor_x, anchor_y = anchors[zone]

                if placed_by_zone[zone]:
                    same_zone_positions = [
                        placement[placed_ref]
                        for placed_ref in placed_by_zone[zone]
                        if placed_ref in placement
                    ]
                    if same_zone_positions:
                        centroid_x = sum(float(pos.get("x", 0.0) or 0.0) for pos in same_zone_positions) / len(same_zone_positions)
                        centroid_y = sum(float(pos.get("y", 0.0) or 0.0) for pos in same_zone_positions) / len(same_zone_positions)
                        anchor_x = (anchor_x * 0.55) + (centroid_x * 0.45)
                        anchor_y = (anchor_y * 0.55) + (centroid_y * 0.45)

                candidates = self._candidate_positions(
                    core_left,
                    core_top,
                    core_right,
                    core_bottom,
                    item_w + _PLACEMENT_CLEARANCE_MM,
                    item_h + _PLACEMENT_CLEARANCE_MM,
                    anchor_x,
                    anchor_y,
                )

                best: Optional[Tuple[float, float, float]] = None
                for cand_x, cand_y in candidates:
                    candidate_pos = {"x": cand_x, "y": cand_y, "rot": 0.0}
                    candidate_rect = self._placement_rect(item, candidate_pos)

                    blocked = False
                    crowding = 0.0
                    for other_ref, other_pos in placement.items():
                        other_item = part_map.get(other_ref)
                        if other_item is None:
                            continue
                        other_rect = self._placement_rect(other_item, other_pos)
                        if self._rects_overlap(candidate_rect, other_rect):
                            blocked = True
                            break
                        dx = cand_x - float(other_pos.get("x", 0.0) or 0.0)
                        dy = cand_y - float(other_pos.get("y", 0.0) or 0.0)
                        distance = math.hypot(dx, dy)
                        if distance < 18.0:
                            weight = 1.25 if other_item.get("zone") == zone else 0.55
                            crowding += (18.0 - distance) * weight
                    if blocked:
                        continue

                    quad = quadrant_for(cand_x, cand_y)
                    same_zone_quad = zone_quadrants[zone][quad]
                    main_bias = 0.0
                    if zone == "main":
                        left_count = zone_quadrants[zone]["tl"] + zone_quadrants[zone]["bl"]
                        right_count = zone_quadrants[zone]["tr"] + zone_quadrants[zone]["br"]
                        if left_count > right_count + 1 and cand_x < core_center_x:
                            main_bias += 7.5
                        if right_count > left_count + 1 and cand_x >= core_center_x:
                            main_bias += 7.5

                    distance_score = math.hypot((cand_x - anchor_x) / 12.0, (cand_y - anchor_y) / 10.0) * 5.0
                    center_pull = math.hypot((cand_x - core_center_x) / 18.0, (cand_y - core_center_y) / 18.0)
                    if zone != "main":
                        center_pull *= 0.35

                    score = distance_score + crowding + center_pull + (same_zone_quad * 2.4) + main_bias
                    if best is None or score < best[0]:
                        best = (score, cand_x, cand_y)

                if best is None:
                    return False, {
                        "reason": "no_candidate",
                        "zone": zone,
                        "ref": ref,
                        "anchors": anchors,
                    }

                _, x, y = best
                placement[ref] = {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "rot": 0.0,
                    "zone": zone,
                    "category": item.get("category"),
                    "edge": None,
                    "group_order": item.get("group_order"),
                }
                placed_by_zone[zone].append(ref)
                zone_quadrants[zone][quadrant_for(x, y)] += 1

        return True, {"anchors": anchors, "placed_by_zone": placed_by_zone}

    def _pack_rect(
        self,
        items: List[Dict[str, Any]],
        left: float,
        top: float,
        right: float,
        bottom: float,
        placement: Dict[str, Dict[str, float]],
    ) -> None:
        if not items or right <= left or bottom <= top:
            return
        avail_w = max(0.0, right - left)
        avail_h = max(0.0, bottom - top)
        zone_w, zone_h, shelf_layout = self._layout_shelves(items, avail_w)
        start_x = left + max(0.0, (avail_w - zone_w) / 2.0)
        start_y = top + max(0.0, (avail_h - zone_h) / 2.0)
        for item, rel_x, rel_y in shelf_layout:
            placement[item["ref"]] = {
                "x": round(start_x + rel_x, 3),
                "y": round(start_y + rel_y, 3),
                "rot": 0.0,
            }

    def _relax_movable_parts(
        self,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        movable_refs: List[str],
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> None:
        movable = [ref for ref in movable_refs if ref in placement and ref in part_map]
        if not movable:
            return

        def dims(ref: str) -> Tuple[float, float]:
            item = part_map[ref]
            return (
                float(item.get("width", 6.0) or 6.0) + _PLACEMENT_CLEARANCE_MM,
                float(item.get("height", 4.0) or 4.0) + _PLACEMENT_CLEARANCE_MM,
            )

        for _ in range(80):
            moved = False
            for i, ref_a in enumerate(movable):
                pos_a = placement[ref_a]
                ax, ay = float(pos_a.get("x", 0.0) or 0.0), float(pos_a.get("y", 0.0) or 0.0)
                aw, ah = dims(ref_a)
                for ref_b, pos_b in placement.items():
                    if ref_b == ref_a or ref_b not in part_map:
                        continue
                    bx, by = float(pos_b.get("x", 0.0) or 0.0), float(pos_b.get("y", 0.0) or 0.0)
                    bw, bh = dims(ref_b)
                    overlap_x = ((aw + bw) / 2.0) - abs(ax - bx)
                    overlap_y = ((ah + bh) / 2.0) - abs(ay - by)
                    if overlap_x <= 0.0 or overlap_y <= 0.0:
                        continue
                    if overlap_x < overlap_y:
                        delta = overlap_x / 2.0 + 0.25
                        ax += delta if ax >= bx else -delta
                    else:
                        delta = overlap_y / 2.0 + 0.25
                        ay += delta if ay >= by else -delta
                    moved = True
                half_w = aw / 2.0
                half_h = ah / 2.0
                ax = min(max(ax, min_x + half_w), max_x - half_w)
                ay = min(max(ay, min_y + half_h), max_y - half_h)
                updated = dict(pos_a)
                updated["x"] = round(ax, 3)
                updated["y"] = round(ay, 3)
                updated["rot"] = float(pos_a.get("rot", 0.0) or 0.0)
                placement[ref_a] = updated
            if not moved:
                break

    def _is_active_like(self, item: Dict[str, Any]) -> bool:
        category = str(item.get("category", "") or "")
        if category == "active":
            return True
        text = str(item.get("text", "") or "")
        return any(
            token in text
            for token in (
                "mcu",
                "microcontroller",
                "controller",
                "regulator",
                "ldo",
                "op amp",
                "amplifier",
                "mosfet",
                "driver",
                "bridge",
            )
        )

    def _is_small_companion(self, item: Dict[str, Any]) -> bool:
        if str(item.get("category", "") or "") not in {"support", "interface_support"}:
            return False
        if str(item.get("placed_edge", "") or "") in {"left", "right", "top", "bottom"}:
            return False
        w = float(item.get("width", 0.0) or 0.0)
        h = float(item.get("height", 0.0) or 0.0)
        return max(w, h) <= _COMPANION_MAX_BODY_MM

    @staticmethod
    def _normalize_net_name(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9_+\-./]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text

    def _item_net_names(self, item: Dict[str, Any]) -> set[str]:
        part = item.get("part") if isinstance(item.get("part"), dict) else {}
        nets: set[str] = set()
        for pin in list(part.get("pins") or []):
            if not isinstance(pin, dict):
                continue
            net = self._normalize_net_name(pin.get("net"))
            if net:
                nets.add(net)
        for sec in list(part.get("support_candidates") or []):
            if not isinstance(sec, dict):
                continue
            net_hint = self._normalize_net_name(sec.get("net_hint"))
            if net_hint:
                nets.add(net_hint)
            for net_name in list(sec.get("source_nets") or []):
                net = self._normalize_net_name(net_name)
                if net:
                    nets.add(net)
        return nets

    def _net_name_is_ground(self, net_name: str) -> bool:
        net = self._normalize_net_name(net_name)
        if not net:
            return False
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", net) if tok}
        return ("gnd" in tokens) or (net in {"agnd", "dgnd", "pgnd", "earth"})

    def _net_name_is_power(self, net_name: str) -> bool:
        net = self._normalize_net_name(net_name)
        if not net or self._net_name_is_ground(net):
            return False
        tokens = {tok for tok in re.split(r"[^a-z0-9]+", net) if tok}
        if tokens & {"vin", "vcc", "vdd", "vss", "avcc", "dvcc", "iovcc", "iovdd", "vio"}:
            return True
        if net in {"5v", "5v0", "3v3", "3v30", "plus5v", "plus3v3"}:
            return True
        if re.match(r"^(?:plus)?\d+v\d*$", net):
            return True
        if re.match(r"^v\d+(?:_\d+)?$", net):
            return True
        return False

    def _build_repair_anchor_sets(
        self,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        anchors: Dict[str, List[str]] = {"active": [], "usb": [], "power": []}
        for ref in sorted(placement.keys()):
            item = part_map.get(ref)
            if item is None:
                continue
            category = str(item.get("category", "") or "")
            zone = str(item.get("zone", "") or "")
            net_names = self._item_net_names(item)
            is_active = self._is_active_like(item)
            if is_active:
                anchors["active"].append(ref)

            is_usb = zone == "usb"
            is_power = (zone == "power") or any(self._net_name_is_power(net_name) for net_name in net_names)

            if is_usb and (is_active or category in {"port", "interface_support"}):
                anchors["usb"].append(ref)
            if is_power and (is_active or category == "port" or category == "interface_support"):
                anchors["power"].append(ref)

        if not anchors["usb"]:
            anchors["usb"] = list(anchors["active"])
        if not anchors["power"]:
            anchors["power"] = list(anchors["active"])
        return anchors

    def _choose_anchor_for_ref(
        self,
        ref: str,
        item: Dict[str, Any],
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        anchors: Dict[str, List[str]],
    ) -> str:
        current_pos = placement.get(ref)
        if current_pos is None:
            return ""

        zone = str(item.get("zone", "") or "")
        item_nets = {
            net_name
            for net_name in self._item_net_names(item)
            if not self._net_name_is_ground(net_name)
        }
        candidates = list(anchors.get("active") or [])
        if zone == "usb" and anchors.get("usb"):
            candidates = list(anchors.get("usb") or candidates)
        elif (zone == "power" or any(self._net_name_is_power(net_name) for net_name in item_nets)) and anchors.get("power"):
            candidates = list(anchors.get("power") or candidates)
        if not candidates:
            return ""

        same_zone = [
            cand
            for cand in candidates
            if str((part_map.get(cand) or {}).get("zone", "") or "") == zone
        ]
        if same_zone:
            candidates = same_zone

        x = float(current_pos.get("x", 0.0) or 0.0)
        y = float(current_pos.get("y", 0.0) or 0.0)

        best_ref = ""
        best_key: Optional[Tuple[int, int, float, str]] = None
        for cand in candidates:
            if cand == ref:
                continue
            pos = placement.get(cand)
            if pos is None:
                continue
            cand_item = part_map.get(cand) or {}
            cand_nets = {
                net_name
                for net_name in self._item_net_names(cand_item)
                if not self._net_name_is_ground(net_name)
            }
            net_overlap = len(item_nets & cand_nets) if item_nets and cand_nets else 0
            power_overlap = len(
                {
                    net_name
                    for net_name in (item_nets & cand_nets)
                    if self._net_name_is_power(net_name)
                }
            )
            d = math.hypot(float(pos.get("x", 0.0) or 0.0) - x, float(pos.get("y", 0.0) or 0.0) - y)
            key = (-net_overlap, -power_overlap, round(d, 4), cand)
            if best_key is None or key < best_key:
                best_key = key
                best_ref = cand
        return best_ref

    def _placement_quality_cost(self, report: Dict[str, Any]) -> float:
        overlaps = int(report.get("overlap_count", 0) or 0)
        out_of_bounds = int(report.get("out_of_bounds_count", 0) or 0)
        clock_far = int(report.get("clock_far_count", 0) or 0)
        companion_far = int(report.get("companion_far_count", 0) or 0)
        return (overlaps * 1000.0) + (out_of_bounds * 1000.0) + (clock_far * 25.0) + (companion_far * 6.0)

    def _verify_placement_quality(
        self,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        board_rect: Tuple[float, float, float, float],
    ) -> Dict[str, Any]:
        overlaps = self._approx_overlap_pairs(placement, part_map)
        out_of_bounds: List[str] = []
        board_left, board_top, board_right, board_bottom = board_rect
        for ref in sorted(placement.keys()):
            item = part_map.get(ref)
            pos = placement.get(ref)
            if item is None or pos is None:
                continue
            rect = self._placement_rect(item, pos, clearance=0.2)
            if rect[0] < board_left or rect[2] > board_right or rect[1] < board_top or rect[3] > board_bottom:
                out_of_bounds.append(ref)

        anchors = self._build_repair_anchor_sets(placement, part_map)
        active_refs = list(anchors.get("active") or [])
        clock_far: List[Dict[str, Any]] = []
        companion_far: List[Dict[str, Any]] = []

        if active_refs:
            for ref in sorted(placement.keys()):
                item = part_map.get(ref)
                pos = placement.get(ref)
                if item is None or pos is None:
                    continue
                category = str(item.get("category", "") or "")
                if category == "clock":
                    anchor_ref = self._choose_anchor_for_ref(ref, item, placement, part_map, anchors)
                    if not anchor_ref:
                        continue
                    anchor_pos = placement.get(anchor_ref) or {}
                    distance = math.hypot(
                        float(pos.get("x", 0.0) or 0.0) - float(anchor_pos.get("x", 0.0) or 0.0),
                        float(pos.get("y", 0.0) or 0.0) - float(anchor_pos.get("y", 0.0) or 0.0),
                    )
                    if distance > _CLOCK_TARGET_MAX_MM:
                        clock_far.append(
                            {
                                "ref": ref,
                                "anchor_ref": anchor_ref,
                                "distance_mm": round(distance, 3),
                            }
                        )

                if self._is_small_companion(item):
                    anchor_ref = self._choose_anchor_for_ref(ref, item, placement, part_map, anchors)
                    if not anchor_ref:
                        continue
                    anchor_pos = placement.get(anchor_ref) or {}
                    distance = math.hypot(
                        float(pos.get("x", 0.0) or 0.0) - float(anchor_pos.get("x", 0.0) or 0.0),
                        float(pos.get("y", 0.0) or 0.0) - float(anchor_pos.get("y", 0.0) or 0.0),
                    )
                    if distance > _COMPANION_TARGET_MAX_MM:
                        companion_far.append(
                            {
                                "ref": ref,
                                "anchor_ref": anchor_ref,
                                "distance_mm": round(distance, 3),
                            }
                        )

        report = {
            "overlap_count": len(overlaps),
            "overlap_pairs_sample": overlaps[:12],
            "out_of_bounds_count": len(out_of_bounds),
            "out_of_bounds_sample": out_of_bounds[:12],
            "clock_far_count": len(clock_far),
            "clock_far_sample": clock_far[:12],
            "companion_far_count": len(companion_far),
            "companion_far_sample": companion_far[:20],
            "anchor_active_count": len(active_refs),
        }
        report["fatal"] = bool(report["overlap_count"] or report["out_of_bounds_count"])
        report["needs_repair"] = bool(
            report["fatal"]
            or report["clock_far_count"]
            or report["companion_far_count"]
        )
        report["quality_cost"] = round(self._placement_quality_cost(report), 3)
        return report

    def _position_legal(
        self,
        ref: str,
        candidate_pos: Dict[str, float],
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        board_rect: Tuple[float, float, float, float],
        core_rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        item = part_map.get(ref)
        if item is None:
            return False

        rect = self._placement_rect(item, candidate_pos)
        board_left, board_top, board_right, board_bottom = board_rect
        if rect[0] < board_left or rect[2] > board_right or rect[1] < board_top or rect[3] > board_bottom:
            return False

        if core_rect is not None:
            core_left, core_top, core_right, core_bottom = core_rect
            if rect[0] < core_left or rect[2] > core_right or rect[1] < core_top or rect[3] > core_bottom:
                return False

        for other_ref, other_pos in placement.items():
            if other_ref == ref:
                continue
            other_item = part_map.get(other_ref)
            if other_item is None:
                continue
            other_rect = self._placement_rect(other_item, other_pos)
            if self._rects_overlap(rect, other_rect):
                return False
        return True

    def _move_ref_near_anchor(
        self,
        ref: str,
        anchor_ref: str,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        board_rect: Tuple[float, float, float, float],
        core_rect: Optional[Tuple[float, float, float, float]],
        target_radius_mm: float,
    ) -> bool:
        pos = placement.get(ref)
        anchor_pos = placement.get(anchor_ref)
        if pos is None or anchor_pos is None:
            return False

        current_x = float(pos.get("x", 0.0) or 0.0)
        current_y = float(pos.get("y", 0.0) or 0.0)
        anchor_x = float(anchor_pos.get("x", 0.0) or 0.0)
        anchor_y = float(anchor_pos.get("y", 0.0) or 0.0)

        best_pos = dict(pos)
        best_score = None
        radii = sorted(
            {
                max(2.5, target_radius_mm - 2.5),
                target_radius_mm,
                target_radius_mm + 2.0,
                target_radius_mm + 4.0,
            }
        )
        angles = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)

        candidates: List[Tuple[float, float]] = [(current_x, current_y)]
        for radius in radii:
            for angle in angles:
                radians = math.radians(angle)
                cx = anchor_x + (radius * math.cos(radians))
                cy = anchor_y + (radius * math.sin(radians))
                candidates.append((round(cx, 3), round(cy, 3)))

        for cx, cy in candidates:
            candidate = dict(pos)
            candidate["x"] = float(cx)
            candidate["y"] = float(cy)
            if not self._position_legal(ref, candidate, placement, part_map, board_rect, core_rect):
                continue

            dist_anchor = math.hypot(cx - anchor_x, cy - anchor_y)
            move_dist = math.hypot(cx - current_x, cy - current_y)
            crowding = 0.0
            for other_ref, other_pos in placement.items():
                if other_ref == ref:
                    continue
                ox = float(other_pos.get("x", 0.0) or 0.0)
                oy = float(other_pos.get("y", 0.0) or 0.0)
                d = math.hypot(cx - ox, cy - oy)
                if d < 14.0:
                    crowding += (14.0 - d) * 0.2
            score = (abs(dist_anchor - target_radius_mm) * 4.0) + (move_dist * 0.2) + crowding
            if best_score is None or score < best_score:
                best_score = score
                best_pos = candidate

        changed = (
            round(float(best_pos.get("x", 0.0) or 0.0), 3) != round(current_x, 3)
            or round(float(best_pos.get("y", 0.0) or 0.0), 3) != round(current_y, 3)
        )
        if changed:
            placement[ref] = best_pos
        return changed

    def _repair_placement_once(
        self,
        placement: Dict[str, Dict[str, float]],
        part_map: Dict[str, Dict[str, Any]],
        board_rect: Tuple[float, float, float, float],
        core_rect: Tuple[float, float, float, float],
    ) -> bool:
        core_left, core_top, core_right, core_bottom = core_rect
        movable_refs = [
            ref
            for ref in sorted(placement.keys())
            if ref in part_map and str(part_map[ref].get("placed_edge", "") or "") not in {"left", "right", "top", "bottom"}
        ]
        if not movable_refs:
            return False

        changed = False
        overlap_before = len(self._approx_overlap_pairs(placement, part_map))
        self._relax_movable_parts(placement, part_map, movable_refs, core_left, core_right, core_top, core_bottom)
        overlap_after = len(self._approx_overlap_pairs(placement, part_map))
        if overlap_after < overlap_before:
            changed = True

        anchors = self._build_repair_anchor_sets(placement, part_map)
        if not anchors.get("active"):
            return changed

        clock_candidates: List[Tuple[float, str, str]] = []
        companion_candidates: List[Tuple[float, str, str]] = []
        for ref in movable_refs:
            item = part_map.get(ref)
            pos = placement.get(ref)
            if item is None or pos is None:
                continue
            anchor_ref = self._choose_anchor_for_ref(ref, item, placement, part_map, anchors)
            if not anchor_ref:
                continue
            anchor_pos = placement.get(anchor_ref)
            if anchor_pos is None:
                continue
            distance = math.hypot(
                float(pos.get("x", 0.0) or 0.0) - float(anchor_pos.get("x", 0.0) or 0.0),
                float(pos.get("y", 0.0) or 0.0) - float(anchor_pos.get("y", 0.0) or 0.0),
            )
            if str(item.get("category", "") or "") == "clock" and distance > _CLOCK_TARGET_MAX_MM:
                clock_candidates.append((distance, ref, anchor_ref))
            elif self._is_small_companion(item) and distance > _COMPANION_TARGET_MAX_MM:
                companion_candidates.append((distance, ref, anchor_ref))

        clock_candidates.sort(key=lambda row: row[0], reverse=True)
        companion_candidates.sort(key=lambda row: row[0], reverse=True)

        for _, ref, anchor_ref in clock_candidates:
            if self._move_ref_near_anchor(
                ref,
                anchor_ref,
                placement,
                part_map,
                board_rect,
                core_rect,
                target_radius_mm=9.0,
            ):
                changed = True

        for _, ref, anchor_ref in companion_candidates[:24]:
            if self._move_ref_near_anchor(
                ref,
                anchor_ref,
                placement,
                part_map,
                board_rect,
                core_rect,
                target_radius_mm=10.5,
            ):
                changed = True

        if changed:
            self._relax_movable_parts(placement, part_map, movable_refs, core_left, core_right, core_top, core_bottom)

        return changed

    # ── LLM zoning ────────────────────────────────────────────────────────

    def _get_llm_zoning_plan(
        self, parts: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        if not self._llm_client:
            logger.warning("No LLM client — using single-group fallback.")
            return {"groups": [{"name": "Default", "refs": [p.get("ref") for p in parts]}]}, "No LLM."

        sys_prompt = (
            "You are the Zoning Planner for a PCB design tool.\n"
            "Break the component manifest into logical functional groups "
            "(e.g. 'MCU', 'Power', 'Connectors', 'Passives').\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"groups":[{"name":"Group_Name","refs":["U1","C1",...]}]}\n'
            "Every ref MUST appear in exactly one group."
        )
        refs_str = ", ".join(
            f"{p.get('ref','')} ({p.get('mpn','?')})"
            for p in parts if p.get("ref")
        )
        try:
            from ...llm.client import LLMMessage
            resp = self._llm_client.chat(
                [LLMMessage(role="user", content=f"Component Manifest:\n{refs_str}")],
                system_prompt=sys_prompt,
            )
            content = resp.content or ""
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                plan = json.loads(m.group(0))
                if isinstance(plan.get("groups"), list):
                    return plan, content
        except Exception as e:
            logger.error("LLM zoning failed: %s", e)

        fallback = {"groups": [{"name": "Default", "refs": [p.get("ref") for p in parts]}]}
        return fallback, "Fallback zoning applied."

    # ── Deterministic packer ───────────────────────────────────────────────

    def _pack_components(
        self,
        classified_parts: List[Dict[str, Any]],
    ) -> Tuple[float, float, Dict[str, Dict[str, float]], Dict[str, Any]]:
        """Place edge-facing parts on the perimeter and anchor-place the interior.

        This remains board-agnostic: decisions are based only on generic
        package/category inference, never on a board template.
        """
        placement: Dict[str, Dict[str, float]] = {}

        left_edge = [p for p in classified_parts if p.get("preferred_edge") == "left"]
        right_edge = [p for p in classified_parts if p.get("preferred_edge") == "right"]
        top_edge = [p for p in classified_parts if p.get("preferred_edge") == "top" and p.get("category") != "long_header"]
        bottom_edge = [p for p in classified_parts if p.get("preferred_edge") == "bottom"]
        long_headers = [p for p in classified_parts if p.get("category") == "long_header"]
        balanced_top, balanced_bottom = self._split_long_edge_headers(long_headers, top_edge, bottom_edge)
        top_edge.extend(balanced_top)
        bottom_edge.extend(balanced_bottom)
        interior = [p for p in classified_parts if p.get("region") != "edge"]
        for item in interior:
            item["zone"] = self._interior_zone(item)
        for item in left_edge + right_edge + top_edge + bottom_edge:
            item["zone"] = "edge"
        usb_zone = [p for p in interior if p.get("zone") == "usb"]
        power_zone = [p for p in interior if p.get("zone") == "power"]
        main_zone = [p for p in interior if p.get("zone") == "main"]
        part_map = {item["ref"]: item for item in classified_parts}

        usb_w, usb_h = self._estimate_zone_layout(usb_zone)
        power_w, power_h = self._estimate_zone_layout(power_zone)
        main_w, main_h = self._estimate_zone_layout(main_zone)
        interior_area = sum(
            (float(item.get("width", 6.0) or 6.0) + _INNER_GAP_MM)
            * (float(item.get("height", 4.0) or 4.0) + _INNER_GAP_MM)
            for item in interior
        )
        base_span = math.sqrt(max(interior_area, 1.0))
        top_span = self._stack_span(top_edge, "top")
        bottom_span = self._stack_span(bottom_edge, "bottom")
        left_span = self._stack_span(left_edge, "left")
        right_span = self._stack_span(right_edge, "right")

        left_band = self._max_depth(left_edge, "left")
        right_band = self._max_depth(right_edge, "right")
        top_band = self._max_depth(top_edge, "top")
        bottom_band = self._max_depth(bottom_edge, "bottom")

        left_band += self._edge_to_core_gap(left_edge)
        right_band += self._edge_to_core_gap(right_edge)
        top_band += self._edge_to_core_gap(top_edge)
        bottom_band += self._edge_to_core_gap(bottom_edge)

        inner_w = max(top_span, bottom_span, main_w + max(usb_w, power_w) * 0.9, base_span * 1.45, 52.0)
        inner_h = max(left_span, right_span, main_h + max(usb_h, power_h) * 0.55, base_span * 1.18, 58.0)
        final_outline_w = 0.0
        final_outline_h = 0.0
        final_placement: Dict[str, Dict[str, float]] = {}
        placement_stats: Dict[str, Any] = {
            "outline_pass_budget": _PACK_OUTLINE_MAX_PASSES,
            "verify_repair_pass_budget": _VERIFY_REPAIR_MAX_PASSES,
            "outline_passes": 0,
            "verify_repair_passes": 0,
            "final_verify": {},
            "used_fallback": False,
        }
        outline_w = 0.0
        outline_h = 0.0

        def place_edge(items: List[Dict[str, Any]], edge: str) -> None:
            if not items:
                return
            total_span = self._stack_span(items, edge)
            if edge in {"top", "bottom"}:
                available = max(0.0, inner_w - (2.0 * _EDGE_END_MARGIN_MM))
                cursor = inner_left + _EDGE_END_MARGIN_MM + max(0.0, (available - total_span) / 2.0)
            else:
                available = max(0.0, inner_h - (2.0 * _EDGE_END_MARGIN_MM))
                cursor = inner_top + _EDGE_END_MARGIN_MM + max(0.0, (available - total_span) / 2.0)
            for idx, item in enumerate(items):
                ref = item["ref"]
                w, h = self._oriented_dims(item, edge)
                span = w if edge in {"top", "bottom"} else h
                depth = h if edge in {"top", "bottom"} else w
                rot = self._edge_rotation(edge, item)
                inset = self._edge_inset(item, edge)
                outward_depth, inward_depth = self._port_edge_depths(item, edge, rot)
                outward_depth_val = round(float(outward_depth), 3) if isinstance(outward_depth, (int, float)) else None
                inward_depth_val = round(float(inward_depth), 3) if isinstance(inward_depth, (int, float)) else None
                resolved_face_dir = item.get("port_face_dir")
                resolved_face_dir_source = item.get("port_face_dir_source")
                if str(item.get("category", "") or "") == "port":
                    outward_dir = self._outward_dir_for_edge(edge)
                    local_face = self._local_dir_for_board_dir(outward_dir, rot)
                    if local_face in {"px", "py", "nx", "ny"}:
                        resolved_face_dir = local_face
                        if local_face != item.get("port_face_dir"):
                            resolved_face_dir_source = "rotation_resolved"
                item["placed_edge"] = edge
                if edge == "left":
                    x = origin_x + _PERIMETER_MARGIN_MM + inset + (depth / 2.0)
                    y = cursor + (span / 2.0)
                elif edge == "right":
                    x = origin_x + outline_w - _PERIMETER_MARGIN_MM - inset - (depth / 2.0)
                    y = cursor + (span / 2.0)
                elif edge == "top":
                    x = cursor + (span / 2.0)
                    y = origin_y + _PERIMETER_MARGIN_MM + inset + (depth / 2.0)
                else:  # bottom
                    x = cursor + (span / 2.0)
                    y = origin_y + outline_h - _PERIMETER_MARGIN_MM - inset - (depth / 2.0)
                placement[ref] = {
                    "x": round(x, 3),
                    "y": round(y, 3),
                    "rot": rot,
                    "zone": item.get("zone"),
                    "category": item.get("category"),
                    "edge": edge,
                    "face_dir": resolved_face_dir,
                    "face_dir_source": resolved_face_dir_source,
                    "footprint_body_dir": item.get("footprint_body_dir"),
                    "footprint_face_dir_guess": item.get("port_face_dir"),
                    "edge_outward_depth": outward_depth_val,
                    "edge_inward_depth": inward_depth_val,
                    "group_order": item.get("group_order"),
                }
                gap = self._edge_item_gap(item, items[idx + 1], edge) if idx + 1 < len(items) else 0.0
                cursor += span + gap

        for pass_idx in range(_PACK_OUTLINE_MAX_PASSES):
            placement_stats["outline_passes"] = pass_idx + 1
            placement = {}
            for item in classified_parts:
                item["placed_edge"] = None

            outline_w = round(left_band + inner_w + right_band + 2 * _PERIMETER_MARGIN_MM, 2)
            outline_h = round(top_band + inner_h + bottom_band + 2 * _PERIMETER_MARGIN_MM, 2)

            origin_x = _PAGE_CX_MM - (outline_w / 2.0)
            origin_y = _PAGE_CY_MM - (outline_h / 2.0)

            inner_left = origin_x + _PERIMETER_MARGIN_MM + left_band
            inner_right = inner_left + inner_w
            inner_top = origin_y + _PERIMETER_MARGIN_MM + top_band
            inner_bottom = inner_top + inner_h

            place_edge(left_edge, "left")
            place_edge(right_edge, "right")
            place_edge(top_edge, "top")
            place_edge(bottom_edge, "bottom")

            core_left = inner_left
            core_right = inner_right
            core_top = inner_top
            core_bottom = inner_bottom

            def reserve_from_items(items: List[Dict[str, Any]], edge: Optional[str], extra_gap: float = 2.5) -> None:
                nonlocal core_left, core_right, core_top, core_bottom
                for item in items:
                    pos = placement.get(item["ref"])
                    if not pos:
                        continue
                    if edge in {"top", "bottom", "left", "right"}:
                        comp_w, comp_h = self._oriented_dims(item, edge)
                    else:
                        comp_w = float(item.get("width", 6.0) or 6.0)
                        comp_h = float(item.get("height", 4.0) or 4.0)
                    half_w = comp_w / 2.0
                    half_h = comp_h / 2.0
                    if edge == "left":
                        core_left = max(core_left, float(pos["x"]) + half_w + extra_gap)
                    elif edge == "right":
                        core_right = min(core_right, float(pos["x"]) - half_w - extra_gap)
                    elif edge == "top":
                        core_top = max(core_top, float(pos["y"]) + half_h + extra_gap)
                    elif edge == "bottom":
                        core_bottom = min(core_bottom, float(pos["y"]) - half_h - extra_gap)
                    else:
                        core_right = min(core_right, float(pos["x"]) - half_w - extra_gap)

            reserve_from_items(left_edge, "left")
            reserve_from_items(right_edge, "right")
            reserve_from_items(top_edge, "top", extra_gap=8.0)
            reserve_from_items(bottom_edge, "bottom", extra_gap=3.0)

            if core_right - core_left < 18.0 or core_bottom - core_top < 18.0:
                inner_w += max(6.0, inner_w * 0.1)
                inner_h += max(6.0, inner_h * 0.1)
                continue

            if interior:
                trial = dict(placement)
                ok, _debug = self._try_place_interior(
                    {"usb": usb_zone, "power": power_zone, "main": main_zone},
                    trial,
                    part_map,
                    core_left,
                    core_top,
                    core_right,
                    core_bottom,
                )
                if not ok:
                    inner_w += max(6.0, inner_w * 0.1)
                    inner_h += max(5.0, inner_h * 0.08)
                    continue

                movable_refs = [item["ref"] for item in interior]
                self._relax_movable_parts(trial, part_map, movable_refs, core_left, core_right, core_top, core_bottom)
                if self._approx_overlap_pairs(trial, part_map):
                    inner_w += max(6.0, inner_w * 0.08)
                    inner_h += max(5.0, inner_h * 0.08)
                    continue
                placement = trial

            board_rect = (origin_x, origin_y, origin_x + outline_w, origin_y + outline_h)
            core_rect = (core_left, core_top, core_right, core_bottom)
            best_placement = {ref: dict(pos) for ref, pos in placement.items()}
            best_verify = self._verify_placement_quality(best_placement, part_map, board_rect)
            repair_runs = 0

            for _ in range(_VERIFY_REPAIR_MAX_PASSES):
                if not bool(best_verify.get("needs_repair")):
                    break
                trial = {ref: dict(pos) for ref, pos in best_placement.items()}
                changed = self._repair_placement_once(
                    trial,
                    part_map,
                    board_rect=board_rect,
                    core_rect=core_rect,
                )
                repair_runs += 1
                if not changed:
                    break
                trial_verify = self._verify_placement_quality(trial, part_map, board_rect)
                if self._placement_quality_cost(trial_verify) <= self._placement_quality_cost(best_verify):
                    best_placement = trial
                    best_verify = trial_verify

            placement_stats["verify_repair_passes"] = max(
                int(placement_stats.get("verify_repair_passes", 0) or 0),
                repair_runs,
            )
            placement_stats["final_verify"] = best_verify

            if bool(best_verify.get("fatal")):
                # Fatal verify failures mean geometry still invalid; grow the
                # interior canvas and retry within the fixed outline pass budget.
                inner_w += max(6.0, inner_w * 0.08)
                inner_h += max(5.0, inner_h * 0.08)
                continue

            final_outline_w = outline_w
            final_outline_h = outline_h
            final_placement = best_placement
            break

        if not final_placement:
            final_outline_w = outline_w
            final_outline_h = outline_h
            final_placement = placement
            placement_stats["used_fallback"] = True
            if not placement_stats.get("final_verify"):
                board_rect = (
                    _PAGE_CX_MM - (final_outline_w / 2.0),
                    _PAGE_CY_MM - (final_outline_h / 2.0),
                    _PAGE_CX_MM + (final_outline_w / 2.0),
                    _PAGE_CY_MM + (final_outline_h / 2.0),
                )
                placement_stats["final_verify"] = self._verify_placement_quality(final_placement, part_map, board_rect)

        return final_outline_w, final_outline_h, final_placement, placement_stats

    # ── Action builders ────────────────────────────────────────────────────

    def _build_outline_action(self, outline_w: float, outline_h: float) -> object:
        """Return a DEFINE_BOARD_OUTLINE DesignAction sized for the actual BOM."""
        try:
            from ..design_actions import DesignAction, DesignActionType
        except Exception as e:
            logger.error("Cannot import DesignAction for outline: %s", e)
            return None
        return DesignAction(
            action_type=DesignActionType.DEFINE_BOARD_OUTLINE,
            description=f"Define board outline ({outline_w:.0f}×{outline_h:.0f} mm)",
            parameters={"width": round(outline_w, 2), "height": round(outline_h, 2)},
            requires_approval=False,
        )

    def _build_add_actions(
        self,
        parts: List[Dict[str, Any]],
        placement: Dict[str, Dict[str, float]],
    ) -> list:
        """Return one ADD_COMPONENT DesignAction per part."""
        try:
            from ..design_actions import DesignAction, DesignActionType
        except Exception as e:
            logger.error("Cannot import DesignAction: %s", e)
            return []

        actions = []
        for part in parts:
            ref = str(part.get("ref", "") or "").strip()
            mpn = str(part.get("mpn", "") or "").strip()
            fp  = str(part.get("footprint", "") or "").strip()
            if not ref:
                continue
            pos = placement.get(ref, {})
            x   = float(pos.get("x", _PAGE_CX_MM))
            y   = float(pos.get("y", _PAGE_CY_MM))
            rot = float(pos.get("rot", 0.0))
            action = DesignAction(
                action_type=DesignActionType.ADD_COMPONENT,
                description=f"Place {ref} ({mpn}) at ({x:.1f}, {y:.1f})",
                parameters={
                    "query":    fp or mpn,
                    "footprint": fp,
                    "package":  fp,
                    "ref":      ref,
                    "location": {"x": x, "y": y},
                    "rotation": rot,
                    "placement_intent": {
                        "source": "place_agent",
                        "edge": pos.get("edge"),
                        "zone": pos.get("zone"),
                        "category": pos.get("category"),
                        "face_dir": pos.get("face_dir"),
                        "face_dir_source": pos.get("face_dir_source"),
                        "footprint_body_dir": pos.get("footprint_body_dir"),
                        "footprint_face_dir_guess": pos.get("footprint_face_dir_guess"),
                        "edge_outward_depth": pos.get("edge_outward_depth"),
                        "edge_inward_depth": pos.get("edge_inward_depth"),
                        "group_order": pos.get("group_order"),
                    },
                },
                requires_approval=False,
            )
            actions.append(action)
        return actions
