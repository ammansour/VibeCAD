"""Handler mixin for DesignAgent action executors."""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .design_actions import DesignAction, DesignActionType

logger = logging.getLogger(__name__)

_NO_CONNECT_NET_TOKENS = frozenset({"nc", "noconnect", "notconnected", "unconnected", "nonet", "none"})


class DesignAgentHandlersMixin:
    @staticmethod
    def _is_valid_board(board: Any) -> bool:
        return bool(
            board is not None
            and hasattr(board, "GetFootprints")
            and hasattr(board, "Add")
            and hasattr(board, "GetFileName")
        )

    @classmethod
    def _get_active_board(cls, context: Dict[str, Any]) -> Any:
        board = context.get('board') if isinstance(context, dict) else None
        if cls._is_valid_board(board):
            return board
        try:
            import pcbnew  # type: ignore
            board = pcbnew.GetBoard()
        except Exception:
            board = None
        return board if cls._is_valid_board(board) else None

    @staticmethod
    def _looks_package_hint(value: str) -> bool:
        raw = str(value or "").strip()
        if not raw:
            return False
        return bool(re.search(r"\b(?:DIP|PDIP|QFN|DFN|TQFP|LQFP|QFP|SOIC|SOT|SSOP|TSSOP|HTSSOP|MSOP|SOP|SO|VSSOP|QFN|LGA|BGA|TO)-?\d", raw, re.IGNORECASE))

    @staticmethod
    def _extract_footprint_id(footprint_obj: Any) -> str:
        if footprint_obj is None:
            return ""
        for name in ("GetFPIDAsString", "GetFootprintIDAsString"):
            fn = getattr(footprint_obj, name, None)
            if callable(fn):
                try:
                    value = str(fn() or "").strip()
                    if value:
                        return value
                except Exception:
                    pass
        try:
            fpid = footprint_obj.GetFPID()
        except Exception:
            fpid = None
        if fpid is None:
            return ""
        nick = ""
        item = ""
        for name in ("GetLibNickname", "GetNickname"):
            fn = getattr(fpid, name, None)
            if callable(fn):
                try:
                    nick = str(fn() or "").strip()
                except Exception:
                    nick = ""
                if nick:
                    break
        for name in ("GetLibItemName", "GetFootprintName", "GetItemName"):
            fn = getattr(fpid, name, None)
            if callable(fn):
                try:
                    item = str(fn() or "").strip()
                except Exception:
                    item = ""
                if item:
                    break
        if nick and item:
            return f"{nick}:{item}"
        for name in ("AsString", "Format"):
            fn = getattr(fpid, name, None)
            if callable(fn):
                try:
                    value = str(fn() or "").strip()
                    if value:
                        return value
                except TypeError:
                    try:
                        value = str(fn(None) or "").strip()
                        if value:
                            return value
                    except Exception:
                        pass
                except Exception:
                    pass
        try:
            return str(fpid or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _remove_board_item(board: Any, item: Any) -> bool:
        if board is None or item is None:
            return False
        for name in ("Remove", "Delete"):
            fn = getattr(board, name, None)
            if callable(fn):
                try:
                    fn(item)
                    return True
                except Exception:
                    continue
        return False

    @staticmethod
    def _canonical_net_token(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if not text:
            return ""
        return re.sub(r"[^a-z0-9]+", "", text)

    @classmethod
    def _is_no_connect_net_name(cls, raw: Any) -> bool:
        return cls._canonical_net_token(raw) in _NO_CONNECT_NET_TOKENS

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
                # VERY CRITICAL FIX: Transfer ownership to C++ so Python GC doesn't delete it
                net_item.thisown = False
                add = getattr(board, 'Add', None)
                if callable(add):
                    try:
                        board.Add(net_item)
                    except TypeError:
                        an = getattr(board, 'AddNet', None)
                        if callable(an):
                            an(net_item)
                else:
                    an = getattr(board, 'AddNet', None)
                    if callable(an):
                        an(net_item)
                return net_item
            except Exception:
                return None


        assigned = 0
        errors: List[str] = []
        skipped_no_connect = 0

        for item in assignments:
            if not isinstance(item, dict):
                continue
            ref_val = item.get('ref') or item.get('reference') or item.get('designator')
            ref = str(ref_val or '').strip().upper()
            pad_val = item.get('pad') or item.get('pin') or item.get('pad_num')
            pad_num = str(pad_val or '').strip()
            net_val = item.get('net') or item.get('net_name') or item.get('netName') or item.get('name')
            net_name = str(net_val or '').strip()

            # Accept "pin_ref" like "U1-3" / "U1:3" / "U1/3".
            if (not ref or not pad_num) and isinstance(item.get('pin_ref'), str):
                pr = str(item.get('pin_ref') or '').strip()
                m = re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", pr)
                if m:
                    ref = m.group(1).upper()
                    pad_num = m.group(2).strip()
            if not ref or not pad_num or not net_name:
                continue
            if self._is_no_connect_net_name(net_name):
                skipped_no_connect += 1
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

        # Aggressively refresh KiCad's PCB view.
        try:
            import pcbnew
            for refresh_fn in (
                lambda: pcbnew.Refresh(),
                lambda: board.GetDesignSettings(),
                lambda: pcbnew.UpdateUserInterface(),
            ):
                try:
                    refresh_fn()
                except Exception:
                    pass
        except Exception:
            pass

        if assigned <= 0:
            suffix = (" Errors: " + "; ".join(errors[:8])) if errors else ""
            if skipped_no_connect:
                suffix += f" Skipped no-connect pseudo-nets: {skipped_no_connect}."
            return False, "No nets assigned." + suffix

        msg = f"Assigned nets to {assigned} pad(s)."
        if errors:
            msg += f" Warnings: {len(errors)} item(s) could not be assigned."
            msg += " Details: " + "; ".join(errors[:5])
        if skipped_no_connect:
            msg += f" Skipped no-connect pseudo-net assignments: {skipped_no_connect}."
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
        if self._is_no_connect_net_name(net_name):
            return True, f"Skipped DEFINE_NET for no-connect pseudo-net '{net_name}'."
        if pads is None:
            pads = []
        if not isinstance(pads, list):
            pads = [pads]

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
                # VERY CRITICAL FIX: Transfer ownership to C++ so Python GC doesn't delete it
                net_item.thisown = False
                add = getattr(board, 'Add', None)
                if callable(add):
                    try:
                        board.Add(net_item)
                    except TypeError:
                        an = getattr(board, 'AddNet', None)
                        if callable(an):
                            an(net_item)
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

        # If no pads were provided, defining the net means "ensure this net exists".
        # (Some LLMs emit DEFINE_NET to create nets before ASSIGN_NETS.)
        if not pads:
            return True, f"Created/ensured net '{net_name}'."

        assigned = 0
        errors: List[str] = []
        invalid: List[str] = []

        for item in pads:
            ref = ''
            pad_num = ''
            if isinstance(item, str):
                text = item.strip()
                # Accept U1/1, U1-1, U1:1.
                m = re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", text)
                if not m:
                    invalid.append(text)
                    continue
                ref, pad_num = m.group(1), m.group(2)
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

        # Aggressively refresh KiCad's PCB view.
        try:
            import pcbnew
            for refresh_fn in (
                lambda: pcbnew.Refresh(),
                lambda: board.GetDesignSettings(),
                lambda: pcbnew.UpdateUserInterface(),
            ):
                try:
                    refresh_fn()
                except Exception:
                    pass
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

        def set_orientation_deg(footprint_obj, angle_deg: float) -> bool:
            try:
                if hasattr(footprint_obj, 'SetOrientationDegrees'):
                    footprint_obj.SetOrientationDegrees(float(angle_deg))
                    return True
                eda_angle = getattr(pcbnew, 'EDA_ANGLE', None)
                if eda_angle is not None:
                    footprint_obj.SetOrientation(eda_angle(float(angle_deg), pcbnew.DEGREES_T))
                    return True
            except Exception:
                return False
            return False

        def rotation_from_dir(source_dir: str, target_dir: str) -> float:
            order = ['px', 'py', 'nx', 'ny']
            if source_dir not in order or target_dir not in order:
                return 0.0
            return float(((order.index(target_dir) - order.index(source_dir)) % 4) * 90)

        def outward_dir_for_edge(edge: str) -> str:
            return {
                'left': 'nx',
                'right': 'px',
                'top': 'ny',
                'bottom': 'py',
            }.get(edge, 'nx')

        def fixed_port_rotation_for_edge(edge: str) -> float:
            return {
                'left': 180.0,
                'right': 0.0,
                'top': 90.0,
                'bottom': 270.0,
            }.get(edge, 0.0)

        # ── extract query ────────────────────────────────────────────────────
        params = action.parameters or {}
        placement_intent = params.get('placement_intent') if isinstance(params.get('placement_intent'), dict) else {}
        planned_edge = str(placement_intent.get('edge', '') or '').strip().lower()
        planned_category = str(placement_intent.get('category', '') or '').strip().lower()
        edge_mount = planned_edge in {'left', 'right', 'top', 'bottom'}
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

        def build_query_variants(raw_query: str) -> List[str]:
            variants: List[str] = []

            def add_variant(value: str) -> None:
                candidate = str(value or '').strip()
                if not candidate:
                    return
                if candidate not in variants:
                    variants.append(candidate)

            add_variant(raw_query)
            add_variant(raw_query.replace(":", " "))
            add_variant(re.sub(r'[_\-/]+', ' ', raw_query))
            if ":" in raw_query:
                lib_part = raw_query.split(":", 1)[1].strip()
                add_variant(lib_part)
                add_variant(re.sub(r'[_\-/]+', ' ', lib_part))
            return variants

        # Guardrail: a package family without a pin count is too ambiguous and
        # tends to resolve to random small footprints (e.g. DIP-4), which then
        # cascades into net-assignment/DRC failures. Force explicit pin count.
        _pkg_families = {
            "DIP", "PDIP", "QFN", "DFN", "TQFP", "LQFP", "QFP",
            "SOIC", "SOT", "SSOP", "TSSOP", "HTSSOP", "MSOP", "SOP", "SO",
        }
        if query.upper() in _pkg_families and not re.search(r"\d", query):
            return False, (
                f"Footprint query '{query}' is too generic. "
                "Specify the pin count/package (e.g. 'DIP-28', 'TQFP-32')."
            )

        # ── resolve board ────────────────────────────────────────────────────
        board = self._get_active_board(context)
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
                if package_hint.upper() in _pkg_families and not re.search(r"\d", package_hint):
                    return False, (
                        f"Package hint '{package_hint}' is too generic. "
                        "Specify the pin count/package (e.g. 'DIP-28', 'TQFP-32')."
                    )

            explicit_footprint_id = bool(package_hint and ":" in package_hint)

            # For explicit library-qualified package hints, the resolver already
            # searches by the package internally. Extra query variants just repeat
            # the same failing search dozens of times and add noise to the log.
            attempt_queries = [query]
            if explicit_footprint_id:
                if package_hint and package_hint != query:
                    attempt_queries.append(package_hint)
            else:
                attempt_queries = build_query_variants(query)
                if package_hint:
                    attempt_queries.extend(build_query_variants(package_hint))
                    attempt_queries.extend(build_query_variants(f"{query} {package_hint}"))
            dedup_attempts: List[str] = []
            for q in attempt_queries:
                if q not in dedup_attempts:
                    dedup_attempts.append(q)

            for attempt_query in dedup_attempts:
                if not attempt_query:
                    continue
                try:
                    resolver = getattr(self._library_manager, 'resolve_best_footprint_item', None)
                    if callable(resolver):
                        item = resolver(attempt_query, package_hint=package_hint or None)
                        if item and getattr(item, 'local_footprint_path', None):
                            if package_hint and not self._library_manager.footprint_path_matches_hint(str(item.local_footprint_path), package_hint):
                                continue
                            fp_path = item.local_footprint_path
                            resolved_mpn = getattr(item, 'mpn', query) or query
                            break
                except Exception:
                    pass

            # Try resolve_best_footprint_path
            if not fp_path:
                for attempt_query in dedup_attempts:
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
                search_queries = dedup_attempts
                for sq in search_queries:
                    try:
                        if hasattr(self._library_manager, 'search_parts'):
                            results = await self._library_manager.search_parts(sq)
                        else:
                            results = self._library_manager.search_parts_sync(sq)
                        for item in (results or []):
                            if getattr(item, 'local_footprint_path', None):
                                if package_hint and not self._library_manager.footprint_path_matches_hint(str(item.local_footprint_path), package_hint):
                                    continue
                                fp_path = item.local_footprint_path
                                resolved_mpn = getattr(item, 'mpn', query) or query
                                break
                        if fp_path:
                            break
                        # Try downloading online candidates
                        if not fp_path and results and not explicit_footprint_id:
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
            # Provide actionable suggestions so the LLM can choose an actually-available footprint
            # in the next iteration (without silently picking a deterministic fallback here).
            suggestions: List[str] = []
            try:
                lm = self._library_manager
                if lm is not None:
                    alt_queries: List[str] = []
                    alt_queries.extend(build_query_variants(query)[:4])
                    ql = query.lower()
                    if "crystal" in ql or ql.startswith("xtal") or ql.startswith("osc"):
                        alt_queries.extend(["Crystal", "Crystal SMD", "Oscillator"])
                    if any(tok in ql for tok in ("header", "socket", "shield header", "pin header", "pin socket")):
                        alt_queries.extend(["PinHeader 2.54", "PinSocket 2.54", "Header 2.54"])

                    seen_alt: set = set()
                    dedup_alt: List[str] = []
                    for aq in alt_queries:
                        aq = str(aq or "").strip()
                        if not aq:
                            continue
                        if aq in seen_alt:
                            continue
                        seen_alt.add(aq)
                        dedup_alt.append(aq)

                    async def _search_one(q: str) -> List[Any]:
                        try:
                            if hasattr(lm, "search_parts"):
                                return await lm.search_parts(q)  # type: ignore[attr-defined]
                            if hasattr(lm, "search_parts_sync"):
                                return lm.search_parts_sync(q)  # type: ignore[attr-defined]
                        except Exception:
                            return []
                        return []

                    for aq in dedup_alt[:6]:
                        results = await _search_one(aq)
                        for item in (results or [])[:25]:
                            try:
                                if not getattr(item, "local_footprint_path", None):
                                    continue
                                name = str(getattr(item, "name", "") or "").strip()
                                if name and name not in suggestions:
                                    suggestions.append(name)
                            except Exception:
                                continue
                        if len(suggestions) >= 8:
                            break
            except Exception:
                suggestions = []

            if suggestions:
                shown = ", ".join(suggestions[:6])
                return False, (
                    f"No footprint found for '{query}'. "
                    f"Try one of these local footprint names: {shown}."
                )
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

        has_explicit_rotation = any(
            key in params and str(params.get(key, "") or "").strip() != ""
            for key in ('rotation', 'rot', 'angle')
        )
        raw_rotation = params.get('rotation', params.get('rot', params.get('angle', 0.0)))
        try:
            requested_rotation = float(raw_rotation or 0.0)
        except Exception:
            return False, f"Invalid rotation value: {raw_rotation!r}"

        if planned_edge in {'left', 'right', 'top', 'bottom'}:
            category_tokens = {
                tok for tok in re.split(r"[^a-z0-9]+", planned_category) if tok
            }
            is_port_category = bool(planned_category.endswith("_port") or ("port" in category_tokens))
            if is_port_category and not has_explicit_rotation:
                hinted_face_dir = str(placement_intent.get('face_dir', '') or '').strip().lower()
                if planned_edge in {'left', 'right'} and hinted_face_dir in {'py', 'ny'}:
                    hinted_face_dir = 'px' if hinted_face_dir == 'py' else 'nx'
                if hinted_face_dir in {'px', 'nx', 'py', 'ny'}:
                    requested_rotation = rotation_from_dir(hinted_face_dir, outward_dir_for_edge(planned_edge))
                else:
                    requested_rotation = fixed_port_rotation_for_edge(planned_edge)

        if not set_orientation_deg(fp, requested_rotation):
            return False, f"Failed to rotate footprint before placement: {requested_rotation}°"

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

        def find_aligned_slot(edge: str) -> Tuple[int, int] | None:
            if edge not in {'left', 'right', 'top', 'bottom'}:
                return None
            line_grid = grid
            if 'header' in planned_category:
                line_grid = mm2iu(2.54)
            elif edge in {'left', 'right'}:
                line_grid = mm2iu(5.0)
            for r in range(1, 24):
                offsets = []
                for delta in range(1, r + 1):
                    offsets.extend((delta, -delta))
                for offset in offsets:
                    nx = target_x
                    ny = target_y
                    if edge in {'top', 'bottom'}:
                        nx = target_x + offset * line_grid
                    else:
                        ny = target_y + offset * line_grid
                    if not would_overlap(nx, ny):
                        return nx, ny
            return None

        grid = mm2iu(10.0)  # 10mm placement grid step
        if would_overlap(target_x, target_y):
            # Spiral outward to find a free slot
            placed = False
            aligned_slot = find_aligned_slot(planned_edge)
            if aligned_slot is not None:
                target_x, target_y = aligned_slot
                placed = True
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
            if hasattr(fp, 'thisown'):
                fp.thisown = False
            board.Add(fp)
        except Exception as e:
            return False, f"board.Add() failed: {e}"

        # ── clamp to board outline if one exists ─────────────────────────────
        board_bounds: Optional[Tuple[int, int, int, int]] = None
        try:
            bb = board.GetBoardEdgesBoundingBox()
            bw, bh = int(bb.GetWidth()), int(bb.GetHeight())
            if bw > mm2iu(5) and bh > mm2iu(5):
                margin = mm2iu(0.6 if edge_mount else 1.0)
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
                    if planned_edge != 'left':
                        bx_min += hw
                    if planned_edge != 'right':
                        bx_max -= hw
                    if planned_edge != 'top':
                        by_min += hh
                    if planned_edge != 'bottom':
                        by_max -= hh
                clamped = False
                if bx_max > bx_min and by_max > by_min:
                    board_bounds = (bx_min, bx_max, by_min, by_max)
                    new_x = max(bx_min, min(bx_max, target_x))
                    new_y = max(by_min, min(by_max, target_y))
                    if new_x != target_x or new_y != target_y:
                        target_x, target_y = new_x, new_y
                        clamped = True
        except Exception:
            pass

        if not set_pos(fp, target_x, target_y):
            return False, "SetPosition failed (KiCad API incompatible)"

        # Post-placement: correct courtyard overshoot using actual world-coordinate bbox.
        # The pre-placement clamp uses fp_bb.GetWidth()//2 which assumes the courtyard is
        # centered on the footprint origin — incorrect for asymmetric parts (connectors,
        # edge-mount parts, etc.). After rotation this can place the courtyard outside the
        # board boundary. Reading the courtyard in world-coords after SetPosition gives the
        # exact overreach and lets us apply a precise correction.
        try:
            _fp_world_bb = None
            for _bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                _fn = getattr(fp, _bbm, None)
                if callable(_fn):
                    try:
                        _tmp = _fn() if _bbm == 'GetCourtyardBoundingBox' else _fn(False, False)
                        if _tmp and int(_tmp.GetWidth()) > 0:
                            _fp_world_bb = _tmp
                            break
                    except Exception:
                        pass
            if _fp_world_bb is not None:
                _brd_bb = board.GetBoardEdgesBoundingBox()
                _brd_bw = int(_brd_bb.GetWidth())
                _brd_bh = int(_brd_bb.GetHeight())
                if _brd_bw > mm2iu(5) and _brd_bh > mm2iu(5):
                    _ecl = mm2iu(0.6)  # KiCad default edge clearance (0.5 mm) + 0.1 mm buffer
                    _bd_x0 = int(_brd_bb.GetX()) + _ecl
                    _bd_y0 = int(_brd_bb.GetY()) + _ecl
                    _bd_x1 = int(_brd_bb.GetX()) + _brd_bw - _ecl
                    _bd_y1 = int(_brd_bb.GetY()) + _brd_bh - _ecl
                    _fp_x0 = int(_fp_world_bb.GetX())
                    _fp_y0 = int(_fp_world_bb.GetY())
                    _fp_x1 = _fp_x0 + int(_fp_world_bb.GetWidth())
                    _fp_y1 = _fp_y0 + int(_fp_world_bb.GetHeight())
                    _dx, _dy = 0, 0
                    if _fp_x0 < _bd_x0 and planned_edge != 'left':
                        _dx = _bd_x0 - _fp_x0
                    elif _fp_x1 > _bd_x1 and planned_edge != 'right':
                        _dx = _bd_x1 - _fp_x1
                    if _fp_y0 < _bd_y0 and planned_edge != 'top':
                        _dy = _bd_y0 - _fp_y0
                    elif _fp_y1 > _bd_y1 and planned_edge != 'bottom':
                        _dy = _bd_y1 - _fp_y1
                    if _dx != 0 or _dy != 0:
                        _cur_pos = fp.GetPosition()
                        _cur_px = int(getattr(_cur_pos, 'x', 0))
                        _cur_py = int(getattr(_cur_pos, 'y', 0))
                        set_pos(fp, _cur_px + _dx, _cur_py + _dy)
                        target_x = _cur_px + _dx
                        target_y = _cur_py + _dy
        except Exception:
            pass

        # Final anti-overlap pass: edge clamping/correction can push a footprint
        # back into a collision after the initial free-slot search.
        if would_overlap(target_x, target_y):
            def _clamp_candidate(nx: int, ny: int) -> Tuple[int, int]:
                if board_bounds is None:
                    return nx, ny
                bx_min, bx_max, by_min, by_max = board_bounds
                if bx_max <= bx_min or by_max <= by_min:
                    return nx, ny
                return (
                    max(bx_min, min(bx_max, nx)),
                    max(by_min, min(by_max, ny)),
                )

            resolved_slot: Optional[Tuple[int, int]] = None
            tried: set[Tuple[int, int]] = {(target_x, target_y)}

            def _try_slot(nx: int, ny: int) -> Optional[Tuple[int, int]]:
                cx, cy = _clamp_candidate(nx, ny)
                key = (cx, cy)
                if key in tried:
                    return None
                tried.add(key)
                if not would_overlap(cx, cy):
                    return key
                return None

            if planned_edge in {'left', 'right', 'top', 'bottom'}:
                line_grid = grid
                if 'header' in planned_category:
                    line_grid = mm2iu(2.54)
                elif planned_edge in {'left', 'right'}:
                    line_grid = mm2iu(5.0)
                for r in range(1, 40):
                    offsets: List[int] = []
                    for delta in range(1, r + 1):
                        offsets.extend((delta, -delta))
                    for offset in offsets:
                        nx, ny = target_x, target_y
                        if planned_edge in {'top', 'bottom'}:
                            nx = target_x + offset * line_grid
                        else:
                            ny = target_y + offset * line_grid
                        slot = _try_slot(nx, ny)
                        if slot is not None:
                            resolved_slot = slot
                            break
                    if resolved_slot is not None:
                        break

            if resolved_slot is None:
                base_x, base_y = target_x, target_y
                for r in range(1, 25):
                    if resolved_slot is not None:
                        break
                    for dx in range(-r, r + 1):
                        if resolved_slot is not None:
                            break
                        for dy in range(-r, r + 1):
                            if abs(dx) != r and abs(dy) != r:
                                continue
                            slot = _try_slot(base_x + dx * grid, base_y + dy * grid)
                            if slot is not None:
                                resolved_slot = slot
                                break

            if resolved_slot is not None and resolved_slot != (target_x, target_y):
                target_x, target_y = resolved_slot
                set_pos(fp, target_x, target_y)

        # Record for future anti-overlap
        self._session_placed_positions.append((target_x, target_y))

        # ── assign reference designator ──────────────────────────────────────
        requested_ref = str(params.get('ref') or params.get('reference') or '').strip().upper()
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

            if requested_ref:
                new_ref = requested_ref
            else:
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
            new_ref = requested_ref or 'U?'

        explicit_fp_hint = ''
        for key in ('footprint', 'footprint_id', 'footprintId', 'kicad_footprint', 'kicadFootprint'):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                explicit_fp_hint = value.strip()
                break
        explicit_package_hint = ''
        package_value = params.get('package')
        if isinstance(package_value, str) and package_value.strip():
            explicit_package_hint = package_value.strip()
        elif self._looks_package_hint(query):
            explicit_package_hint = query.strip()

        actual_fp_id = self._extract_footprint_id(fp)
        resolved_fp_id = f"{fp_file.parent.stem}:{fp_file.stem}"
        expected_fp_id = explicit_fp_hint or resolved_fp_id
        footprint_hint_match = False
        package_hint_match = False
        resolved_fp_match = False
        try:
            if self._library_manager is not None:
                candidate_fp_id = actual_fp_id or expected_fp_id
                resolved_fp_match = bool(
                    self._library_manager.assess_footprint_id_compatibility(resolved_fp_id, candidate_fp_id).matched
                )
                if explicit_fp_hint:
                    footprint_hint_match = bool(
                        self._library_manager.assess_footprint_id_compatibility(explicit_fp_hint, candidate_fp_id).matched
                    )
                if explicit_package_hint:
                    package_hint_match = bool(
                        self._library_manager.assess_footprint_id_compatibility(explicit_package_hint, candidate_fp_id).matched
                    )
        except Exception:
            footprint_hint_match = False
            package_hint_match = False
            resolved_fp_match = False
        footprint_conflict = bool(explicit_fp_hint) and actual_fp_id and not (
            footprint_hint_match or resolved_fp_match
        )
        package_conflict = bool(explicit_package_hint) and actual_fp_id and not (
            package_hint_match or resolved_fp_match
        )
        if footprint_conflict or package_conflict:
            self._remove_board_item(board, fp)
            expected_desc = explicit_fp_hint if footprint_conflict else explicit_package_hint
            expected_kind = 'footprint' if footprint_conflict else 'package'
            return False, (
                f"Imported footprint for {new_ref} did not match the requested {expected_kind}. "
                f"Expected {expected_desc}, but KiCad loaded {actual_fp_id or expected_fp_id}. "
                "The component was removed instead of keeping a wrong package on the board."
            )

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
                verbose = bool((context or {}).get('verbose'))

                def _is_footprint_candidate(item: Any) -> bool:
                    try:
                        if getattr(item, 'local_footprint_path', None):
                            return True
                        if getattr(item, 'footprint_url', None):
                            return True
                        category = str(getattr(item, 'category', '') or '').lower()
                        if 'footprint' in category:
                            return True
                    except Exception:
                        pass
                    return False

                ranked = sorted(
                    list(results),
                    key=lambda item: (
                        0 if _is_footprint_candidate(item) else 1,
                        0 if query.lower() in str(getattr(item, 'name', '') or '').lower() else 1,
                    ),
                )
                max_show = min(len(ranked), 20 if verbose else 6)
                lines = [f"Found {len(results)} matching parts for '{query}':"]
                serialized_results = []
                for item in ranked[:max_show]:
                    label = (getattr(item, 'name', '') or getattr(item, 'mpn', '')).strip()
                    pkg = (getattr(item, 'package', '') or '').strip()
                    src = ''
                    try:
                        src = getattr(getattr(item, 'source', None), 'value', '')
                    except Exception:
                        src = ''
                    is_footprint = _is_footprint_candidate(item)
                    suffix = ""
                    if pkg:
                        suffix += f" ({pkg})"
                    if src:
                        suffix += f" [{src}]"
                    if not verbose:
                        suffix += " [fp]" if is_footprint else " [sym]"
                    lines.append(f"- {label}{suffix}")
                    serialized_results.append({
                        "name": label,
                        "package": pkg,
                        "source": src,
                        "is_footprint_candidate": bool(is_footprint),
                    })

                # Provide a machine-pickable list of footprint identifiers so the LLM can
                # choose (or refine search) without guessing library names from memory.
                footprint_candidates: List[str] = []
                for r in serialized_results:
                    try:
                        if not r.get("is_footprint_candidate"):
                            continue
                        name = str(r.get("name", "") or "").strip()
                        if not name:
                            continue
                        # Prefer explicit "Lib:Footprint" identifiers; allow others if present.
                        if ":" in name:
                            footprint_candidates.append(name)
                        elif name not in footprint_candidates:
                            footprint_candidates.append(name)
                    except Exception:
                        continue

                if footprint_candidates:
                    show_n = 12 if verbose else 6
                    lines.append("")
                    lines.append("Footprint candidates (use EXACTLY one for ADD_COMPONENT.parameters.query):")
                    for fp in footprint_candidates[:show_n]:
                        lines.append(f"- {fp}")
                    try:
                        lines.append(f"FOOTPRINT_CANDIDATES_JSON: {json.dumps(footprint_candidates[:12])}")
                    except Exception:
                        pass
                else:
                    lines.append("")
                    lines.append(
                        "No obvious footprint candidates were found. "
                        "Refine SEARCH_PART with an explicit package/pin count/pitch (e.g. 'TQFP-32 7x7 0.8mm')."
                    )
                if len(results) > max_show:
                    lines.append(f"…and {len(results) - max_show} more")
                logger.info(
                    "SEARCH_PART query=%r results=%d shown=%d",
                    query, len(results), max_show
                )
                # Persist structured search output for downstream phases.
                store = context.setdefault("search_part_results", {})
                if isinstance(store, dict):
                    store[query] = serialized_results
                return True, "\n".join(lines)
            else:
                return True, "No parts found matching the query. Tip: try a shorter part number, a description, or a manufacturer name."
        except Exception as e:
            return False, f"Search failed: {e}"

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
                if hasattr(t, 'thisown'):
                    t.thisown = False
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
        # Accept a few common key variants; other code paths use "reference".
        ref_val = (
            params.get('ref')
            or params.get('reference')
            or params.get('designator')
            or params.get('component')
        )
        ref = str(ref_val or '').strip().upper()

        location = params.get('location')
        if location in (None, ""):
            # Common alternate encodings: x/y pairs or "pos"/"at".
            if "x" in params and "y" in params:
                location = {"x": params.get("x"), "y": params.get("y")}
            elif "x_mm" in params and "y_mm" in params:
                location = {"x": params.get("x_mm"), "y": params.get("y_mm")}
            elif "pos" in params:
                location = params.get("pos")
            elif "at" in params:
                location = params.get("at")

        if location in (None, ""):
            # Last-resort: try to extract coordinates from the action description.
            try:
                m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", action.description or "")
                if m:
                    location = f"{m.group(1)},{m.group(2)}"
            except Exception:
                pass

        strategy = str(params.get("strategy", "") or "").strip().lower()
        if (location in (None, "")) and strategy:
            # Strategy-based movement: used by VerificationAgent when it can't
            # safely invent coordinates. We compute a reasonable target.
            #
            # Supported strategies:
            # - edge_inset: clamp inside the board outline with a small margin
            # - resolve_overlap / increase_spacing: nudge away from overlapping footprints
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

            def mm2iu(mm_val: float) -> int:
                try:
                    return int(pcbnew.FromMM(float(mm_val)))
                except Exception:
                    return int(float(mm_val) * 1e6)

            def iu2mm(iu_val: int) -> float:
                try:
                    to_mm = getattr(pcbnew, "ToMM", None)
                    if callable(to_mm):
                        return float(to_mm(iu_val))
                except Exception:
                    pass
                return float(iu_val) / 1e6

            def _get_rect(fp_obj):
                """Return (l,t,r,b,cx,cy,hw,hh) in IU for a footprint."""
                bb = None
                for bbm in ("GetCourtyardBoundingBox", "GetBoundingBox"):
                    fn = getattr(fp_obj, bbm, None)
                    if callable(fn):
                        try:
                            bb = fn() if bbm == "GetCourtyardBoundingBox" else fn(False, False)
                            if bb and bb.GetWidth() > 0:
                                break
                            bb = None
                        except Exception:
                            bb = None
                if bb is None:
                    # Fallback: 10mm square around position.
                    try:
                        pos = fp_obj.GetPosition()
                        x = int(getattr(pos, "x", pos.GetX()))
                        y = int(getattr(pos, "y", pos.GetY()))
                    except Exception:
                        x, y = 0, 0
                    half = mm2iu(5.0)
                    l, t, r, b = x - half, y - half, x + half, y + half
                    return (l, t, r, b, x, y, half, half)

                l = int(bb.GetLeft())
                t = int(bb.GetTop())
                r = int(bb.GetRight())
                b = int(bb.GetBottom())
                cx = int(bb.GetX()) + int(bb.GetWidth()) // 2
                cy = int(bb.GetY()) + int(bb.GetHeight()) // 2
                hw = int(bb.GetWidth()) // 2
                hh = int(bb.GetHeight()) // 2
                return (l, t, r, b, cx, cy, hw, hh)

            def _clamp_to_outline(x_iu: int, y_iu: int, hw: int, hh: int) -> Tuple[int, int]:
                try:
                    bb = board.GetBoardEdgesBoundingBox()
                    bw = int(bb.GetWidth())
                    bh = int(bb.GetHeight())
                    if bw <= mm2iu(5) or bh <= mm2iu(5):
                        return x_iu, y_iu
                    margin = mm2iu(1.0)
                    bx_min = int(bb.GetX()) + margin + hw
                    bx_max = int(bb.GetX()) + bw - margin - hw
                    by_min = int(bb.GetY()) + margin + hh
                    by_max = int(bb.GetY()) + bh - margin - hh
                    if bx_max <= bx_min or by_max <= by_min:
                        return x_iu, y_iu
                    x_iu = max(bx_min, min(bx_max, x_iu))
                    y_iu = max(by_min, min(by_max, y_iu))
                except Exception:
                    pass
                return x_iu, y_iu

            # Find the footprint now (we'll set location below).
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

            l, t, r, b, cx, cy, hw, hh = _get_rect(footprint)

            # Compute a new target.
            new_x = cx
            new_y = cy
            if strategy in {"edge_inset"}:
                new_x, new_y = _clamp_to_outline(new_x, new_y, hw, hh)
            elif strategy in {"resolve_overlap", "increase_spacing"}:
                clearance = mm2iu(0.5)
                push_x = 0.0
                push_y = 0.0
                for other in list(board.GetFootprints() or []):
                    try:
                        if other is footprint:
                            continue
                        ol, ot, or_, ob, ocx, ocy, ohw, ohh = _get_rect(other)
                    except Exception:
                        continue

                    # AABB overlap with clearance.
                    if (r + clearance <= ol) or (l - clearance >= or_) or (b + clearance <= ot) or (t - clearance >= ob):
                        continue

                    dx = float(cx - ocx)
                    dy = float(cy - ocy)
                    if dx == 0.0 and dy == 0.0:
                        dx = 1.0
                    mag = (dx * dx + dy * dy) ** 0.5
                    if mag > 0:
                        push_x += dx / mag
                        push_y += dy / mag

                # If we detected overlaps, nudge by 5mm away from the crowd.
                step = mm2iu(5.0)
                mag = (push_x * push_x + push_y * push_y) ** 0.5
                if mag > 0.0:
                    new_x = int(cx + (push_x / mag) * step)
                    new_y = int(cy + (push_y / mag) * step)

                new_x, new_y = _clamp_to_outline(new_x, new_y, hw, hh)
            else:
                return False, f"Unknown MOVE_COMPONENT strategy: {strategy}"

            location = {"x": round(iu2mm(new_x), 2), "y": round(iu2mm(new_y), 2)}

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

        def _parse_relative_location_mm(raw: Any) -> Optional[Tuple[float, float]]:
            if not isinstance(raw, str):
                return None
            text = str(raw or "").strip().lower()
            if not text:
                return None

            # Examples supported:
            # - "right 5 in"
            # - "left 12.7 mm"
            # - "up 250 mil"
            # - "down by 3.0"
            m = re.search(
                r"\b(?P<dir>right|left|up|down)\b\s*(?:by\s*)?"
                r"(?P<dist>-?\d+(?:\.\d+)?)\s*"
                r"(?P<unit>mm|mil|mils|in|inch|inches)?\b",
                text,
                flags=re.IGNORECASE,
            )
            if not m:
                return None

            direction = str(m.group("dir") or "").strip().lower()
            try:
                distance = float(m.group("dist"))
            except Exception:
                return None
            unit = str(m.group("unit") or "mm").strip().lower()

            if unit in {"in", "inch", "inches"}:
                distance_mm = distance * 25.4
            elif unit in {"mil", "mils"}:
                distance_mm = distance * 0.0254
            else:
                distance_mm = distance

            target_fp = None
            try:
                for fp in board.GetFootprints():
                    try:
                        if str(fp.GetReference()).upper() == ref:
                            target_fp = fp
                            break
                    except Exception:
                        continue
            except Exception:
                target_fp = None
            if target_fp is None:
                return None

            try:
                pos = target_fp.GetPosition()
                x_iu = int(getattr(pos, "x", pos.GetX()))
                y_iu = int(getattr(pos, "y", pos.GetY()))
            except Exception:
                return None

            to_mm = getattr(pcbnew, "ToMM", None)
            if callable(to_mm):
                x_mm = float(to_mm(x_iu))
                y_mm = float(to_mm(y_iu))
            else:
                x_mm = float(x_iu) / 1e6
                y_mm = float(y_iu) / 1e6

            dx = 0.0
            dy = 0.0
            if direction == "right":
                dx = distance_mm
            elif direction == "left":
                dx = -distance_mm
            elif direction == "up":
                dy = -distance_mm
            elif direction == "down":
                dy = distance_mm
            else:
                return None

            return (x_mm + dx, y_mm + dy)

        parsed = self._parse_location_mm(location)
        if parsed is None:
            parsed = _parse_relative_location_mm(location)
        if parsed is None:
            return (
                False,
                f"Could not parse location '{location}'. Use 'x,y' in mm (e.g., '50,25') "
                "or a relative move like 'right 5 mm'.",
            )

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

        # Post-move: correct courtyard overshoot using actual world-coordinate bbox.
        # Reads the real KiCad courtyard bounds after positioning, so it correctly handles
        # asymmetric footprints where the courtyard is not centered on the origin.
        try:
            _mv_fp_bb = None
            for _bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                _fn = getattr(footprint, _bbm, None)
                if callable(_fn):
                    try:
                        _tmp = _fn() if _bbm == 'GetCourtyardBoundingBox' else _fn(False, False)
                        if _tmp and int(_tmp.GetWidth()) > 0:
                            _mv_fp_bb = _tmp
                            break
                    except Exception:
                        pass
            if _mv_fp_bb is not None:
                _mv_brd_bb = board.GetBoardEdgesBoundingBox()
                _mv_bw = int(_mv_brd_bb.GetWidth())
                _mv_bh = int(_mv_brd_bb.GetHeight())
                if _mv_bw > int(from_mm(5)) and _mv_bh > int(from_mm(5)):
                    _ecl = int(from_mm(0.6))
                    _bd_x0 = int(_mv_brd_bb.GetX()) + _ecl
                    _bd_y0 = int(_mv_brd_bb.GetY()) + _ecl
                    _bd_x1 = int(_mv_brd_bb.GetX()) + _mv_bw - _ecl
                    _bd_y1 = int(_mv_brd_bb.GetY()) + _mv_bh - _ecl
                    _fp_x0 = int(_mv_fp_bb.GetX())
                    _fp_y0 = int(_mv_fp_bb.GetY())
                    _fp_x1 = _fp_x0 + int(_mv_fp_bb.GetWidth())
                    _fp_y1 = _fp_y0 + int(_mv_fp_bb.GetHeight())
                    _dx, _dy = 0, 0
                    if _fp_x0 < _bd_x0:
                        _dx = _bd_x0 - _fp_x0
                    elif _fp_x1 > _bd_x1:
                        _dx = _bd_x1 - _fp_x1
                    if _fp_y0 < _bd_y0:
                        _dy = _bd_y0 - _fp_y0
                    elif _fp_y1 > _bd_y1:
                        _dy = _bd_y1 - _fp_y1
                    if _dx != 0 or _dy != 0:
                        _mv_pos = footprint.GetPosition()
                        _mv_px = int(getattr(_mv_pos, 'x', 0))
                        _mv_py = int(getattr(_mv_pos, 'y', 0))
                        footprint.SetPosition(pcbnew.VECTOR2I(_mv_px + _dx, _mv_py + _dy))
                        _to_mm_fn = getattr(pcbnew, 'ToMM', None)
                        if callable(_to_mm_fn):
                            x_mm = round(float(_to_mm_fn(_mv_px + _dx)), 2)
                            y_mm = round(float(_to_mm_fn(_mv_py + _dy)), 2)
                        else:
                            x_mm = round((_mv_px + _dx) / 1e6, 2)
                            y_mm = round((_mv_py + _dy) / 1e6, 2)
        except Exception:
            pass

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

    async def _handle_rotate_component(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Rotate a footprint by an angle in degrees.

        This must be executed on KiCad's GUI thread (pcbnew is not thread-safe).
        """
        params = action.parameters or {}
        ref_val = (
            params.get('ref')
            or params.get('reference')
            or params.get('designator')
            or params.get('component')
        )
        ref = str(ref_val or '').strip().upper()
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
          1. Use kicad-cli pcb drc (KiCad 7/8/9+) for JSON output
        """
        import subprocess
        import tempfile
        import json as _json

        params = action.parameters or {}
        focus = str(params.get("focus", "") or params.get("mode", "") or "").strip().lower()
        placement_only = focus in {"placement", "overlap"}

        def _is_connectivity_only_issue(description: Any) -> bool:
            text = str(description or "").strip().lower()
            if not text:
                return False
            return (
                "missing connection between items" in text
                or "unconnected item" in text
                or "unconnected items" in text
            )

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
                    if placement_only and _is_connectivity_only_issue(desc):
                        continue
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

                if not placement_only:
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

        if not drc_ran:
            return False, (
                "DRC could not be run (kicad-cli failed or produced no JSON report). "
                "This check currently requires kicad-cli."
            )

        return (len(errors) == 0), self._format_drc_results(errors, warnings)

    async def _handle_run_erc(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Run PCB-native electrical sanity checks.

        VibeCAD currently avoids schematic searching/analysis. This check flags
        pads with no net assigned (netcode 0), which is actionable for PCB-only
        sessions and avoids schematic vs PCB drift.
        """
        try:
            import pcbnew as _pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available (this action must run inside KiCad)"

        board = context.get('board')
        if board is None:
            try:
                board = _pcbnew.GetBoard()
            except Exception:
                board = None
        if board is None:
            return False, "No active board found (cannot run electrical checks)"

        return self._run_pcb_electrical_check(board, _pcbnew)

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

        if hasattr(via, 'thisown'):
            via.thisown = False
        board.Add(via)
        return True, f"Via placed at ({x_mm}, {y_mm}) mm"

    async def _handle_define_board_outline(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Draw an Edge.Cuts outline centered on the page.

        Supported shapes:
        - rectangle (default)
        - rounded_rectangle (uses corner_radius)
        - circle (uses diameter or min(width, height))
        """
        try:
            import pcbnew  # type: ignore
        except Exception:
            return False, "pcbnew not available"

        params = action.parameters or {}

        def parse_mm(value: Any, default: float) -> float:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                parsed = float(value)
                return parsed if parsed > 0 else default
            if isinstance(value, str):
                text = value.strip().lower()
                if not text:
                    return default
                try:
                    parsed = float(text)
                    return parsed if parsed > 0 else default
                except Exception:
                    pass
                match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(mm|mil)?\s*$", text)
                if match:
                    parsed = float(match.group(1))
                    unit = match.group(2) or "mm"
                    if unit == "mil":
                        parsed *= 0.0254
                    return parsed if parsed > 0 else default
            return default

        def first_value(*keys: str) -> Any:
            for k in keys:
                if k in params and params.get(k) not in (None, ""):
                    return params.get(k)
            return None

        def parse_int(value: Any, default: int, min_value: int, max_value: int) -> int:
            try:
                iv = int(float(value))
            except Exception:
                iv = int(default)
            if iv < min_value:
                return min_value
            if iv > max_value:
                return max_value
            return iv

        shape_raw = str(first_value("shape", "outline_shape", "shape_type") or "rectangle").strip().lower()
        shape_aliases = {
            "rect": "rectangle",
            "box": "rectangle",
            "rounded": "rounded_rectangle",
            "rounded_rect": "rounded_rectangle",
            "rounded-rectangle": "rounded_rectangle",
            "roundrect": "rounded_rectangle",
            "round-rect": "rounded_rectangle",
            "circle": "circle",
            "circular": "circle",
        }
        shape = shape_aliases.get(shape_raw, shape_raw)
        if shape not in {"rectangle", "rounded_rectangle", "circle"}:
            shape = "rectangle"

        width = parse_mm(first_value("width", "width_mm", "w"), 100.0)
        height = parse_mm(first_value("height", "height_mm", "h"), 80.0)
        diameter = parse_mm(first_value("diameter", "diameter_mm"), min(width, height))
        corner_radius = parse_mm(
            first_value("corner_radius", "corner_radius_mm", "radius", "fillet_radius", "r"),
            0.0,
        )
        corner_segments = parse_int(first_value("corner_segments", "arc_segments"), default=8, min_value=2, max_value=64)
        circle_segments = parse_int(first_value("circle_segments", "segments"), default=64, min_value=16, max_value=256)

        if shape == "rectangle" and corner_radius > 0:
            shape = "rounded_rectangle"

        if shape == "circle":
            diameter = max(1.0, diameter)
            width = diameter
            height = diameter

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

        edge_cuts = getattr(pcbnew, 'Edge_Cuts', 44)
        try:
            edge_cuts = board.GetLayerID('Edge.Cuts')
        except Exception:
            pass

        # Replace existing Edge.Cuts outline instead of duplicating.
        removed = 0
        try:
            drawings = getattr(board, 'GetDrawings', None)
            if callable(drawings):
                for d in list(drawings()):
                    try:
                        gl = getattr(d, 'GetLayer', None)
                        if callable(gl) and int(gl()) == int(edge_cuts):
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

        def mk(x_mm: float, y_mm: float):
            ix, iy = mm2iu(x_mm), mm2iu(y_mm)
            for ctor in ('VECTOR2I', 'wxPoint'):
                try:
                    return getattr(pcbnew, ctor)(ix, iy)
                except Exception:
                    continue
            return None

        def arc_points(cx: float, cy: float, radius: float, start_deg: float, end_deg: float, steps: int) -> List[Tuple[float, float]]:
            pts: List[Tuple[float, float]] = []
            for i in range(steps + 1):
                t = float(i) / float(max(steps, 1))
                ang_deg = start_deg + (end_deg - start_deg) * t
                ang = math.radians(ang_deg)
                pts.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))
            return pts

        points: List[Tuple[float, float]] = []
        outline_desc = ""

        if shape == "circle":
            radius = max(0.5, diameter / 2.0)
            for i in range(circle_segments):
                ang = (2.0 * math.pi * float(i)) / float(circle_segments)
                points.append((cx_mm + radius * math.cos(ang), cy_mm + radius * math.sin(ang)))
            outline_desc = f"circle d={diameter:.2f}mm seg={circle_segments}"

        elif shape == "rounded_rectangle":
            radius = min(max(0.0, corner_radius), width / 2.0, height / 2.0)
            if radius <= 0.0:
                shape = "rectangle"
            else:
                left, top = ox_mm, oy_mm
                right, bottom = ox_mm + width, oy_mm + height

                points.append((left + radius, top))
                points.append((right - radius, top))
                points.extend(arc_points(right - radius, top + radius, radius, -90.0, 0.0, corner_segments)[1:])
                points.append((right, bottom - radius))
                points.extend(arc_points(right - radius, bottom - radius, radius, 0.0, 90.0, corner_segments)[1:])
                points.append((left + radius, bottom))
                points.extend(arc_points(left + radius, bottom - radius, radius, 90.0, 180.0, corner_segments)[1:])
                points.append((left, top + radius))
                points.extend(arc_points(left + radius, top + radius, radius, 180.0, 270.0, corner_segments)[1:])

                outline_desc = (
                    f"rounded_rectangle {width:.2f}x{height:.2f}mm "
                    f"r={radius:.2f}mm seg={corner_segments}"
                )

        if shape == "rectangle":
            points = [
                (ox_mm, oy_mm),
                (ox_mm + width, oy_mm),
                (ox_mm + width, oy_mm + height),
                (ox_mm, oy_mm + height),
            ]
            outline_desc = f"rectangle {width:.2f}x{height:.2f}mm"

        if len(points) < 3:
            return False, "Failed to build outline geometry"

        segment_count = 0
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
                continue
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

            p1 = mk(x1, y1)
            p2 = mk(x2, y2)
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
                if hasattr(seg, 'thisown'):
                    seg.thisown = False
                board.Add(seg)
                segment_count += 1
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
            "DEFINE_BOARD_OUTLINE DONE: shape=%s page=(%.2f,%.2f) center=(%.2f,%.2f) origin=(%.2f,%.2f) segments=%d removed=%d detail=%s",
            shape,
            float(pw_mm),
            float(ph_mm),
            float(cx_mm),
            float(cy_mm),
            float(ox_mm),
            float(oy_mm),
            int(segment_count),
            int(removed),
            outline_desc,
        )

        return True, (
            f"Board outline ({shape}) centered at ({cx_mm:.1f}, {cy_mm:.1f}) mm. "
            f"Size: {width:.1f}x{height:.1f} mm. Segments: {segment_count}."
        )

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
        refs_val = params.get('refs')
        direction = str(params.get('direction', 'horizontally') or '').strip().lower()

        if not refs_val:
            return False, "No component references specified"

        if not isinstance(refs_val, list):
            return False, "ALIGN_COMPONENTS.parameters.refs must be a list of references"

        refs = [str(r).strip().upper() for r in refs_val if str(r or "").strip()]
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

        if hasattr(pcb_text, 'thisown'):
            pcb_text.thisown = False
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
        
        if hasattr(zone, 'thisown'):
            zone.thisown = False
        board.Add(zone)

        return True, f"Copper zone added on {layer_name} for net {net_name} ({width}x{height} mm)"

    async def _handle_autoroute(self, action: DesignAction, context: Dict) -> Tuple[bool, str]:
        """Autoroute the board via Freerouting.

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

        result = autoroute(board_path, allow_manhattan_fallback=False)
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
            logger.info("SEARCH_WEB query=%r", query)
            from .component_search import ComponentWebSearch
            searcher = ComponentWebSearch()
            ql = query.lower()
            github_kicad_lookup = (
                ("github.com" in ql or "github" in ql)
                and "kicad" in ql
                and ("symbol" in ql or "footprint" in ql)
            )
            if github_kicad_lookup:
                results = searcher.search_github(query, limit=5)
            else:
                results = searcher.search(query, limit=5)
            if not results:
                logger.info("SEARCH_WEB query=%r results=0", query)
                return True, f"No results found for '{query}'. Try a different search term or MPN."
            logger.info("SEARCH_WEB query=%r results=%d", query, len(results))
            try:
                store = context.setdefault("search_web_results", {})
                if isinstance(store, dict):
                    store[query] = [
                        (r.to_dict() if hasattr(r, "to_dict") else {})
                        for r in results
                    ]
            except Exception:
                pass
            text_parts = [f"## Web Search Results for '{query}'\n"]
            for r in results:
                text_parts.append(r.to_text())
                text_parts.append("")
            return True, "\n".join(text_parts)
        except Exception as e:
            logger.exception("Web search failed for query=%r", query)
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
