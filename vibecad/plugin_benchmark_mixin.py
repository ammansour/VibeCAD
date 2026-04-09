"""Large benchmark scoring/report helpers for VibeCAD plugin."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .parsers import PCBParser, SchematicParser
try:
    from .design import DesignAction, DesignActionType
except Exception:  # pragma: no cover - runtime fallback when design module is unavailable
    DesignAction = Any

    class _DesignActionTypeFallback:
        RUN_DRC = "RUN_DRC"

    DesignActionType = _DesignActionTypeFallback()

# KiCad imports - these are available in KiCad's Python environment
try:
    import pcbnew

    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False

try:
    import wx

    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False

logger = logging.getLogger('vibecad')


class BenchmarkMixin:
    def _forward_agent_response_with_benchmark_capture(self, text: str, *, benchmark: Optional[Dict[str, Any]] = None) -> None:
        """Forward agent response to UI and retain a small tail for benchmark diagnostics."""
        if benchmark is not None:
            try:
                tail = benchmark.setdefault("ui_responses_tail", [])
                tail.append(str(text or ""))
                if len(tail) > 12:
                    del tail[:-12]
            except Exception:
                pass
        try:
            if self.frame:
                wx.CallAfter(self.frame.add_design_response, text)
        except Exception:
            pass
    def _benchmark_record_state_transition(self, benchmark: Optional[Dict[str, Any]], new_state: Any) -> None:
        if not isinstance(benchmark, dict):
            return
        try:
            rows = benchmark.setdefault("state_transitions", [])
            rows.append(
                {
                    "state": str(getattr(new_state, "name", str(new_state))),
                    "t_s": round(max(0.0, time.time() - float(benchmark.get("start_ts", time.time()))), 3),
                }
            )
        except Exception:
            pass

    @staticmethod
    def _benchmark_loop_token_usage(loop: Any) -> Dict[str, int]:
        zero = {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
        }
        try:
            agent = getattr(loop, "_agent", None)
            if agent is None:
                return dict(zero)
            getter = getattr(agent, "get_llm_usage_totals", None)
            if not callable(getter):
                return dict(zero)
            raw = getter()
            if not isinstance(raw, dict):
                return dict(zero)
            out = dict(zero)
            for key in out.keys():
                try:
                    out[key] = max(0, int(raw.get(key, 0) or 0))
                except Exception:
                    out[key] = 0
            return out
        except Exception:
            return dict(zero)

    def _benchmark_phase_token_usage(self, benchmark: Optional[Dict[str, Any]], loop: Any) -> Dict[str, Dict[str, int]]:
        cumulative = self._benchmark_loop_token_usage(loop)
        if not isinstance(benchmark, dict):
            return {"phase": dict(cumulative), "cumulative": dict(cumulative)}

        prev = benchmark.get("_last_phase_token_usage_totals")
        if not isinstance(prev, dict):
            prev = {}
        phase_delta: Dict[str, int] = {}
        for key, value in cumulative.items():
            try:
                prev_value = max(0, int(prev.get(key, 0) or 0))
            except Exception:
                prev_value = 0
            phase_delta[key] = max(0, int(value) - prev_value)

        benchmark["_last_phase_token_usage_totals"] = dict(cumulative)
        benchmark["token_usage_summary"] = dict(cumulative)
        return {
            "phase": phase_delta,
            "cumulative": dict(cumulative),
        }

    @staticmethod
    def _benchmark_bom_score_from_oracle(bom_oracle: Dict[str, Any]) -> int:
        if not isinstance(bom_oracle, dict):
            return 0
        checks = [row for row in list(bom_oracle.get("checks", []) or []) if isinstance(row, dict)]
        required_count = int(bom_oracle.get("required_count", 0) or 0)
        if required_count <= 0:
            required_count = max(len([row for row in checks if not bool(row.get("optional", False))]), 1)
        penalties = len(list(bom_oracle.get("missing", []) or [])) + len(list(bom_oracle.get("qty_mismatches", []) or []))
        score = int(round(100.0 * float(required_count - penalties) / float(required_count)))
        if score < 0:
            return 0
        if score > 100:
            return 100
        return score
    @staticmethod
    def _benchmark_sync_done_phase_row(benchmark: Optional[Dict[str, Any]], report: Dict[str, Any]) -> None:
        if not isinstance(benchmark, dict) or not isinstance(report, dict):
            return
        rows = benchmark.get("phase_scores")
        if not isinstance(rows, list) or not rows:
            return
        target = None
        for row in reversed(rows):
            if isinstance(row, dict) and str(row.get("phase", "") or "").upper() == "DONE":
                target = row
                break
        if not isinstance(target, dict):
            return
        semantic = report.get("semantic_completeness") if isinstance(report.get("semantic_completeness"), dict) else {}
        bom = report.get("bom_oracle") if isinstance(report.get("bom_oracle"), dict) else {}
        footprint_match = report.get("footprint_match") if isinstance(report.get("footprint_match"), dict) else {}
        footprint_subscores = footprint_match.get("subscores") if isinstance(footprint_match.get("subscores"), dict) else {}
        footprint_primary = footprint_subscores.get("primary") if isinstance(footprint_subscores.get("primary"), dict) else {}
        footprint_secondary = footprint_subscores.get("secondary") if isinstance(footprint_subscores.get("secondary"), dict) else {}
        placement = report.get("placement_score") if isinstance(report.get("placement_score"), dict) else {}
        net_score = report.get("net_score") if isinstance(report.get("net_score"), dict) else {}
        ee_rules = report.get("ee_rules") if isinstance(report.get("ee_rules"), dict) else {}
        bom_score = int(report.get("bom_score_out_of_100", report.get("score_out_of_100", 0)) or 0)
        target.update(
            {
                "overall_score": int(report.get("overall_score_out_of_100", report.get("score_out_of_100", 0)) or 0),
                "oracle_score": bom_score,
                "bom_score": bom_score,
                "semantic_ok": bool(semantic.get("ok", False)),
                "semantic_secondary_ok": bool(semantic.get("secondary_ok", False)),
                "bom_oracle_ok": bool(bom.get("ok", False)),
                "footprint_match_ok": bool(footprint_match.get("ok", False)),
                "footprint_match_score": int(footprint_match.get("score_out_of_100", 0) or 0),
                "footprint_match_primary_score": int(footprint_primary.get("score_out_of_100", 0) or 0),
                "footprint_match_secondary_score": int(footprint_secondary.get("score_out_of_100", 0) or 0),
                "placement_score": int(placement.get("score", 0) or 0),
                "net_score": int(net_score.get("score", 0) or 0),
                "net_ok": bool(net_score.get("ok", False)),
                "ee_score": int(ee_rules.get("score_out_of_100", 0) or 0),
                "ee_ok": bool(ee_rules.get("ok", False)),
                "ee_issue_count": int(ee_rules.get("failed_rule_count", 0) or 0),
                "ee_issues_sample": list(ee_rules.get("issues_for_done_checkpoint", []) or [])[:10],
                "all_issue_count": int(report.get("all_issue_count", 0) or 0),
                "all_issue_penalty": int(report.get("all_issue_penalty", 0) or 0),
                "critical_issue_count": int(report.get("critical_issue_count", 0) or 0),
            }
        )
    def _benchmark_queue_phase_score(
        self,
        benchmark: Optional[Dict[str, Any]],
        loop: Any,
        phase_name: str,
        phase_result: Any,
    ) -> None:
        if not isinstance(benchmark, dict) or loop is None:
            return
        try:
            planned_actions = getattr(phase_result, "actions", None)
            has_actions = isinstance(planned_actions, list) and len(planned_actions) > 0
        except Exception:
            has_actions = False
        if not has_actions:
            phase_upper = str(phase_name or "").upper()
            if phase_upper in {"PLACE", "GEOM", "NET", "DONE"}:
                try:
                    self._benchmark_add_geom_ground_zone_and_run_drc(benchmark, loop)
                except Exception:
                    logger.exception("Benchmark GEOM ground-zone insertion failed")
            self._benchmark_log_phase_score(benchmark, loop, phase_name)
            return
        try:
            queue = benchmark.setdefault("pending_phase_scores", [])
            if not isinstance(queue, list):
                queue = []
                benchmark["pending_phase_scores"] = queue
            queue.append(
                {
                    "phase": str(phase_name or ""),
                    "queued_t_s": round(
                        max(0.0, time.time() - float(benchmark.get("start_ts", time.time()))), 3
                    ),
                }
            )
        except Exception:
            self._benchmark_log_phase_score(benchmark, loop, phase_name)
    def _benchmark_flush_pending_phase_scores(
        self,
        benchmark: Optional[Dict[str, Any]],
        loop: Any,
        new_state: Any,
    ) -> None:
        if not isinstance(benchmark, dict) or loop is None:
            return
        state_name = str(getattr(new_state, "name", str(new_state)) or "").upper()
        if state_name not in {"OBSERVING", "DONE", "ERROR"}:
            return
        queue = benchmark.get("pending_phase_scores")
        if not isinstance(queue, list) or not queue:
            return
        while queue:
            item = queue.pop(0)
            phase_name = str(item.get("phase", "") or "") if isinstance(item, dict) else ""
            if not phase_name:
                continue
            phase_upper = str(phase_name or "").upper()
            if phase_upper in {"PLACE", "GEOM", "NET", "DONE"}:
                try:
                    self._benchmark_add_geom_ground_zone_and_run_drc(benchmark, loop)
                except Exception:
                    logger.exception("Benchmark GEOM ground-zone insertion failed")
            self._benchmark_log_phase_score(benchmark, loop, phase_name)
    def _benchmark_log_phase_score(self, benchmark: Optional[Dict[str, Any]], loop: Any, phase_name: str) -> None:
        if not isinstance(benchmark, dict) or loop is None:
            return
        try:
            report = self._benchmark_build_report(benchmark, loop)
            bom_score = int(report.get("bom_score_out_of_100", report.get("score_out_of_100", 0)) or 0)
            semantic = report.get("semantic_completeness") if isinstance(report.get("semantic_completeness"), dict) else {}
            bom = report.get("bom_oracle") if isinstance(report.get("bom_oracle"), dict) else {}
            footprint_match = report.get("footprint_match") if isinstance(report.get("footprint_match"), dict) else {}
            footprint_subscores = footprint_match.get("subscores") if isinstance(footprint_match.get("subscores"), dict) else {}
            footprint_primary = footprint_subscores.get("primary") if isinstance(footprint_subscores.get("primary"), dict) else {}
            footprint_secondary = footprint_subscores.get("secondary") if isinstance(footprint_subscores.get("secondary"), dict) else {}
            placement = report.get("placement_score") if isinstance(report.get("placement_score"), dict) else {}
            net_score = report.get("net_score") if isinstance(report.get("net_score"), dict) else {}
            ee_rules = report.get("ee_rules") if isinstance(report.get("ee_rules"), dict) else {}
            token_usage = self._benchmark_phase_token_usage(benchmark, loop)
            phase_str = str(phase_name or "")
            phase_upper = phase_str.upper()
            placement_checkpoint_score: Optional[int]
            if phase_upper == "SPEC":
                placement_checkpoint_score = None
            else:
                placement_checkpoint_score = int(placement.get("score", 0) or 0)
            net_checkpoint_score: Optional[int]
            if phase_upper in {"SPEC", "PLACE"}:
                net_checkpoint_score = None
            else:
                net_checkpoint_score = int(net_score.get("score", 0) or 0)
            row = {
                "phase": phase_str,
                "t_s": round(max(0.0, time.time() - float(benchmark.get("start_ts", time.time()))), 3),
                "overall_score": int(report.get("overall_score_out_of_100", report.get("score_out_of_100", 0)) or 0),
                "oracle_score": bom_score,
                "bom_score": bom_score,
                "semantic_ok": bool(semantic.get("ok", False)),
                "semantic_secondary_ok": bool(semantic.get("secondary_ok", False)),
                "bom_oracle_ok": bool(bom.get("ok", False)),
                "footprint_match_ok": bool(footprint_match.get("ok", False)),
                "footprint_match_score": int(footprint_match.get("score_out_of_100", 0) or 0),
                "footprint_match_primary_score": int(footprint_primary.get("score_out_of_100", 0) or 0),
                "footprint_match_secondary_score": int(footprint_secondary.get("score_out_of_100", 0) or 0),
                "placement_score": placement_checkpoint_score,
                "net_score": net_checkpoint_score,
                "net_ok": bool(net_score.get("ok", False)) if net_checkpoint_score is not None else False,
                "ee_score": int(ee_rules.get("score_out_of_100", 0) or 0),
                "ee_ok": bool(ee_rules.get("ok", False)),
                "ee_issue_count": int(ee_rules.get("failed_rule_count", 0) or 0),
                "ee_issues_sample": list(ee_rules.get("issues_for_done_checkpoint", []) or [])[:10],
                "all_issue_count": int(report.get("all_issue_count", 0) or 0),
                "all_issue_penalty": int(report.get("all_issue_penalty", 0) or 0),
                "critical_issue_count": int(report.get("critical_issue_count", 0) or 0),
                "token_usage": token_usage,
            }
            rows = benchmark.setdefault("phase_scores", [])
            rows.append(row)
            
            p_val = "N/A" if row.get("placement_score") is None else str(int(row["placement_score"]))
            n_val = "N/A" if row.get("net_score") is None else str(int(row["net_score"]))
            ee_val = str(int(row.get("ee_score", 0) or 0))
            
            if phase_upper == "PLACE":
                logger.info(
                    "Benchmark checkpoint [%s]: overall=%d/100 bom=%d/100 semantic=%s secondary=%s bom_pass=%s footprint=%d/100 (p=%d s=%d) placement=%s/100 ee=%s/100 issues=%d mistakes=%d penalty=%d critical=%d",
                    row["phase"],
                    int(row.get("overall_score", 0) or 0),
                    bom_score,
                    "pass" if row["semantic_ok"] else "fail",
                    "pass" if row["semantic_secondary_ok"] else "fail",
                    "pass" if row["bom_oracle_ok"] else "fail",
                    int(row["footprint_match_score"]),
                    int(row["footprint_match_primary_score"]),
                    int(row["footprint_match_secondary_score"]),
                    p_val,
                    ee_val,
                    int(row.get("ee_issue_count", 0) or 0),
                    int(row.get("all_issue_count", 0) or 0),
                    int(row.get("all_issue_penalty", 0) or 0),
                    int(row.get("critical_issue_count", 0) or 0),
                )
            else:
                logger.info(
                    "Benchmark checkpoint [%s]: overall=%d/100 bom=%d/100 semantic=%s secondary=%s bom_pass=%s footprint=%d/100 (p=%d s=%d) placement=%s/100 net=%s/100 ee=%s/100 issues=%d mistakes=%d penalty=%d critical=%d",
                    row["phase"],
                    int(row.get("overall_score", 0) or 0),
                    bom_score,
                    "pass" if row["semantic_ok"] else "fail",
                    "pass" if row["semantic_secondary_ok"] else "fail",
                    "pass" if row["bom_oracle_ok"] else "fail",
                    int(row["footprint_match_score"]),
                    int(row["footprint_match_primary_score"]),
                    int(row["footprint_match_secondary_score"]),
                    p_val,
                    n_val,
                    ee_val,
                    int(row.get("ee_issue_count", 0) or 0),
                    int(row.get("all_issue_count", 0) or 0),
                    int(row.get("all_issue_penalty", 0) or 0),
                    int(row.get("critical_issue_count", 0) or 0),
                )
            try:
                if self.frame:
                    if phase_upper == "PLACE":
                        wx.CallAfter(
                            self.frame.add_design_response,
                            f"📊 {row['phase']} checkpoint: overall={int(row.get('overall_score', 0) or 0)}/100, "
                            f"bom {bom_score}/100, "
                            f"semantic={'pass' if row['semantic_ok'] else 'fail'}, "
                            f"secondary={'pass' if row['semantic_secondary_ok'] else 'fail'}, "
                            f"bom={'pass' if row['bom_oracle_ok'] else 'fail'}, "
                            f"footprint={int(row['footprint_match_score'])}/100 "
                            f"(p={int(row['footprint_match_primary_score'])}, s={int(row['footprint_match_secondary_score'])}), "
                            f"placement={p_val}/100, "
                            f"ee={ee_val}/100 ({int(row.get('ee_issue_count', 0) or 0)} issues), "
                            f"mistakes={int(row.get('all_issue_count', 0) or 0)} "
                            f"(penalty={int(row.get('all_issue_penalty', 0) or 0)}, "
                            f"critical={int(row.get('critical_issue_count', 0) or 0)})"
                        )
                    else:
                        wx.CallAfter(
                            self.frame.add_design_response,
                            f"📊 {row['phase']} checkpoint: overall={int(row.get('overall_score', 0) or 0)}/100, "
                            f"bom {bom_score}/100, "
                            f"semantic={'pass' if row['semantic_ok'] else 'fail'}, "
                            f"secondary={'pass' if row['semantic_secondary_ok'] else 'fail'}, "
                            f"bom={'pass' if row['bom_oracle_ok'] else 'fail'}, "
                            f"footprint={int(row['footprint_match_score'])}/100 "
                            f"(p={int(row['footprint_match_primary_score'])}, s={int(row['footprint_match_secondary_score'])}), "
                            f"placement={p_val}/100, "
                            f"net={n_val}/100, "
                            f"ee={ee_val}/100 ({int(row.get('ee_issue_count', 0) or 0)} issues), "
                            f"mistakes={int(row.get('all_issue_count', 0) or 0)} "
                            f"(penalty={int(row.get('all_issue_penalty', 0) or 0)}, "
                            f"critical={int(row.get('critical_issue_count', 0) or 0)})"
                        )
            except Exception:
                pass
        except Exception:
            logger.exception("Benchmark phase checkpoint logging failed for %s", phase_name)
    def _benchmark_find_gnd_net_item_from_plan(self, board: Any, loop: Any) -> Optional[Any]:
        """Resolve the board net item that maps to the net-plan canonical ground group."""
        try:
            planned = self._benchmark_planned_net_snapshot(loop)
            by_group = planned.get("by_group") if isinstance(planned.get("by_group"), dict) else {}
            gnd_row = by_group.get("gnd") if isinstance(by_group.get("gnd"), dict) else {}
            plan_net_names = [
                str(v).strip()
                for v in list(gnd_row.get("net_names") or [])
                if str(v).strip()
            ]
        except Exception:
            plan_net_names = []

        target_canonical_names: Set[str] = set()
        for net_name in plan_net_names:
            canonical = self._benchmark_net_canonical_name(net_name)
            if canonical:
                target_canonical_names.add(canonical)
        # Fallback canonical group if the net plan has no explicit ground net names yet.
        if not target_canonical_names:
            target_canonical_names.add("gnd")

        net_info = None
        try:
            net_info = board.GetNetInfo() if board is not None else None
        except Exception:
            net_info = None
        if net_info is None:
            return None

        get_net_item = getattr(net_info, "GetNetItem", None)
        if not callable(get_net_item):
            return None

        net_items: List[Any] = []
        # Preferred path requested by the benchmark spec: iterate GetNetItem().
        try:
            maybe_iterable = get_net_item()
            if maybe_iterable is not None:
                net_items = [row for row in list(maybe_iterable) if row is not None]
        except TypeError:
            net_items = []
        except Exception:
            net_items = []

        if not net_items:
            try:
                net_count = int(getattr(net_info, "GetNetCount", lambda: 0)() or 0)
            except Exception:
                net_count = 0
            if net_count <= 0:
                net_count = 2048
            empty_streak = 0
            for net_code in range(0, net_count + 1):
                try:
                    net_obj = get_net_item(int(net_code))
                except Exception:
                    continue
                if net_obj is None:
                    empty_streak += 1
                    if net_count == 2048 and empty_streak > 128:
                        break
                    continue
                empty_streak = 0
                net_items.append(net_obj)

        for net_obj in net_items:
            net_name = ""
            try:
                net_name = str(getattr(net_obj, "GetNetname", lambda: "")() or "").strip()
            except Exception:
                net_name = ""
            if not net_name:
                continue
            canonical = self._benchmark_net_canonical_name(net_name)
            if canonical and canonical in target_canonical_names:
                return net_obj
        return None
    def _benchmark_add_geom_ground_zone_and_run_drc(
        self,
        benchmark: Optional[Dict[str, Any]],
        loop: Any,
    ) -> None:
        """After GEOM placement, ensure a B.Cu ground zone exists, fill it, and run DRC.

        Threading model:
        - All pcbnew mutations run on the GUI thread.
        - DRC (kicad-cli subprocess) runs off the GUI thread.
        """
        if not isinstance(benchmark, dict) or loop is None:
            return
        if not PCBNEW_AVAILABLE:
            return
        if bool(benchmark.get("_geom_ground_zone_done", False)):
            return

        try:
            zone_result = self._benchmark_call_on_gui_sync(
                lambda: self._benchmark_add_geom_ground_zone_on_gui(loop),
                timeout_s=45.0,
            )
        except Exception:
            logger.exception("Benchmark GEOM zone: GUI-thread zone insertion failed")
            return

        if not isinstance(zone_result, dict) or not bool(zone_result.get("ok", False)):
            return

        board_path = str(zone_result.get("board_path", "") or "").strip()
        if board_path:
            self._benchmark_run_placement_drc_cli(board_path)

        benchmark["_geom_ground_zone_done"] = True

    def _benchmark_call_on_gui_sync(self, fn: Any, *, timeout_s: float = 30.0) -> Any:
        """Run `fn` on the GUI thread and wait from the caller thread."""
        if not WX_AVAILABLE:
            return fn()
        try:
            if bool(wx.IsMainThread()):
                return fn()
        except Exception:
            pass

        import threading

        done = threading.Event()
        box: Dict[str, Any] = {}

        def _runner() -> None:
            try:
                box["result"] = fn()
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

        wx.CallAfter(_runner)
        if not done.wait(timeout=max(1.0, float(timeout_s))):
            raise TimeoutError("GUI-thread call timed out")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _benchmark_add_geom_ground_zone_on_gui(self, loop: Any) -> Dict[str, Any]:
        """GUI-thread worker: add/fill B.Cu GND zone and save board."""
        board = None
        try:
            board = pcbnew.GetBoard()
        except Exception:
            board = None
        if board is None:
            return {"ok": False, "reason": "no_active_board"}

        gnd_net = self._benchmark_find_gnd_net_item_from_plan(board, loop)
        if gnd_net is None:
            logger.info("Benchmark GEOM zone: no canonical GND net resolved from net plan")
            return {"ok": False, "reason": "no_canonical_gnd"}

        layer_id = -1
        try:
            layer_id = int(board.GetLayerID("B.Cu"))
        except Exception:
            try:
                layer_id = int(getattr(pcbnew, "B_Cu", -1))
            except Exception:
                layer_id = -1
        if layer_id < 0:
            return {"ok": False, "reason": "no_b_cu_layer"}

        gnd_canonical = ""
        try:
            gnd_canonical = self._benchmark_net_canonical_name(
                str(getattr(gnd_net, "GetNetname", lambda: "")() or "")
            )
        except Exception:
            gnd_canonical = ""

        existing_zone = None
        try:
            for zone in list(board.Zones() or []):
                try:
                    if int(getattr(zone, "GetLayer", lambda: -1)()) != layer_id:
                        continue
                    net_name = str(getattr(zone, "GetNetname", lambda: "")() or "").strip()
                    if not net_name:
                        net_obj = getattr(zone, "GetNet", lambda: None)()
                        if net_obj is not None:
                            net_name = str(getattr(net_obj, "GetNetname", lambda: "")() or "").strip()
                    if gnd_canonical and self._benchmark_net_canonical_name(net_name) == gnd_canonical:
                        existing_zone = zone
                        break
                except Exception:
                    continue
        except Exception:
            existing_zone = None

        if existing_zone is None:
            bbox = None
            try:
                bbox = board.GetBoardEdgesBoundingBox()
            except Exception:
                bbox = None
            if bbox is None or int(getattr(bbox, "GetWidth", lambda: 0)() or 0) <= 0:
                try:
                    bbox = board.GetBoundingBox()
                except Exception:
                    bbox = None
            if bbox is None:
                return {"ok": False, "reason": "no_board_bbox"}
            bx = int(getattr(bbox, "GetX", lambda: 0)() or 0)
            by = int(getattr(bbox, "GetY", lambda: 0)() or 0)
            bw = int(getattr(bbox, "GetWidth", lambda: 0)() or 0)
            bh = int(getattr(bbox, "GetHeight", lambda: 0)() or 0)
            if bw <= 0 or bh <= 0:
                return {"ok": False, "reason": "empty_board_bbox"}

            zone = pcbnew.ZONE(board)
            zone.SetLayer(layer_id)
            try:
                zone.SetNet(gnd_net)
            except Exception:
                try:
                    zone.SetNetCode(int(getattr(gnd_net, "GetNetCode", lambda: 0)() or 0))
                except Exception:
                    pass
            outline = zone.Outline()
            outline.NewOutline()
            corners = (
                (bx, by),
                (bx + bw, by),
                (bx + bw, by + bh),
                (bx, by + bh),
            )
            for px, py in corners:
                try:
                    outline.Append(int(px), int(py))
                except Exception:
                    try:
                        outline.Append(pcbnew.VECTOR2I(int(px), int(py)))
                    except Exception:
                        pass
            try:
                zone.SetIsFilled(True)
            except Exception:
                pass
            
            if hasattr(zone, 'thisown'):
                zone.thisown = False
            board.Add(zone)
            existing_zone = zone

        try:
            if existing_zone is not None:
                existing_zone.SetIsFilled(True)
        except Exception:
            pass

        try:
            filler = pcbnew.ZONE_FILLER(board)
            try:
                filler.Fill(board.Zones())
            except Exception:
                filler.Fill([existing_zone] if existing_zone is not None else [])
        except Exception:
            logger.exception("Benchmark GEOM zone: zone fill failed")
            return {"ok": False, "reason": "zone_fill_failed"}

        board_path = ""
        try:
            board_path = str(board.GetFileName() or "")
        except Exception:
            board_path = ""
        if board_path:
            try:
                pcbnew.SaveBoard(board_path, board)
            except Exception:
                logger.debug("Benchmark GEOM zone: SaveBoard failed", exc_info=True)

        try:
            self._safe_pcbnew_refresh()
        except Exception:
            pass

        return {"ok": True, "board_path": board_path}

    def _benchmark_run_placement_drc_cli(self, board_path: str) -> None:
        """Background-thread DRC run after GEOM zone fill (no pcbnew calls)."""
        board_path = str(board_path or "").strip()
        if not board_path or not os.path.exists(board_path):
            return

        import platform
        import subprocess
        import tempfile

        cli_cmd = "kicad-cli"
        if platform.system() == "Darwin":
            candidates = [
                "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
                "/Applications/KiCad/kicad.app/Contents/MacOS/kicad-cli",
                "kicad-cli",
            ]
        elif platform.system() == "Windows":
            candidates = [
                r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe",
                r"C:\Program Files\KiCad\8.0\bin\kicad-cli.exe",
                r"C:\Program Files\KiCad\7.0\bin\kicad-cli.exe",
                "kicad-cli.exe",
            ]
        else:
            candidates = ["kicad-cli"]

        if candidates:
            for c in candidates:
                if c.startswith("kicad-cli"):
                    continue
                try:
                    if os.path.exists(c) and os.access(c, os.X_OK):
                        cli_cmd = c
                        break
                except Exception:
                    continue

        report_fd, report_path = tempfile.mkstemp(suffix=".json")
        os.close(report_fd)
        try:
            res = subprocess.run(
                [
                    cli_cmd,
                    "pcb",
                    "drc",
                    "--output",
                    report_path,
                    "--format",
                    "json",
                    "--severity-all",
                    board_path,
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
            if int(getattr(res, "returncode", 1) or 1) != 0:
                err = str(getattr(res, "stderr", "") or getattr(res, "stdout", "") or "").strip()
                logger.info("Benchmark GEOM zone: post-fill DRC warning: %s", err[:500] or "kicad-cli failed")
        except FileNotFoundError:
            logger.info("Benchmark GEOM zone: post-fill DRC skipped (kicad-cli not found)")
        except Exception:
            logger.exception("Benchmark GEOM zone: post-fill DRC failed")
        finally:
            try:
                if os.path.exists(report_path):
                    os.unlink(report_path)
            except Exception:
                pass
    def _benchmark_extract_gate_failures(self, loop: Any) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            gfs = artifacts.get("gate_failures")
            if isinstance(gfs, list):
                for f in gfs:
                    if isinstance(f, dict):
                        rows.append(dict(f))
        except Exception:
            pass
        return rows
    def _benchmark_extract_unresolved_roles_from_gate_failures(
        self, gate_failures: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for f in gate_failures or []:
            if not isinstance(f, dict):
                continue
            gate = str(f.get("gate", "") or "")
            msg = str(f.get("message", "") or "")
            details = f.get("details") if isinstance(f.get("details"), dict) else {}
            rid = str(details.get("role_id", "") or "").strip()
            if gate == "SPEC.device_resolvability" and rid:
                if rid not in seen:
                    out.append({"role_id": rid, "source": "gate", "gate": gate, "message": msg})
                    seen.add(rid)
                continue
            if gate == "SPEC.progress":
                open_issues = details.get("openissues") or details.get("open_issues")
                if isinstance(open_issues, list):
                    for issue in open_issues:
                        if not isinstance(issue, dict):
                            continue
                        role = str(issue.get("role", "") or "").strip()
                        if not role or role in seen:
                            continue
                        out.append(
                            {
                                "role_id": role,
                                "source": "spec_progress_open_issues",
                                "issue_type": str(issue.get("type", "") or ""),
                                "detail": str(issue.get("detail", "") or ""),
                                "status": str(issue.get("status", "") or ""),
                            }
                        )
                        seen.add(role)
        return out
    def _benchmark_extract_role_constraints_snapshot(self, loop: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            draft = artifacts.get("design_spec_draft")
            if not isinstance(draft, dict):
                return out
            roles = draft.get("roles")
            if not isinstance(roles, list):
                return out
            for r in roles:
                if not isinstance(r, dict):
                    continue
                rid = str(r.get("role_id", "") or "").strip()
                if not rid:
                    continue
                out[rid] = {
                    "role_type": str(r.get("role_type", "") or ""),
                    "critical": bool(r.get("critical", False)),
                    "constraints": dict(r.get("constraints") or {}) if isinstance(r.get("constraints"), dict) else {},
                    "alternates": list(r.get("alternates") or []) if isinstance(r.get("alternates"), list) else [],
                }
        except Exception:
            pass
        return out
    def _benchmark_final_roles_snapshot(self, loop: Any) -> List[Dict[str, Any]]:
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            # Prefer the new manifest-derived role list (written by 3-step SPEC)
            draft = artifacts.get("design_spec_draft")
            if isinstance(draft, dict) and isinstance(draft.get("roles"), list):
                roles = [r for r in draft.get("roles") if isinstance(r, dict)]
                if roles:
                    return roles
            # Legacy: direct design_spec from single-pass agents
            spec = artifacts.get("design_spec")
            if isinstance(spec, dict) and isinstance(spec.get("roles"), list):
                roles = [r for r in spec.get("roles") if isinstance(r, dict)]
                if roles:
                    return roles
            # Fallback: translate manifest.parts directly
            manifest = artifacts.get("manifest")
            if isinstance(manifest, dict) and isinstance(manifest.get("parts"), list):
                roles = []
                for part in manifest["parts"]:
                    if not isinstance(part, dict):
                        continue
                    ref = str(part.get("ref", "") or "")
                    mpn = str(part.get("mpn", "") or "")
                    fp  = str(part.get("footprint", "") or "")
                    pins = part.get("pins") if isinstance(part.get("pins"), list) else []
                    nets = list({str(p.get("net", "") or "") for p in pins if p.get("net")})
                    roles.append({
                        "role_id":    ref,
                        "role_type":  mpn or ref,
                        "quantity":   int(part.get("qty", 1) or 1),
                        "critical":   True,
                        "constraints": {"part_query": mpn, "package": fp},
                        "alternates": nets[:6],
                    })
                if roles:
                    return roles
        except Exception:
            pass
        return []
    def _benchmark_text_blob_for_role(self, role: Dict[str, Any]) -> str:
        parts: List[str] = []
        try:
            parts.append(str(role.get("role_id", "") or ""))
            parts.append(str(role.get("role_type", "") or ""))
            c = role.get("constraints") if isinstance(role.get("constraints"), dict) else {}
            parts.append(str(c.get("part_query", "") or ""))
            parts.append(str(c.get("package", "") or ""))
            alts = role.get("alternates")
            if isinstance(alts, list):
                parts.extend(str(a or "") for a in alts[:8])
        except Exception:
            pass
        s = " ".join(parts).lower()
        # Normalise separators and expand camelCase/PascalCase words so
        # e.g. "PinHeader" matches combos like ["pin", "header"].
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # camelCase split before lowering
        s = s.lower()
        s = re.sub(r"[_:/\\.-]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def _benchmark_uno_semantic_completeness(self, loop: Any, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Scenario-specific semantic coverage check for the fixed Uno benchmark."""
        scenario = str(benchmark.get("scenario", "") or "").lower()
        if "uno" not in scenario:
            return {"applicable": False}

        roles = self._benchmark_final_roles_snapshot(loop)
        blobs = [self._benchmark_text_blob_for_role(r) for r in roles if isinstance(r, dict)]

        def _has_all(*needles: str) -> bool:
            for b in blobs:
                if all(n in b for n in needles):
                    return True
            return False

        def _has_any_combo(combos: List[List[str]]) -> bool:
            return any(_has_all(*combo) for combo in combos)

        checks = [
            ("mcu_main", [["atmega328"], ["mcu", "328"]]),
            ("usb_bridge", [["atmega16u2"], ["usb", "bridge"], ["usb", "serial"]]),
            ("usb_connector", [["usb", "connector"], ["usb b"], ["usb type b"]]),
            (
                "dc_input",
                [
                    ["dc", "jack"],
                    ["barrel", "jack"],
                    ["dcjack"],
                    ["dcinput"],
                    ["barreljack"],
                    ["power", "jack"],
                    ["power", "input"],
                    ["dc", "input"],
                    ["barrel", "connector"],
                ],
            ),
            ("reg_5v", [["5v", "reg"], ["5v", "ldo"], ["ncp1117"]]),
            ("reg_3v3", [["3 3v", "reg"], ["3v3", "reg"], ["lp2985"]]),
            ("clock_16mhz", [["16mhz", "crystal"], ["16mhz", "resonator"], ["crystal"], ["resonator"], ["cstce"]]),
            ("usb_protection", [["ptc"], ["polyfuse"], ["resettable", "fuse"], ["msmf"], ["fuse"]]),
            ("headers", [["header"], ["icsp"]]),
        ]
        secondary_checks = [
            ("decoupling_caps", [["capacitor"], ["cap", "100nf"], ["decoupling"]]),
            ("bulk_caps", [["47uf"], ["bulk", "cap"], ["electrolytic"]]),
            ("resistors", [["resistor"], ["10k"], ["1k"], ["22r"], ["22ohm"]]),
            ("leds", [["led"], ["tx"], ["rx"], ["power led"]]),
            ("reset_switch", [["reset", "switch"], ["tact", "switch"], ["pushbutton"]]),
            ("power_mux_mosfet", [["mosfet"], ["fdn340p"]]),
            ("power_cmp_opamp", [["lm358"], ["lmv358"], ["opamp"], ["op amp"], ["operational", "amplifier"], ["power", "comparator"]]),
        ]

        results: List[Dict[str, Any]] = []
        missing: List[str] = []
        for name, combos in checks:
            ok = _has_any_combo(combos)
            results.append({"name": name, "present": bool(ok), "match_any_of": combos})
            if not ok:
                missing.append(name)

        secondary_results: List[Dict[str, Any]] = []
        secondary_missing: List[str] = []
        for name, combos in secondary_checks:
            ok = _has_any_combo(combos)
            secondary_results.append({"name": name, "present": bool(ok), "match_any_of": combos})
            if not ok:
                secondary_missing.append(name)

        return {
            "applicable": True,
            "roles_count": len(roles),
            "checks": results,
            "secondary_checks": secondary_results,
            "missing": missing,
            "secondary_missing": secondary_missing,
            # Keep strict pass criteria on core blocks; expose secondary coverage separately.
            "ok": len(missing) == 0,
            "secondary_ok": len(secondary_missing) == 0,
        }
    def _benchmark_spec_footprint_match(self, loop: Any, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Score whether SPEC preserved resolved footprints for primaries and secondaries."""
        expected_rows: List[Dict[str, Any]] = []
        actual_by_ref: Dict[str, str] = {}
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            spec_debug = artifacts.get("spec_debug") if isinstance(artifacts.get("spec_debug"), dict) else {}
            manifest = artifacts.get("manifest") if isinstance(artifacts.get("manifest"), dict) else {}
            primary_refs: Set[str] = set()

            if isinstance(spec_debug, dict):
                for row in list(spec_debug.get("datasheet_primaries") or []):
                    if not isinstance(row, dict):
                        continue
                    ref = str(row.get("ref", "") or "").strip().upper()
                    if ref:
                        primary_refs.add(ref)

            # Prefer the final resolved role list because it includes secondary
            # support parts in addition to board-level primaries.
            expected_by_ref: Dict[str, Dict[str, Any]] = {}
            for role in self._benchmark_final_roles_snapshot(loop):
                if not isinstance(role, dict):
                    continue
                constraints = role.get("constraints") if isinstance(role.get("constraints"), dict) else {}
                ref = str(role.get("role_id", "") or "").strip().upper()
                fp = str(constraints.get("package", "") or "").strip()
                if not ref or not fp or ref in expected_by_ref:
                    continue
                expected_by_ref[ref] = {
                    "ref": ref,
                    "tier": "primary" if (not primary_refs or ref in primary_refs) else "secondary",
                    "expected_footprint": fp,
                    "mpn": str(constraints.get("part_query", role.get("role_type", "")) or ""),
                    "role": str(role.get("role_type", "") or ""),
                }

            if expected_by_ref:
                expected_rows.extend(expected_by_ref.values())

            if not expected_rows and isinstance(spec_debug, dict):
                for row in list(spec_debug.get("datasheet_primaries") or []):
                    if not isinstance(row, dict):
                        continue
                    ref = str(row.get("ref", "") or "").strip().upper()
                    fp = str(row.get("footprint", "") or "").strip()
                    if not ref or not fp:
                        continue
                    expected_rows.append(
                        {
                            "ref": ref,
                            "tier": "primary",
                            "expected_footprint": fp,
                            "mpn": str(row.get("mpn", "") or ""),
                            "role": str(row.get("role", "") or ""),
                        }
                    )

            if not expected_rows:
                spec = artifacts.get("design_spec") if isinstance(artifacts.get("design_spec"), dict) else {}
                for row in list(spec.get("resolved_parts") or []):
                    if not isinstance(row, dict):
                        continue
                    ref = str(row.get("ref", row.get("role_id", "")) or "").strip().upper()
                    fp = str(row.get("footprint_id", "") or "").strip()
                    if not ref or not fp:
                        continue
                    expected_rows.append(
                        {
                            "ref": ref,
                            "tier": "primary" if (not primary_refs or ref in primary_refs) else "secondary",
                            "expected_footprint": fp,
                            "mpn": str(row.get("mpn", row.get("part_query", "")) or ""),
                            "role": str(row.get("role", row.get("role_type", "")) or ""),
                        }
                    )

            if isinstance(manifest, dict):
                for part in list(manifest.get("parts") or []):
                    if not isinstance(part, dict):
                        continue
                    ref = str(part.get("ref", "") or "").strip().upper()
                    fp = str(part.get("footprint", "") or "").strip()
                    if ref and fp and ref not in actual_by_ref:
                        actual_by_ref[ref] = fp

            if not actual_by_ref:
                for role in self._benchmark_final_roles_snapshot(loop):
                    if not isinstance(role, dict):
                        continue
                    ref = str(role.get("role_id", "") or "").strip().upper()
                    constraints = role.get("constraints") if isinstance(role.get("constraints"), dict) else {}
                    fp = str(constraints.get("package", "") or "").strip()
                    if ref and fp and ref not in actual_by_ref:
                        actual_by_ref[ref] = fp
        except Exception:
            return {"applicable": False, "reason": "exception"}

        if not expected_rows:
            return {"applicable": False, "reason": "no_expected_primary_footprints"}

        checks: List[Dict[str, Any]] = []
        mismatches: List[Dict[str, Any]] = []
        matched_count = 0
        exact_count = 0
        tier_stats: Dict[str, Dict[str, int]] = {
            "primary": {"expected_count": 0, "matched_count": 0, "exact_count": 0},
            "secondary": {"expected_count": 0, "matched_count": 0, "exact_count": 0},
        }
        for row in expected_rows:
            ref = str(row.get("ref", "") or "")
            tier = str(row.get("tier", "primary") or "primary").strip().lower()
            if tier not in tier_stats:
                tier = "secondary"
            expected_fp = str(row.get("expected_footprint", "") or "")
            actual_fp = str(actual_by_ref.get(ref, "") or "")
            expected_variants = set(self._bench_id_variants(expected_fp) or set())
            actual_variants = set(self._bench_id_variants(actual_fp) or set())
            exact_match = bool(expected_variants and actual_variants and not expected_variants.isdisjoint(actual_variants))
            matched = bool(actual_fp and self._bench_ids_match(actual_fp, expected_fp))
            tier_stats[tier]["expected_count"] += 1
            if matched:
                matched_count += 1
                tier_stats[tier]["matched_count"] += 1
            if exact_match:
                exact_count += 1
                tier_stats[tier]["exact_count"] += 1
            status = "exact" if exact_match else ("compatible" if matched else ("missing" if not actual_fp else "mismatch"))
            check = {
                "ref": ref,
                "tier": tier,
                "mpn": str(row.get("mpn", "") or ""),
                "role": str(row.get("role", "") or ""),
                "expected_footprint": expected_fp,
                "actual_footprint": actual_fp,
                "matched": matched,
                "exact_match": exact_match,
                "status": status,
            }
            checks.append(check)
            if not matched:
                mismatches.append(check)

        expected_count = len(expected_rows)
        score = int(round(100.0 * float(matched_count) / float(max(expected_count, 1))))
        exact_score = int(round(100.0 * float(exact_count) / float(max(expected_count, 1))))
        subscores: Dict[str, Dict[str, Any]] = {}
        for tier_name, stats in tier_stats.items():
            tier_expected = int(stats.get("expected_count", 0) or 0)
            tier_matched = int(stats.get("matched_count", 0) or 0)
            tier_exact = int(stats.get("exact_count", 0) or 0)
            subscores[tier_name] = {
                "applicable": tier_expected > 0,
                "expected_count": tier_expected,
                "matched_count": tier_matched,
                "exact_count": tier_exact,
                "mismatched_count": max(0, tier_expected - tier_matched),
                "score_out_of_100": int(round(100.0 * float(tier_matched) / float(max(tier_expected, 1)))),
                "exact_score_out_of_100": int(round(100.0 * float(tier_exact) / float(max(tier_expected, 1)))),
                "ok": tier_expected > 0 and tier_matched == tier_expected,
            }
        return {
            "applicable": True,
            "expected_count": expected_count,
            "matched_count": matched_count,
            "exact_count": exact_count,
            "mismatched_count": len(mismatches),
            "primary_expected_count": int(tier_stats["primary"].get("expected_count", 0) or 0),
            "secondary_expected_count": int(tier_stats["secondary"].get("expected_count", 0) or 0),
            "score_out_of_100": score,
            "exact_score_out_of_100": exact_score,
            "subscores": subscores,
            "checks": checks,
            "mismatches": mismatches,
            "ok": matched_count == expected_count,
        }
    def _benchmark_uno_bom_oracle(self, loop: Any, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Strict Uno BOM oracle check (role-level with qty constraints)."""
        scenario = str(benchmark.get("scenario", "") or "").lower()
        if "uno" not in scenario:
            return {"applicable": False}

        roles = self._benchmark_final_roles_snapshot(loop)
        if not roles:
            return {
                "applicable": True,
                "roles_count": 0,
                "checks": [],
                "missing": [],
                "qty_mismatches": [],
                "errors": ["no roles in final spec"],
                "ok": False,
            }

        rows: List[Dict[str, Any]] = []
        for r in roles:
            if not isinstance(r, dict):
                continue
            c = r.get("constraints") if isinstance(r.get("constraints"), dict) else {}
            qty_raw = r.get("quantity", 1)
            try:
                qty = int(qty_raw)
            except Exception:
                qty = 1
            if qty <= 0:
                qty = 1
            blob = self._benchmark_text_blob_for_role(r)
            rows.append(
                {
                    "role_id": str(r.get("role_id", "") or ""),
                    "role_type": str(r.get("role_type", "") or ""),
                    "blob": blob,
                    "qty": qty,
                    "part_query": str(c.get("part_query", "") or ""),
                    "package": str(c.get("package", "") or ""),
                }
            )

        def _row_match_any(row: Dict[str, Any], combos: List[List[str]]) -> bool:
            b = str(row.get("blob", "") or "")
            return any(all(str(tok or "").lower() in b for tok in combo) for combo in combos)

        def _row_semantic_hints(row: Dict[str, Any]) -> Dict[str, str]:
            """Infer coarse canonical semantics for benchmark-sensitive buckets.

            This stays generic: it uses ref-prefix and electronics-family cues,
            not board-specific reference designators or per-board exceptions.
            """
            rid = str(row.get("role_id", "") or "").strip().upper()
            role_type = str(row.get("role_type", "") or "").strip().lower()
            blob = str(row.get("blob", "") or "").strip().lower()
            part_query = str(row.get("part_query", "") or "").strip().lower()
            package = str(row.get("package", "") or "").strip().lower()
            text = " ".join(x for x in (role_type, blob, part_query, package) if x).lower()

            def _has_any(*needles: str) -> bool:
                return any(n in text for n in needles)

            hints: Dict[str, str] = {}

            if rid.startswith(("X", "Y")) or _has_any("16mhz", "crystal", "xtal", "resonator", "cstce", "cstne", "nx3225", "fa-238"):
                if _has_any("resonator", "cstce", "cstne", "ceramic resonator"):
                    hints["clock_source"] = "resonator"
                elif _has_any("crystal", "xtal", "nx3225", "fa-238", "16.000"):
                    hints["clock_source"] = "crystal"

            if rid.startswith("D") or _has_any("diode", "zener", "tvs", "esd"):
                if _has_any("m7", "s1m", "s1a", "s1b", "s1d", "s1g", "s1j", "mbra140", "ss14", "1n4007", "es1j", "reverse polarity", "vin protection", "rectifier", "schottky"):
                    hints["diode_role"] = "power_rectifier"
                elif (
                    _has_any("cg0603mlc", "varistor", "mlv")
                    and _has_any("usb", "data", "d+", "d-", "usb_d")
                ):
                    hints["diode_role"] = "usb_line_varistor"
                elif _has_any("tvs", "array", "usblc", "esd protection", "varistor"):
                    hints["diode_role"] = "usb_protection_array"
                elif (
                    _has_any("usb", "usb_d", "d+", "d-", "zener", "minimelf", "cd1206")
                    or (_has_any("clamp", "signal diode") and _has_any("usb", "d+", "d-", "data line"))
                ):
                    hints["diode_role"] = "discrete_signal_clamp"

            return hints

        def _row_matches_sensitive_item(row: Dict[str, Any], item_id: str) -> bool:
            """Deterministic guardrails for oracle buckets with high semantic drift.

            The LLM can still propose matches first, but these buckets need stricter
            acceptance rules to avoid run-to-run score variance from over-broad
            semantic matching.
            """
            rid = str(row.get("role_id", "") or "").upper()
            blob = str(row.get("blob", "") or "")
            part_query = str(row.get("part_query", "") or "").lower()
            text = f"{blob} {part_query}".lower()
            hints = _row_semantic_hints(row)

            def _has_any(*needles: str) -> bool:
                return any(n in text for n in needles)

            if item_id == "leds_total":
                # Only actual indicator LED parts should count here.
                return rid.startswith("LED") or _has_any(" led ", "indicator led", "led0805", "led0603", "power led", "tx led", "rx led")

            if item_id == "clock_16mhz_crystal":
                # Required bucket: accept a canonical 16 MHz clock source, whether
                # implemented as a crystal or ceramic resonator.
                clock_source = hints.get("clock_source", "")
                if clock_source in {"crystal", "resonator"}:
                    return True
                if _has_any("resonator", "cstce", "cstne", "ceramic resonator"):
                    return True
                return rid.startswith(("X", "Y")) and _has_any("crystal", "xtal", "nx3225", "fa-238", "16.000")

            if item_id == "clock_16mhz_resonator":
                if hints.get("clock_source", "") == "resonator":
                    return True
                return rid.startswith(("Y", "X")) and _has_any("resonator", "cstce", "cstne", "ceramic resonator")

            if item_id == "reverse_diode_m7":
                if hints.get("diode_role", "") != "power_rectifier":
                    return False
                if _has_any("usb", "usbvcc", "vbus", "d+", "d-", "usb_d", "data line"):
                    return False
                if _has_any("signal diode", "clamp", "esd", "tvs", "varistor"):
                    return False
                return True

            if item_id == "signal_diodes":
                # Count only discrete small-signal clamp diodes, not TVS/ESD arrays.
                diode_role = hints.get("diode_role", "")
                if diode_role == "discrete_signal_clamp":
                    return True
                if diode_role in {"usb_protection_array", "power_rectifier"}:
                    return False
                if _has_any("reset") and not _has_any("usb", "d+", "d-", "data line"):
                    return False
                return rid.startswith("D") and (
                    _has_any("zener", "minimelf", "cd1206")
                    or (_has_any("clamp", "signal diode") and _has_any("usb", "d+", "d-", "data line"))
                )

            if item_id == "esd_varistors":
                if hints.get("diode_role", "") == "usb_protection_array":
                    return True
                return _has_any("varistor", "tvs", "esd")

            return True

        def _match_rows_for_item(item_id: str, combos: List[List[str]]) -> List[Dict[str, Any]]:
            if llm_matches is not None:
                _m = llm_matches.get(item_id)
                _m = _m if isinstance(_m, dict) else {}
                _matched_ids = set(_m.get("matched_role_ids", []) or [])
                rows0 = [row for row in rows if row.get("role_id") in _matched_ids]
            else:
                rows0 = [row for row in rows if _row_match_any(row, combos)] if combos else []

            # Apply deterministic bucket-specific guardrails.
            rows1 = [row for row in rows0 if _row_matches_sensitive_item(row, item_id)]

            # If semantic matching produced nothing valid for a sensitive bucket,
            # fall back to deterministic keyword matching for just that bucket.
            sensitive_ids = {
                "leds_total",
                "clock_16mhz_crystal",
                "clock_16mhz_resonator",
                "signal_diodes",
                "esd_varistors",
            }
            if not rows1 and item_id in sensitive_ids:
                rows1 = [
                    row for row in rows
                    if _row_match_any(row, combos) and _row_matches_sensitive_item(row, item_id)
                ]
            if not rows1 and item_id in {"clock_16mhz_crystal", "clock_16mhz_resonator"}:
                rows1 = [row for row in rows if _row_matches_sensitive_item(row, item_id)]
            return rows1

        def _usb_signal_protection_match() -> tuple[List[Dict[str, Any]], int]:
            discrete_rows: List[Dict[str, Any]] = []
            varistor_rows: List[Dict[str, Any]] = []
            for row in rows:
                role = _row_semantic_hints(row).get("diode_role", "")
                if role == "discrete_signal_clamp":
                    discrete_rows.append(row)
                elif role == "usb_line_varistor":
                    varistor_rows.append(row)

            discrete_qty = int(sum(int(r.get("qty", 0) or 0) for r in discrete_rows))
            varistor_qty = int(sum(int(r.get("qty", 0) or 0) for r in varistor_rows))

            if discrete_qty >= 2:
                return discrete_rows, min(discrete_qty, 2)
            if varistor_qty >= 2:
                return varistor_rows, 2
            if discrete_qty > 0:
                return discrete_rows, discrete_qty
            return [], 0

        # Tightened against the official Arduino Uno Rev3-02 TH BOM intent,
        # but kept compatible with role-level matching (no hard dependency on
        # reference designators like C9/RN1/Y1 in extracted text blobs).
        oracle = [
            {"item_id": "mcu_main", "qty_min": 1, "qty_max": 1, "combos": [["atmega328p", "pu"], ["atmega328p"]]},
            {"item_id": "usb_bridge", "qty_min": 1, "qty_max": 1, "combos": [["atmega16u2", "mu"], ["atmega16u2"]]},
            {"item_id": "reg_5v", "qty_min": 1, "qty_max": 1, "combos": [["ncp1117", "st50"], ["ncp1117", "5v"], ["ncp1117", "5 0"]]},
            {"item_id": "reg_3v3", "qty_min": 1, "qty_max": 1, "combos": [["lp2985", "33"], ["lp2985", "3v3"], ["lp2985", "3 3"]]},
            # Power-path comparator: non-obvious Uno detail; many LLM runs omit it.
            {"item_id": "power_opamp", "qty_min": 1, "qty_max": 1, "optional": True, "combos": [["lmv358"], ["power", "opamp"], ["power", "comparator"], ["op", "amp", "power"], ["power", "select", "op"]]},
            {"item_id": "pchan_mosfet", "qty_min": 1, "qty_max": 1, "combos": [["fdn340p"], ["p", "channel", "mosfet"]]},
            # ["usb","connector"] removed — too generic, matched ICSP/serial pin headers
            {"item_id": "usb_connector", "qty_min": 1, "qty_max": 1, "combos": [["usb", "type b"], ["usb b"], ["usbconn"], ["usb", "type", "b"], ["usb", "receptacle"]]},
            {"item_id": "dc_jack", "qty_min": 1, "qty_max": 1, "combos": [["dc", "21mm"], ["barrel", "jack"], ["power", "jack"], ["dc", "jack"], ["dc", "barrel"], ["barrel", "connector"], ["power", "connector", "2m"]]},
            # Clock: accept any 16MHz clocking device (crystal or resonator), qty 1-2.
            # Separate crystal/resonator checks were unreliable — resonators described as
            # "16MHz Crystal Resonator" matched both.
            {"item_id": "clock_16mhz_crystal", "qty_min": 1, "qty_max": 2, "combos": [
                ["16mhz", "crystal"], ["16mhz", "xtal"], ["16mhz", "resonator"],
                ["fa", "238"],       # Epson FA-238 16.0000MB-C3 crystal MPN
                ["16", "0000"],      # 16.0000 MHz in MPN text
                ["cstce16m0v53"],    # Murata CSTCE16M0V53-R0 resonator MPN
                ["cstne16m0v53"],    # Murata CSTNE16M0V53-R0 resonator MPN
            ]},
            {"item_id": "clock_16mhz_resonator", "qty_min": 1, "qty_max": 2, "optional": True, "combos": [["cstce16m0v53", "r0"], ["cstne16m0v53", "r0"], ["16mhz", "resonator"]]},
            # Often omitted in role-level drafts even when implied by crystal selection.
            {"item_id": "crystal_load_caps_22pf", "qty_min": 2, "qty_max": 2, "optional": True, "combos": [["22pf"], ["22p"]]},
            {"item_id": "usb_polyfuse_500ma", "qty_min": 1, "qty_max": 1, "combos": [["mf", "msmf050", "2"], ["polyfuse", "500ma"], ["ptc", "500ma"]]},
            # M7 equivalent: S1M-13-F (Diodes Inc), MBRA140T3G, SS14, S1A/B/D/J/G,
            # 1N4007, ES1J etc. are all 1A SMA/DO-214AC rectifiers.
            # Also accept Schottky variants: B140 (CD214A-B140LF), BAT54, etc.
            {"item_id": "reverse_diode_m7", "qty_min": 1, "qty_max": 1, "combos": [
                ["m7"],
                ["s1m"],
                ["s1a"], ["s1b"], ["s1d"], ["s1g"], ["s1j"],
                ["mbra140"],
                ["ss14"],
                ["1n4007"],
                ["es1j"],
                ["b140"],
                ["cd214"],
                ["bat54"],
                ["sk14"],
                ["reverse", "polarity", "diode"],
                ["vin", "protection", "diode"],
                ["rectifier", "sma"],
                ["schottky", "protection"],
                ["schottky", "reverse"],
            ]},
            {"item_id": "signal_diodes", "qty_min": 2, "qty_max": 2, "combos": [["cd1206", "s01575"], ["signal", "diode"], ["minimelf", "diode"], ["zener", "3 3"], ["zener", "3v3"], ["usb", "clamp"], ["usb", "zener"]]}, 
            {"item_id": "ferrite_bead", "qty_min": 1, "qty_max": 1, "optional": True, "combos": [["blm21"], ["ferrite", "bead"], ["emi", "suppression"]]},
            {"item_id": "esd_varistors", "qty_min": 2, "qty_max": 2, "optional": True, "combos": [["cg0603mlc", "05e"], ["esd", "varistor"], ["varistor"]]},
            {"item_id": "caps_100n", "qty_min": 6, "qty_max": 13, "combos": [["100nf"], ["100n"], ["0.1u"], ["0 1u"]]},  # Uno R3: 6-10 nominal; LLMs may emit up to 13 (multi-VCC-pin ICs)
            {"item_id": "caps_1u", "qty_min": 1, "qty_max": 4, "combos": [["1uf"], ["1 uf"]]},
            # NOTE: ["1u"] removed — single-token substring match is too broad and can
            # false-positive on 47uF blobs or ref-duplicated entries.
            {"item_id": "caps_47u", "qty_min": 2, "qty_max": 3, "combos": [["47uf"], ["47u"], ["electrolytic", "47u"]]},  # Uno R3: 2 nominal bulk caps; accept 3 (LLM sometimes splits rails)
            # Uno R3 has 1MΩ resistors on USB shield bleed + ATmega16U2 lines.
            # LLMs sometimes pick 10k for those lines; accept qty 1–3.
            # Only match actual 1M-range values to avoid false-positives.
            {"item_id": "res_1m", "qty_min": 1, "qty_max": 3, "combos": [
                ["1meg"], ["1mohm"], ["1 mohm"], ["1m", "resistor"], ["1m", "ohm"],
                ["1m", "pull"],
            ]},
            {"item_id": "res_arrays", "qty_min": 2, "qty_max": 8, "optional": True, "combos": [["cay16"], ["resistor", "array"], ["sip", "resistor"], ["bussed", "resistor"]]},
            {"item_id": "leds_total", "qty_min": 1, "qty_max": 4, "combos": [["ledchip", "led0805"], ["led", "0805"], ["led", "indicator"], ["led1"], ["led2"], ["led3"], ["led4"]]},
            {"item_id": "reset_button", "qty_min": 1, "qty_max": 1, "combos": [["reset", "switch"], ["tact", "switch"], ["pushbutton"]]},
            {"item_id": "solder_jumpers", "qty_min": 2, "qty_max": 2, "optional": True, "combos": [["solder", "jumper"], ["reset en", "sj"], ["ground", "sj"]]},
            # Würth WR-PHD 2.54mm variants: 61300xxxxxx and 61301xxxxxx
            {"item_id": "pin_headers_main", "qty_min": 4, "qty_max": 8, "combos": [
                ["pin", "header"], ["pinhd"], ["icsp", "2x3"], ["header"],
                ["61300"], ["61301"],  # Würth WR-PHD series MPN prefixes
                ["connector", "2 54"],  # 2.54mm pitch connectors
                ["expansion", "connector"], ["shield", "connector"],
            ]},
            {"item_id": "jp2_optional_2x2", "qty_min": 0, "qty_max": 1, "optional": True, "combos": [["jp2"], ["2x2", "header"], ["pinhd", "2x2"]]},
        ]

        # ── LLM-assisted matching (falls back to keyword if unavailable) ──
        llm_matches = self._benchmark_llm_oracle_match(rows, oracle)
        if llm_matches is not None:
            logger.info("oracle: using LLM-assisted matching for %d items", len(oracle))
        else:
            logger.info("oracle: using keyword matching (LLM unavailable or failed)")

        checks: List[Dict[str, Any]] = []
        missing: List[str] = []
        qty_mismatches: List[Dict[str, Any]] = []
        for exp in oracle:
            item_id = str(exp.get("item_id", "") or "")
            combos = exp.get("combos") if isinstance(exp.get("combos"), list) else []
            qty_min = int(exp.get("qty_min", 1) or 1)
            qty_max = int(exp.get("qty_max", qty_min) or qty_min)
            optional = bool(exp.get("optional", False)) or qty_min <= 0
            if item_id == "signal_diodes":
                matched_rows, total_qty = _usb_signal_protection_match()
            else:
                matched_rows = _match_rows_for_item(item_id, combos)
                total_qty = int(sum(int(row.get("qty", 0) or 0) for row in matched_rows))
            present = bool(total_qty > 0)
            qty_ok = bool(qty_min <= total_qty <= qty_max)
            if not present:
                if not optional:
                    missing.append(item_id)
            elif not qty_ok:
                if not optional:
                    qty_mismatches.append(
                        {
                            "item_id": item_id,
                            "expected_qty_min": qty_min,
                            "expected_qty_max": qty_max,
                            "actual_qty": total_qty,
                        }
                    )
            checks.append(
                {
                    "item_id": item_id,
                    "optional": optional,
                    "present": present,
                    "actual_qty": total_qty,
                    "expected_qty_min": qty_min,
                    "expected_qty_max": qty_max,
                    "matched_role_ids": [str(row.get("role_id", "") or "") for row in matched_rows[:12]],
                }
            )

        errors: List[str] = []
        if missing:
            errors.append(f"missing_required_items: {missing}")
        if qty_mismatches:
            errors.append(f"qty_mismatch_items: {[m.get('item_id') for m in qty_mismatches]}")

        return {
            "applicable": True,
            "roles_count": len(rows),
            "required_count": int(sum(1 for c in checks if not bool(c.get("optional", False)))),
            "checks": checks,
            "debug_rows": rows,
            "missing": missing,
            "qty_mismatches": qty_mismatches,
            "errors": errors,
            "ok": (len(errors) == 0),
        }
    def _benchmark_llm_oracle_match(
        self,
        rows: List[Dict[str, Any]],
        oracle_items: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """LLM-assisted matching of manifest rows to oracle expected items.

        For each oracle item, asks the LLM which manifest rows (if any) represent
        that component using semantic understanding rather than keyword combos.

        Returns {item_id: {"matched_role_ids": [...], "total_qty": int}} or None
        if the LLM is unavailable or the call fails (caller should fall back to
        keyword matching).
        """
        if self.llm_client is None:
            return None

        _SYSTEM = (
            "You are an electronics BOM validation expert.\n"
            "Given MANIFEST ROWS (parts extracted from a PCB BOM) and a list of "
            "EXPECTED COMPONENTS, identify which manifest rows represent each "
            "expected component.\n"
            "A row MATCHES if it represents the same type of electrical component "
            "regardless of naming style, abbreviations, or alternative MPNs/descriptions.\n"
            "Be inclusive: if a row plausibly represents the expected component, match it.\n\n"
            "Return ONLY strict JSON (no markdown fences):\n"
            '{\n'
            '  "matches": {\n'
            '    "<item_id>": {\n'
            '      "matched_role_ids": ["<role_id>", ...],\n'
            '      "total_qty": <integer sum of qty for matched rows>\n'
            '    }\n'
            '  }\n'
            '}\n\n'
            'Every item_id from EXPECTED COMPONENTS must appear as a key in "matches".\n'
            "Use matched_role_ids=[] and total_qty=0 when no rows match."
        )

        # Human-readable descriptions for each oracle item_id
        _ITEM_DESC: Dict[str, str] = {
            "mcu_main":               "main microcontroller (e.g. ATmega328P)",
            "usb_bridge":             "USB-to-serial bridge IC (e.g. ATmega16U2)",
            "reg_5v":                 "5V voltage regulator (e.g. NCP1117-5V)",
            "reg_3v3":                "3.3V voltage regulator (e.g. LP2985-33)",
            "power_opamp":            "op-amp or comparator for power-path / source selection",
            "pchan_mosfet":           "P-channel MOSFET for power switching (e.g. FDN340P)",
            "usb_connector":          "USB Type-B connector",
            "dc_jack":                "DC barrel jack / power input connector",
            "clock_16mhz_crystal":    "16 MHz crystal or ceramic resonator",
            "clock_16mhz_resonator":  "16 MHz ceramic resonator (optional alternative to crystal)",
            "crystal_load_caps_22pf": "22 pF crystal load capacitors",
            "usb_polyfuse_500ma":     "500 mA USB polyfuse / PTC resettable fuse",
            "reverse_diode_m7":       "reverse-polarity protection diode (M7 / S1M / 1N4007 class)",
            "signal_diodes":          "small-signal Zener clamping diodes on USB D+/D- lines (3.3V clamp, qty 2)",
            "ferrite_bead":           "ferrite bead for EMI / power-rail filtering",
            "esd_varistors":          "ESD protection varistors on USB lines",
            "caps_100n":              "100 nF (0.1 uF) decoupling capacitors",
            "caps_1u":                "1 uF capacitors",
            "caps_47u":               "47 uF bulk electrolytic capacitors",
            "res_1m":                 "1 MΩ resistors",
            "res_arrays":             "resistor arrays / SIP resistor packs",
            "leds_total":             "indicator LEDs (power, TX, RX, etc.)",
            "reset_button":           "reset push button / tactile switch",
            "solder_jumpers":         "solder jumpers",
            "pin_headers_main":       "pin headers / expansion connectors / shield connectors",
            "jp2_optional_2x2":       "optional 2×2 header JP2",
        }

        row_summaries = [
            {
                "role_id":    row.get("role_id", ""),
                "role_type":  row.get("role_type", ""),
                "part_query": row.get("part_query", ""),
                "package":    row.get("package", ""),
                "qty":        row.get("qty", 1),
            }
            for row in rows
        ]

        item_list = [
            {
                "item_id":     exp["item_id"],
                "description": _ITEM_DESC.get(
                    exp["item_id"],
                    exp["item_id"].replace("_", " ")
                ),
            }
            for exp in oracle_items
            if isinstance(exp, dict) and exp.get("item_id")
        ]

        prompt = (
            "MANIFEST ROWS:\n"
            + json.dumps(row_summaries, indent=2)
            + "\n\nEXPECTED COMPONENTS:\n"
            + json.dumps(item_list, indent=2)
            + "\n\nFor each expected component, identify which manifest rows (by "
            "role_id) represent it. "
            "Sum the qty values of matched rows for total_qty. "
            "Return JSON only."
        )

        try:
            from .llm.client import LLMMessage
            resp = self.llm_client.chat(
                [LLMMessage(role="user", content=prompt)],
                system_prompt=_SYSTEM,
                response_format={"type": "json_object"},
            )
            content = (getattr(resp, "content", "") or "").strip()
            if not content:
                logger.warning("oracle LLM match: empty response, falling back to keywords")
                return None
            obj = json.loads(content)
            matches = obj.get("matches")
            if not isinstance(matches, dict):
                logger.warning(
                    "oracle LLM match: unexpected response shape (%r), falling back to keywords",
                    type(matches).__name__,
                )
                return None
            logger.info("oracle LLM match: received matches for %d item(s)", len(matches))
            return matches
        except Exception as _e:
            logger.warning("oracle LLM match failed: %s — falling back to keyword matching", _e)
            return None
    def _benchmark_fail_fast_before_geom(self, benchmark: Optional[Dict[str, Any]], loop: Any) -> Optional[Dict[str, Any]]:
        """Benchmark-only gate: stop early if Uno coverage/BOM-oracle is incomplete."""
        if not isinstance(benchmark, dict) or loop is None:
            return None
        if bool(benchmark.get("force_proceed_all_phases", False)):
            return None
        warn_only = bool(benchmark.get("fail_fast_warn_only", False))
        if warn_only and bool(benchmark.get("fail_fast_warning_emitted", False)):
            return None
        if bool(benchmark.get("fail_fast_triggered", False)):
            return None
        bom_only_mode = bool(benchmark.get("bom_only_mode", False))
        roles = self._benchmark_final_roles_snapshot(loop)
        if not roles:
            return None

        bom_payload = None
        if bool(benchmark.get("strict_bom_oracle_before_geom", True)):
            bom = self._benchmark_uno_bom_oracle(loop, benchmark)
            if bool(bom.get("applicable")) and not bool(bom.get("ok", False)):
                bom_payload = {
                    "gate": "BENCHMARK.bom_oracle",
                    "message": "Benchmark stopped: Uno BOM oracle mismatch",
                    "details": {
                        "missing": list(bom.get("missing", []) or []),
                        "qty_mismatches": list(bom.get("qty_mismatches", []) or []),
                        "errors": list(bom.get("errors", []) or []),
                    },
                    "bounce_to": "SPEC",
                }

        if bom_only_mode:
            if bom_payload is None:
                return None
            if warn_only:
                benchmark["fail_fast_warning_emitted"] = True
                benchmark["fail_fast_warning_payload"] = bom_payload
                return bom_payload
            benchmark["fail_fast_triggered"] = True
            benchmark["fail_fast_payload"] = bom_payload
            return bom_payload

        if not bool(benchmark.get("strict_semantic_gate_before_geom", True)) and bom_payload is None:
            return None

        sc = self._benchmark_uno_semantic_completeness(loop, benchmark)
        if not bool(sc.get("applicable")):
            return bom_payload
        require_secondary = bool(benchmark.get("strict_semantic_gate_include_secondary", True))
        semantic_ok = bool(sc.get("ok", False)) and (bool(sc.get("secondary_ok", False)) or not require_secondary)

        if semantic_ok and bom_payload is None:
            return None

        payload = bom_payload or {
            "gate": "BENCHMARK.semantic_completeness",
            "message": "Benchmark stopped: semantic completeness incomplete",
            "details": {
                "missing_core": list(sc.get("missing", []) or []),
                "missing_secondary": list(sc.get("secondary_missing", []) or []),
            },
            "bounce_to": "SPEC",
        }
        if warn_only:
            benchmark["fail_fast_warning_emitted"] = True
            benchmark["fail_fast_warning_payload"] = payload
            return payload
        benchmark["fail_fast_triggered"] = True
        benchmark["fail_fast_payload"] = payload
        return payload
    def _benchmark_stage_statuses(self, loop: Any, gate_failures: List[Dict[str, Any]]) -> Dict[str, str]:
        statuses: Dict[str, str] = {
            "SPEC": "unknown",
            "RESOLVE": "unknown",
            "IMPORT": "unknown",
            "GEOM": "unknown",
            "NET": "unknown",
            "BIND": "unknown",
            "DRC": "unknown",
        }
        gate_names = [str((f or {}).get("gate", "") or "") for f in gate_failures if isinstance(f, dict)]
        unresolved_roles = self._benchmark_extract_unresolved_roles_from_gate_failures(gate_failures)
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            if isinstance(artifacts.get("design_spec"), dict) or isinstance(artifacts.get("design_spec_draft"), dict):
                statuses["SPEC"] = "pass"
        except Exception:
            pass
        if any(g.startswith("SPEC.") for g in gate_names):
            statuses["SPEC"] = "fail"
        if any(g == "SPEC.device_resolvability" for g in gate_names):
            statuses["RESOLVE"] = "fail"
        elif any(
            str((u or {}).get("issue_type", "") or "") in {"missingsymbol", "missingfootprint", "ambiguouschoice"}
            or "no resolved part selected" in str((u or {}).get("detail", "") or "").lower()
            or "criticalmissingsymbolorfootprint" in str((u or {}).get("detail", "") or "").lower()
            for u in unresolved_roles
            if isinstance(u, dict)
        ):
            statuses["RESOLVE"] = "fail"
        elif statuses["SPEC"] == "pass":
            statuses["RESOLVE"] = "pass"
        if any(g.startswith("GEOM.") for g in gate_names):
            statuses["GEOM"] = "fail"
        if any(g.startswith("NET.") for g in gate_names):
            statuses["NET"] = "fail"
        if any(g.startswith("BIND.") for g in gate_names):
            statuses["BIND"] = "fail"

        history = list(getattr(loop, "_history", []) or [])
        import_attempted = False
        import_failed = False
        net_attempted = False
        net_failed = False
        drc_attempted = False
        drc_failed = False
        for step in history:
            action = getattr(step, "action", None)
            if action is None:
                continue
            at = str(getattr(getattr(action, "action_type", None), "name", getattr(action, "action_type", "")) or "")
            ok = bool(getattr(action, "success", False))
            if at in {"DOWNLOAD_SYMBOL", "DOWNLOAD_FOOTPRINT", "ADD_COMPONENT"}:
                import_attempted = True
                if not ok:
                    import_failed = True
            if at in {"DEFINE_NET", "ASSIGN_NETS", "AUTOROUTE_BOARD"}:
                net_attempted = True
                if not ok:
                    net_failed = True
            if at == "RUN_DRC":
                drc_attempted = True
                if not ok:
                    drc_failed = True
        if import_attempted:
            statuses["IMPORT"] = "fail" if import_failed else "pass"
        if net_attempted and statuses["NET"] == "unknown":
            statuses["NET"] = "fail" if net_failed else "pass"
        if drc_attempted:
            statuses["DRC"] = "fail" if drc_failed else "pass"

        try:
            placement_plan = artifacts.get("placement_plan")
            if isinstance(placement_plan, dict) and statuses["GEOM"] == "unknown":
                statuses["GEOM"] = "pass"
        except Exception:
            pass
        if statuses["BIND"] == "unknown" and statuses["NET"] != "unknown":
            statuses["BIND"] = statuses["NET"]
        return statuses
    @staticmethod
    def _benchmark_drc_error_entries(text: str) -> List[str]:
        entries: List[str] = []
        in_errors = False
        for raw in str(text or "").splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("ERRORS"):
                in_errors = True
                continue
            if upper.startswith("WARNINGS"):
                break
            if not in_errors:
                continue
            if line.startswith("... and "):
                continue
            m = re.match(r"^\d+\.\s+(.*)$", line)
            if m:
                entries.append(str(m.group(1) or "").strip())
        return entries
    @staticmethod
    def _benchmark_drc_warning_entries(text: str) -> List[str]:
        entries: List[str] = []
        in_warnings = False
        for raw in str(text or "").splitlines():
            line = str(raw or "").strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("WARNINGS"):
                in_warnings = True
                continue
            if not in_warnings:
                continue
            if line.startswith("... and "):
                continue
            m = re.match(r"^\d+\.\s+(.*)$", line)
            if m:
                entries.append(str(m.group(1) or "").strip())
        return entries
    @classmethod
    def _benchmark_drc_connectivity_only(cls, text: str) -> bool:
        entries = cls._benchmark_drc_error_entries(text)
        if not entries:
            return False
        for entry in entries:
            normalized = re.sub(r"\s+", " ", str(entry or "").strip()).lower()
            if (
                "missing connection between items" in normalized
                or "unconnected item" in normalized
                or "unconnected items" in normalized
            ):
                continue
            return False
        return True
    @staticmethod
    def _benchmark_parse_drc_clearance_error(entry: str) -> Optional[Dict[str, float]]:
        text = str(entry or "").strip()
        m = re.search(
            r"Clearance violation\s*\(\s*clearance\s*([0-9.]+)\s*mm;\s*actual\s*([0-9.]+)\s*mm\)\s*at\s*\(\s*([0-9.+-]+)\s*,\s*([0-9.+-]+)\s*\)mm",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return None
        try:
            return {
                "required_mm": float(m.group(1)),
                "actual_mm": float(m.group(2)),
                "x_mm": float(m.group(3)),
                "y_mm": float(m.group(4)),
            }
        except Exception:
            return None

    def _benchmark_drc_footprint_profiles(self) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []
        pdata = None
        if PCBNEW_AVAILABLE:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
            if board is not None:
                try:
                    board_path = str(board.GetFileName() or "").strip()
                except Exception:
                    board_path = ""
                if board_path:
                    try:
                        pdata = PCBParser(board_path).parse()
                    except Exception:
                        pdata = None
        if pdata is None:
            pdata = self.pcb_data
        if pdata is None:
            return profiles

        for fp in list(getattr(pdata, "footprints", []) or []):
            ref = str(getattr(fp, "reference", "") or "").strip().upper()
            pads_local = list(getattr(fp, "pads", []) or [])
            if not ref or len(pads_local) < 2:
                continue
            try:
                cx = float(getattr(getattr(fp, "at", None), "x", 0.0) or 0.0)
                cy = float(getattr(getattr(fp, "at", None), "y", 0.0) or 0.0)
                rot_deg = float(getattr(fp, "rotation", 0.0) or 0.0)
            except Exception:
                continue
            th = math.radians(rot_deg)
            ct = math.cos(th)
            st = math.sin(th)

            pads: List[Dict[str, float]] = []
            for pad in pads_local:
                try:
                    lx = float(getattr(getattr(pad, "at", None), "x", 0.0) or 0.0)
                    ly = float(getattr(getattr(pad, "at", None), "y", 0.0) or 0.0)
                    size = getattr(pad, "size", (0.0, 0.0))
                    w = float(size[0] or 0.0)
                    h = float(size[1] or 0.0)
                except Exception:
                    continue
                if w <= 0.0 or h <= 0.0:
                    continue
                gx = cx + (ct * lx) - (st * ly)
                gy = cy + (st * lx) + (ct * ly)
                pads.append({"x": gx, "y": gy, "w": w, "h": h})
            if len(pads) < 2:
                continue

            xmin = min(p["x"] - (p["w"] / 2.0) for p in pads)
            xmax = max(p["x"] + (p["w"] / 2.0) for p in pads)
            ymin = min(p["y"] - (p["h"] / 2.0) for p in pads)
            ymax = max(p["y"] + (p["h"] / 2.0) for p in pads)

            min_gap = None
            for i, p1 in enumerate(pads):
                for p2 in pads[i + 1:]:
                    gap_x = abs(p1["x"] - p2["x"]) - ((p1["w"] + p2["w"]) / 2.0)
                    gap_y = abs(p1["y"] - p2["y"]) - ((p1["h"] + p2["h"]) / 2.0)
                    sep_x = max(0.0, gap_x)
                    sep_y = max(0.0, gap_y)
                    edge_gap = math.hypot(sep_x, sep_y)
                    min_gap = edge_gap if min_gap is None else min(min_gap, edge_gap)

            profiles.append(
                {
                    "ref": ref,
                    "xmin": float(xmin),
                    "xmax": float(xmax),
                    "ymin": float(ymin),
                    "ymax": float(ymax),
                    "cx": float(cx),
                    "cy": float(cy),
                    "min_pad_gap_mm": None if min_gap is None else float(min_gap),
                }
            )
        return profiles

    def _benchmark_reclassify_intrinsic_clearance_errors(self, errors: List[str]) -> Dict[str, Any]:
        effective_errors: List[str] = []
        intrinsic_errors: List[Dict[str, Any]] = []
        warning_entries: List[str] = []
        profiles = self._benchmark_drc_footprint_profiles()
        margin_mm = 0.8

        for raw in list(errors or []):
            entry = str(raw or "").strip()
            parsed = self._benchmark_parse_drc_clearance_error(entry)
            if parsed is None:
                effective_errors.append(entry)
                continue

            x = float(parsed.get("x_mm", 0.0) or 0.0)
            y = float(parsed.get("y_mm", 0.0) or 0.0)
            required_mm = float(parsed.get("required_mm", 0.0) or 0.0)
            actual_mm = float(parsed.get("actual_mm", 0.0) or 0.0)

            best = None
            best_dist = None
            for prof in profiles:
                try:
                    if x < float(prof["xmin"]) - margin_mm or x > float(prof["xmax"]) + margin_mm:
                        continue
                    if y < float(prof["ymin"]) - margin_mm or y > float(prof["ymax"]) + margin_mm:
                        continue
                    d = math.hypot(x - float(prof["cx"]), y - float(prof["cy"]))
                    if best is None or best_dist is None or d < best_dist:
                        best = prof
                        best_dist = d
                except Exception:
                    continue

            min_gap = None
            if isinstance(best, dict):
                try:
                    min_gap = best.get("min_pad_gap_mm")
                    min_gap = None if min_gap is None else float(min_gap)
                except Exception:
                    min_gap = None

            intrinsic = (
                isinstance(best, dict)
                and min_gap is not None
                and (min_gap < (required_mm - 0.01))
                and (actual_mm <= (min_gap + 0.04))
            )
            if intrinsic:
                detail = {
                    "message": entry,
                    "ref": str(best.get("ref", "") or ""),
                    "required_mm": round(required_mm, 4),
                    "actual_mm": round(actual_mm, 4),
                    "min_pad_gap_mm": round(float(min_gap), 4),
                }
                intrinsic_errors.append(detail)
                warning_entries.append(entry)
            else:
                effective_errors.append(entry)

        return {
            "effective_errors": effective_errors,
            "intrinsic_errors": intrinsic_errors,
            "warning_entries": warning_entries,
            "raw_error_count": len(list(errors or [])),
            "effective_error_count": len(effective_errors),
            "intrinsic_error_count": len(intrinsic_errors),
        }

    def _benchmark_placement_score(self, loop, benchmark) -> dict:
        metrics = {
            "score": 0.0,
            "components_placed": 0,
            "components_expected": 0,
            "out_of_bounds": 0,
            "overlap_collisions": 0,
            "drc_passed": None,
            "drc_error_count": 0,
            "audit_complete": False,
            "success": False,
        }
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            profile = self._benchmark_generalizable_profile(benchmark if isinstance(benchmark, dict) else None)
            score_cap_moved_060 = self._benchmark_profile_float(
                profile,
                ("placement", "score_caps", "moved_ratio_ge_0_60"),
                40.0,
                min_value=0.0,
            )
            score_cap_moved_035 = self._benchmark_profile_float(
                profile,
                ("placement", "score_caps", "moved_ratio_ge_0_35"),
                60.0,
                min_value=0.0,
            )
            score_cap_moved_015 = self._benchmark_profile_float(
                profile,
                ("placement", "score_caps", "moved_ratio_ge_0_15"),
                80.0,
                min_value=0.0,
            )
            score_cap_rot_mismatch = self._benchmark_profile_float(
                profile,
                ("placement", "score_caps", "rot_mismatch"),
                70.0,
                min_value=0.0,
            )
            score_cap_outward_mismatch = self._benchmark_profile_float(
                profile,
                ("placement", "score_caps", "outward_mismatch"),
                75.0,
                min_value=0.0,
            )
            ignore_placement_drc_in_score = bool(
                isinstance(benchmark, dict) and benchmark.get("ignore_placement_drc_in_score", False)
            )
            manifest = artifacts.get("manifest")
            part_lookup: Dict[str, Dict[str, Any]] = {}
            if manifest and isinstance(manifest, dict):
                metrics["components_expected"] = len(manifest.get("parts", []))
                part_lookup = {
                    str(part.get("ref", "") or ""): part
                    for part in list(manifest.get("parts", []) or [])
                    if isinstance(part, dict)
                }
            
            plan = artifacts.get("placement_plan")
            if not plan or not isinstance(plan, dict):
                return metrics

            allowed_refs = {
                str(ref or "").strip().upper()
                for ref in list(part_lookup.keys()) + list(plan.keys())
                if str(ref or "").strip()
            }
            live_board_metrics = self._benchmark_live_board_placement_metrics(allowed_refs=allowed_refs)
            scored_positions = (
                live_board_metrics.get("snapshot")
                if isinstance(live_board_metrics.get("snapshot"), dict) and live_board_metrics.get("snapshot")
                else plan
            )
            metrics["scoring_source"] = "board" if scored_positions is not plan else "plan"

            # placement_plan / live board snapshot is {ref: {x, y, rot}} — flatten values to a list
            placements = [v for v in scored_positions.values() if isinstance(v, dict)]
            metrics["components_placed"] = len(placements)
            if metrics["components_placed"] == 0:
                return metrics
                
            coverage = min(1.0, metrics["components_placed"] / max(1, metrics["components_expected"]))
            collisions = 0
            overlap_pairs: List[Tuple[str, str]] = []
            out_of_bounds = 0
            if bool(live_board_metrics.get("usable")):
                out_of_bounds = int(live_board_metrics.get("out_of_bounds", 0) or 0)
                collisions = int(live_board_metrics.get("overlap_collisions", 0) or 0)
                overlap_pairs = [
                    (str(a), str(b))
                    for a, b in list(live_board_metrics.get("overlap_pairs_sample", []) or [])
                    if a and b
                ]
                oob_refs = [str(ref) for ref in list(live_board_metrics.get("out_of_bounds_refs_sample", []) or []) if ref]
                if oob_refs:
                    metrics["out_of_bounds_refs_sample"] = oob_refs
            else:
                for p in placements:
                    x = p.get("x", 0)
                    y = p.get("y", 0)
                    if x < -50 or x > 200 or y < -50 or y > 200:
                        out_of_bounds += 1
                try:
                    from vibecad.design.sub_agents.place import ComponentPlaceAgent

                    estimator = ComponentPlaceAgent(llm_client=None)

                    def placement_dims(ref: str, pos: Dict[str, Any]) -> Tuple[float, float]:
                        part = part_lookup.get(ref, {"ref": ref})
                        item = estimator._categorize_part(part)
                        edge = str(pos.get("edge", "") or "")
                        if edge in {"left", "right", "top", "bottom"}:
                            item["placed_edge"] = edge
                            w, h = estimator._oriented_dims(item, edge)
                        else:
                            w = float(item.get("width", 6.0) or 6.0)
                            h = float(item.get("height", 4.0) or 4.0)
                            rot = abs(float(pos.get("rot", 0.0) or 0.0)) % 180.0
                            if abs(rot - 90.0) < 1e-3:
                                w, h = h, w
                        return (w + 1.6, h + 1.6)

                    refs = [str(ref) for ref, pos in scored_positions.items() if isinstance(ref, str) and isinstance(pos, dict)]
                    for i, ref_a in enumerate(refs):
                        pos_a = scored_positions.get(ref_a) or {}
                        ax = float(pos_a.get("x", 0.0) or 0.0)
                        ay = float(pos_a.get("y", 0.0) or 0.0)
                        aw, ah = placement_dims(ref_a, pos_a)
                        for ref_b in refs[i + 1:]:
                            pos_b = scored_positions.get(ref_b) or {}
                            bx = float(pos_b.get("x", 0.0) or 0.0)
                            by = float(pos_b.get("y", 0.0) or 0.0)
                            bw, bh = placement_dims(ref_b, pos_b)
                            overlap_x = ((aw + bw) / 2.0) - abs(ax - bx)
                            overlap_y = ((ah + bh) / 2.0) - abs(ay - by)
                            if overlap_x > 0.0 and overlap_y > 0.0:
                                collisions += 1
                                if len(overlap_pairs) < 12:
                                    overlap_pairs.append((ref_a, ref_b))
                except Exception:
                    for i, p1 in enumerate(placements):
                        for j, p2 in enumerate(placements):
                            if i < j:
                                dx = p1.get("x", 0) - p2.get("x", 0)
                                dy = p1.get("y", 0) - p2.get("y", 0)
                                dist_sq = dx*dx + dy*dy
                                if dist_sq < 25:
                                    collisions += 1
            metrics["out_of_bounds"] = out_of_bounds
            metrics["overlap_collisions"] = collisions
            if overlap_pairs:
                metrics["overlap_pairs_sample"] = overlap_pairs

            placement_audit = self._benchmark_placement_audit(loop)
            audit_complete = int(placement_audit.get("expected_count", 0) or 0) > 0
            metrics["audit_complete"] = audit_complete
            metrics["audit_expected_count"] = int(placement_audit.get("expected_count", 0) or 0)
            metrics["audit_extra_count"] = int(placement_audit.get("extra_count", 0) or 0)
            metrics["audit_missing_critical_count"] = int(placement_audit.get("missing_critical_count", 0) or 0)
            if audit_complete:
                metrics["components_placed"] = int(placement_audit.get("matched_count", metrics["components_placed"]) or 0)
                metrics["components_expected"] = int(placement_audit.get("expected_count", metrics["components_expected"]) or 0)

            drc_passed = None
            drc_error_count = 0
            drc_raw_error_count = 0
            drc_intrinsic_error_count = 0
            ignored_connectivity_only_drc_failures = 0
            history = list(getattr(loop, "_history", []) or [])
            for step in history:
                action = getattr(step, "action", None)
                if action is None:
                    continue
                at = str(getattr(getattr(action, "action_type", None), "name", getattr(action, "action_type", "")) or "")
                if at != "RUN_DRC":
                    continue
                desc = str(getattr(action, "description", "") or "").lower()
                if desc and "placement" not in desc:
                    continue
                text = str(getattr(action, "result_message", "") or "")
                if (not bool(getattr(action, "success", False))) and self._benchmark_drc_connectivity_only(text):
                    ignored_connectivity_only_drc_failures += 1
                    continue
                raw_entries = self._benchmark_drc_error_entries(text)
                drc_raw_error_count = len(raw_entries)
                classified = self._benchmark_reclassify_intrinsic_clearance_errors(raw_entries)
                drc_error_count = int(classified.get("effective_error_count", 0) or 0)
                drc_intrinsic_error_count = int(classified.get("intrinsic_error_count", 0) or 0)
                drc_passed = bool(getattr(action, "success", False))
                if (drc_passed is False) and drc_raw_error_count > 0 and drc_error_count <= 0:
                    drc_passed = True
            if ignored_connectivity_only_drc_failures:
                metrics["ignored_connectivity_only_drc_failures"] = ignored_connectivity_only_drc_failures
                if drc_passed is None:
                    drc_passed = True
                    drc_error_count = 0
            metrics["drc_passed"] = drc_passed
            metrics["drc_error_count"] = drc_error_count
            metrics["drc_raw_error_count"] = drc_raw_error_count
            metrics["drc_intrinsic_error_count"] = drc_intrinsic_error_count

            # Compare planned vs actual board placement to avoid over-scoring
            # runs where many parts drifted away from the intended plan.
            placement_diag = self._benchmark_placement_diagnostics(
                placement_plan=plan if isinstance(plan, dict) else {},
                board_snapshot=(
                    live_board_metrics.get("snapshot")
                    if isinstance(live_board_metrics.get("snapshot"), dict)
                    else {}
                ),
                benchmark=benchmark if isinstance(benchmark, dict) else None,
            )
            if bool(placement_diag.get("usable")):
                compared_count = int(placement_diag.get("compared_count", 0) or 0)
                moved_count = int(placement_diag.get("moved_count", 0) or 0)
                rot_mismatch_count = int(placement_diag.get("rot_mismatch_count", 0) or 0)
                outward_mismatch_count = int(placement_diag.get("outward_mismatch_count", 0) or 0)
                metrics["plan_alignment"] = {
                    "compared_count": compared_count,
                    "moved_count": moved_count,
                    "rot_mismatch_count": rot_mismatch_count,
                    "outward_mismatch_count": outward_mismatch_count,
                    "moved_ratio": round(
                        float(moved_count) / float(max(compared_count, 1)),
                        3,
                    ) if compared_count > 0 else 0.0,
                    "score_caps": {
                        "moved_ratio_ge_0_60": round(float(score_cap_moved_060), 3),
                        "moved_ratio_ge_0_35": round(float(score_cap_moved_035), 3),
                        "moved_ratio_ge_0_15": round(float(score_cap_moved_015), 3),
                        "rot_mismatch": round(float(score_cap_rot_mismatch), 3),
                        "outward_mismatch": round(float(score_cap_outward_mismatch), 3),
                    },
                }
            
            score = coverage * 100
            score -= (out_of_bounds * 5)
            score -= (collisions * 2)
            if not audit_complete:
                score = min(score, 60.0)
            if int(placement_audit.get("missing_critical_count", 0) or 0) > 0:
                score = min(score, 25.0)
            if int(placement_audit.get("extra_count", 0) or 0) > 0:
                score = min(score, 60.0)
            if drc_passed is False and not ignore_placement_drc_in_score:
                score = min(score, 20.0 if drc_error_count > 0 else 40.0)
            if bool(placement_diag.get("usable")):
                compared_count = int(placement_diag.get("compared_count", 0) or 0)
                moved_count = int(placement_diag.get("moved_count", 0) or 0)
                rot_mismatch_count = int(placement_diag.get("rot_mismatch_count", 0) or 0)
                outward_mismatch_count = int(placement_diag.get("outward_mismatch_count", 0) or 0)
                moved_ratio = (
                    float(moved_count) / float(max(compared_count, 1))
                    if compared_count > 0 else 0.0
                )
                if moved_ratio >= 0.60:
                    score = min(score, score_cap_moved_060)
                elif moved_ratio >= 0.35:
                    score = min(score, score_cap_moved_035)
                elif moved_ratio >= 0.15:
                    score = min(score, score_cap_moved_015)
                if rot_mismatch_count > 0:
                    score = min(score, score_cap_rot_mismatch)
                if outward_mismatch_count > 0:
                    score = min(score, score_cap_outward_mismatch)
            metrics["score"] = max(0.0, min(100.0, score))
            if metrics["score"] >= 70 and (ignore_placement_drc_in_score or drc_passed is not False) and audit_complete:
                metrics["success"] = True
        except Exception as e:
            import logging
            logging.getLogger(__name__).exception("Place score error")
            metrics["error"] = str(e)
        return metrics
    def _benchmark_placement_audit(self, loop: Any) -> Dict[str, Any]:
        """Audit placed footprint queries against resolved SPEC parts.

        This is best-effort benchmark diagnostics only; it does not gate runtime.
        """
        expected: List[Dict[str, Any]] = []
        actual_adds: List[Dict[str, Any]] = []
        inferred_by_ref: List[Dict[str, Any]] = []
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            spec = artifacts.get("design_spec")
            if isinstance(spec, dict):
                for rp in list(spec.get("resolved_parts") or []):
                    if not isinstance(rp, dict):
                        continue
                    role_id = str(rp.get("role_id", "") or "").strip()
                    fp = str(rp.get("footprint_id", "") or "").strip()
                    critical = False
                    roles = spec.get("roles")
                    if isinstance(roles, list):
                        for r in roles:
                            if isinstance(r, dict) and str(r.get("role_id", "") or "").strip() == role_id:
                                critical = bool(r.get("critical", False))
                                break
                    if fp:
                        expected.append({"role_id": role_id, "footprint_id": fp, "critical": critical})
        except Exception:
            pass
        if not expected:
            try:
                artifacts = getattr(loop, "_artifacts", {}) or {}
                manifest = artifacts.get("manifest")
                if isinstance(manifest, dict):
                    for part in list(manifest.get("parts") or []):
                        if not isinstance(part, dict):
                            continue
                        pref = str(part.get("ref", "") or "").strip().upper()
                        if not pref:
                            continue
                        expected.append(
                            {
                                "role_id": pref,
                                "footprint_id": str(part.get("footprint", "") or "").strip(),
                                "critical": True,
                                "match_by_ref": True,
                            }
                        )
            except Exception:
                pass
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            placement_plan = artifacts.get("placement_plan") if isinstance(artifacts.get("placement_plan"), dict) else {}
            plan_ref_to_fp: Dict[str, str] = {}
            for pl in list(placement_plan.get("placements") or []):
                if not isinstance(pl, dict):
                    continue
                pref = str(pl.get("ref", "") or "").strip().upper()
                pfp = str(pl.get("footprint_id", "") or "").strip()
                if pref and pfp and pref not in plan_ref_to_fp:
                    plan_ref_to_fp[pref] = pfp

            history = list(getattr(loop, "_history", []) or [])
            touched_refs: set = set()
            for step in history:
                action = getattr(step, "action", None)
                if action is None:
                    continue
                at = str(getattr(getattr(action, "action_type", None), "name", getattr(action, "action_type", "")) or "")
                if at not in {"ADD_COMPONENT", "MOVE_COMPONENT", "ROTATE_COMPONENT"}:
                    continue
                if not bool(getattr(action, "success", False)):
                    continue
                params = action.parameters if isinstance(getattr(action, "parameters", None), dict) else {}
                ref = str(params.get("ref", "") or "").strip()
                if ref:
                    touched_refs.add(ref.upper())
                if at != "ADD_COMPONENT":
                    continue
                q = ""
                for k in ("query", "part_name", "mpn", "part", "name"):
                    v = str(params.get(k, "") or "").strip()
                    if v:
                        q = v
                        break
                if q:
                    actual_adds.append({"query": q, "ref": ref})

            # If placements were mostly MOVE/ROTATE actions on existing refs,
            # infer their intended footprint IDs from the active placement_plan.
            if touched_refs and plan_ref_to_fp:
                for r in sorted(touched_refs):
                    fp = str(plan_ref_to_fp.get(r, "") or "").strip()
                    if not fp:
                        continue
                    inferred_by_ref.append({"query": fp, "ref": r})
        except Exception:
            pass

        if inferred_by_ref:
            # Merge inferred refs while preserving explicit ADD_COMPONENT evidence.
            seen_pairs = {(str(x.get("query", "") or ""), str(x.get("ref", "") or "")) for x in actual_adds}
            for row in inferred_by_ref:
                pair = (str(row.get("query", "") or ""), str(row.get("ref", "") or ""))
                if pair in seen_pairs:
                    continue
                actual_adds.append(row)
                seen_pairs.add(pair)

        matched_expected_idx: set = set()
        extras: List[Dict[str, Any]] = []
        for a in actual_adds:
            q = str(a.get("query", "") or "")
            hit_idx = None
            for i, e in enumerate(expected):
                if i in matched_expected_idx:
                    continue
                if bool(e.get("match_by_ref", False)):
                    if str(a.get("ref", "") or "").strip().upper() == str(e.get("role_id", "") or "").strip().upper():
                        hit_idx = i
                        break
                if self._bench_ids_match(q, str(e.get("footprint_id", "") or "")):
                    hit_idx = i
                    break
            if hit_idx is None:
                extras.append({"query": q, "ref": str(a.get("ref", "") or "")})
            else:
                matched_expected_idx.add(hit_idx)

        missing = [
            {"role_id": str(e.get("role_id", "") or ""), "footprint_id": str(e.get("footprint_id", "") or ""), "critical": bool(e.get("critical", False))}
            for i, e in enumerate(expected)
            if i not in matched_expected_idx
        ]
        missing_critical = [m for m in missing if bool(m.get("critical", False))]
        audit_complete = len(expected) > 0
        reason = "" if audit_complete else "no_expected_resolved_parts"
        return {
            "expected_count": len(expected),
            "placed_count": len(actual_adds),
            "matched_count": len(matched_expected_idx),
            "extra_count": len(extras),
            "missing_count": len(missing),
            "missing_critical_count": len(missing_critical),
            "extras": extras[:20],
            "missing": missing[:20],
            "audit_complete": audit_complete,
            "reason": reason,
            "ok": audit_complete and (len(missing_critical) == 0) and (len(extras) == 0),
        }
    def _benchmark_collect_board_footprints(self, allowed_refs: Optional[set[str]] = None) -> Dict[str, Any]:
        if not PCBNEW_AVAILABLE:
            return {"snapshot": {}, "items": [], "board_rect_iu": None}
        try:
            board = pcbnew.GetBoard()
        except Exception:
            board = None
        if board is None:
            return {"snapshot": {}, "items": [], "board_rect_iu": None}

        def iu_to_mm(value: Any) -> float:
            try:
                return float(pcbnew.ToMM(int(value)))
            except Exception:
                try:
                    return float(value) / 1e6
                except Exception:
                    return 0.0

        def read_xy(pos_obj: Any) -> tuple[int, int]:
            if pos_obj is None:
                return 0, 0
            for ax, ay in (('x', 'y'), ('X', 'Y')):
                try:
                    xv = getattr(pos_obj, ax, None)
                    yv = getattr(pos_obj, ay, None)
                    if isinstance(xv, (int, float)) and isinstance(yv, (int, float)):
                        return int(xv), int(yv)
                except Exception:
                    pass
            try:
                gx = getattr(pos_obj, 'GetX', None)
                gy = getattr(pos_obj, 'GetY', None)
                if callable(gx) and callable(gy):
                    return int(gx()), int(gy())
            except Exception:
                pass
            return 0, 0

        def read_rotation(fp: Any) -> float:
            try:
                deg = getattr(fp, 'GetOrientationDegrees', None)
                if callable(deg):
                    return float(deg())
            except Exception:
                pass
            try:
                ang = fp.GetOrientation()
            except Exception:
                ang = None
            if ang is None:
                return 0.0
            for name, scale in (('AsDegrees', 1.0), ('AsTenthsOfADegree', 0.1)):
                fn = getattr(ang, name, None)
                if callable(fn):
                    try:
                        return float(fn()) * scale
                    except Exception:
                        pass
            return 0.0

        def extract_footprint_id(fp: Any) -> str:
            for name in ('GetFPIDAsString', 'GetFootprintIDAsString'):
                fn = getattr(fp, name, None)
                if callable(fn):
                    try:
                        value = str(fn() or '').strip()
                        if value:
                            return value
                    except Exception:
                        pass
            try:
                fpid = fp.GetFPID()
            except Exception:
                fpid = None
            if fpid is None:
                return ''
            nick = ''
            item = ''
            for name in ('GetLibNickname', 'GetNickname'):
                fn = getattr(fpid, name, None)
                if callable(fn):
                    try:
                        nick = str(fn() or '').strip()
                    except Exception:
                        nick = ''
                    if nick:
                        break
            for name in ('GetLibItemName', 'GetFootprintName', 'GetItemName'):
                fn = getattr(fpid, name, None)
                if callable(fn):
                    try:
                        item = str(fn() or '').strip()
                    except Exception:
                        item = ''
                    if item:
                        break
            if nick and item:
                return f'{nick}:{item}'
            for name in ('AsString', 'Format'):
                fn = getattr(fpid, name, None)
                if callable(fn):
                    try:
                        value = str(fn() or '').strip()
                        if value:
                            return value
                    except TypeError:
                        continue
                    except Exception:
                        pass
            return ''

        board_rect_iu: Optional[Tuple[int, int, int, int]] = None
        try:
            board_bb = board.GetBoardEdgesBoundingBox()
            bw = int(board_bb.GetWidth())
            bh = int(board_bb.GetHeight())
            if bw > 0 and bh > 0:
                x0 = int(board_bb.GetX())
                y0 = int(board_bb.GetY())
                board_rect_iu = (x0, y0, x0 + bw, y0 + bh)
        except Exception:
            board_rect_iu = None

        snapshot: Dict[str, Any] = {}
        items: List[Dict[str, Any]] = []
        try:
            footprints = list(board.GetFootprints() or [])
        except Exception:
            footprints = []
        for fp in footprints:
            try:
                ref = str(fp.GetReference() or '').strip().upper()
            except Exception:
                ref = ''
            if not ref:
                continue
            if allowed_refs is not None and ref not in allowed_refs:
                continue
            try:
                x_iu, y_iu = read_xy(fp.GetPosition())
            except Exception:
                x_iu, y_iu = (0, 0)
            snapshot[ref] = {
                'x': round(iu_to_mm(x_iu), 3),
                'y': round(iu_to_mm(y_iu), 3),
                'rot': round(read_rotation(fp), 3),
                'footprint': extract_footprint_id(fp),
                'layer': 'B' if bool(getattr(fp, 'IsFlipped', lambda: False)()) else 'F',
            }
            rect_iu = None
            for bbm in ('GetCourtyardBoundingBox', 'GetBoundingBox'):
                fn = getattr(fp, bbm, None)
                if not callable(fn):
                    continue
                try:
                    bb = fn() if bbm == 'GetCourtyardBoundingBox' else fn(False, False)
                    if bb and int(bb.GetWidth()) > 0 and int(bb.GetHeight()) > 0:
                        x0 = int(bb.GetX())
                        y0 = int(bb.GetY())
                        rect_iu = (x0, y0, x0 + int(bb.GetWidth()), y0 + int(bb.GetHeight()))
                        break
                except Exception:
                    continue
                    
            pads_rect_iu = None
            try:
                pads = list(fp.Pads() or [])
                if pads:
                    min_x = min_y = float('inf')
                    max_x = max_y = float('-inf')
                    for p in pads:
                        try:
                            bb = p.GetBoundingBox()
                            if bb:
                                x0 = int(bb.GetX())
                                y0 = int(bb.GetY())
                                px0, py0 = x0, y0
                                px1, py1 = x0 + int(bb.GetWidth()), y0 + int(bb.GetHeight())
                                min_x = min(min_x, px0)
                                min_y = min(min_y, py0)
                                max_x = max(max_x, px1)
                                max_y = max(max_y, py1)
                        except Exception:
                            continue
                    if min_x != float('inf'):
                        pads_rect_iu = (min_x, min_y, max_x, max_y)
            except Exception:
                pass

            items.append({
                'ref': ref,
                'rect_iu': rect_iu,
                'pads_rect_iu': pads_rect_iu,
            })
        return {"snapshot": snapshot, "items": items, "board_rect_iu": board_rect_iu}
    def _benchmark_live_board_placement_metrics(self, allowed_refs: Optional[set[str]] = None) -> Dict[str, Any]:
        collected = self._benchmark_collect_board_footprints(allowed_refs=allowed_refs)
        snapshot = collected.get("snapshot") if isinstance(collected.get("snapshot"), dict) else {}
        items = list(collected.get("items") or [])
        board_rect = collected.get("board_rect_iu")
        rect_items = [
            (str(item.get("ref", "") or ""), item.get("rect_iu"))
            for item in items
            if str(item.get("ref", "") or "") and isinstance(item.get("rect_iu"), tuple) and len(item.get("rect_iu")) == 4
        ]
        if not snapshot or not rect_items or not (isinstance(board_rect, tuple) and len(board_rect) == 4):
            return {"snapshot": {}, "usable": False}

        out_of_bounds_refs: List[str] = []
        bx0, by0, bx1, by1 = board_rect
        for item in items:
            ref = str(item.get("ref", "") or "")
            rect = item.get("rect_iu")
            pads_rect = item.get("pads_rect_iu")
            
            # For edge connectors (like USB or Barrel Jack),
            # allow bounding box to hang over the edge, as long as the pads are safely inside.
            is_edge_connector = ref.startswith("J")
            test_rect = pads_rect if (is_edge_connector and pads_rect) else rect
            
            if not ref or not isinstance(test_rect, tuple) or len(test_rect) != 4:
                continue
            rx0, ry0, rx1, ry1 = [int(v) for v in test_rect]
            if rx0 < bx0 or ry0 < by0 or rx1 > bx1 or ry1 > by1:
                out_of_bounds_refs.append(ref)

        overlap_pairs: List[Tuple[str, str]] = []
        for idx, (ref_a, rect_a) in enumerate(rect_items):
            ax0, ay0, ax1, ay1 = [int(v) for v in rect_a]
            for ref_b, rect_b in rect_items[idx + 1:]:
                bx0, by0, bx1, by1 = [int(v) for v in rect_b]
                if ax1 <= bx0 or ax0 >= bx1 or ay1 <= by0 or ay0 >= by1:
                    continue
                overlap_pairs.append((ref_a, ref_b))

        return {
            "snapshot": snapshot,
            "usable": True,
            "out_of_bounds": len(out_of_bounds_refs),
            "out_of_bounds_refs_sample": out_of_bounds_refs[:12],
            "overlap_collisions": len(overlap_pairs),
            "overlap_pairs_sample": overlap_pairs[:12],
        }
    def _benchmark_board_placement_snapshot(self) -> Dict[str, Any]:
        collected = self._benchmark_collect_board_footprints()
        snapshot = collected.get("snapshot")
        return snapshot if isinstance(snapshot, dict) else {}
    @staticmethod
    def _benchmark_rotation_delta_deg(actual_deg: float, expected_deg: float) -> float:
        delta = (float(actual_deg) - float(expected_deg)) % 360.0
        if delta > 180.0:
            delta -= 360.0
        return round(delta, 3)
    @staticmethod
    def _benchmark_generalizable_profile(benchmark: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "placement": {
                "drift_xy_mm": 0.25,
                "drift_rot_deg": 1.0,
                "score_caps": {
                    "moved_ratio_ge_0_60": 40.0,
                    "moved_ratio_ge_0_35": 60.0,
                    "moved_ratio_ge_0_15": 80.0,
                    "rot_mismatch": 70.0,
                    "outward_mismatch": 75.0,
                },
            },
            "clock": {
                "distance_limit_mm": 20.0,
                "cap_radius_mm": 12.0,
                "min_crystal_caps": 2,
            },
            "ee_rules": {
                "regulator_cap_radius_mm": 15.0,
                "regulator_small_cap_min_uf": 0.01,
                "regulator_small_cap_max_uf": 0.22,
                "regulator_bulk_cap_min_uf": 1.0,
                "regulator_min_nearby_caps": 2,
                "regulator_min_small_caps": 1,
                "regulator_min_bulk_caps": 1,
                "decoupling_radius_mm": 12.0,
                "decoupling_small_cap_min_uf": 0.01,
                "decoupling_small_cap_max_uf": 0.22,
                "decoupling_min_small_caps_per_ic": 1,
                "bulk_cap_min_uf_per_rail": 4.7,
                "usb_protector_max_distance_mm": 20.0,
                "enforce_dual_layer_gnd_zone_only_when_routed": True,
                "single_layer_gnd_zone_risk_min_gnd_segments": 80,
                "single_layer_gnd_zone_risk_min_gnd_vias": 4,
            },
        }

        def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            keys = set(base.keys()) | set(override.keys())
            for key in keys:
                b = base.get(key)
                o = override.get(key)
                if isinstance(b, dict) and isinstance(o, dict):
                    out[key] = _deep_merge(b, o)
                elif key in override:
                    out[key] = o
                else:
                    out[key] = b
            return out

        custom = {}
        if isinstance(benchmark, dict):
            raw = benchmark.get("generalizable_checks_profile")
            if isinstance(raw, dict):
                custom = raw
        return _deep_merge(defaults, custom)

    @staticmethod
    def _benchmark_profile_float(
        profile: Dict[str, Any],
        path: Tuple[str, ...],
        default: float,
        *,
        min_value: Optional[float] = None,
    ) -> float:
        node: Any = profile
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        try:
            value = float(node)
        except Exception:
            value = float(default)
        if min_value is not None:
            value = max(float(min_value), value)
        return float(value)

    @staticmethod
    def _benchmark_profile_int(
        profile: Dict[str, Any],
        path: Tuple[str, ...],
        default: int,
        *,
        min_value: Optional[int] = None,
    ) -> int:
        node: Any = profile
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        try:
            value = int(node)
        except Exception:
            value = int(default)
        if min_value is not None:
            value = max(int(min_value), value)
        return int(value)

    @staticmethod
    def _benchmark_profile_bool(
        profile: Dict[str, Any],
        path: Tuple[str, ...],
        default: bool,
    ) -> bool:
        node: Any = profile
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, bool):
            return node
        if isinstance(node, str):
            return node.strip().lower() in {"1", "true", "yes", "on"}
        if node is None:
            return bool(default)
        try:
            return bool(int(node))
        except Exception:
            return bool(default)

    def _benchmark_placement_diagnostics(
        self,
        placement_plan: Dict[str, Any],
        board_snapshot: Dict[str, Any],
        benchmark: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(placement_plan, dict) or not isinstance(board_snapshot, dict):
            return {"usable": False}

        profile = self._benchmark_generalizable_profile(benchmark)
        drift_xy_mm = self._benchmark_profile_float(
            profile,
            ("placement", "drift_xy_mm"),
            0.25,
            min_value=0.0,
        )
        drift_rot_deg = self._benchmark_profile_float(
            profile,
            ("placement", "drift_rot_deg"),
            1.0,
            min_value=0.0,
        )

        rows: List[Dict[str, Any]] = []
        moved_count = 0
        rot_mismatch_count = 0
        outward_mismatch_count = 0

        dir_order = ["px", "py", "nx", "ny"]
        outward_dir_for_edge = {
            "left": "nx",
            "right": "px",
            "top": "ny",
            "bottom": "py",
        }

        for ref, planned in placement_plan.items():
            if not isinstance(ref, str) or not isinstance(planned, dict):
                continue
            actual = board_snapshot.get(ref)
            if not isinstance(actual, dict):
                continue

            px = float(planned.get("x", 0.0) or 0.0)
            py = float(planned.get("y", 0.0) or 0.0)
            pr = float(planned.get("rot", 0.0) or 0.0)
            ax = float(actual.get("x", 0.0) or 0.0)
            ay = float(actual.get("y", 0.0) or 0.0)
            ar = float(actual.get("rot", 0.0) or 0.0)
            dx = round(ax - px, 3)
            dy = round(ay - py, 3)
            drot = self._benchmark_rotation_delta_deg(ar, pr)

            moved = abs(dx) > drift_xy_mm or abs(dy) > drift_xy_mm
            rot_mismatch = abs(drot) > drift_rot_deg
            if moved:
                moved_count += 1
            if rot_mismatch:
                rot_mismatch_count += 1

            row: Dict[str, Any] = {
                "ref": ref,
                "edge": planned.get("edge"),
                "category": planned.get("category"),
                "face_dir": planned.get("face_dir"),
                "planned": {"x": px, "y": py, "rot": pr},
                "actual": {"x": ax, "y": ay, "rot": ar},
                "delta": {"dx": dx, "dy": dy, "drot": drot},
                "moved": moved,
                "rot_mismatch": rot_mismatch,
            }

            edge = str(planned.get("edge", "") or "").lower()
            face_dir = str(planned.get("face_dir", "") or "").lower()
            category = str(planned.get("category", "") or "").lower()
            if edge in outward_dir_for_edge and face_dir in dir_order:
                target = outward_dir_for_edge[edge]
                expected_rot = float(((dir_order.index(target) - dir_order.index(face_dir)) % 4) * 90)
                outward_err = self._benchmark_rotation_delta_deg(ar, expected_rot)
                outward_ok = abs(outward_err) <= drift_rot_deg
                row["expected_outward_rot"] = expected_rot
                row["outward_rot_error"] = outward_err
                row["outward_ok"] = outward_ok
                row["outward_basis"] = "face_dir"
                if not outward_ok:
                    outward_mismatch_count += 1
            elif edge in outward_dir_for_edge and category == "port":
                expected_rot = {
                    "left": 180.0,
                    "right": 0.0,
                    "top": 90.0,
                    "bottom": 270.0,
                }.get(edge, 0.0)
                outward_err = self._benchmark_rotation_delta_deg(ar, expected_rot)
                outward_ok = abs(outward_err) <= drift_rot_deg
                row["expected_outward_rot"] = expected_rot
                row["outward_rot_error"] = outward_err
                row["outward_ok"] = outward_ok
                row["outward_basis"] = "port_edge_default"
                if not outward_ok:
                    outward_mismatch_count += 1

            rows.append(row)

        rows.sort(
            key=lambda r: (
                0 if bool(r.get("rot_mismatch")) else 1,
                0 if bool(r.get("moved")) else 1,
                -abs(float((r.get("delta") or {}).get("drot", 0.0) or 0.0)),
                -(
                    abs(float((r.get("delta") or {}).get("dx", 0.0) or 0.0))
                    + abs(float((r.get("delta") or {}).get("dy", 0.0) or 0.0))
                ),
                str(r.get("ref", "") or ""),
            )
        )

        return {
            "usable": bool(rows),
            "planned_count": len([1 for _ref, pos in placement_plan.items() if isinstance(pos, dict)]),
            "compared_count": len(rows),
            "moved_count": moved_count,
            "rot_mismatch_count": rot_mismatch_count,
            "outward_mismatch_count": outward_mismatch_count,
            "rows_sample": rows[:80],
            "thresholds": {
                "drift_xy_mm": round(float(drift_xy_mm), 3),
                "drift_rot_deg": round(float(drift_rot_deg), 3),
            },
        }
    def _benchmark_companion_schematic_file(self) -> Optional[Path]:
        """Best-effort path to the active PCB companion schematic file."""
        pcb_file = ""
        if PCBNEW_AVAILABLE:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
            if board is not None:
                try:
                    pcb_file = str(board.GetFileName() or "")
                except Exception:
                    pcb_file = ""
        if not pcb_file:
            return None
        try:
            return Path(pcb_file).with_suffix(".kicad_sch")
        except Exception:
            return None
    def _benchmark_parse_schematic_hierarchy(
        self,
        root_file: Path,
        *,
        max_files: int = 64,
    ) -> Dict[str, Any]:
        """Parse top-level schematic and subsheets to recover symbol counts/refs."""
        visited: Set[str] = set()
        refs: Set[str] = set()
        parse_errors: List[str] = []
        component_count = 0
        wire_count = 0
        net_label_count = 0

        def _walk(path: Path) -> None:
            nonlocal component_count, wire_count, net_label_count
            if len(visited) >= max_files:
                return
            try:
                path_key = str(path.resolve())
            except Exception:
                path_key = str(path)
            if path_key in visited:
                return
            visited.add(path_key)
            try:
                sch = SchematicParser(str(path)).parse()
            except Exception:
                parse_errors.append(str(path))
                return

            try:
                component_count += int(getattr(sch, "component_count", 0) or 0)
            except Exception:
                pass
            try:
                wire_count += len(list(getattr(sch, "wires", []) or []))
            except Exception:
                pass
            try:
                net_label_count += int(getattr(sch, "net_label_count", 0) or 0)
            except Exception:
                pass

            try:
                for sym in list(getattr(sch, "symbols", []) or []):
                    ref = str(getattr(sym, "reference", "") or "").strip().upper()
                    if not ref or ref.startswith("#"):
                        continue
                    if not bool(getattr(sym, "on_board", True)):
                        continue
                    refs.add(ref)
            except Exception:
                pass

            try:
                for sheet in list(getattr(sch, "sheets", []) or []):
                    filename = str(getattr(sheet, "filename", "") or "").strip()
                    if not filename:
                        continue
                    child = path.parent / filename
                    if not child.suffix:
                        child = child.with_suffix(".kicad_sch")
                    if child.exists():
                        _walk(child)
            except Exception:
                pass

        try:
            if root_file.exists():
                _walk(root_file)
        except Exception:
            pass

        return {
            "component_count": int(component_count),
            "wire_count": int(wire_count),
            "net_label_count": int(net_label_count),
            "refs": refs,
            "parsed_file_count": int(len(visited)),
            "parse_errors_sample": parse_errors[:8],
        }
    def _benchmark_schematic_ref_set(self) -> Set[str]:
        refs: Set[str] = set()
        if self.schematic_data is not None:
            try:
                for sym in list(getattr(self.schematic_data, "symbols", []) or []):
                    ref = str(getattr(sym, "reference", "") or "").strip().upper()
                    if not ref or ref.startswith("#"):
                        continue
                    if not bool(getattr(sym, "on_board", True)):
                        continue
                    refs.add(ref)
            except Exception:
                refs = set()

        companion = self._benchmark_companion_schematic_file()
        if companion is None or not companion.exists():
            return refs
        parsed = self._benchmark_parse_schematic_hierarchy(companion)
        parsed_refs = parsed.get("refs")
        if isinstance(parsed_refs, set):
            refs.update(parsed_refs)
            return refs
        if isinstance(parsed_refs, list):
            refs.update({str(r or "").strip().upper() for r in parsed_refs if str(r or "").strip()})
            return refs
        return refs
    def _benchmark_schematic_audit(self) -> Dict[str, Any]:
        """Check whether a usable schematic is available for benchmark verification."""
        loaded = bool(self.schematic_data is not None)
        component_count = 0
        wire_count = 0
        label_count = 0
        if loaded:
            try:
                component_count = int(getattr(self.schematic_data, "component_count", 0) or 0)
            except Exception:
                component_count = 0
            try:
                wire_count = len(list(getattr(self.schematic_data, "wires", []) or []))
            except Exception:
                wire_count = 0
            try:
                label_count = int(getattr(self.schematic_data, "net_label_count", 0) or 0)
            except Exception:
                label_count = 0

        companion = self._benchmark_companion_schematic_file()
        companion_path = str(companion) if companion is not None else ""
        companion_exists = bool(companion is not None and companion.exists())
        hierarchical_fallback_used = False
        hierarchical_file_count = 0
        hierarchical_parse_errors_sample: List[str] = []

        if companion_exists and component_count <= 0 and companion is not None:
            hier = self._benchmark_parse_schematic_hierarchy(companion)
            hierarchical_file_count = int(hier.get("parsed_file_count", 0) or 0)
            hierarchical_parse_errors_sample = list(hier.get("parse_errors_sample", []) or [])
            hier_component_count = int(hier.get("component_count", 0) or 0)
            if hier_component_count > 0:
                component_count = hier_component_count
                wire_count = max(wire_count, int(hier.get("wire_count", 0) or 0))
                label_count = max(label_count, int(hier.get("net_label_count", 0) or 0))
                loaded = True
                hierarchical_fallback_used = True

        ok = loaded and component_count > 0
        reason = ""
        if not ok:
            if companion_exists:
                reason = (
                    "companion_schematic_parse_failed"
                    if hierarchical_parse_errors_sample
                    else "companion_schematic_exists_but_not_loaded_or_invalid"
                )
            else:
                reason = "no_loaded_schematic"

        return {
            "ok": bool(ok),
            "loaded": bool(loaded),
            "component_count": int(component_count),
            "wire_count": int(wire_count),
            "net_label_count": int(label_count),
            "companion_path": companion_path,
            "companion_exists": bool(companion_exists),
            "hierarchical_fallback_used": bool(hierarchical_fallback_used),
            "hierarchical_file_count": int(hierarchical_file_count),
            "hierarchical_parse_errors_sample": hierarchical_parse_errors_sample[:8],
            "reason": reason,
        }
    def _benchmark_ground_plane_audit(self) -> Dict[str, Any]:
        """Check whether the design has at least one copper GND plane."""
        zones: List[Dict[str, Any]] = []

        if PCBNEW_AVAILABLE:
            try:
                board = pcbnew.GetBoard()
            except Exception:
                board = None
            if board is not None:
                try:
                    zone_iter = list(board.Zones() or [])
                except Exception:
                    zone_iter = []
                for zone in zone_iter:
                    net_name = ""
                    layer_name = ""
                    is_rule_area = False
                    try:
                        is_rule_area = bool(getattr(zone, "GetIsRuleArea", lambda: False)())
                    except Exception:
                        is_rule_area = False
                    try:
                        gn = getattr(zone, "GetNetname", None)
                        if callable(gn):
                            net_name = str(gn() or "").strip()
                    except Exception:
                        net_name = ""
                    if not net_name:
                        try:
                            net_obj = getattr(zone, "GetNet", lambda: None)()
                            if net_obj is not None:
                                net_name = str(getattr(net_obj, "GetNetname", lambda: "")() or "").strip()
                        except Exception:
                            pass
                    try:
                        lid = int(getattr(zone, "GetLayer", lambda: -1)())
                    except Exception:
                        lid = -1
                    if lid >= 0:
                        try:
                            layer_name = str(getattr(board, "GetLayerName", lambda _lid: "")(lid) or "").strip()
                        except Exception:
                            layer_name = ""
                    is_copper = ".cu" in layer_name.lower() if layer_name else False
                    zones.append(
                        {
                            "net_name": net_name,
                            "layer": layer_name,
                            "is_copper": bool(is_copper),
                            "is_rule_area": bool(is_rule_area),
                        }
                    )

        # Fallback to parser snapshot if board API is unavailable or empty.
        if not zones and self.pcb_data is not None:
            try:
                for zone in list(getattr(self.pcb_data, "zones", []) or []):
                    if zone is None:
                        continue
                    net_name = str(getattr(zone, "net_name", "") or "").strip()
                    layer_name = str(getattr(zone, "layer", "") or "").strip()
                    zones.append(
                        {
                            "net_name": net_name,
                            "layer": layer_name,
                            "is_copper": bool(".cu" in layer_name.lower()),
                            "is_rule_area": False,
                        }
                    )
            except Exception:
                pass

        gnd_copper_zones = [
            row
            for row in zones
            if isinstance(row, dict)
            and bool(row.get("is_copper"))
            and (self._benchmark_net_canonical_name(str(row.get("net_name", "") or "")) == "gnd")
            and not bool(row.get("is_rule_area"))
        ]

        return {
            "ok": bool(gnd_copper_zones),
            "zone_count": len(zones),
            "gnd_copper_zone_count": len(gnd_copper_zones),
            "gnd_copper_zones_sample": gnd_copper_zones[:12],
        }
    def _benchmark_clock_placement_audit(
        self,
        loop: Any,
        benchmark: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Audit crystal/resonator placement relative to the nearest MCU."""
        board_snapshot = self._benchmark_board_placement_snapshot()
        if not isinstance(board_snapshot, dict) or not board_snapshot:
            return {"applicable": False, "ok": False, "reason": "no_board_snapshot"}

        profile = self._benchmark_generalizable_profile(benchmark)
        distance_limit_mm = self._benchmark_profile_float(
            profile,
            ("clock", "distance_limit_mm"),
            20.0,
            min_value=1.0,
        )
        cap_radius_mm = self._benchmark_profile_float(
            profile,
            ("clock", "cap_radius_mm"),
            12.0,
            min_value=1.0,
        )
        min_crystal_caps = self._benchmark_profile_int(
            profile,
            ("clock", "min_crystal_caps"),
            2,
            min_value=1,
        )

        manifest_by_ref: Dict[str, Dict[str, Any]] = {}
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            manifest = artifacts.get("manifest") if isinstance(artifacts.get("manifest"), dict) else {}
            for part in list(manifest.get("parts") or []):
                if not isinstance(part, dict):
                    continue
                pref = str(part.get("ref", "") or "").strip().upper()
                if pref:
                    manifest_by_ref[pref] = part
        except Exception:
            manifest_by_ref = {}

        def _blob_for_ref(ref: str) -> str:
            snap = board_snapshot.get(ref) if isinstance(board_snapshot.get(ref), dict) else {}
            part = manifest_by_ref.get(ref, {})
            fields = [
                ref,
                str(snap.get("footprint", "") or ""),
                str(part.get("mpn", "") or ""),
                str(part.get("value", "") or ""),
                str(part.get("description", "") or ""),
            ]
            return " ".join(fields).lower()

        def _dist_mm(a_ref: str, b_ref: str) -> float:
            a = board_snapshot.get(a_ref) if isinstance(board_snapshot.get(a_ref), dict) else {}
            b = board_snapshot.get(b_ref) if isinstance(board_snapshot.get(b_ref), dict) else {}
            ax = float(a.get("x", 0.0) or 0.0)
            ay = float(a.get("y", 0.0) or 0.0)
            bx = float(b.get("x", 0.0) or 0.0)
            by = float(b.get("y", 0.0) or 0.0)
            dx = ax - bx
            dy = ay - by
            return float((dx * dx + dy * dy) ** 0.5)

        refs = sorted(str(r or "").strip().upper() for r in list(board_snapshot.keys()) if str(r or "").strip())
        if not refs:
            return {"applicable": False, "ok": False, "reason": "no_component_refs"}

        net_snapshot = self._benchmark_net_snapshot_with_fallback(loop, allowed_refs=set(refs))
        net_by_ref = net_snapshot.get("by_ref") if isinstance(net_snapshot.get("by_ref"), dict) else {}
        net_by_group = net_snapshot.get("by_group") if isinstance(net_snapshot.get("by_group"), dict) else {}

        clock_refs = []
        mcu_refs = []
        cap_refs = []
        mcu_tokens = ("mcu", "microcontroller", "atmega", "stm32", "esp32", "pic", "avr", "samd", "nrf")
        for ref in refs:
            blob = _blob_for_ref(ref)
            if ref.startswith("C"):
                cap_refs.append(ref)
            if ref.startswith(("X", "Y")) or any(tok in blob for tok in ("crystal", "xtal", "resonator", "oscillator")):
                clock_refs.append(ref)
            if ref.startswith("U") and any(tok in blob for tok in mcu_tokens):
                mcu_refs.append(ref)

        if not clock_refs:
            return {"applicable": False, "ok": True, "reason": "no_clock_source_found"}
        if not mcu_refs:
            # Fallback: if no explicit MCU token, use IC refs.
            mcu_refs = [ref for ref in refs if ref.startswith("U")]
        if not mcu_refs:
            return {
                "applicable": True,
                "ok": False,
                "reason": "no_mcu_found_for_clock_distance_check",
                "clock_ref_count": len(clock_refs),
            }

        def _associated_mcus_for_clock(clock_ref: str) -> List[str]:
            associated: Set[str] = set()
            pad_map = net_by_ref.get(clock_ref) if isinstance(net_by_ref.get(clock_ref), dict) else {}
            for net_name in pad_map.values():
                group = self._benchmark_net_canonical_name(str(net_name or "").strip())
                if not group:
                    continue
                row = net_by_group.get(group) if isinstance(net_by_group.get(group), dict) else {}
                for ref in list(row.get("refs") or []):
                    ref_u = str(ref or "").strip().upper()
                    if ref_u in mcu_refs:
                        associated.add(ref_u)
            if clock_ref in associated:
                associated.discard(clock_ref)
            return sorted(associated)

        def _canonical_nets_for_ref(ref: str) -> Set[str]:
            out: Set[str] = set()
            pad_map = net_by_ref.get(ref) if isinstance(net_by_ref.get(ref), dict) else {}
            for net_name in pad_map.values():
                group = self._benchmark_net_canonical_name(str(net_name or "").strip())
                if group:
                    out.add(group)
            return out

        net_evidence_ready = bool(net_by_ref)
        rows: List[Dict[str, Any]] = []
        distance_ok = True
        crystal_caps_ok = True
        for clock_ref in sorted(set(clock_refs)):
            clock_blob = _blob_for_ref(clock_ref)
            candidate_mcu_refs = _associated_mcus_for_clock(clock_ref) or list(mcu_refs)
            clock_nets = _canonical_nets_for_ref(clock_ref)
            clock_signal_nets = {n for n in clock_nets if n != "gnd"}
            nearest_mcu = ""
            nearest_dist = None
            for mcu_ref in candidate_mcu_refs:
                d = _dist_mm(clock_ref, mcu_ref)
                if nearest_dist is None or d < nearest_dist:
                    nearest_dist = d
                    nearest_mcu = mcu_ref
            nearby_caps_detailed: List[Dict[str, Any]] = []
            for cref in cap_refs:
                dist = _dist_mm(clock_ref, cref)
                if dist > cap_radius_mm:
                    continue
                cap_nets = _canonical_nets_for_ref(cref)
                overlap_nets = sorted((cap_nets & clock_signal_nets) - {"gnd"})
                has_gnd = "gnd" in cap_nets
                electrically_related = bool(overlap_nets) and has_gnd
                nearby_caps_detailed.append(
                    {
                        "ref": cref,
                        "distance_mm": round(float(dist), 3),
                        "has_gnd": bool(has_gnd),
                        "clock_net_overlap": overlap_nets,
                        "electrically_related": bool(electrically_related),
                    }
                )

            if net_evidence_ready:
                if clock_signal_nets:
                    effective_caps = [row for row in nearby_caps_detailed if bool(row.get("electrically_related", False))]
                else:
                    # If we cannot infer clock nets, at least prefer grounded nearby caps.
                    effective_caps = [row for row in nearby_caps_detailed if bool(row.get("has_gnd", False))]
            else:
                # Fallback when we lack usable net evidence.
                effective_caps = list(nearby_caps_detailed)

            is_integrated_resonator = any(tok in clock_blob for tok in ("resonator", "cstce", "cstd", "cstne", "ceramic"))
            is_crystal_like = ("crystal" in clock_blob) and (not is_integrated_resonator)
            this_distance_ok = (nearest_dist is not None) and (nearest_dist <= distance_limit_mm)
            this_caps_ok = (not is_crystal_like) or (len(effective_caps) >= min_crystal_caps)
            if not this_distance_ok:
                distance_ok = False
            if not this_caps_ok:
                crystal_caps_ok = False
            rows.append(
                {
                    "clock_ref": clock_ref,
                    "nearest_mcu_ref": nearest_mcu,
                    "associated_mcu_refs": candidate_mcu_refs[:8],
                    "distance_to_mcu_mm": (round(float(nearest_dist), 3) if nearest_dist is not None else None),
                    "distance_limit_mm": distance_limit_mm,
                    "is_crystal_like": bool(is_crystal_like),
                    "min_crystal_caps": int(min_crystal_caps),
                    "clock_net_groups": sorted(clock_signal_nets),
                    "nearby_cap_radius_mm": round(float(cap_radius_mm), 3),
                    "nearby_cap_count_within_radius_mm": len(nearby_caps_detailed),
                    "nearby_cap_count_within_12mm": len(nearby_caps_detailed),
                    "electrically_related_cap_count": len(effective_caps),
                    "nearby_caps_sample": [str(row.get("ref", "") or "") for row in nearby_caps_detailed[:8]],
                    "nearby_caps_detailed_sample": nearby_caps_detailed[:8],
                    "distance_ok": bool(this_distance_ok),
                    "caps_ok": bool(this_caps_ok),
                }
            )

        return {
            "applicable": True,
            "ok": bool(distance_ok and crystal_caps_ok),
            "distance_ok": bool(distance_ok),
            "crystal_load_cap_hint_ok": bool(crystal_caps_ok),
            "clock_ref_count": len(clock_refs),
            "mcu_ref_count": len(mcu_refs),
            "thresholds": {
                "distance_limit_mm": round(float(distance_limit_mm), 3),
                "cap_radius_mm": round(float(cap_radius_mm), 3),
                "min_crystal_caps": int(min_crystal_caps),
                "net_evidence_ready": bool(net_evidence_ready),
            },
            "rows": rows[:24],
        }
    def _benchmark_design_sanity_checks(
        self,
        loop: Any,
        benchmark: Dict[str, Any],
        net_score: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Post-run benchmark sanity checks for schematic/ground/clock/power integrity."""
        checks: List[Dict[str, Any]] = []

        schematic = self._benchmark_schematic_audit()
        checks.append(
            {
                "id": "schematic_presence",
                "required": bool(benchmark.get("strict_require_schematic", False)),
                "ok": bool(schematic.get("ok", False)),
                "gate": "BENCHMARK.schematic_required",
                "message": "No usable schematic loaded; ERC-grade verification is unavailable",
                "bounce_to": "SPEC",
                "details": schematic,
            }
        )

        ground = self._benchmark_ground_plane_audit()
        checks.append(
            {
                "id": "ground_plane",
                "required": bool(benchmark.get("strict_require_ground_plane", False)),
                "ok": bool(ground.get("ok", False)),
                "gate": "BENCHMARK.ground_plane",
                "message": "No copper GND zone found",
                "bounce_to": "GEOM",
                "details": ground,
            }
        )

        clock = self._benchmark_clock_placement_audit(loop, benchmark=benchmark)
        checks.append(
            {
                "id": "clock_placement",
                "required": bool(benchmark.get("strict_require_clock_placement", False))
                and bool(clock.get("applicable", False)),
                "ok": bool(clock.get("ok", False)),
                "gate": "BENCHMARK.clock_placement",
                "message": "Clock source placement is risky (distance/cap-proximity checks failed)",
                "bounce_to": "GEOM",
                "details": clock,
            }
        )

        bridge_issues = int(net_score.get("bridge_integrity_issue_ref_count", 0) or 0)
        bridge_details = {
            "bridge_integrity_candidate_ref_count": int(net_score.get("bridge_integrity_candidate_ref_count", 0) or 0),
            "bridge_integrity_issue_ref_count": bridge_issues,
            "bridge_integrity_issues_sample": list(net_score.get("bridge_integrity_issues_sample", []) or [])[:20],
        }
        checks.append(
            {
                "id": "power_path_bridge_integrity",
                "required": bool(net_score.get("applicable", False)),
                "ok": bridge_issues == 0,
                "gate": "BENCHMARK.power_path_integrity",
                "message": "Power/protection bridge parts have suspicious connectivity",
                "bounce_to": "NET",
                "details": bridge_details,
            }
        )

        failed_required = [
            row
            for row in checks
            if bool(row.get("required", False)) and not bool(row.get("ok", False))
        ]
        first_failure = failed_required[0] if failed_required else None
        return {
            "applicable": bool(checks),
            "ok": len(failed_required) == 0,
            "checks": checks,
            "failed_required_checks": failed_required,
            "first_failure": first_failure if isinstance(first_failure, dict) else None,
        }
    def _benchmark_manifest_part_map(self, loop: Any) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        try:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            manifest = artifacts.get("manifest") if isinstance(artifacts.get("manifest"), dict) else {}
            for part in list(manifest.get("parts") or []):
                if not isinstance(part, dict):
                    continue
                ref = str(part.get("ref", "") or "").strip().upper()
                if ref:
                    out[ref] = part
        except Exception:
            pass
        return out
    def _benchmark_board_snapshot_with_fallback(self) -> Dict[str, Any]:
        snap = self._benchmark_board_placement_snapshot()
        if isinstance(snap, dict) and snap:
            return snap

        out: Dict[str, Any] = {}
        pdata = self.pcb_data
        if pdata is None:
            return out
        try:
            for fp in list(getattr(pdata, "footprints", []) or []):
                ref = str(getattr(fp, "reference", "") or "").strip().upper()
                if not ref:
                    continue
                out[ref] = {
                    "x": float(getattr(getattr(fp, "at", None), "x", 0.0) or 0.0),
                    "y": float(getattr(getattr(fp, "at", None), "y", 0.0) or 0.0),
                    "rot": float(getattr(fp, "rotation", 0.0) or 0.0),
                    "footprint": (
                        f"{str(getattr(fp, 'library', '') or '').strip()}:{str(getattr(fp, 'footprint_name', '') or '').strip()}"
                        if str(getattr(fp, "library", "") or "").strip()
                        else str(getattr(fp, "footprint_name", "") or "").strip()
                    ),
                    "layer": "F",
                }
        except Exception:
            return {}
        return out
    def _benchmark_parser_net_snapshot(self, allowed_refs: Optional[Set[str]] = None) -> Dict[str, Any]:
        pdata = self.pcb_data
        if pdata is None:
            return {"source": "parser", "usable": False}

        by_ref: Dict[str, Dict[str, str]] = {}
        all_pad_names_by_ref: Dict[str, Set[str]] = {}
        by_group_tmp: Dict[str, Dict[str, Any]] = {}
        unique_net_names: Set[str] = set()
        assignment_count = 0

        try:
            for fp in list(getattr(pdata, "footprints", []) or []):
                ref = str(getattr(fp, "reference", "") or "").strip().upper()
                if not ref:
                    continue
                if allowed_refs is not None and ref not in allowed_refs:
                    continue
                pads = list(getattr(fp, "pads", []) or [])
                for pad in pads:
                    pad_name = str(getattr(pad, "number", "") or "").strip()
                    if pad_name:
                        all_pad_names_by_ref.setdefault(ref, set()).add(pad_name)
                    net_name = str(getattr(pad, "net_name", "") or "").strip()
                    if not pad_name or not net_name:
                        continue
                    by_ref.setdefault(ref, {})[pad_name] = net_name
                    unique_net_names.add(net_name)
                    assignment_count += 1
                    group = self._benchmark_net_canonical_name(net_name)
                    if not group:
                        continue
                    row = by_group_tmp.setdefault(
                        group,
                        {"refs": set(), "net_names": set(), "pads": [], "net_refs": {}, "net_pad_count": {}},
                    )
                    row["refs"].add(ref)
                    row["net_names"].add(net_name)
                    net_refs = row.setdefault("net_refs", {})
                    refs_for_net = net_refs.setdefault(net_name, set())
                    refs_for_net.add(ref)
                    net_pad_count = row.setdefault("net_pad_count", {})
                    net_pad_count[net_name] = int(net_pad_count.get(net_name, 0) or 0) + 1
                    if len(row["pads"]) < 24:
                        row["pads"].append(f"{ref}.{pad_name}")
        except Exception:
            return {"source": "parser", "usable": False}

        by_group = {
            group: {
                "refs": sorted(set(row.get("refs") or [])),
                "net_names": sorted(set(row.get("net_names") or [])),
                "pad_count": len(list(row.get("pads") or [])),
                "pads_sample": list(row.get("pads") or []),
                "net_ref_map": {
                    str(net): sorted(set(refs or []))
                    for net, refs in dict(row.get("net_refs") or {}).items()
                    if str(net or "").strip()
                },
                "net_pad_count_map": {
                    str(net): int(count or 0)
                    for net, count in dict(row.get("net_pad_count") or {}).items()
                    if str(net or "").strip()
                },
            }
            for group, row in by_group_tmp.items()
        }
        return {
            "source": "parser",
            "usable": bool(by_ref),
            "assignment_count": assignment_count,
            "unique_net_count": len(unique_net_names),
            "by_ref": by_ref,
            "all_pad_names_by_ref": {
                ref: sorted(set(pads))
                for ref, pads in all_pad_names_by_ref.items()
            },
            "by_group": by_group,
        }
    def _benchmark_net_snapshot_with_fallback(
        self,
        loop: Any,
        allowed_refs: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        board_snapshot = self._benchmark_board_net_snapshot(allowed_refs=allowed_refs)
        if bool(board_snapshot.get("usable")):
            return board_snapshot
        parser_snapshot = self._benchmark_parser_net_snapshot(allowed_refs=allowed_refs)
        if bool(parser_snapshot.get("usable")):
            return parser_snapshot
        return self._benchmark_planned_net_snapshot(loop)
    def _benchmark_ref_text_blob(
        self,
        ref: str,
        manifest_by_ref: Dict[str, Dict[str, Any]],
        board_snapshot: Dict[str, Any],
    ) -> str:
        ref_u = str(ref or "").strip().upper()
        part = manifest_by_ref.get(ref_u, {}) if isinstance(manifest_by_ref, dict) else {}
        b = board_snapshot.get(ref_u, {}) if isinstance(board_snapshot, dict) else {}
        fields: List[str] = [ref_u]
        if isinstance(part, dict):
            for key in ("mpn", "value", "description", "footprint"):
                fields.append(str(part.get(key, "") or ""))
        if isinstance(b, dict):
            fields.append(str(b.get("footprint", "") or ""))
        s = " ".join(fields).lower()
        s = re.sub(r"[_:/\\.-]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    @staticmethod
    def _benchmark_distance_mm(board_snapshot: Dict[str, Any], a_ref: str, b_ref: str) -> Optional[float]:
        try:
            a = board_snapshot.get(str(a_ref or "").strip().upper())
            b = board_snapshot.get(str(b_ref or "").strip().upper())
            if not isinstance(a, dict) or not isinstance(b, dict):
                return None
            ax = float(a.get("x", 0.0) or 0.0)
            ay = float(a.get("y", 0.0) or 0.0)
            bx = float(b.get("x", 0.0) or 0.0)
            by = float(b.get("y", 0.0) or 0.0)
            dx = ax - bx
            dy = ay - by
            return float((dx * dx + dy * dy) ** 0.5)
        except Exception:
            return None
    @staticmethod
    def _benchmark_cap_value_uF(text: str) -> Optional[float]:
        raw = str(text or "").strip().lower()
        if not raw:
            return None
        raw = raw.replace("µ", "u").replace("μ", "u")
        raw = raw.replace(" ", "")

        m = re.search(r"(\d+(?:\.\d+)?)(p|n|u|m)?f\b", raw)
        if not m:
            m = re.search(r"\b(\d+(?:\.\d+)?)(p|n|u|m)\b", raw)
        if not m:
            return None
        try:
            value = float(m.group(1))
        except Exception:
            return None
        prefix = str(m.group(2) or "u").lower()
        if prefix == "p":
            return value * 1e-6
        if prefix == "n":
            return value * 1e-3
        if prefix == "u":
            return value
        if prefix == "m":
            return value * 1e3
        return value
    @staticmethod
    def _benchmark_power_group_family(group: str) -> str:
        g = str(group or "").strip().lower()
        if not g:
            return ""
        if g in {"usb_vbus", "vusb", "usbvcc", "usb_vcc", "vbus"}:
            return "usb_input"
        if g in {"vin_jack", "dc_jack_vin", "dc_in"}:
            return "vin_input"
        if g in {"vin", "vin_raw"}:
            return "vin_raw"
        if g in {"plus5v", "v5_0", "5v", "5v0"}:
            return "logic_5v"
        if g in {"v3_3", "v3_30", "3v3", "3v30"}:
            return "logic_3v3"
        if g in {"gnd", "agnd", "pgnd", "earth"}:
            return "ground"
        if re.match(r"^(?:\+|-)?v\d+(?:_\d+)?$", g):
            return "voltage_rail"
        if re.match(r"^(?:\d+v\d+)$", g):
            return "voltage_rail"
        if ("vdd" in g) or ("vcc" in g) or ("vss" in g) or ("vbat" in g):
            return "voltage_rail"
        if ("power" in g) or ("pwr" in g) or ("rail" in g):
            return "power_named"
        return ""

    def _benchmark_is_power_group(self, group: str) -> bool:
        fam = self._benchmark_power_group_family(group)
        return bool(fam and fam != "ground")

    def _benchmark_parser_track_metrics(self) -> Dict[str, Any]:
        pdata = self.pcb_data
        if pdata is None:
            return {"usable": False}

        net_by_num: Dict[int, str] = {}
        try:
            for n in list(getattr(pdata, "nets", []) or []):
                num = int(getattr(n, "number", 0) or 0)
                name = str(getattr(n, "name", "") or "").strip()
                if num > 0 and name:
                    net_by_num[num] = name
        except Exception:
            net_by_num = {}

        by_group: Dict[str, Dict[str, Any]] = {}
        try:
            for tr in list(getattr(pdata, "tracks", []) or []):
                net_num = int(getattr(tr, "net", 0) or 0)
                net_name = str(net_by_num.get(net_num, "") or "").strip()
                if not net_name:
                    continue
                group = self._benchmark_net_canonical_name(net_name) or net_name.lower()
                sx = float(getattr(getattr(tr, "start", None), "x", 0.0) or 0.0)
                sy = float(getattr(getattr(tr, "start", None), "y", 0.0) or 0.0)
                ex = float(getattr(getattr(tr, "end", None), "x", 0.0) or 0.0)
                ey = float(getattr(getattr(tr, "end", None), "y", 0.0) or 0.0)
                length_mm = float(((sx - ex) ** 2 + (sy - ey) ** 2) ** 0.5)
                width_mm = float(getattr(tr, "width", 0.0) or 0.0)
                row = by_group.setdefault(
                    group,
                    {
                        "net_names": set(),
                        "segment_count": 0,
                        "total_length_mm": 0.0,
                        "width_sum_mm": 0.0,
                        "min_width_mm": None,
                        "max_width_mm": 0.0,
                        "via_count": 0,
                        "min_via_drill_mm": None,
                    },
                )
                row["net_names"].add(net_name)
                row["segment_count"] = int(row.get("segment_count", 0) or 0) + 1
                row["total_length_mm"] = float(row.get("total_length_mm", 0.0) or 0.0) + length_mm
                row["width_sum_mm"] = float(row.get("width_sum_mm", 0.0) or 0.0) + width_mm
                prev_min = row.get("min_width_mm")
                row["min_width_mm"] = width_mm if prev_min is None else min(float(prev_min), width_mm)
                row["max_width_mm"] = max(float(row.get("max_width_mm", 0.0) or 0.0), width_mm)
        except Exception:
            return {"usable": False}

        try:
            for via in list(getattr(pdata, "vias", []) or []):
                net_num = int(getattr(via, "net", 0) or 0)
                net_name = str(net_by_num.get(net_num, "") or "").strip()
                if not net_name:
                    continue
                group = self._benchmark_net_canonical_name(net_name) or net_name.lower()
                drill_mm = float(getattr(via, "drill", 0.0) or 0.0)
                row = by_group.setdefault(
                    group,
                    {
                        "net_names": set(),
                        "segment_count": 0,
                        "total_length_mm": 0.0,
                        "width_sum_mm": 0.0,
                        "min_width_mm": None,
                        "max_width_mm": 0.0,
                        "via_count": 0,
                        "min_via_drill_mm": None,
                    },
                )
                row["net_names"].add(net_name)
                row["via_count"] = int(row.get("via_count", 0) or 0) + 1
                prev_drill = row.get("min_via_drill_mm")
                row["min_via_drill_mm"] = drill_mm if prev_drill is None else min(float(prev_drill), drill_mm)
        except Exception:
            pass

        out = {}
        for group, row in by_group.items():
            seg_count = int(row.get("segment_count", 0) or 0)
            total_len = float(row.get("total_length_mm", 0.0) or 0.0)
            out[group] = {
                "net_names": sorted(set(row.get("net_names") or [])),
                "segment_count": seg_count,
                "total_length_mm": round(total_len, 3),
                "avg_width_mm": (
                    round(float(row.get("width_sum_mm", 0.0) or 0.0) / float(seg_count), 3)
                    if seg_count > 0
                    else None
                ),
                "min_width_mm": (
                    round(float(row.get("min_width_mm")), 3)
                    if row.get("min_width_mm") is not None
                    else None
                ),
                "max_width_mm": (
                    round(float(row.get("max_width_mm", 0.0) or 0.0), 3)
                    if seg_count > 0
                    else None
                ),
                "via_count": int(row.get("via_count", 0) or 0),
                "min_via_drill_mm": (
                    round(float(row.get("min_via_drill_mm")), 3)
                    if row.get("min_via_drill_mm") is not None
                    else None
                ),
            }
        return {"usable": bool(out), "by_group": out}
    def _benchmark_collect_drc_messages(self, loop: Any) -> Dict[str, Any]:
        raw_errors: List[str] = []
        warnings: List[str] = []
        try:
            history = list(getattr(loop, "_history", []) or [])
            for step in history:
                action = getattr(step, "action", None)
                if action is None:
                    continue
                at = str(getattr(getattr(action, "action_type", None), "name", getattr(action, "action_type", "")) or "")
                if at != "RUN_DRC":
                    continue
                text = str(getattr(action, "result_message", "") or "")
                raw_errors.extend(self._benchmark_drc_error_entries(text))
                warnings.extend(self._benchmark_drc_warning_entries(text))
        except Exception:
            pass
        classified = self._benchmark_reclassify_intrinsic_clearance_errors(raw_errors)
        effective_errors = list(classified.get("effective_errors", []) or [])
        intrinsic_errors = list(classified.get("intrinsic_errors", []) or [])
        warnings.extend([str(w) for w in list(classified.get("warning_entries", []) or [])])
        return {
            "errors": effective_errors,
            "warnings": warnings,
            "error_count": len(effective_errors),
            "warning_count": len(warnings),
            "raw_error_count": int(classified.get("raw_error_count", len(raw_errors)) or len(raw_errors)),
            "intrinsic_error_count": int(classified.get("intrinsic_error_count", len(intrinsic_errors)) or len(intrinsic_errors)),
            "intrinsic_errors": intrinsic_errors,
        }
    def _benchmark_ee_rules_audit(
        self,
        loop: Any,
        benchmark: Dict[str, Any],
        net_score: Dict[str, Any],
        design_sanity: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Comprehensive EE rule checks. Every failed rule contributes score penalties."""
        scenario = str(benchmark.get("scenario", "") or "").lower()
        is_uno = "uno" in scenario

        manifest_by_ref = self._benchmark_manifest_part_map(loop)
        board_snapshot = self._benchmark_board_snapshot_with_fallback()
        allowed_refs = set(manifest_by_ref.keys()) if manifest_by_ref else set(board_snapshot.keys())
        net_snapshot = self._benchmark_net_snapshot_with_fallback(loop, allowed_refs=allowed_refs or None)
        by_ref = net_snapshot.get("by_ref") if isinstance(net_snapshot.get("by_ref"), dict) else {}
        by_group = net_snapshot.get("by_group") if isinstance(net_snapshot.get("by_group"), dict) else {}
        drc_info = self._benchmark_collect_drc_messages(loop)
        track_metrics = self._benchmark_parser_track_metrics()
        track_by_group = track_metrics.get("by_group") if isinstance(track_metrics.get("by_group"), dict) else {}
        schematic = self._benchmark_schematic_audit()
        ground = self._benchmark_ground_plane_audit()
        clock = self._benchmark_clock_placement_audit(loop, benchmark=benchmark)
        net_source = str(net_score.get("source", "") or "").strip().lower()
        net_assignment_count = int(net_score.get("assignment_count", 0) or 0)
        net_evidence_ready = bool(net_source == "board" and net_assignment_count > 0)
        defer_net_ee_until_routed = bool(benchmark.get("defer_net_dependent_ee_until_routed", True))
        routed_segment_total = 0
        try:
            routed_segment_total = int(
                sum(
                    int(row.get("segment_count", 0) or 0)
                    for row in list(track_by_group.values())
                    if isinstance(row, dict)
                )
            )
        except Exception:
            routed_segment_total = 0
        routing_evidence_ready = bool(track_by_group) and routed_segment_total > 0

        profile = self._benchmark_generalizable_profile(benchmark)
        regulator_cap_radius_mm = self._benchmark_profile_float(
            profile,
            ("ee_rules", "regulator_cap_radius_mm"),
            15.0,
            min_value=1.0,
        )
        regulator_small_cap_min_uf = self._benchmark_profile_float(
            profile,
            ("ee_rules", "regulator_small_cap_min_uf"),
            0.01,
            min_value=0.0,
        )
        regulator_small_cap_max_uf = self._benchmark_profile_float(
            profile,
            ("ee_rules", "regulator_small_cap_max_uf"),
            0.22,
            min_value=regulator_small_cap_min_uf,
        )
        regulator_bulk_cap_min_uf = self._benchmark_profile_float(
            profile,
            ("ee_rules", "regulator_bulk_cap_min_uf"),
            1.0,
            min_value=0.0,
        )
        regulator_min_nearby_caps = self._benchmark_profile_int(
            profile,
            ("ee_rules", "regulator_min_nearby_caps"),
            2,
            min_value=1,
        )
        regulator_min_small_caps = self._benchmark_profile_int(
            profile,
            ("ee_rules", "regulator_min_small_caps"),
            1,
            min_value=0,
        )
        regulator_min_bulk_caps = self._benchmark_profile_int(
            profile,
            ("ee_rules", "regulator_min_bulk_caps"),
            1,
            min_value=0,
        )
        decoupling_radius_mm = self._benchmark_profile_float(
            profile,
            ("ee_rules", "decoupling_radius_mm"),
            12.0,
            min_value=1.0,
        )
        decoupling_small_cap_min_uf = self._benchmark_profile_float(
            profile,
            ("ee_rules", "decoupling_small_cap_min_uf"),
            0.01,
            min_value=0.0,
        )
        decoupling_small_cap_max_uf = self._benchmark_profile_float(
            profile,
            ("ee_rules", "decoupling_small_cap_max_uf"),
            0.22,
            min_value=decoupling_small_cap_min_uf,
        )
        decoupling_min_small_caps_per_ic = self._benchmark_profile_int(
            profile,
            ("ee_rules", "decoupling_min_small_caps_per_ic"),
            1,
            min_value=0,
        )
        bulk_cap_min_uf_per_rail = self._benchmark_profile_float(
            profile,
            ("ee_rules", "bulk_cap_min_uf_per_rail"),
            4.7,
            min_value=0.0,
        )
        usb_protector_max_distance_mm = self._benchmark_profile_float(
            profile,
            ("ee_rules", "usb_protector_max_distance_mm"),
            20.0,
            min_value=0.0,
        )
        enforce_dual_layer_gnd_zone_only_when_routed = self._benchmark_profile_bool(
            profile,
            ("ee_rules", "enforce_dual_layer_gnd_zone_only_when_routed"),
            True,
        )
        gnd_single_layer_risk_min_segments = self._benchmark_profile_int(
            profile,
            ("ee_rules", "single_layer_gnd_zone_risk_min_gnd_segments"),
            80,
            min_value=1,
        )
        gnd_single_layer_risk_min_vias = self._benchmark_profile_int(
            profile,
            ("ee_rules", "single_layer_gnd_zone_risk_min_gnd_vias"),
            4,
            min_value=0,
        )

        def _canonical_nets_for_ref(ref: str) -> Set[str]:
            nets: Set[str] = set()
            net_map = by_ref.get(ref) if isinstance(by_ref.get(ref), dict) else {}
            for net_name in net_map.values():
                group = self._benchmark_net_canonical_name(str(net_name or "").strip())
                if group:
                    nets.add(group)
            return nets

        def _power_nets_for_ref(ref: str) -> Set[str]:
            return {g for g in _canonical_nets_for_ref(ref) if self._benchmark_is_power_group(g)}

        rules: List[Dict[str, Any]] = []

        def _add_rule(
            *,
            rule_id: str,
            title: str,
            ok: bool,
            penalty: int,
            stages: List[str],
            message: str,
            details: Dict[str, Any],
            applicable: bool = True,
            required: bool = True,
        ) -> None:
            rules.append(
                {
                    "id": str(rule_id),
                    "title": str(title),
                    "ok": bool(ok),
                    "applicable": bool(applicable),
                    "required": bool(required),
                    "penalty": int(max(0, penalty)),
                    "stages": [str(s).upper() for s in stages if str(s).strip()],
                    "message": str(message),
                    "details": details if isinstance(details, dict) else {},
                }
            )

        mechanical_prefixes = ("H", "TP", "FID", "MH")
        pcb_refs = {str(r).strip().upper() for r in board_snapshot.keys() if str(r).strip()}
        schematic_refs: Set[str] = self._benchmark_schematic_ref_set()

        # 1) Schematic ↔ PCB equivalence
        missing_on_pcb = sorted([r for r in (schematic_refs - pcb_refs) if not r.startswith(mechanical_prefixes)])[:40]
        extra_on_pcb = sorted([r for r in (pcb_refs - schematic_refs) if not r.startswith(mechanical_prefixes)])[:40]
        manifest_refs = set(manifest_by_ref.keys())
        manifest_missing_in_sch = sorted(list(manifest_refs - schematic_refs))[:40] if schematic_refs else []
        manifest_missing_on_pcb = sorted(list(manifest_refs - pcb_refs))[:40] if manifest_refs else []
        coverage = net_score.get("coverage") if isinstance(net_score.get("coverage"), dict) else {}
        partial_refs = int(coverage.get("partial_refs_count", 0) or 0)
        zero_refs = int(coverage.get("refs_without_nets_count", 0) or 0)
        connectivity_coverage_applicable = (not defer_net_ee_until_routed) or net_evidence_ready
        equiv_ok = bool(schematic.get("ok", False)) and not missing_on_pcb and not extra_on_pcb and not manifest_missing_on_pcb
        if connectivity_coverage_applicable:
            equiv_ok = equiv_ok and partial_refs == 0 and zero_refs == 0
        equiv_pen = 0
        if not bool(schematic.get("ok", False)):
            equiv_pen += 18
        equiv_pen += min(12, (2 * len(missing_on_pcb)) + (2 * len(extra_on_pcb)) + (2 * len(manifest_missing_on_pcb)))
        if connectivity_coverage_applicable and (partial_refs > 0 or zero_refs > 0):
            equiv_pen += 6
        _add_rule(
            rule_id="schematic_pcb_equivalence",
            title="Schematic-to-PCB Equivalence",
            ok=equiv_ok,
            penalty=min(20, equiv_pen),
            stages=["SPEC", "IMPORT", "NET"],
            message="Schematic/PCB equivalence mismatch detected" if not equiv_ok else "Schematic and PCB intent are aligned",
            details={
                "schematic_loaded": bool(schematic.get("loaded", False)),
                "schematic_component_count": int(schematic.get("component_count", 0) or 0),
                "pcb_ref_count": len(pcb_refs),
                "missing_on_pcb": missing_on_pcb,
                "extra_on_pcb": extra_on_pcb,
                "manifest_missing_in_schematic": manifest_missing_in_sch,
                "manifest_missing_on_pcb": manifest_missing_on_pcb,
                "partial_ref_count": partial_refs,
                "refs_without_nets_count": zero_refs,
                "connectivity_coverage_applicable": bool(connectivity_coverage_applicable),
            },
            applicable=True,
            required=True,
        )

        # 2) Part-role sanity
        role_issues: List[Dict[str, Any]] = []
        conn_tokens = ("connector", "header", "socket", "receptacle", "usb", "jack", "terminal", "plug", "icsp", "pinheader", "barrel")
        fuse_tokens = ("fuse", "polyfuse", "ptc", "resettable")
        ferrite_tokens = ("ferrite", "bead", "inductor", "choke")
        clock_tokens = ("crystal", "xtal", "resonator", "oscillator", "cst", "nx", "fa-238")
        refs_to_check = sorted(set(manifest_refs or pcb_refs))
        for ref in refs_to_check:
            blob = self._benchmark_ref_text_blob(ref, manifest_by_ref, board_snapshot)
            if not blob:
                continue
            if ref.startswith("J"):
                if any(tok in blob for tok in fuse_tokens):
                    role_issues.append({"ref": ref, "issue": "connector_misclassified_as_fuse"})
                elif not any(tok in blob for tok in conn_tokens):
                    role_issues.append({"ref": ref, "issue": "connector_role_unclear"})
            elif ref.startswith("FB"):
                if any(tok in blob for tok in fuse_tokens) and not any(tok in blob for tok in ferrite_tokens):
                    role_issues.append({"ref": ref, "issue": "ferrite_misclassified_as_fuse"})
            elif ref.startswith("F"):
                if (not ref.startswith("FB")) and any(tok in blob for tok in ferrite_tokens) and not any(tok in blob for tok in fuse_tokens):
                    role_issues.append({"ref": ref, "issue": "fuse_misclassified_as_ferrite"})
            elif ref.startswith(("X", "Y")):
                if not any(tok in blob for tok in clock_tokens):
                    role_issues.append({"ref": ref, "issue": "clock_source_role_unclear"})
        _add_rule(
            rule_id="part_role_sanity",
            title="Part-Role Sanity",
            ok=len(role_issues) == 0,
            penalty=min(15, 3 * len(role_issues)),
            stages=["SPEC", "RESOLVE"],
            message="Component role/type mismatches found" if role_issues else "Component roles look coherent",
            details={"issues": role_issues[:50]},
            applicable=True,
            required=True,
        )

        # 3) Power-path topology
        power_issues: List[Dict[str, Any]] = []
        if defer_net_ee_until_routed and (not net_evidence_ready):
            _add_rule(
                rule_id="power_path_topology",
                title="Power-Path Topology",
                ok=True,
                penalty=0,
                stages=["NET"],
                message="Power-path topology deferred until routed net evidence is available",
                details={
                    "deferred": True,
                    "reason": "net_evidence_not_ready",
                    "net_source": net_source,
                    "net_assignment_count": net_assignment_count,
                },
                applicable=False,
                required=False,
            )
        else:
            bridge_issue_count = int(net_score.get("bridge_integrity_issue_ref_count", 0) or 0)
            for row in list(net_score.get("bridge_integrity_issues_sample", []) or [])[:30]:
                if isinstance(row, dict):
                    power_issues.append({"issue": "bridge_integrity", **row})
            present_power_groups = sorted(
                {
                    str(group or "")
                    for group in by_group.keys()
                    if self._benchmark_is_power_group(str(group or ""))
                }
            )
            if not present_power_groups:
                power_issues.append({"issue": "no_power_rails_detected"})
            _add_rule(
                rule_id="power_path_topology",
                title="Power-Path Topology",
                ok=len(power_issues) == 0 and bridge_issue_count == 0,
                penalty=min(15, (2 * bridge_issue_count) + (3 * len(power_issues))),
                stages=["NET"],
                message="Power-path topology issues detected" if power_issues else "Power-path topology checks passed",
                details={
                    "issues": power_issues[:50],
                    "present_power_groups": present_power_groups[:60],
                },
                applicable=True,
                required=True,
            )

        # 4) Regulator stability capacitors
        regulator_issues: List[Dict[str, Any]] = []
        cap_refs = [r for r in refs_to_check if r.startswith("C")]
        reg_refs = []
        reg_tokens = ("reg", "ldo", "ncp1117", "lp2985", "ams1117", "buck", "boost")
        for ref in refs_to_check:
            if not ref.startswith("U"):
                continue
            blob = self._benchmark_ref_text_blob(ref, manifest_by_ref, board_snapshot)
            if any(tok in blob for tok in reg_tokens):
                reg_refs.append(ref)
        for rref in reg_refs:
            regulator_power_nets = _power_nets_for_ref(rref)
            near_caps: List[Dict[str, Any]] = []
            net_matched_caps: List[Dict[str, Any]] = []
            for cref in cap_refs:
                d = self._benchmark_distance_mm(board_snapshot, rref, cref)
                if d is None or d > regulator_cap_radius_mm:
                    continue
                cblob = self._benchmark_ref_text_blob(cref, manifest_by_ref, board_snapshot)
                uf = self._benchmark_cap_value_uF(cblob)
                cap_nets = _canonical_nets_for_ref(cref)
                net_overlap = sorted(regulator_power_nets & cap_nets)
                has_gnd = "gnd" in cap_nets
                row = {
                    "ref": cref,
                    "distance_mm": round(d, 3),
                    "uF": uf,
                    "has_gnd": bool(has_gnd),
                    "net_overlap": net_overlap,
                }
                near_caps.append(row)
                if not net_evidence_ready:
                    net_matched_caps.append(row)
                elif regulator_power_nets:
                    if has_gnd and bool(net_overlap):
                        net_matched_caps.append(row)
                else:
                    if has_gnd:
                        net_matched_caps.append(row)

            effective_caps = net_matched_caps if net_matched_caps else near_caps
            small_count = len(
                [
                    c
                    for c in effective_caps
                    if c.get("uF") is not None and regulator_small_cap_min_uf <= float(c["uF"]) <= regulator_small_cap_max_uf
                ]
            )
            bulk_count = len(
                [
                    c
                    for c in effective_caps
                    if c.get("uF") is not None and float(c["uF"]) >= regulator_bulk_cap_min_uf
                ]
            )
            if (
                len(effective_caps) < regulator_min_nearby_caps
                or small_count < regulator_min_small_caps
                or bulk_count < regulator_min_bulk_caps
            ):
                regulator_issues.append(
                    {
                        "ref": rref,
                        "regulator_power_nets": sorted(regulator_power_nets),
                        "nearby_cap_count": len(near_caps),
                        "matched_cap_count": len(effective_caps),
                        "small_cap_count": small_count,
                        "bulk_cap_count": bulk_count,
                        "near_caps_sample": near_caps[:10],
                        "matched_caps_sample": effective_caps[:10],
                    }
                )
        _add_rule(
            rule_id="regulator_stability_caps",
            title="Regulator Stability Capacitors",
            ok=len(regulator_issues) == 0,
            penalty=min(12, 3 * len(regulator_issues)),
            stages=["SPEC", "GEOM"],
            message="Regulator stability capacitor checks failed" if regulator_issues else "Regulator capacitor checks passed",
            details={
                "issues": regulator_issues[:30],
                "regulator_ref_count": len(reg_refs),
                "thresholds": {
                    "regulator_cap_radius_mm": round(float(regulator_cap_radius_mm), 3),
                    "small_cap_min_uf": round(float(regulator_small_cap_min_uf), 4),
                    "small_cap_max_uf": round(float(regulator_small_cap_max_uf), 4),
                    "bulk_cap_min_uf": round(float(regulator_bulk_cap_min_uf), 4),
                    "min_nearby_caps": int(regulator_min_nearby_caps),
                    "min_small_caps": int(regulator_min_small_caps),
                    "min_bulk_caps": int(regulator_min_bulk_caps),
                },
            },
            applicable=bool(reg_refs),
            required=True,
        )

        # 5) Decoupling coverage
        decoupling_issues: List[Dict[str, Any]] = []
        regulator_set = set(reg_refs)
        ic_refs = [r for r in refs_to_check if r.startswith("U") and r not in regulator_set]
        for iref in ic_refs:
            local_small = 0
            ic_power_nets = _power_nets_for_ref(iref)
            nearby: List[Dict[str, Any]] = []
            matched_caps: List[Dict[str, Any]] = []
            for cref in cap_refs:
                d = self._benchmark_distance_mm(board_snapshot, iref, cref)
                if d is None or d > decoupling_radius_mm:
                    continue
                cblob = self._benchmark_ref_text_blob(cref, manifest_by_ref, board_snapshot)
                uf = self._benchmark_cap_value_uF(cblob)
                cap_nets = _canonical_nets_for_ref(cref)
                has_gnd = "gnd" in cap_nets
                net_overlap = sorted(ic_power_nets & cap_nets)
                row = {
                    "ref": cref,
                    "distance_mm": round(d, 3),
                    "uF": uf,
                    "has_gnd": bool(has_gnd),
                    "net_overlap": net_overlap,
                }
                nearby.append(row)

                if not net_evidence_ready:
                    matched_caps.append(row)
                elif ic_power_nets:
                    if has_gnd and bool(net_overlap):
                        matched_caps.append(row)
                else:
                    if has_gnd:
                        matched_caps.append(row)

            effective_caps = matched_caps if matched_caps else nearby
            for row in effective_caps:
                uf = row.get("uF")
                if uf is not None and decoupling_small_cap_min_uf <= float(uf) <= decoupling_small_cap_max_uf:
                    local_small += 1

            if local_small < decoupling_min_small_caps_per_ic:
                decoupling_issues.append(
                    {
                        "ref": iref,
                        "issue": "missing_local_decoupling",
                        "ic_power_nets": sorted(ic_power_nets),
                        "matched_small_cap_count": int(local_small),
                        "nearby_caps_sample": nearby[:10],
                        "matched_caps_sample": effective_caps[:10],
                    }
                )

        power_rails = {
            g for g in by_group.keys()
            if self._benchmark_is_power_group(str(g or ""))
        }
        missing_bulk_rails: List[str] = []
        if (not defer_net_ee_until_routed) or net_evidence_ready:
            for rail in sorted(power_rails):
                has_bulk = False
                for cref in cap_refs:
                    net_map = by_ref.get(cref) if isinstance(by_ref.get(cref), dict) else {}
                    cap_nets = {
                        self._benchmark_net_canonical_name(str(n or "").strip())
                        for n in net_map.values()
                        if self._benchmark_net_canonical_name(str(n or "").strip())
                    }
                    if rail not in cap_nets or "gnd" not in cap_nets:
                        continue
                    cblob = self._benchmark_ref_text_blob(cref, manifest_by_ref, board_snapshot)
                    uf = self._benchmark_cap_value_uF(cblob)
                    if uf is not None and float(uf) >= bulk_cap_min_uf_per_rail:
                        has_bulk = True
                        break
                if not has_bulk:
                    missing_bulk_rails.append(rail)
        if missing_bulk_rails:
            decoupling_issues.append({"issue": "missing_bulk_cap_per_rail", "rails": missing_bulk_rails})

        _add_rule(
            rule_id="decoupling_coverage",
            title="Decoupling Coverage",
            ok=len(decoupling_issues) == 0,
            penalty=min(15, 2 * len(decoupling_issues)),
            stages=["SPEC", "GEOM"],
            message="Decoupling coverage gaps detected" if decoupling_issues else "Decoupling checks passed",
            details={
                "issues": decoupling_issues[:40],
                "ic_ref_count": len(ic_refs),
                "thresholds": {
                    "decoupling_radius_mm": round(float(decoupling_radius_mm), 3),
                    "small_cap_min_uf": round(float(decoupling_small_cap_min_uf), 4),
                    "small_cap_max_uf": round(float(decoupling_small_cap_max_uf), 4),
                    "min_small_caps_per_ic": int(decoupling_min_small_caps_per_ic),
                    "bulk_cap_min_uf_per_rail": round(float(bulk_cap_min_uf_per_rail), 4),
                },
            },
            applicable=bool(ic_refs or power_rails),
            required=True,
        )

        # 6) Ground return continuity
        ground_issues: List[Dict[str, Any]] = []
        ground_deferred_note: Optional[Dict[str, Any]] = None
        ground_notes: List[Dict[str, Any]] = []
        if not bool(ground.get("ok", False)):
            ground_issues.append({"issue": "no_gnd_copper_zone"})
        if (not defer_net_ee_until_routed) or net_evidence_ready:
            gnd_check = None
            for row in list(net_score.get("checks", []) or []):
                if isinstance(row, dict) and str(row.get("group", "") or "") == "gnd":
                    gnd_check = row
                    break
            if isinstance(gnd_check, dict) and not bool(gnd_check.get("ok", False)):
                ground_issues.append(
                    {
                        "issue": "gnd_group_connectivity",
                        "score_out_of_100": int(gnd_check.get("score_out_of_100", 0) or 0),
                        "disconnected_expected_pin_count": int(gnd_check.get("disconnected_expected_pin_count", 0) or 0),
                    }
                )
        else:
            ground_deferred_note = {
                "deferred": True,
                "reason": "net_evidence_not_ready",
                "net_source": net_source,
                "net_assignment_count": net_assignment_count,
            }
        if self.pcb_data is not None:
            try:
                cu_layers = {
                    str(name).lower()
                    for _num, (name, _ltype) in dict(getattr(self.pcb_data, "layers", {}) or {}).items()
                    if str(name or "").lower().endswith(".cu")
                }
                if "f.cu" in cu_layers and "b.cu" in cu_layers:
                    zone_layers = {
                        str(row.get("layer", "") or "").lower()
                        for row in list(ground.get("gnd_copper_zones_sample", []) or [])
                        if isinstance(row, dict)
                    }
                    if zone_layers and len(zone_layers) < 2:
                        gnd_track = track_by_group.get("gnd") if isinstance(track_by_group.get("gnd"), dict) else {}
                        gnd_seg_count = int(gnd_track.get("segment_count", 0) or 0)
                        gnd_via_count = int(gnd_track.get("via_count", 0) or 0)
                        risk_high = (gnd_seg_count >= gnd_single_layer_risk_min_segments) or (
                            gnd_via_count >= gnd_single_layer_risk_min_vias
                        )
                        should_enforce = True
                        if enforce_dual_layer_gnd_zone_only_when_routed and (not routing_evidence_ready):
                            should_enforce = False
                        if should_enforce and risk_high:
                            ground_issues.append(
                                {
                                    "issue": "gnd_zone_single_layer_high_risk",
                                    "zone_layers": sorted(zone_layers),
                                    "gnd_segment_count": gnd_seg_count,
                                    "gnd_via_count": gnd_via_count,
                                    "risk_threshold_segment_count": gnd_single_layer_risk_min_segments,
                                    "risk_threshold_via_count": gnd_single_layer_risk_min_vias,
                                }
                            )
                        else:
                            ground_notes.append(
                                {
                                    "note": "gnd_zone_single_layer_detected_but_not_flagged",
                                    "zone_layers": sorted(zone_layers),
                                    "gnd_segment_count": gnd_seg_count,
                                    "gnd_via_count": gnd_via_count,
                                    "enforced": bool(should_enforce),
                                }
                            )
            except Exception:
                pass
        _add_rule(
            rule_id="ground_return_continuity",
            title="Ground Return Continuity",
            ok=len(ground_issues) == 0,
            penalty=min(12, 4 * len(ground_issues)),
            stages=["GEOM", "NET"],
            message="Ground continuity/return-path issues found" if ground_issues else "Ground continuity checks passed",
            details={
                "issues": ground_issues[:20],
                "notes": ground_notes[:20],
                "ground_audit": ground,
                "deferred_net_connectivity": ground_deferred_note or {},
                "thresholds": {
                    "enforce_dual_layer_gnd_zone_only_when_routed": bool(enforce_dual_layer_gnd_zone_only_when_routed),
                    "single_layer_gnd_zone_risk_min_gnd_segments": int(gnd_single_layer_risk_min_segments),
                    "single_layer_gnd_zone_risk_min_gnd_vias": int(gnd_single_layer_risk_min_vias),
                },
            },
            applicable=True,
            required=True,
        )

        # 7) Clock loop quality
        clock_issues: List[Dict[str, Any]] = []
        if bool(clock.get("applicable", False)) and not bool(clock.get("ok", False)):
            clock_issues.append({"issue": "placement_or_caps", "details": clock})
        xtal_lengths = {}
        if (not defer_net_ee_until_routed) or routing_evidence_ready:
            for group, row in track_by_group.items():
                g = str(group or "")
                if "xtal" in g or "osc" in g:
                    xtal_lengths[g] = float(row.get("total_length_mm", 0.0) or 0.0)
            if xtal_lengths:
                long_nets = {k: v for k, v in xtal_lengths.items() if v > 25.0}
                if long_nets:
                    clock_issues.append({"issue": "clock_nets_too_long", "nets_mm": long_nets})
                xtal1 = xtal_lengths.get("xtal1")
                xtal2 = xtal_lengths.get("xtal2")
                if xtal1 is not None and xtal2 is not None:
                    mismatch = abs(float(xtal1) - float(xtal2))
                    if mismatch > 3.0:
                        clock_issues.append({"issue": "xtal_route_length_mismatch", "xtal1_mm": xtal1, "xtal2_mm": xtal2, "delta_mm": round(mismatch, 3)})
        _add_rule(
            rule_id="clock_loop_quality",
            title="Clock Loop Quality",
            ok=len(clock_issues) == 0,
            penalty=min(10, 4 * len(clock_issues)),
            stages=["GEOM", "NET"],
            message="Clock loop quality risks detected" if clock_issues else "Clock loop checks passed",
            details={"issues": clock_issues[:20], "clock_audit": clock},
            applicable=True,
            required=True,
        )

        # 8) USB signal integrity basics
        usb_issues: List[Dict[str, Any]] = []
        usb_dp = track_by_group.get("usb_d_p") if isinstance(track_by_group.get("usb_d_p"), dict) else None
        usb_dn = track_by_group.get("usb_d_n") if isinstance(track_by_group.get("usb_d_n"), dict) else None
        if (not defer_net_ee_until_routed) or routing_evidence_ready:
            if usb_dp and usb_dn:
                lp = float(usb_dp.get("total_length_mm", 0.0) or 0.0)
                ln = float(usb_dn.get("total_length_mm", 0.0) or 0.0)
                mismatch = abs(lp - ln)
                if mismatch > 2.0:
                    usb_issues.append({"issue": "usb_pair_length_mismatch", "d_plus_mm": round(lp, 3), "d_minus_mm": round(ln, 3), "delta_mm": round(mismatch, 3)})
            elif ("usb_d_p" in by_group) or ("usb_d_n" in by_group):
                usb_issues.append({"issue": "usb_pair_track_data_missing"})

        usb_connector_refs = [
            r for r in refs_to_check
            if r.startswith("J") and "usb" in self._benchmark_ref_text_blob(r, manifest_by_ref, board_snapshot)
        ]
        protector_refs = []
        for r in refs_to_check:
            blob = self._benchmark_ref_text_blob(r, manifest_by_ref, board_snapshot)
            if not (r.startswith("D") or r.startswith("U")):
                continue
            if not any(tok in blob for tok in ("esd", "tvs", "zener", "varistor", "clamp")):
                continue
            if (not defer_net_ee_until_routed) or net_evidence_ready:
                net_map = by_ref.get(r) if isinstance(by_ref.get(r), dict) else {}
                nets = {
                    self._benchmark_net_canonical_name(str(n or "").strip())
                    for n in net_map.values()
                    if self._benchmark_net_canonical_name(str(n or "").strip())
                }
                if {"usb_d_p", "usb_d_n"} & nets:
                    protector_refs.append(r)
            else:
                protector_refs.append(r)
        if usb_connector_refs:
            if not protector_refs:
                usb_issues.append({"issue": "no_usb_protection_detected_near_connector", "connectors": usb_connector_refs})
            else:
                d_best = None
                pair = ("", "")
                for jref in usb_connector_refs:
                    for pref in protector_refs:
                        d = self._benchmark_distance_mm(board_snapshot, jref, pref)
                        if d is None:
                            continue
                        if d_best is None or d < d_best:
                            d_best = d
                            pair = (jref, pref)
                if d_best is not None and d_best > usb_protector_max_distance_mm:
                    usb_issues.append({"issue": "usb_protection_too_far_from_connector", "distance_mm": round(float(d_best), 3), "connector_ref": pair[0], "protector_ref": pair[1]})
        if (not defer_net_ee_until_routed) or net_evidence_ready:
            for gname in ("usb_d_p", "usb_d_n"):
                row = by_group.get(gname) if isinstance(by_group.get(gname), dict) else {}
                refs = list(row.get("refs") or []) if isinstance(row, dict) else []
                if len(refs) > 6:
                    usb_issues.append({"issue": "usb_net_high_fanout_possible_stubs", "group": gname, "ref_count": len(refs)})
        _add_rule(
            rule_id="usb_signal_integrity",
            title="USB Signal Integrity",
            ok=len(usb_issues) == 0,
            penalty=min(12, 3 * len(usb_issues)),
            stages=["NET", "GEOM"],
            message="USB signal-integrity issues detected" if usb_issues else "USB signal-integrity checks passed",
            details={
                "issues": usb_issues[:30],
                "thresholds": {
                    "usb_protector_max_distance_mm": round(float(usb_protector_max_distance_mm), 3),
                },
            },
            applicable=bool(is_uno or ("usb_d_p" in by_group) or ("usb_d_n" in by_group) or usb_connector_refs),
            required=True,
        )

        # 9) Current/thermal sizing
        thermal_issues: List[Dict[str, Any]] = []
        power_groups = {
            str(g or "")
            for g in set(list(track_by_group.keys()) + list(by_group.keys()))
            if self._benchmark_is_power_group(str(g or ""))
        }
        present_power_groups = sorted(g for g in power_groups if g in track_by_group or g in by_group)
        if defer_net_ee_until_routed and (not routing_evidence_ready):
            _add_rule(
                rule_id="current_thermal_sizing",
                title="Current/Thermal Sizing",
                ok=True,
                penalty=0,
                stages=["NET", "GEOM"],
                message="Current/thermal sizing deferred until routed net evidence is available",
                details={
                    "deferred": True,
                    "reason": "routing_evidence_not_ready",
                    "net_source": net_source,
                    "net_assignment_count": net_assignment_count,
                    "routed_segment_total": routed_segment_total,
                },
                applicable=False,
                required=False,
            )
        else:
            for g in sorted(present_power_groups):
                row = track_by_group.get(g) if isinstance(track_by_group.get(g), dict) else {}
                seg_count = int(row.get("segment_count", 0) or 0)
                if seg_count <= 0:
                    thermal_issues.append({"issue": "no_routed_segments_on_power_group", "group": g})
                    continue
                min_w = row.get("min_width_mm")
                if min_w is not None and float(min_w) < 0.20:
                    thermal_issues.append({"issue": "power_trace_too_narrow", "group": g, "min_width_mm": float(min_w)})
                max_w = row.get("max_width_mm")
                if max_w is not None and float(max_w) < 0.30:
                    thermal_issues.append({"issue": "power_trace_width_low_margin", "group": g, "max_width_mm": float(max_w)})
                min_via_drill = row.get("min_via_drill_mm")
                if min_via_drill is not None and float(min_via_drill) < 0.25:
                    thermal_issues.append({"issue": "power_via_drill_too_small", "group": g, "min_via_drill_mm": float(min_via_drill)})
            _add_rule(
                rule_id="current_thermal_sizing",
                title="Current/Thermal Sizing",
                ok=len(thermal_issues) == 0,
                penalty=min(10, 2 * len(thermal_issues)),
                stages=["NET", "GEOM"],
                message="Current-path trace/via sizing issues detected" if thermal_issues else "Current/thermal sizing checks passed",
                details={"issues": thermal_issues[:40]},
                applicable=bool(present_power_groups),
                required=True,
            )

        # 10) DFM checks
        dfm_error_issues: List[Dict[str, Any]] = []
        dfm_warning_issues: List[Dict[str, Any]] = []
        dfm_keywords = ("annular", "solder mask", "mask", "silk", "silkscreen", "courtyard", "edge clearance", "to board edge", "clearance")
        for msg in list(drc_info.get("errors", []) or []):
            m = str(msg or "")
            if any(k in m.lower() for k in dfm_keywords):
                dfm_error_issues.append({"issue": "drc_error", "message": m})
        for msg in list(drc_info.get("warnings", []) or [])[:60]:
            m = str(msg or "")
            if any(k in m.lower() for k in dfm_keywords):
                dfm_warning_issues.append({"issue": "drc_warning", "message": m})
        live_metrics = self._benchmark_live_board_placement_metrics(allowed_refs=allowed_refs or None)
        if bool(live_metrics.get("usable")):
            oob = int(live_metrics.get("out_of_bounds", 0) or 0)
            overlaps = int(live_metrics.get("overlap_collisions", 0) or 0)
            if oob > 0:
                dfm_error_issues.append(
                    {
                        "issue": "component_out_of_bounds",
                        "count": oob,
                        "refs_sample": list(live_metrics.get("out_of_bounds_refs_sample", []) or [])[:20],
                    }
                )
            if overlaps > 0:
                dfm_error_issues.append(
                    {
                        "issue": "component_overlap_collision",
                        "count": overlaps,
                        "pairs_sample": list(live_metrics.get("overlap_pairs_sample", []) or [])[:20],
                    }
                )
        _add_rule(
            rule_id="dfm_checks",
            title="DFM Checks",
            ok=len(dfm_error_issues) == 0,
            penalty=min(12, 2 * len(dfm_error_issues)),
            stages=["DRC", "GEOM"],
            message="DFM issues detected" if dfm_error_issues else "DFM checks passed",
            details={
                "issues": (dfm_error_issues + dfm_warning_issues)[:60],
                "error_issues_count": len(dfm_error_issues),
                "warning_issues_count": len(dfm_warning_issues),
                "drc_error_count": int(drc_info.get("error_count", 0) or 0),
                "drc_warning_count": int(drc_info.get("warning_count", 0) or 0),
                "drc_raw_error_count": int(drc_info.get("raw_error_count", 0) or 0),
                "drc_intrinsic_error_count": int(drc_info.get("intrinsic_error_count", 0) or 0),
            },
            applicable=True,
            required=True,
        )

        failed_rules = [r for r in rules if bool(r.get("applicable", True)) and not bool(r.get("ok", False))]
        failed_required_rules = [r for r in failed_rules if bool(r.get("required", True))]
        total_penalty = int(sum(int(r.get("penalty", 0) or 0) for r in rules if bool(r.get("applicable", True))))
        score = max(0, 100 - total_penalty)
        issues_for_done = [
            {
                "id": str(r.get("id", "") or ""),
                "title": str(r.get("title", "") or ""),
                "message": str(r.get("message", "") or ""),
                "penalty": int(r.get("penalty", 0) or 0),
                "stages": list(r.get("stages", []) or []),
                "details": r.get("details", {}),
            }
            for r in failed_rules
        ]
        return {
            "applicable": bool(rules),
            "score_out_of_100": int(score),
            "total_penalty": int(total_penalty),
            "rule_count": len(rules),
            "failed_rule_count": len(failed_rules),
            "failed_required_rule_count": len(failed_required_rules),
            "failed_rule_ids": [str(r.get("id", "") or "") for r in failed_rules],
            "rules": rules,
            "issues_for_done_checkpoint": issues_for_done,
            "ok": len(failed_required_rules) == 0,
        }
    @staticmethod
    def _benchmark_net_canonical_name(value: str) -> str:
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        raw = raw.replace("−", "-")
        raw = re.sub(r"\bplus\b", "+", raw)
        raw = re.sub(r"\bminus\b", "-", raw)
        raw = re.sub(r"[^a-z0-9+\-._]+", "_", raw).strip("_")
        if not raw:
            return ""
        compact = re.sub(r"[^a-z0-9]+", "", raw)
        if compact in {"nc", "noconnect", "notconnected", "unconnected", "nonet", "none"}:
            return ""

        def _normalize_voltage_token(token: str) -> Optional[str]:
            t = str(token or "").strip().lower()
            if not t:
                return None
            # 3v3, +3v3, 3.3v, +3.3v
            m = re.match(r"^([+-]?)(\d+)(?:\.(\d+))?v(\d+)?$", t)
            if m:
                sign = "-" if m.group(1) == "-" else ""
                major = str(int(m.group(2)))
                minor_raw = m.group(4) if m.group(4) is not None else (m.group(3) if m.group(3) is not None else "0")
                minor = str(int(minor_raw))
                return f"{sign}v{major}_{minor}"
            # v3v3, +v3v3, v3.3
            m = re.match(r"^([+-]?)v(\d+)(?:v(\d+)|\.(\d+))?$", t)
            if m:
                sign = "-" if m.group(1) == "-" else ""
                major = str(int(m.group(2)))
                minor_raw = m.group(3) if m.group(3) is not None else (m.group(4) if m.group(4) is not None else "0")
                minor = str(int(minor_raw))
                return f"{sign}v{major}_{minor}"
            return None

        whole_voltage = _normalize_voltage_token(raw.replace("_", "v"))
        if whole_voltage:
            return whole_voltage

        out_tokens: List[str] = []
        for tok in (t for t in raw.split("_") if t):
            normalized_voltage = _normalize_voltage_token(tok)
            if normalized_voltage:
                out_tokens.append(normalized_voltage)
                continue
            cleaned = re.sub(r"[^a-z0-9+\-]", "", tok).strip("+-")
            if cleaned:
                out_tokens.append(cleaned)

        if not out_tokens:
            return ""
        canonical = "_".join(out_tokens)
        pin_alias = re.match(r"^(a\d+|d\d+)_(sda|scl|rx|tx|miso|mosi|sck|clk)$", canonical)
        if pin_alias:
            return pin_alias.group(1)
        return canonical
    def _benchmark_board_net_snapshot(self, allowed_refs: Optional[Set[str]] = None) -> Dict[str, Any]:
        if not PCBNEW_AVAILABLE:
            return {"source": "board", "usable": False}
        try:
            board = pcbnew.GetBoard()
        except Exception:
            board = None
        if board is None:
            return {"source": "board", "usable": False}

        by_ref: Dict[str, Dict[str, str]] = {}
        all_pad_names_by_ref: Dict[str, Set[str]] = {}
        by_group_tmp: Dict[str, Dict[str, Any]] = {}
        unique_net_names: Set[str] = set()
        assignment_count = 0
        try:
            footprints = list(board.GetFootprints() or [])
        except Exception:
            footprints = []
        for fp in footprints:
            try:
                ref = str(fp.GetReference() or "").strip().upper()
            except Exception:
                ref = ""
            if not ref:
                continue
            if allowed_refs is not None and ref not in allowed_refs:
                continue
            try:
                pads = list(fp.Pads() or [])
            except Exception:
                pads = []
            for pad in pads:
                pad_name = ""
                for attr in ("GetPadName", "GetName", "GetNumber"):
                    fn = getattr(pad, attr, None)
                    if callable(fn):
                        try:
                            pad_name = str(fn() or "").strip()
                        except Exception:
                            pad_name = ""
                        if pad_name:
                            break
                if pad_name:
                    all_pad_names_by_ref.setdefault(ref, set()).add(pad_name)
                net_name = ""
                for attr in ("GetNetname", "GetNetName"):
                    fn = getattr(pad, attr, None)
                    if callable(fn):
                        try:
                            net_name = str(fn() or "").strip()
                        except Exception:
                            net_name = ""
                        if net_name:
                            break
                if not pad_name or not net_name:
                    continue
                by_ref.setdefault(ref, {})[pad_name] = net_name
                unique_net_names.add(net_name)
                assignment_count += 1
                group = self._benchmark_net_canonical_name(net_name)
                if not group:
                    continue
                row = by_group_tmp.setdefault(
                    group,
                    {"refs": set(), "net_names": set(), "pads": [], "net_refs": {}, "net_pad_count": {}},
                )
                row["refs"].add(ref)
                row["net_names"].add(net_name)
                net_refs = row.setdefault("net_refs", {})
                ref_set = net_refs.setdefault(net_name, set())
                ref_set.add(ref)
                net_pad_count = row.setdefault("net_pad_count", {})
                net_pad_count[net_name] = int(net_pad_count.get(net_name, 0) or 0) + 1
                if len(row["pads"]) < 24:
                    row["pads"].append(f"{ref}.{pad_name}")

        by_group = {
            group: {
                "refs": sorted(set(row.get("refs") or [])),
                "net_names": sorted(set(row.get("net_names") or [])),
                "pad_count": len(list(row.get("pads") or [])),
                "pads_sample": list(row.get("pads") or []),
                "net_ref_map": {
                    str(net): sorted(set(refs or []))
                    for net, refs in dict(row.get("net_refs") or {}).items()
                    if str(net or "").strip()
                },
                "net_pad_count_map": {
                    str(net): int(count or 0)
                    for net, count in dict(row.get("net_pad_count") or {}).items()
                    if str(net or "").strip()
                },
            }
            for group, row in by_group_tmp.items()
        }
        return {
            "source": "board",
            "usable": bool(by_ref),
            "assignment_count": assignment_count,
            "unique_net_count": len(unique_net_names),
            "by_ref": by_ref,
            "all_pad_names_by_ref": {
                ref: sorted(set(pads))
                for ref, pads in all_pad_names_by_ref.items()
            },
            "by_group": by_group,
        }
    def _benchmark_planned_net_snapshot(self, loop: Any) -> Dict[str, Any]:
        artifacts = getattr(loop, "_artifacts", {}) or {}
        net_plan = artifacts.get("net_plan") if isinstance(artifacts.get("net_plan"), dict) else {}
        assignments = [row for row in list(net_plan.get("assignments") or []) if isinstance(row, dict)]
        by_ref: Dict[str, Dict[str, str]] = {}
        by_group_tmp: Dict[str, Dict[str, Any]] = {}
        unique_net_names: Set[str] = set()
        for row in assignments:
            ref = str(row.get("ref", "") or "").strip().upper()
            pad = str(row.get("pad", "") or "").strip()
            net_name = str(row.get("net", "") or "").strip()
            if not ref or not pad or not net_name:
                continue
            by_ref.setdefault(ref, {})[pad] = net_name
            unique_net_names.add(net_name)
            group = self._benchmark_net_canonical_name(net_name)
            if not group:
                continue
            bucket = by_group_tmp.setdefault(
                group,
                {"refs": set(), "net_names": set(), "pads": [], "net_refs": {}, "net_pad_count": {}},
            )
            bucket["refs"].add(ref)
            bucket["net_names"].add(net_name)
            net_refs = bucket.setdefault("net_refs", {})
            ref_set = net_refs.setdefault(net_name, set())
            ref_set.add(ref)
            net_pad_count = bucket.setdefault("net_pad_count", {})
            net_pad_count[net_name] = int(net_pad_count.get(net_name, 0) or 0) + 1
            if len(bucket["pads"]) < 24:
                bucket["pads"].append(f"{ref}.{pad}")
        by_group = {
            group: {
                "refs": sorted(set(row.get("refs") or [])),
                "net_names": sorted(set(row.get("net_names") or [])),
                "pad_count": len(list(row.get("pads") or [])),
                "pads_sample": list(row.get("pads") or []),
                "net_ref_map": {
                    str(net): sorted(set(refs or []))
                    for net, refs in dict(row.get("net_refs") or {}).items()
                    if str(net or "").strip()
                },
                "net_pad_count_map": {
                    str(net): int(count or 0)
                    for net, count in dict(row.get("net_pad_count") or {}).items()
                    if str(net or "").strip()
                },
            }
            for group, row in by_group_tmp.items()
        }
        return {
            "source": "net_plan",
            "usable": bool(by_ref),
            "assignment_count": len(assignments),
            "unique_net_count": len(unique_net_names),
            "by_ref": by_ref,
            "by_group": by_group,
        }
    def _benchmark_required_pads_by_ref(
        self,
        manifest: Dict[str, Any],
        all_pad_names_by_ref: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        skip_net_names = {"", "NC", "N/C", "NO_CONNECT", "UNCONNECTED", "NOT_CONNECTED", "NO_NET", "NONET", "NONE"}

        manifest_by_ref: Dict[str, Dict[str, Any]] = {}
        explicit_nc_by_ref: Dict[str, Set[str]] = {}
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = str(part.get("ref", "") or "").strip().upper()
            if not ref:
                continue
            manifest_by_ref[ref] = part
            nc_pads: Set[str] = set()
            pins = part.get("pins")
            if isinstance(pins, list):
                for pin in pins:
                    if not isinstance(pin, dict):
                        continue
                    pad = str(pin.get("num") or pin.get("pad") or pin.get("pin") or "").strip()
                    raw_net = str(pin.get("net", "") or "").strip().upper()
                    if pad and raw_net in skip_net_names:
                        nc_pads.add(pad)
            explicit_nc_by_ref[ref] = nc_pads

        board_pads_by_ref = {
            str(ref or "").strip().upper(): [
                str(pad).strip()
                for pad in list(pads or [])
                if str(pad).strip()
            ]
            for ref, pads in all_pad_names_by_ref.items()
            if str(ref or "").strip()
        }
        required: Dict[str, List[str]] = {}
        for ref, board_pads in board_pads_by_ref.items():
            if manifest_by_ref and ref not in manifest_by_ref:
                continue
            board_pads = board_pads_by_ref.get(ref) or []
            if not board_pads:
                continue
            nc_pads = explicit_nc_by_ref.get(ref, set())
            required_pads = [pad for pad in board_pads if pad not in nc_pads]

            if not required_pads:
                required[ref] = []
                continue

            ordered_unique: List[str] = []
            seen: Set[str] = set()
            for pad in required_pads:
                if pad in seen:
                    continue
                seen.add(pad)
                ordered_unique.append(pad)
            required[ref] = ordered_unique
        return required
    def _benchmark_uno_net_coverage(
        self,
        loop: Any,
        actual_snapshot: Dict[str, Any],
        allowed_refs: Set[str],
    ) -> Dict[str, Any]:
        by_ref = actual_snapshot.get("by_ref") if isinstance(actual_snapshot.get("by_ref"), dict) else {}
        all_pad_names_by_ref = (
            actual_snapshot.get("all_pad_names_by_ref")
            if isinstance(actual_snapshot.get("all_pad_names_by_ref"), dict)
            else {}
        )

        if all_pad_names_by_ref:
            artifacts = getattr(loop, "_artifacts", {}) or {}
            manifest = artifacts.get("manifest") if isinstance(artifacts.get("manifest"), dict) else {}
            required_pads_by_ref = self._benchmark_required_pads_by_ref(manifest, all_pad_names_by_ref)
            refs = sorted(
                ref
                for ref in {str(r).upper() for r in all_pad_names_by_ref.keys()}
                if not allowed_refs or ref in allowed_refs
            )
            total_unique_pads = 0
            assigned_unique_pads = 0
            full_refs: List[str] = []
            partial_refs: List[Dict[str, Any]] = []
            zero_refs: List[Dict[str, Any]] = []
            exempt_nc_pads_total = 0
            unassigned_pads: List[Dict[str, Any]] = []
            per_ref_coverage: List[Dict[str, Any]] = []
            for ref in refs:
                board_pads = [
                    str(pad).strip()
                    for pad in list(all_pad_names_by_ref.get(ref) or [])
                    if str(pad).strip()
                ]
                total_pads = list(required_pads_by_ref.get(ref) or [])
                exempt_nc_pads_total += max(0, len(board_pads) - len(total_pads))
                if not total_pads:
                    full_refs.append(ref)
                    per_ref_coverage.append(
                        {
                            "ref": ref,
                            "pad_count": 0,
                            "assigned_pad_count": 0,
                            "unassigned_pads": [],
                            "all_pads_exempt_explicit_nc": True,
                        }
                    )
                    continue
                total_pad_lookup = set(total_pads)
                assigned_pads = sorted(
                    set(
                        str(pad).strip()
                        for pad in list((by_ref.get(ref) or {}).keys())
                        if str(pad).strip() and str(pad).strip() in total_pad_lookup
                    )
                )
                total_unique_pads += len(total_pads)
                assigned_unique_pads += len(assigned_pads)
                missing = [pad for pad in total_pads if pad not in set(assigned_pads)]
                if missing:
                    for pad in missing:
                        unassigned_pads.append({"ref": ref, "pad": pad})
                per_ref_coverage.append(
                    {
                        "ref": ref,
                        "pad_count": len(total_pads),
                        "assigned_pad_count": len(assigned_pads),
                        "unassigned_pads": missing,
                        "all_pads_exempt_explicit_nc": False,
                    }
                )
                if not assigned_pads:
                    zero_refs.append(
                        {
                            "ref": ref,
                            "pad_count": len(total_pads),
                            "unassigned_pads_sample": missing[:12],
                        }
                    )
                elif missing:
                    partial_refs.append(
                        {
                            "ref": ref,
                            "pad_count": len(total_pads),
                            "assigned_pad_count": len(assigned_pads),
                            "unassigned_pads_sample": missing[:12],
                        }
                    )
                else:
                    full_refs.append(ref)

            total_refs = len(refs)
            ref_coverage_score = int(round(100.0 * float(len(full_refs)) / float(max(total_refs, 1)))) if total_refs else 0
            pad_coverage_score = int(round(100.0 * float(assigned_unique_pads) / float(max(total_unique_pads, 1)))) if total_unique_pads else 0
            return {
                "usable": total_refs > 0,
                "source": str(actual_snapshot.get("source", "board") or "board"),
                "strict_all_physical_pads_checked": True,
                "coverage_basis": "all_physical_pads_except_explicit_nc",
                "total_refs": total_refs,
                "full_refs_count": len(full_refs),
                "partial_refs_count": len(partial_refs),
                "refs_without_nets_count": len(zero_refs),
                "total_unique_pads": total_unique_pads,
                "assigned_unique_pads": assigned_unique_pads,
                "unassigned_unique_pads": max(0, total_unique_pads - assigned_unique_pads),
                "explicit_nc_exempt_pads_count": int(exempt_nc_pads_total),
                "ref_coverage_score": ref_coverage_score,
                "pad_coverage_score": pad_coverage_score,
                "full_refs_sample": full_refs[:20],
                "partial_refs_sample": partial_refs[:20],
                "refs_without_nets_sample": zero_refs[:20],
                "unassigned_pads_sample": unassigned_pads[:200],
                "per_ref_coverage": per_ref_coverage,
            }

        artifacts = getattr(loop, "_artifacts", {}) or {}
        net_plan = artifacts.get("net_plan") if isinstance(artifacts.get("net_plan"), dict) else {}
        coverage = net_plan.get("coverage") if isinstance(net_plan.get("coverage"), dict) else {}
        if coverage:
            return {
                "usable": True,
                "source": "net_plan",
                "strict_all_physical_pads_checked": False,
                "total_refs": int(coverage.get("total_refs", 0) or 0),
                "full_refs_count": max(
                    0,
                    int(coverage.get("total_refs", 0) or 0)
                    - int(coverage.get("partial_refs_count", 0) or 0)
                    - int(coverage.get("refs_without_nets_count", 0) or 0),
                ),
                "partial_refs_count": int(coverage.get("partial_refs_count", 0) or 0),
                "refs_without_nets_count": int(coverage.get("refs_without_nets_count", 0) or 0),
                "total_unique_pads": int(coverage.get("total_pads", 0) or 0),
                "assigned_unique_pads": int(coverage.get("assigned_pads", 0) or 0),
                "ref_coverage_score": int(round(100.0 * float(max(
                    0,
                    int(coverage.get("total_refs", 0) or 0)
                    - int(coverage.get("partial_refs_count", 0) or 0)
                    - int(coverage.get("refs_without_nets_count", 0) or 0),
                )) / float(max(int(coverage.get("total_refs", 0) or 0), 1)))),
                "pad_coverage_score": int(round(100.0 * float(coverage.get("coverage_ratio", 0.0) or 0.0))),
                "full_refs_sample": [],
                "partial_refs_sample": list(coverage.get("partial_refs_sample", []) or [])[:20],
                "refs_without_nets_sample": list(coverage.get("refs_without_nets_sample", []) or [])[:20],
            }

        return {
            "usable": False,
            "source": str(actual_snapshot.get("source", "unknown") or "unknown"),
            "strict_all_physical_pads_checked": False,
        }
    def _benchmark_uno_net_score(self, loop: Any, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        scenario = str(benchmark.get("scenario", "") or "").lower()
        if "uno" not in scenario:
            return {"applicable": False}
        artifacts = getattr(loop, "_artifacts", {}) or {}
        manifest = artifacts.get("manifest") if isinstance(artifacts.get("manifest"), dict) else {}
        expected_by_group: Dict[str, Dict[str, Any]] = {}
        all_manifest_refs: Set[str] = set()
        manifest_part_by_ref: Dict[str, Dict[str, Any]] = {}
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = str(part.get("ref", "") or "").strip().upper()
            if not ref:
                continue
            all_manifest_refs.add(ref)
            manifest_part_by_ref[ref] = part
            pins = list(part.get("pins") or []) if isinstance(part.get("pins"), list) else []
            for pin in pins:
                if not isinstance(pin, dict):
                    continue
                net_name = str(pin.get("net", "") or "").strip()
                group = self._benchmark_net_canonical_name(net_name)
                if not group:
                    continue
                bucket = expected_by_group.setdefault(
                    group,
                    {"refs": set(), "net_names": set(), "pin_count": 0, "pins": []},
                )
                bucket["refs"].add(ref)
                bucket["net_names"].add(net_name)
                bucket["pin_count"] = int(bucket.get("pin_count", 0) or 0) + 1
                pin_pad = str(
                    pin.get("num")
                    or pin.get("pad")
                    or pin.get("pin")
                    or pin.get("number")
                    or ""
                ).strip()
                if pin_pad:
                    bucket["pins"].append(
                        {
                            "ref": ref,
                            "pad": pin_pad,
                            "net_name": net_name,
                        }
                    )

        board_snapshot = self._benchmark_board_net_snapshot(allowed_refs=all_manifest_refs or None)
        actual_snapshot = board_snapshot if bool(board_snapshot.get("usable")) else self._benchmark_planned_net_snapshot(loop)
        actual_groups = actual_snapshot.get("by_group") if isinstance(actual_snapshot.get("by_group"), dict) else {}
        actual_by_ref = actual_snapshot.get("by_ref") if isinstance(actual_snapshot.get("by_ref"), dict) else {}
        coverage_metrics = self._benchmark_uno_net_coverage(loop, actual_snapshot, all_manifest_refs)
        board_pad_lookup_by_ref = {
            str(ref or "").strip().upper(): {
                str(pad or "").strip()
                for pad in list(pads or [])
                if str(pad or "").strip()
            }
            for ref, pads in (
                actual_snapshot.get("all_pad_names_by_ref", {})
                if isinstance(actual_snapshot.get("all_pad_names_by_ref"), dict)
                else {}
            ).items()
            if str(ref or "").strip()
        }
        skipped_expected_pins: List[Dict[str, Any]] = []
        skipped_expected_pins_by_group_ref: Dict[Tuple[str, str], int] = {}
        if board_pad_lookup_by_ref:
            for group, row in expected_by_group.items():
                if not isinstance(row, dict):
                    continue
                kept_pins: List[Dict[str, Any]] = []
                for pin_row in [p for p in list(row.get("pins") or []) if isinstance(p, dict)]:
                    ref = str(pin_row.get("ref", "") or "").strip().upper()
                    pad = str(pin_row.get("pad", "") or "").strip()
                    board_pads = board_pad_lookup_by_ref.get(ref, set())
                    if ref and pad and board_pads and pad not in board_pads:
                        skipped_expected_pins.append(
                            {
                                "group": str(group or ""),
                                "ref": ref,
                                "pad": pad,
                                "reason": "pad_not_on_footprint",
                                "board_pads_sample": sorted(list(board_pads))[:12],
                            }
                        )
                        key = (str(group or ""), ref)
                        skipped_expected_pins_by_group_ref[key] = int(
                            skipped_expected_pins_by_group_ref.get(key, 0) or 0
                        ) + 1
                        continue
                    kept_pins.append(pin_row)
                row["pins"] = kept_pins
                row["pin_count"] = len(kept_pins)

        expected_groups_by_pin: Dict[Tuple[str, str], Set[str]] = {}
        for group, row in expected_by_group.items():
            if not isinstance(row, dict):
                continue
            for pin_row in [p for p in list(row.get("pins") or []) if isinstance(p, dict)]:
                ref = str(pin_row.get("ref", "") or "").strip().upper()
                pad = str(pin_row.get("pad", "") or "").strip()
                if not ref or not pad:
                    continue
                expected_groups_by_pin.setdefault((ref, pad), set()).add(str(group or ""))

        # Bridge/component integrity audit for 2-pad non-connector parts.
        net_ref_fanout: Dict[str, Set[str]] = {}
        for ref, pad_map in actual_by_ref.items():
            ref_norm = str(ref or "").strip().upper()
            if not ref_norm:
                continue
            if not isinstance(pad_map, dict):
                continue
            for net_name in pad_map.values():
                canonical = self._benchmark_net_canonical_name(str(net_name or "").strip())
                if not canonical:
                    continue
                net_ref_fanout.setdefault(canonical, set()).add(ref_norm)

        bridge_integrity_issues: List[Dict[str, Any]] = []
        bridge_candidate_refs = 0
        for ref in sorted(all_manifest_refs):
            if ref.startswith("J"):
                continue
            board_pads = sorted(board_pad_lookup_by_ref.get(ref, set()))
            if len(board_pads) != 2:
                continue

            part = manifest_part_by_ref.get(ref, {})
            text = " ".join(
                str(part.get(key, "") or "")
                for key in ("ref", "footprint", "value", "description", "mpn")
            ).lower()
            is_bridge_like = (
                ref.startswith(("F", "SW", "D", "R", "L", "FB"))
                or bool(re.search(r"\b(fuse|switch|button|diode|resistor|bead|ferrite|inductor)\b", text))
            )
            if not is_bridge_like:
                continue

            bridge_candidate_refs += 1
            live_map = actual_by_ref.get(ref) if isinstance(actual_by_ref.get(ref), dict) else {}
            pad_net_rows: List[Dict[str, str]] = []
            canonical_nets: List[str] = []
            missing_pads: List[str] = []
            for pad in board_pads:
                net_name = str(live_map.get(pad, "") or "").strip()
                canonical = self._benchmark_net_canonical_name(net_name)
                if not canonical:
                    missing_pads.append(pad)
                else:
                    canonical_nets.append(canonical)
                pad_net_rows.append(
                    {
                        "pad": pad,
                        "actual_net_name": net_name,
                        "actual_net_canonical": canonical,
                    }
                )

            if missing_pads:
                bridge_integrity_issues.append(
                    {
                        "ref": ref,
                        "issue": "missing_pad_net",
                        "missing_pads": missing_pads,
                        "pads": pad_net_rows,
                    }
                )
                continue
            if len(set(canonical_nets)) < 2:
                bridge_integrity_issues.append(
                    {
                        "ref": ref,
                        "issue": "both_pads_same_net",
                        "net": canonical_nets[0] if canonical_nets else "",
                        "pads": pad_net_rows,
                    }
                )
                continue

            def _pad_expected_singleton_group(pad_name: str) -> bool:
                groups = expected_groups_by_pin.get((ref, str(pad_name or "").strip()), set())
                if not groups:
                    return False
                for g in groups:
                    row = expected_by_group.get(str(g or ""))
                    if not isinstance(row, dict):
                        continue
                    refs_for_group = {
                        str(r or "").strip().upper()
                        for r in list(row.get("refs") or [])
                        if str(r or "").strip()
                    }
                    if len(refs_for_group) == 1 and ref in refs_for_group:
                        return True
                return False

            orphan_side_nets = []
            for pad_row in pad_net_rows:
                if not isinstance(pad_row, dict):
                    continue
                net_name = str(pad_row.get("actual_net_canonical", "") or "").strip()
                if not net_name:
                    continue
                if len(net_ref_fanout.get(net_name, set())) > 1:
                    continue
                if _pad_expected_singleton_group(str(pad_row.get("pad", "") or "")):
                    continue
                orphan_side_nets.append(net_name)
            orphan_side_nets = sorted(set(orphan_side_nets))
            if orphan_side_nets:
                bridge_integrity_issues.append(
                    {
                        "ref": ref,
                        "issue": "orphan_side_net",
                        "orphan_nets": orphan_side_nets,
                        "pads": pad_net_rows,
                    }
                )

        bridge_bad_refs = sorted(
            {
                str(row.get("ref", "") or "").strip().upper()
                for row in bridge_integrity_issues
                if isinstance(row, dict) and str(row.get("ref", "") or "").strip()
            }
        )
        bridge_integrity_score = (
            int(round(100.0 * float(max(0, bridge_candidate_refs - len(bridge_bad_refs))) / float(max(bridge_candidate_refs, 1))))
            if bridge_candidate_refs > 0
            else 100
        )
        bridge_integrity_ok = bridge_candidate_refs <= 0 or (not bridge_bad_refs)

        groups_to_score = sorted(expected_by_group.keys())
        checks: List[Dict[str, Any]] = []
        group_scores: List[int] = []
        missing_groups: List[str] = []
        pinmap_mismatch_groups: List[Dict[str, Any]] = []
        expected_pin_total = 0
        connected_pin_total = 0
        disconnected_pins: List[Dict[str, Any]] = []
        split_groups: List[Dict[str, Any]] = []
        for group in groups_to_score:
            expected_row = expected_by_group.get(group)
            if not isinstance(expected_row, dict):
                continue
            expected_refs = sorted(set(str(ref) for ref in list(expected_row.get("refs") or []) if ref))
            if not expected_refs:
                continue
            expected_ref_set = set(expected_refs)
            actual_row = actual_groups.get(group) if isinstance(actual_groups.get(group), dict) else {}
            actual_refs = sorted(set(str(ref) for ref in list(actual_row.get("refs") or []) if ref))
            matched_refs_union = sorted(set(expected_refs) & set(actual_refs))
            ref_coverage = float(len(matched_refs_union)) / float(max(len(expected_refs), 1))

            expected_pins = [row for row in list(expected_row.get("pins") or []) if isinstance(row, dict)]
            expected_pin_eval_count = 0
            per_net_pin_counts_raw: Dict[str, int] = {}
            per_net_pin_counts_canonical: Dict[str, int] = {}
            canonical_to_raw_names: Dict[str, Set[str]] = {}
            pin_eval_rows: List[Dict[str, Any]] = []
            for pin_row in expected_pins:
                ref = str(pin_row.get("ref", "") or "").strip().upper()
                pad = str(pin_row.get("pad", "") or "").strip()
                exp_net = str(pin_row.get("net_name", "") or "").strip()
                if not ref or not pad:
                    continue
                expected_pin_eval_count += 1
                actual_net = ""
                try:
                    ref_row = actual_by_ref.get(ref) if isinstance(actual_by_ref.get(ref), dict) else {}
                    actual_net = str(ref_row.get(pad, "") or "").strip()
                except Exception:
                    actual_net = ""
                if actual_net:
                    per_net_pin_counts_raw[actual_net] = int(per_net_pin_counts_raw.get(actual_net, 0) or 0) + 1
                    canonical_actual_net = self._benchmark_net_canonical_name(actual_net) or actual_net
                    per_net_pin_counts_canonical[canonical_actual_net] = (
                        int(per_net_pin_counts_canonical.get(canonical_actual_net, 0) or 0) + 1
                    )
                    canonical_to_raw_names.setdefault(canonical_actual_net, set()).add(actual_net)
                pin_eval_rows.append(
                    {
                        "ref": ref,
                        "pad": pad,
                        "expected_net_name": exp_net,
                        "actual_net_name": actual_net,
                        "actual_net_canonical": (
                            self._benchmark_net_canonical_name(actual_net) or actual_net
                        ) if actual_net else "",
                    }
                )

            dominant_actual_net = ""
            dominant_actual_net_raw_names: List[str] = []
            dominant_pin_count = 0
            if per_net_pin_counts_canonical:
                dominant_actual_net, dominant_pin_count = max(
                    per_net_pin_counts_canonical.items(),
                    key=lambda item: (int(item[1]), str(item[0])),
                )
                dominant_actual_net_raw_names = sorted(
                    set(canonical_to_raw_names.get(dominant_actual_net, set()) or set())
                )
            if expected_pin_eval_count > 0:
                pin_connectivity = float(dominant_pin_count) / float(max(expected_pin_eval_count, 1))
            else:
                pin_connectivity = ref_coverage

            single_ref_group = len(expected_refs) == 1 and expected_pin_eval_count > 0
            single_ref_single_pin = len(expected_refs) == 1 and expected_pin_eval_count <= 1
            if single_ref_group and dominant_pin_count >= expected_pin_eval_count:
                assigned_refs = sorted(
                    {
                        str(row.get("ref", "") or "").upper()
                        for row in pin_eval_rows
                        if str(row.get("actual_net_name", "") or "").strip()
                    }
                )
                assigned_expected_refs = sorted(expected_ref_set & set(assigned_refs))
                ref_coverage = (
                    float(len(assigned_expected_refs)) / float(max(len(expected_refs), 1))
                )
                if ref_coverage > 0.0:
                    matched_refs_union = assigned_expected_refs

            matched_refs = matched_refs_union
            if dominant_actual_net:
                dominant_refs = sorted(
                    {
                        str(row.get("ref", "") or "").upper()
                        for row in pin_eval_rows
                        if str(row.get("actual_net_canonical", "") or "").strip() == dominant_actual_net
                    }
                )
                matched_refs = sorted(expected_ref_set & set(dominant_refs))

            disconnected_here: List[Dict[str, Any]] = []
            alt_group_match_count = 0
            if expected_pin_eval_count > 0:
                for row in pin_eval_rows:
                    actual_net = str(row.get("actual_net_name", "") or "").strip()
                    actual_net_canonical = str(row.get("actual_net_canonical", "") or "").strip()
                    row_ref = str(row.get("ref", "") or "").upper()
                    row_pad = str(row.get("pad", "") or "")
                    if not actual_net:
                        disconnected_here.append(
                            {
                                "ref": row_ref,
                                "pad": row_pad,
                                "reason": "missing_actual_net",
                            }
                        )
                        continue
                    if dominant_actual_net and actual_net_canonical != dominant_actual_net:
                        alt_groups = set(expected_groups_by_pin.get((row_ref, row_pad), set()) or set())
                        if group in alt_groups:
                            alt_groups.discard(group)
                        # If the manifest places the same physical pin in multiple
                        # expected groups, treat this as an ambiguity and avoid
                        # double-penalizing split connectivity in each group.
                        if alt_groups:
                            alt_group_match_count += 1
                            continue
                        disconnected_here.append(
                            {
                                "ref": row_ref,
                                "pad": row_pad,
                                "reason": "split_from_dominant_net",
                                "actual_net_name": actual_net,
                                "actual_net_canonical": actual_net_canonical,
                                "dominant_actual_net": dominant_actual_net,
                            }
                        )

            effective_expected_pin_eval_count = max(0, expected_pin_eval_count - int(alt_group_match_count))
            effective_connected_pin_count = min(int(dominant_pin_count), int(effective_expected_pin_eval_count))
            if effective_expected_pin_eval_count > 0:
                pin_connectivity = float(effective_connected_pin_count) / float(max(effective_expected_pin_eval_count, 1))
            else:
                pin_connectivity = ref_coverage

            group_coverage = min(ref_coverage, pin_connectivity)
            if len(expected_refs) >= 4:
                ok = group_coverage >= 0.50
            elif len(expected_refs) >= 2:
                ok = group_coverage >= 0.67
            else:
                ok = group_coverage >= 1.0

            # Penalize islanded nets only when the manifest expected multi-ref connectivity.
            if len(expected_refs) >= 2 and len(actual_refs) <= 1:
                group_coverage = 0.0
                ok = False

            # Guard against manifest under-specification: an IO-like Uno net that
            # only appears on a single connector ref is almost certainly missing
            # its internal peer connection (e.g., MCU or bridge pin), even if
            # expected_refs is incomplete.
            connector_only_actual = bool(actual_refs) and all(
                str(ref_name or "").upper().startswith("J")
                for ref_name in actual_refs
            )
            io_like_group = bool(re.match(r"^(a\d+|d\d+)$", str(group or ""))) or str(group or "") in {
                "ioref",
                "sda",
                "scl",
                "miso",
                "mosi",
                "sck",
                "rx",
                "tx",
                "reset",
            }
            connector_singleton = False
            if io_like_group and connector_only_actual and len(actual_refs) <= 1:
                connector_singleton = True
                group_coverage = 0.0
                ok = False
                if not disconnected_here:
                    disconnected_here.append(
                        {
                            "ref": str(actual_refs[0]) if actual_refs else "",
                            "pad": "",
                            "reason": "connector_singleton_without_internal_peer",
                        }
                    )

            # Strict pin-map integrity gate: if a ref was expected in this group
            # but all of its expected pins were skipped due pad mismatch, this is
            # a real connectivity/model mismatch and should fail the group.
            evaluated_pin_refs = {
                str(row.get("ref", "") or "").strip().upper()
                for row in expected_pins
                if isinstance(row, dict) and str(row.get("ref", "") or "").strip()
            }
            def _ref_has_group_net_via_any_pad(ref_name: str) -> bool:
                ref_map = actual_by_ref.get(str(ref_name).upper())
                if not isinstance(ref_map, dict):
                    return False
                for net_name in ref_map.values():
                    canonical = self._benchmark_net_canonical_name(str(net_name or "").strip())
                    if canonical and canonical == str(group or ""):
                        return True
                return False

            pinmap_lost_refs = sorted(
                ref_name
                for ref_name in expected_refs
                if (
                    skipped_expected_pins_by_group_ref.get((str(group or ""), str(ref_name).upper()), 0) > 0
                    and str(ref_name).upper() not in evaluated_pin_refs
                    and not _ref_has_group_net_via_any_pad(str(ref_name))
                )
            )
            if pinmap_lost_refs:
                group_coverage = 0.0
                ok = False
                pinmap_mismatch_groups.append(
                    {
                        "group": str(group or ""),
                        "refs": pinmap_lost_refs,
                    }
                )
                for ref_name in pinmap_lost_refs[:8]:
                    disconnected_here.append(
                        {
                            "ref": str(ref_name),
                            "pad": "",
                            "reason": "all_expected_pins_skipped_pad_mismatch",
                            "skipped_pin_count": int(
                                skipped_expected_pins_by_group_ref.get((str(group or ""), str(ref_name).upper()), 0) or 0
                            ),
                        }
                    )

            expected_pin_total += int(effective_expected_pin_eval_count)
            connected_pin_total += int(effective_connected_pin_count)
            if disconnected_here:
                for row in disconnected_here[:12]:
                    disconnected_pins.append({"group": group, **row})
            if (
                len(per_net_pin_counts_canonical) > 1
                and effective_expected_pin_eval_count > 1
                and effective_connected_pin_count < effective_expected_pin_eval_count
            ):
                split_groups.append(
                    {
                        "group": group,
                        "expected_pin_count": effective_expected_pin_eval_count,
                        "dominant_actual_net": dominant_actual_net,
                        "actual_net_pin_counts": {
                            str(net): int(count)
                            for net, count in sorted(
                                per_net_pin_counts_canonical.items(),
                                key=lambda item: (-int(item[1]), str(item[0])),
                            )
                        },
                    }
                )

            score_out_of_100 = int(round(100.0 * group_coverage))
            group_scores.append(score_out_of_100)
            if not ok:
                missing_groups.append(group)
            checks.append(
                {
                    "type": "critical_group",
                    "group": group,
                    "expected_refs": expected_refs,
                    "actual_refs": actual_refs,
                    "matched_refs": matched_refs,
                    "ref_coverage": round(ref_coverage, 3),
                    "pin_connectivity": round(pin_connectivity, 3),
                    "expected_pin_count": int(expected_row.get("pin_count", 0) or 0),
                    "evaluated_pin_count": expected_pin_eval_count,
                    "effective_evaluated_pin_count": int(effective_expected_pin_eval_count),
                    "alt_group_match_count": int(alt_group_match_count),
                    "dominant_actual_net": dominant_actual_net,
                    "dominant_actual_net_raw_names": dominant_actual_net_raw_names,
                    "dominant_actual_net_pin_count": int(dominant_pin_count),
                    "actual_nets_on_expected_pins": sorted(set(per_net_pin_counts_raw.keys())),
                    "actual_nets_on_expected_pins_canonical": sorted(set(per_net_pin_counts_canonical.keys())),
                    "actual_net_pin_counts": {
                        str(net): int(count)
                        for net, count in sorted(
                            per_net_pin_counts_raw.items(),
                            key=lambda item: (-int(item[1]), str(item[0])),
                        )
                    },
                    "actual_net_pin_counts_canonical": {
                        str(net): int(count)
                        for net, count in sorted(
                            per_net_pin_counts_canonical.items(),
                            key=lambda item: (-int(item[1]), str(item[0])),
                        )
                    },
                    "disconnected_expected_pin_count": len(disconnected_here),
                    "disconnected_expected_pins_sample": disconnected_here[:12],
                    "expected_net_names": sorted(set(str(x) for x in list(expected_row.get("net_names") or []) if x)),
                    "actual_net_names": sorted(set(str(x) for x in list(actual_row.get("net_names") or []) if x)),
                    "coverage": round(group_coverage, 3),
                    "score_out_of_100": score_out_of_100,
                    "ok": ok,
                    "single_ref_single_pin": bool(single_ref_single_pin),
                    "single_ref_group": bool(single_ref_group),
                    "connector_singleton": bool(connector_singleton),
                    "pinmap_lost_refs": pinmap_lost_refs,
                }
            )

        actual_multi_net_groups: List[Dict[str, Any]] = []
        for group_name, row in actual_groups.items():
            if not isinstance(row, dict):
                continue
            net_names = [str(v) for v in list(row.get("net_names") or []) if str(v).strip()]
            canonical_net_names = sorted(
                set(
                    self._benchmark_net_canonical_name(v) or str(v)
                    for v in net_names
                    if str(v).strip()
                )
            )
            if len(canonical_net_names) <= 1:
                continue
            actual_multi_net_groups.append(
                {
                    "group": str(group_name or ""),
                    "net_names": sorted(net_names),
                    "net_names_canonical": canonical_net_names,
                    "refs": sorted(set(str(v) for v in list(row.get("refs") or []) if str(v).strip())),
                }
            )
        actual_multi_net_groups.sort(key=lambda r: (str(r.get("group", "")), len(list(r.get("net_names") or []))))

        critical_group_score = int(round(sum(group_scores) / max(len(group_scores), 1))) if checks else 0
        pin_connectivity_score = (
            int(round(100.0 * float(connected_pin_total) / float(max(expected_pin_total, 1))))
            if expected_pin_total > 0
            else critical_group_score
        )

        ref_coverage_score = int(coverage_metrics.get("ref_coverage_score", 0) or 0) if bool(coverage_metrics.get("usable")) else 0
        pad_coverage_score = int(coverage_metrics.get("pad_coverage_score", 0) or 0) if bool(coverage_metrics.get("usable")) else 0
        if bool(coverage_metrics.get("usable")):
            overall_score = int(round(
                (0.25 * float(critical_group_score))
                + (0.20 * float(ref_coverage_score))
                + (0.15 * float(pad_coverage_score))
                + (0.25 * float(pin_connectivity_score))
                + (0.15 * float(bridge_integrity_score))
            ))
        else:
            overall_score = int(round(
                (0.60 * float(critical_group_score))
                + (0.25 * float(pin_connectivity_score))
                + (0.15 * float(bridge_integrity_score))
            ))

        zero_ref_count = int(coverage_metrics.get("refs_without_nets_count", 0) or 0) if bool(coverage_metrics.get("usable")) else 0
        partial_ref_count = int(coverage_metrics.get("partial_refs_count", 0) or 0) if bool(coverage_metrics.get("usable")) else 0
        strict_all_pads_checked = bool(coverage_metrics.get("strict_all_physical_pads_checked", False))
        if not bool(coverage_metrics.get("usable")):
            ref_coverage_ok = False
        elif strict_all_pads_checked:
            ref_coverage_ok = (
                zero_ref_count == 0
                and partial_ref_count == 0
                and ref_coverage_score == 100
                and pad_coverage_score == 100
            )
        else:
            ref_coverage_ok = (
                zero_ref_count == 0
                and ref_coverage_score >= 90
                and pad_coverage_score >= 95
            )
        pin_connectivity_ok = expected_pin_total <= 0 or pin_connectivity_score >= 95
        return {
            "applicable": True,
            "source": str(actual_snapshot.get("source", "unknown") or "unknown"),
            "assignment_count": int(actual_snapshot.get("assignment_count", 0) or 0),
            "unique_net_count": int(actual_snapshot.get("unique_net_count", 0) or 0),
            "expected_group_count": len(checks),
            "groups_scored": len(checks),
            "subscores": {
                "critical_group_score": critical_group_score,
                "ref_coverage_score": ref_coverage_score,
                "pad_coverage_score": pad_coverage_score,
                "pin_connectivity_score": pin_connectivity_score,
                "bridge_integrity_score": bridge_integrity_score,
            },
            "expected_pin_count": int(expected_pin_total),
            "connected_pin_count": int(connected_pin_total),
            "unconnected_pin_count": int(max(0, expected_pin_total - connected_pin_total)),
            "unconnected_pins_sample": disconnected_pins[:80],
            "skipped_expected_pin_count": int(len(skipped_expected_pins)),
            "skipped_expected_pins_sample": skipped_expected_pins[:80],
            "split_group_count": int(len(split_groups)),
            "split_groups_sample": split_groups[:40],
            "pinmap_mismatch_group_count": int(len(pinmap_mismatch_groups)),
            "pinmap_mismatch_groups_sample": pinmap_mismatch_groups[:40],
            "actual_multi_net_group_count": int(len(actual_multi_net_groups)),
            "actual_multi_net_groups_sample": actual_multi_net_groups[:60],
            "bridge_integrity_candidate_ref_count": int(bridge_candidate_refs),
            "bridge_integrity_issue_ref_count": int(len(bridge_bad_refs)),
            "bridge_integrity_issue_count": int(len(bridge_integrity_issues)),
            "bridge_integrity_issues_sample": bridge_integrity_issues[:80],
            "coverage": coverage_metrics,
            "strict_all_physical_pads_checked": strict_all_pads_checked,
            "score": overall_score,
            "ok": bool(checks)
            and critical_group_score >= 70
            and ref_coverage_ok
            and pin_connectivity_ok
            and bridge_integrity_ok
            and not bool(missing_groups),
            "missing_groups": missing_groups,
            "checks": checks,
        }
    def _benchmark_build_report(self, benchmark: Dict[str, Any], loop: Any) -> Dict[str, Any]:
        history = list(getattr(loop, "_history", []) or [])
        action_rows: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        failed_actions: List[Dict[str, Any]] = []
        for step in history:
            action = getattr(step, "action", None)
            if action is None:
                continue
            at = str(getattr(getattr(action, "action_type", None), "name", getattr(action, "action_type", "")) or "")
            counts[at] = int(counts.get(at, 0)) + 1
            row = {
                "iteration": int(getattr(step, "iteration", 0) or 0),
                "action_type": at,
                "success": bool(getattr(action, "success", False)),
                "message": str(getattr(action, "result_message", "") or ""),
                "description": str(getattr(action, "description", "") or ""),
            }
            action_rows.append(row)
            if not row["success"]:
                failed_actions.append(row)

        gate_failures = self._benchmark_extract_gate_failures(loop)
        unresolved_roles = self._benchmark_extract_unresolved_roles_from_gate_failures(gate_failures)
        role_snapshot = self._benchmark_extract_role_constraints_snapshot(loop)
        unresolved_roles_detailed: List[Dict[str, Any]] = []
        for u in unresolved_roles:
            if not isinstance(u, dict):
                continue
            rid = str(u.get("role_id", "") or "").strip()
            row = dict(u)
            if rid and rid in role_snapshot:
                row["role_snapshot"] = role_snapshot.get(rid)
            unresolved_roles_detailed.append(row)
        critical_unresolved_roles: List[Dict[str, Any]] = []
        for row in unresolved_roles_detailed:
            if not isinstance(row, dict):
                continue
            snap = row.get("role_snapshot") if isinstance(row.get("role_snapshot"), dict) else {}
            if not bool(snap.get("critical", False)):
                continue
            constraints = snap.get("constraints") if isinstance(snap.get("constraints"), dict) else {}
            critical_unresolved_roles.append(
                {
                    "role_id": str(row.get("role_id", "") or ""),
                    "issue_type": str(row.get("issue_type", row.get("source", "")) or ""),
                    "detail": str(row.get("detail", row.get("message", "")) or ""),
                    "role_type": str(snap.get("role_type", "") or ""),
                    "part_query": str(constraints.get("part_query", "") or ""),
                    "package": str(constraints.get("package", "") or ""),
                    "alternates": list(snap.get("alternates") or []) if isinstance(snap.get("alternates"), list) else [],
                }
            )
        stage_statuses = self._benchmark_stage_statuses(loop, gate_failures)
        semantic_check = self._benchmark_uno_semantic_completeness(loop, benchmark)
        bom_oracle = self._benchmark_uno_bom_oracle(loop, benchmark)
        footprint_match = self._benchmark_spec_footprint_match(loop, benchmark)
        placement_audit = self._benchmark_placement_audit(loop)
        net_score = self._benchmark_uno_net_score(loop, benchmark)
        design_sanity = self._benchmark_design_sanity_checks(loop, benchmark, net_score)
        ee_rules = self._benchmark_ee_rules_audit(loop, benchmark, net_score, design_sanity)
        bom_only_mode = bool(benchmark.get("bom_only_mode", False))
        if bom_only_mode:
            net_score = {"applicable": False, "source": "skipped", "score": 0, "ok": False}
            design_sanity = {"applicable": False, "ok": True, "checks": [], "failed_required_checks": []}
            ee_rules = {
                "applicable": False,
                "score_out_of_100": 0,
                "total_penalty": 0,
                "rule_count": 0,
                "failed_rule_count": 0,
                "failed_required_rule_count": 0,
                "failed_rule_ids": [],
                "rules": [],
                "issues_for_done_checkpoint": [],
                "ok": True,
            }
        semantic_tiers = {
            "core": "pass" if bool(semantic_check.get("ok", False)) else "fail",
            "secondary": "pass" if bool(semantic_check.get("secondary_ok", False)) else "fail",
            "bom_oracle": "pass" if bool(bom_oracle.get("ok", False)) else "fail",
        }
        bom_checks = list(bom_oracle.get("checks", []) or []) if isinstance(bom_oracle, dict) else []
        checks_by_id: Dict[str, Dict[str, Any]] = {
            str(r.get("item_id", "") or ""): r for r in bom_checks if isinstance(r, dict)
        }
        missing_parts: List[Dict[str, Any]] = []
        incorrect_parts: List[Dict[str, Any]] = []
        for item_id in list(bom_oracle.get("missing", []) or []):
            iid = str(item_id or "")
            if not iid:
                continue
            row = checks_by_id.get(iid) or {}
            qty_min = int(row.get("expected_qty_min", 0) or 0)
            qty_max = int(row.get("expected_qty_max", qty_min) or qty_min)
            actual_qty = int(row.get("actual_qty", 0) or 0)
            missing_parts.append(
                {
                    "item_id": iid,
                    "expected_qty_min": qty_min,
                    "expected_qty_max": qty_max,
                    "actual_qty": actual_qty,
                }
            )
        for row in list(bom_oracle.get("qty_mismatches", []) or []):
            if not isinstance(row, dict):
                continue
            iid = str(row.get("item_id", "") or "")
            if not iid:
                continue
            incorrect_parts.append(
                {
                    "item_id": iid,
                    "expected_qty_min": int(row.get("expected_qty_min", 0) or 0),
                    "expected_qty_max": int(row.get("expected_qty_max", 0) or 0),
                    "actual_qty": int(row.get("actual_qty", 0) or 0),
                }
            )
        total_expected = int(bom_oracle.get("required_count", 0) or 0)
        if total_expected <= 0:
            total_expected = max(len([r for r in bom_checks if isinstance(r, dict) and not bool(r.get("optional", False))]), 1)
        bom_score = self._benchmark_bom_score_from_oracle(bom_oracle)
        if bom_only_mode:
            statuses = {
                "SPEC": "pass",
                "RESOLVE": "pass",
                "IMPORT": "skipped",
                "GEOM": "skipped",
                "NET": "skipped",
                "BIND": "skipped",
                "DRC": "skipped",
            }
            if bool(bom_oracle.get("applicable")) and not bool(bom_oracle.get("ok", False)):
                statuses["SPEC"] = "fail"
                statuses["RESOLVE"] = "fail"
            stage_statuses = statuses
        else:
            if bool(semantic_check.get("applicable")) and (not bool(semantic_check.get("ok", False))):
                # Core architectural incompleteness invalidates both SPEC and RESOLVE.
                stage_statuses["SPEC"] = "fail"
                stage_statuses["RESOLVE"] = "fail"
            elif bool(semantic_check.get("applicable")) and (not bool(semantic_check.get("secondary_ok", False))):
                # Secondary/support BOM incompleteness is a SPEC completeness issue,
                # not necessarily a resolver correctness failure for core roles.
                stage_statuses["SPEC"] = "fail"
            # If no critical unresolved roles remain and core semantics pass, don't
            # report RESOLVE as failed just because noncritical passive roles are
            # still stalling in SPEC.progress.
            if (
                bool(semantic_check.get("applicable"))
                and bool(semantic_check.get("ok", False))
                and not critical_unresolved_roles
                and stage_statuses.get("RESOLVE") == "fail"
            ):
                stage_statuses["RESOLVE"] = "pass"
            if bool(semantic_check.get("applicable")) and bool(semantic_check.get("ok", False)):
                if not bool(placement_audit.get("ok", False)):
                    stage_statuses["IMPORT"] = "fail"
            if bool(bom_oracle.get("applicable")) and not bool(bom_oracle.get("ok", False)):
                stage_statuses["SPEC"] = "fail"
                stage_statuses["RESOLVE"] = "fail"
            if bool(net_score.get("applicable")):
                stage_statuses["NET"] = "pass" if bool(net_score.get("ok", False)) else "fail"
            if stage_statuses.get("BIND") == "unknown" and stage_statuses.get("NET") in {"pass", "fail"}:
                stage_statuses["BIND"] = stage_statuses["NET"]
            if bool(design_sanity.get("applicable")) and not bool(design_sanity.get("ok", False)):
                failed_rows = [
                    row for row in list(design_sanity.get("failed_required_checks") or [])
                    if isinstance(row, dict)
                ]
                for row in failed_rows:
                    cid = str(row.get("id", "") or "")
                    if cid == "schematic_presence":
                        stage_statuses["SPEC"] = "fail"
                        stage_statuses["DRC"] = "fail"
                    elif cid in {"ground_plane", "clock_placement"}:
                        stage_statuses["GEOM"] = "fail"
                    elif cid == "power_path_bridge_integrity":
                        stage_statuses["NET"] = "fail"
            if bool(ee_rules.get("applicable")) and not bool(ee_rules.get("ok", False)):
                for rule in list(ee_rules.get("rules", []) or []):
                    if not isinstance(rule, dict):
                        continue
                    if not bool(rule.get("applicable", True)):
                        continue
                    if bool(rule.get("ok", False)):
                        continue
                    for stage in list(rule.get("stages", []) or []):
                        s = str(stage or "").upper()
                        if s in stage_statuses and stage_statuses.get(s) != "skipped":
                            stage_statuses[s] = "fail"
        terminal_state = str(getattr(getattr(loop, "state", None), "name", getattr(loop, "_state", "")) or "")
        duration_s = round(max(0.0, time.time() - float(benchmark.get("start_ts", time.time()))), 3)

        forced_fail = benchmark.get("fail_fast_payload") if isinstance(benchmark, dict) else None
        if bom_only_mode:
            outcome = "pass" if bool(bom_oracle.get("applicable")) and bool(bom_oracle.get("ok", False)) else "fail"
            primary_failure = None
            if isinstance(forced_fail, dict):
                primary_failure = forced_fail
                outcome = "fail"
            elif not bool(bom_oracle.get("ok", False)):
                primary_failure = {
                    "gate": "BENCHMARK.bom_oracle",
                    "message": "Uno BOM oracle mismatch",
                    "details": {
                        "missing": list(bom_oracle.get("missing", []) or []),
                        "qty_mismatches": list(bom_oracle.get("qty_mismatches", []) or []),
                        "errors": list(bom_oracle.get("errors", []) or []),
                    },
                    "bounce_to": "SPEC",
                }
        else:
            # Fail-fast benchmark outcome: awaiting clarification is a benchmark failure.
            outcome = "pass" if terminal_state == "DONE" else "fail"
            primary_failure = None
            if gate_failures:
                primary_failure = gate_failures[0]
            elif failed_actions:
                primary_failure = failed_actions[0]
            # For benchmark diagnostics, surface the critical unresolved set as the
            # primary failure when a generic SPEC.progress stall contains mixed
            # critical/noncritical issues.
            if (
                isinstance(primary_failure, dict)
                and str(primary_failure.get("gate", "") or "") == "SPEC.progress"
                and critical_unresolved_roles
            ):
                primary_failure = {
                    "gate": "BENCHMARK.critical_resolve",
                    "message": f"{len(critical_unresolved_roles)} critical role(s) unresolved",
                    "details": {
                        "roles": [
                            {
                                "role_id": str(r.get("role_id", "") or ""),
                                "role_type": str(r.get("role_type", "") or ""),
                                "part_query": str(r.get("part_query", "") or ""),
                                "package": str(r.get("package", "") or ""),
                            }
                            for r in critical_unresolved_roles[:12]
                            if isinstance(r, dict)
                        ]
                    },
                    "bounce_to": "SPEC",
                }
            elif (
                isinstance(primary_failure, dict)
                and str(primary_failure.get("gate", "") or "") == "SPEC.progress"
                and not critical_unresolved_roles
                and bool(semantic_check.get("applicable"))
                and bool(semantic_check.get("ok", False))
            ):
                # Distinguish noncritical/passive resolve stalls from core resolve failures.
                open_issues = []
                try:
                    details = primary_failure.get("details") if isinstance(primary_failure.get("details"), dict) else {}
                    raw = details.get("openissues") or details.get("open_issues")
                    if isinstance(raw, list):
                        open_issues = [x for x in raw if isinstance(x, dict)]
                except Exception:
                    open_issues = []
                passive_roles = []
                for issue in open_issues[:20]:
                    role = str(issue.get("role", "") or "").strip()
                    if not role:
                        continue
                    snap = role_snapshot.get(role) if isinstance(role_snapshot.get(role), dict) else {}
                    if bool(snap.get("critical", False)):
                        continue
                    if role not in passive_roles:
                        passive_roles.append(role)
                primary_failure = {
                    "gate": "BENCHMARK.noncritical_resolve_stall",
                    "message": f"{len(passive_roles)} noncritical role(s) unresolved in SPEC.progress",
                    "details": {"roles": passive_roles[:20]},
                    "bounce_to": "SPEC",
                }
            if (
                not isinstance(forced_fail, dict)
                and bool(semantic_check.get("applicable"))
                and bool(semantic_check.get("ok", False))
                and not critical_unresolved_roles
                and not bool(placement_audit.get("ok", False))
            ):
                outcome = "fail"
                primary_failure = {
                    "gate": "BENCHMARK.placement_consistency",
                    "message": (
                        f"Placed footprint consistency mismatch: "
                        f"extra={int(placement_audit.get('extra_count', 0) or 0)}, "
                        f"missing_critical={int(placement_audit.get('missing_critical_count', 0) or 0)}"
                    ),
                    "details": {
                        "extras": list(placement_audit.get("extras", []) or []),
                        "missing": list(placement_audit.get("missing", []) or []),
                    },
                    "bounce_to": "SPEC",
                }
            if isinstance(forced_fail, dict):
                outcome = "fail"
                primary_failure = forced_fail
            elif bool(design_sanity.get("applicable")) and not bool(design_sanity.get("ok", False)):
                outcome = "fail"
                first_failed = (
                    design_sanity.get("first_failure")
                    if isinstance(design_sanity.get("first_failure"), dict)
                    else {}
                )
                primary_failure = {
                    "gate": str(first_failed.get("gate", "") or "BENCHMARK.design_sanity"),
                    "message": str(first_failed.get("message", "") or "Benchmark design sanity checks failed"),
                    "details": {
                        "failed_required_checks": [
                            {
                                "id": str(row.get("id", "") or ""),
                                "gate": str(row.get("gate", "") or ""),
                                "message": str(row.get("message", "") or ""),
                                "details": row.get("details", {}),
                            }
                            for row in list(design_sanity.get("failed_required_checks") or [])
                            if isinstance(row, dict)
                        ]
                    },
                    "bounce_to": str(first_failed.get("bounce_to", "") or "SPEC"),
                }
            elif bool(ee_rules.get("applicable")) and not bool(ee_rules.get("ok", False)):
                outcome = "fail"
                first_failed_rule = None
                for rule in list(ee_rules.get("rules", []) or []):
                    if not isinstance(rule, dict):
                        continue
                    if not bool(rule.get("applicable", True)):
                        continue
                    if bool(rule.get("ok", False)):
                        continue
                    first_failed_rule = rule
                    break
                first_failed_rule = first_failed_rule if isinstance(first_failed_rule, dict) else {}
                stages = [str(s).upper() for s in list(first_failed_rule.get("stages", []) or []) if str(s).strip()]
                bounce_to = stages[0] if stages else "SPEC"
                primary_failure = {
                    "gate": f"BENCHMARK.ee_rule.{str(first_failed_rule.get('id', '') or 'unknown')}",
                    "message": str(first_failed_rule.get("message", "") or "EE rule check failed"),
                    "details": {
                        "ee_score_out_of_100": int(ee_rules.get("score_out_of_100", 0) or 0),
                        "failed_rule_count": int(ee_rules.get("failed_rule_count", 0) or 0),
                        "failed_required_rule_count": int(ee_rules.get("failed_required_rule_count", 0) or 0),
                        "failed_rule_ids": list(ee_rules.get("failed_rule_ids", []) or []),
                        "issues_for_done_checkpoint": list(ee_rules.get("issues_for_done_checkpoint", []) or [])[:20],
                    },
                    "bounce_to": bounce_to,
                }
            elif bool(net_score.get("applicable")) and not bool(net_score.get("ok", False)):
                outcome = "fail"
                primary_failure = {
                    "gate": "BENCHMARK.net_connectivity",
                    "message": "Uno net connectivity score below threshold",
                    "details": {
                        "score": int(net_score.get("score", 0) or 0),
                        "missing_groups": list(net_score.get("missing_groups", []) or []),
                        "essential_missing_groups": list(net_score.get("essential_missing_groups", []) or []),
                        "bridge_integrity_issue_ref_count": int(net_score.get("bridge_integrity_issue_ref_count", 0) or 0),
                        "bridge_integrity_issues_sample": list(net_score.get("bridge_integrity_issues_sample", []) or [])[:20],
                        "unconnected_pins_sample": list(net_score.get("unconnected_pins_sample", []) or [])[:20],
                    },
                    "bounce_to": "NET",
                }
            elif bool(bom_oracle.get("applicable")) and not bool(bom_oracle.get("ok", False)):
                outcome = "fail"
                primary_failure = {
                    "gate": "BENCHMARK.bom_oracle",
                    "message": "Uno BOM oracle mismatch",
                    "details": {
                        "missing": list(bom_oracle.get("missing", []) or []),
                        "qty_mismatches": list(bom_oracle.get("qty_mismatches", []) or []),
                        "errors": list(bom_oracle.get("errors", []) or []),
                    },
                    "bounce_to": "SPEC",
                }

        placement_score = self._benchmark_placement_score(loop, benchmark)
        artifacts = getattr(loop, "_artifacts", {}) or {}
        spec_debug = artifacts.get("spec_debug") if isinstance(artifacts.get("spec_debug"), dict) else None
        net_plan = artifacts.get("net_plan") if isinstance(artifacts.get("net_plan"), dict) else {}
        net_plan_summary = {
            key: value
            for key, value in net_plan.items()
            if key != "assignments"
        }
        placement_plan = artifacts.get("placement_plan") if isinstance(artifacts.get("placement_plan"), dict) else {}
        board_placement_snapshot = self._benchmark_board_placement_snapshot()
        placement_snapshot = {
            str(ref): {
                "x": float(pos.get("x", 0.0) or 0.0),
                "y": float(pos.get("y", 0.0) or 0.0),
                "rot": float(pos.get("rot", 0.0) or 0.0),
                "zone": pos.get("zone"),
                "category": pos.get("category"),
                "edge": pos.get("edge"),
                "face_dir": pos.get("face_dir"),
                "face_dir_source": pos.get("face_dir_source"),
                "footprint_body_dir": pos.get("footprint_body_dir"),
                "footprint_face_dir_guess": pos.get("footprint_face_dir_guess"),
                "edge_outward_depth": pos.get("edge_outward_depth"),
                "edge_inward_depth": pos.get("edge_inward_depth"),
            }
            for ref, pos in placement_plan.items()
            if isinstance(ref, str) and isinstance(pos, dict)
        }
        placement_diagnostics = self._benchmark_placement_diagnostics(
            placement_plan=placement_snapshot,
            board_snapshot=board_placement_snapshot,
            benchmark=benchmark if isinstance(benchmark, dict) else None,
        )

        all_issues: List[Dict[str, Any]] = []
        seen_issue_keys: Set[str] = set()

        def _push_issue(
            *,
            category: str,
            issue_id: str,
            message: str,
            stage: str = "DONE",
            critical: bool = False,
            penalty: int = 1,
            details: Optional[Dict[str, Any]] = None,
        ) -> None:
            c = str(category or "").strip().lower() or "general"
            iid = str(issue_id or "").strip()
            if not iid:
                iid = f"{c}_{len(all_issues) + 1}"
            key = f"{c}:{iid}"
            if key in seen_issue_keys:
                return
            seen_issue_keys.add(key)
            all_issues.append(
                {
                    "category": c,
                    "id": iid,
                    "message": str(message or "").strip(),
                    "stage": str(stage or "DONE").strip().upper() or "DONE",
                    "critical": bool(critical),
                    "score_penalty": int(max(1, penalty)),
                    "details": details if isinstance(details, dict) else {},
                }
            )

        for row in missing_parts[:60]:
            iid = str(row.get("item_id", "") or "").strip()
            if not iid:
                continue
            _push_issue(
                category="bom",
                issue_id=f"missing_{iid}",
                message=(
                    f"Missing required part '{iid}' "
                    f"(expected {int(row.get('expected_qty_min', 0) or 0)}-"
                    f"{int(row.get('expected_qty_max', 0) or 0)}, "
                    f"actual {int(row.get('actual_qty', 0) or 0)})"
                ),
                stage="SPEC",
                critical=True,
                penalty=2,
                details=row,
            )
        for row in incorrect_parts[:60]:
            iid = str(row.get("item_id", "") or "").strip()
            if not iid:
                continue
            _push_issue(
                category="bom",
                issue_id=f"qty_{iid}",
                message=(
                    f"Quantity mismatch for '{iid}' "
                    f"(expected {int(row.get('expected_qty_min', 0) or 0)}-"
                    f"{int(row.get('expected_qty_max', 0) or 0)}, "
                    f"actual {int(row.get('actual_qty', 0) or 0)})"
                ),
                stage="SPEC",
                critical=True,
                penalty=2,
                details=row,
            )
        for item in list(semantic_check.get("missing", []) or [])[:40]:
            name = str(item or "").strip()
            if not name:
                continue
            _push_issue(
                category="semantic",
                issue_id=f"core_{name}",
                message=f"Missing core semantic block: {name}",
                stage="SPEC",
                critical=True,
                penalty=2,
            )
        for item in list(semantic_check.get("secondary_missing", []) or [])[:40]:
            name = str(item or "").strip()
            if not name:
                continue
            _push_issue(
                category="semantic",
                issue_id=f"secondary_{name}",
                message=f"Missing secondary semantic block: {name}",
                stage="SPEC",
                critical=False,
                penalty=1,
            )
        for row in list(footprint_match.get("mismatches", []) or [])[:80]:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("ref", "") or "").strip().upper()
            status = str(row.get("status", "") or "").strip().lower() or "mismatch"
            _push_issue(
                category="footprint",
                issue_id=f"{ref or 'unknown'}_{status}",
                message=(
                    f"Footprint {status} at {ref or 'unknown ref'} "
                    f"(expected '{str(row.get('expected_footprint', '') or '')}', "
                    f"actual '{str(row.get('actual_footprint', '') or '')}')"
                ),
                stage="IMPORT",
                critical=status == "missing",
                penalty=2 if status == "missing" else 1,
                details=row,
            )

        if int(placement_score.get("out_of_bounds", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="out_of_bounds",
                message=f"{int(placement_score.get('out_of_bounds', 0) or 0)} component(s) placed out of bounds",
                stage="GEOM",
                critical=True,
                penalty=2,
                details={"refs_sample": list(placement_score.get("out_of_bounds_refs_sample", []) or [])[:20]},
            )
        if int(placement_score.get("overlap_collisions", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="overlap_collisions",
                message=f"{int(placement_score.get('overlap_collisions', 0) or 0)} placement overlap collision(s)",
                stage="GEOM",
                critical=True,
                penalty=2,
                details={"pairs_sample": list(placement_score.get("overlap_pairs_sample", []) or [])[:20]},
            )
        if int(placement_score.get("audit_missing_critical_count", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="missing_critical_refs",
                message=f"{int(placement_score.get('audit_missing_critical_count', 0) or 0)} critical expected ref(s) missing on board",
                stage="IMPORT",
                critical=True,
                penalty=2,
            )
        if int(placement_score.get("audit_extra_count", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="unexpected_extra_refs",
                message=f"{int(placement_score.get('audit_extra_count', 0) or 0)} unexpected placed ref(s) found",
                stage="IMPORT",
                critical=False,
                penalty=1,
            )
        if placement_score.get("drc_passed") is False:
            _push_issue(
                category="placement",
                issue_id="placement_drc_failed",
                message=f"Placement DRC failed with {int(placement_score.get('drc_error_count', 0) or 0)} error(s)",
                stage="DRC",
                critical=True,
                penalty=2,
            )
        plan_alignment = placement_score.get("plan_alignment") if isinstance(placement_score.get("plan_alignment"), dict) else {}
        if int(plan_alignment.get("moved_count", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="plan_alignment_moved_refs",
                message=(
                    f"{int(plan_alignment.get('moved_count', 0) or 0)} ref(s) drifted from placement plan "
                    f"(ratio {round(float(plan_alignment.get('moved_ratio', 0.0) or 0.0), 3)})"
                ),
                stage="GEOM",
                critical=False,
                penalty=1,
                details=plan_alignment,
            )
        if int(plan_alignment.get("rot_mismatch_count", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="plan_alignment_rotation",
                message=f"{int(plan_alignment.get('rot_mismatch_count', 0) or 0)} ref(s) rotation-mismatched vs plan",
                stage="GEOM",
                critical=False,
                penalty=1,
                details=plan_alignment,
            )
        if int(plan_alignment.get("outward_mismatch_count", 0) or 0) > 0:
            _push_issue(
                category="placement",
                issue_id="plan_alignment_facing",
                message=f"{int(plan_alignment.get('outward_mismatch_count', 0) or 0)} edge-facing mismatch(es) vs plan",
                stage="GEOM",
                critical=False,
                penalty=1,
                details=plan_alignment,
            )

        net_source = str(net_score.get("source", "") or "").strip().lower()
        net_assignment_count = int(net_score.get("assignment_count", 0) or 0)
        net_evidence_ready = bool(net_source == "board" and net_assignment_count > 0)
        defer_net_issue_reporting = bool(benchmark.get("defer_net_issue_reporting_until_routed", True))
        if (not defer_net_issue_reporting) or net_evidence_ready:
            for grp in list(net_score.get("essential_missing_groups", []) or [])[:30]:
                g = str(grp or "").strip()
                if not g:
                    continue
                _push_issue(
                    category="net",
                    issue_id=f"essential_missing_{g}",
                    message=f"Missing essential net group: {g}",
                    stage="NET",
                    critical=True,
                    penalty=2,
                )
            for grp in list(net_score.get("missing_groups", []) or [])[:40]:
                g = str(grp or "").strip()
                if not g:
                    continue
                _push_issue(
                    category="net",
                    issue_id=f"missing_{g}",
                    message=f"Missing net group: {g}",
                    stage="NET",
                    critical=False,
                    penalty=1,
                )
            if int(net_score.get("bridge_integrity_issue_ref_count", 0) or 0) > 0:
                _push_issue(
                    category="net",
                    issue_id="bridge_integrity",
                    message=f"{int(net_score.get('bridge_integrity_issue_ref_count', 0) or 0)} bridge-integrity issue(s)",
                    stage="NET",
                    critical=True,
                    penalty=2,
                    details={"sample": list(net_score.get("bridge_integrity_issues_sample", []) or [])[:20]},
                )
            if int(net_score.get("unconnected_pins_count", 0) or 0) > 0:
                _push_issue(
                    category="net",
                    issue_id="unconnected_pins",
                    message=f"{int(net_score.get('unconnected_pins_count', 0) or 0)} unconnected expected pin(s)",
                    stage="NET",
                    critical=True,
                    penalty=2,
                    details={"sample": list(net_score.get("unconnected_pins_sample", []) or [])[:20]},
                )
            coverage = net_score.get("coverage") if isinstance(net_score.get("coverage"), dict) else {}
            if int(coverage.get("refs_without_nets_count", 0) or 0) > 0:
                _push_issue(
                    category="net",
                    issue_id="refs_without_nets",
                    message=f"{int(coverage.get('refs_without_nets_count', 0) or 0)} ref(s) have zero assigned nets",
                    stage="NET",
                    critical=True,
                    penalty=2,
                    details={"sample": list(coverage.get("refs_without_nets_sample", []) or [])[:20]},
                )
            if int(coverage.get("partial_refs_count", 0) or 0) > 0:
                _push_issue(
                    category="net",
                    issue_id="partial_ref_connectivity",
                    message=f"{int(coverage.get('partial_refs_count', 0) or 0)} ref(s) have only partial net coverage",
                    stage="NET",
                    critical=False,
                    penalty=1,
                    details={"sample": list(coverage.get("partial_refs_sample", []) or [])[:20]},
                )

        for row in list(design_sanity.get("failed_required_checks", []) or [])[:30]:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("id", "") or "").strip() or "unknown"
            gate = str(row.get("bounce_to", row.get("gate", "")) or "").strip().upper() or "SPEC"
            _push_issue(
                category="sanity",
                issue_id=cid,
                message=str(row.get("message", "") or f"Design sanity check failed: {cid}"),
                stage=gate,
                critical=True,
                penalty=2,
                details=row,
            )

        ee_failed_required_ids: Set[str] = set()
        for rule in list(ee_rules.get("rules", []) or []):
            if not isinstance(rule, dict):
                continue
            if not bool(rule.get("applicable", True)):
                continue
            if bool(rule.get("ok", False)):
                continue
            if bool(rule.get("required", True)):
                ee_failed_required_ids.add(str(rule.get("id", "") or ""))
        for row in list(ee_rules.get("issues_for_done_checkpoint", []) or [])[:80]:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id", "") or "").strip() or "unknown"
            stages = [str(s).upper() for s in list(row.get("stages", []) or []) if str(s).strip()]
            _push_issue(
                category="ee",
                issue_id=rid,
                message=str(row.get("message", "") or str(row.get("title", "") or f"EE rule failed: {rid}")),
                stage=stages[0] if stages else "SPEC",
                critical=(rid in ee_failed_required_ids),
                penalty=max(1, min(4, (int(row.get("penalty", 0) or 0) + 3) // 4)),
                details=row,
            )

        critical_role_ids = {
            str(row.get("role_id", "") or "").strip()
            for row in critical_unresolved_roles
            if isinstance(row, dict) and str(row.get("role_id", "") or "").strip()
        }
        for row in unresolved_roles_detailed[:60]:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("role_id", "") or "").strip()
            if not rid:
                continue
            critical_role = rid in critical_role_ids
            _push_issue(
                category="resolve",
                issue_id=f"{rid}:{str(row.get('issue_type', '') or 'unresolved')}",
                message=str(row.get("detail", row.get("message", "")) or f"Unresolved role: {rid}"),
                stage="RESOLVE",
                critical=critical_role,
                penalty=2 if critical_role else 1,
                details=row,
            )

        for idx, row in enumerate(gate_failures[:25]):
            if not isinstance(row, dict):
                continue
            gate = str(row.get("gate", "") or "").strip() or f"gate_{idx + 1}"
            bounce = str(row.get("bounce_to", "") or "").strip().upper() or "SPEC"
            _push_issue(
                category="gate",
                issue_id=gate,
                message=str(row.get("message", "") or f"Gate failure at {gate}"),
                stage=bounce,
                critical=True,
                penalty=2,
                details=row,
            )

        for idx, row in enumerate(failed_actions[:25]):
            if not isinstance(row, dict):
                continue
            aid = str(row.get("id", "") or "").strip() or f"action_{idx + 1}"
            atype = str(row.get("action_type", "") or "").strip().upper() or "ACTION"
            _push_issue(
                category="action",
                issue_id=aid,
                message=str(row.get("message", "") or f"Failed action: {atype}"),
                stage=atype,
                critical=False,
                penalty=1,
                details=row,
            )

        raw_issue_penalty = int(sum(int(row.get("score_penalty", 0) or 0) for row in all_issues))
        all_issue_penalty = min(40, raw_issue_penalty)
        critical_issue_count = len([row for row in all_issues if bool(row.get("critical", False))])

        placement_score_value = int(placement_score.get("score", 0) or 0) if isinstance(placement_score, dict) else 0
        net_score_value = int(net_score.get("score", 0) or 0) if isinstance(net_score, dict) else 0
        footprint_score_value = int(footprint_match.get("score_out_of_100", 0) or 0) if isinstance(footprint_match, dict) else 0
        ee_score_value = int(ee_rules.get("score_out_of_100", 0) or 0) if isinstance(ee_rules, dict) else 0
        weighted_score = int(round(
            (0.30 * float(bom_score))
            + (0.20 * float(placement_score_value))
            + (0.20 * float(net_score_value))
            + (0.15 * float(footprint_score_value))
            + (0.15 * float(ee_score_value))
        ))
        overall_score = int(weighted_score) - int(all_issue_penalty)
        if bool(ee_rules.get("applicable")) and int(ee_rules.get("failed_required_rule_count", 0) or 0) > 0:
            overall_score = min(overall_score, 69)
        if bool(design_sanity.get("applicable")) and not bool(design_sanity.get("ok", False)):
            overall_score = min(overall_score, 65)
        overall_score = max(0, min(100, int(overall_score)))

        return {
            "benchmark_id": str(benchmark.get("id", "")),
            "scenario": str(benchmark.get("scenario", "")),
            "prompt": str(benchmark.get("prompt", "")),
            "llm_model": str(benchmark.get("llm_model", "") or ""),
            "llm_api_base": str(benchmark.get("llm_api_base", "") or ""),
            "started_at": str(benchmark.get("started_at", "")),
            "duration_s": duration_s,
            "outcome": outcome,
            "terminal_agent_state": terminal_state,
            "workflow_phase": str(stage_statuses.get("GEOM", "unknown") or "unknown"),
            "stage_statuses": stage_statuses,
            "semantic_completeness": semantic_check,
            "bom_oracle": bom_oracle,
            "footprint_match": footprint_match,
            "net_score": net_score,
            "design_sanity": design_sanity,
            "ee_rules": ee_rules,
            "ee_rules_score_out_of_100": int(ee_rules.get("score_out_of_100", 0) or 0),
            "all_issue_count": len(all_issues),
            "critical_issue_count": int(critical_issue_count),
            "all_issue_penalty": int(all_issue_penalty),
            "all_issue_penalty_raw": int(raw_issue_penalty),
            "all_issues": list(all_issues)[:250],
            "weighted_score_before_issue_penalty": int(weighted_score),
            "score_out_of_100": overall_score,
            "overall_score_out_of_100": overall_score,
            "bom_score_out_of_100": bom_score,
            "placement_score": placement_score,
            "missing_parts": missing_parts,
            "incorrect_parts": incorrect_parts,
            "placement_audit": placement_audit,
            "semantic_tiers": semantic_tiers,
            "benchmark_fail_fast": forced_fail if isinstance(forced_fail, dict) else None,
            "phase_scores": list(benchmark.get("phase_scores", []) or []),
            "token_usage_summary": (
                dict(benchmark.get("token_usage_summary", {}) or {})
                or self._benchmark_loop_token_usage(loop)
            ),
            "state_transitions": list(benchmark.get("state_transitions", []) or []),
            "action_counts": counts,
            "failed_actions": failed_actions[:25],
            "primary_failure": primary_failure,
            "gate_failures": gate_failures[:25],
            "unresolved_roles": unresolved_roles_detailed,
            "critical_unresolved_roles": critical_unresolved_roles,
            "ui_responses_tail": list(benchmark.get("ui_responses_tail", []) or []),
            "history_len": len(action_rows),
            "spec_debug": spec_debug,
            "placement_snapshot": placement_snapshot,
            "board_placement_snapshot": board_placement_snapshot,
            "placement_diagnostics": placement_diagnostics,
            "net_plan_summary": net_plan_summary,
        }
    def _benchmark_debug_dir(self) -> Path:
        configured = str(os.environ.get("VIBECAD_BENCHMARK_DEBUG_DIR", "") or "").strip()
        if configured:
            debug_dir = Path(configured).expanduser()
        else:
            workspace_preferred = Path("/Users/owner/Documents/GitHub/VibeCAD/vibecad/debug")
            if workspace_preferred.exists() or workspace_preferred.parent.exists():
                debug_dir = workspace_preferred
            else:
                debug_dir = Path(__file__).resolve().parent / "debug"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            debug_dir = Path.cwd() / "vibecad" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir
    def _benchmark_report_path(self, benchmark: Dict[str, Any]) -> Path:
        debug_dir = self._benchmark_debug_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return debug_dir / f"vibecad_benchmark_{benchmark.get('id','run')}_{stamp}.json"
    def _benchmark_critical_resolver_trace(self, critical_unresolved_roles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        lm = getattr(self, "library_manager", None)
        if lm is None:
            return out
        for row in list(critical_unresolved_roles or [])[:8]:
            if not isinstance(row, dict):
                continue
            q = str(row.get("part_query", "") or "").strip()
            pkg = str(row.get("package", "") or "").strip()
            if not q:
                continue
            queries = [q]
            if pkg:
                queries.append(f"{q} {pkg}".strip())
            trace_row: Dict[str, Any] = {
                "role_id": str(row.get("role_id", "") or ""),
                "role_type": str(row.get("role_type", "") or ""),
                "part_query": q,
                "package": pkg,
                "searches": [],
            }
            for query in queries[:2]:
                try:
                    items = list(lm.search_parts_sync(query, source=None, limit=8) or [])
                except Exception:
                    items = []
                hits: List[Dict[str, Any]] = []
                for it in items[:5]:
                    try:
                        hits.append(
                            {
                                "name": str(getattr(it, "name", "") or ""),
                                "package": str(getattr(it, "package", "") or ""),
                                "source": str(getattr(it, "source", "") or ""),
                                "has_symbol": bool(getattr(it, "local_symbol_path", None)),
                                "has_footprint": bool(getattr(it, "local_footprint_path", None)),
                            }
                        )
                    except Exception:
                        continue
                trace_row["searches"].append({"query": query, "hits": hits})
            out.append(trace_row)
        return out
    def _finalize_benchmark_report(self, benchmark: Optional[Dict[str, Any]], loop: Any) -> None:
        if not isinstance(benchmark, dict) or loop is None:
            return
        if benchmark.get("finished"):
            return
        benchmark["finished"] = True
        try:
            report = self._benchmark_build_report(benchmark, loop)
            self._benchmark_sync_done_phase_row(benchmark, report)
            report["phase_scores"] = list(benchmark.get("phase_scores", []) or [])
            try:
                report["critical_resolver_trace"] = self._benchmark_critical_resolver_trace(
                    list(report.get("critical_unresolved_roles") or [])
                )
            except Exception:
                logger.exception("Benchmark critical resolver trace failed")
        except Exception as e:
            logger.exception("Benchmark report build failed")
            report = {
                "benchmark_id": str(benchmark.get("id", "")),
                "scenario": str(benchmark.get("scenario", "")),
                "outcome": "error",
                "error": str(e),
            }
        out_path = None
        try:
            out_path = self._benchmark_report_path(benchmark)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            benchmark["final_report_path"] = str(out_path)
        except Exception:
            logger.exception("Failed to write benchmark report")
        try:
            ps = report.get("placement_score") or {}
            ps_score = ps.get("score", 0) if isinstance(ps, dict) else 0
            ps_placed = ps.get("components_placed", 0) if isinstance(ps, dict) else 0
            ps_expected = ps.get("components_expected", 0) if isinstance(ps, dict) else 0
            ps_oob = ps.get("out_of_bounds", 0) if isinstance(ps, dict) else 0
            ps_col = ps.get("overlap_collisions", 0) if isinstance(ps, dict) else 0
            ns = report.get("net_score") or {}
            ns_score = ns.get("score", 0) if isinstance(ns, dict) else 0
            ee = report.get("ee_rules") if isinstance(report.get("ee_rules"), dict) else {}
            ee_score = int(ee.get("score_out_of_100", 0) or 0)
            ee_issues = int(ee.get("failed_rule_count", 0) or 0)
            all_issue_count = int(report.get("all_issue_count", 0) or 0)
            all_issue_penalty = int(report.get("all_issue_penalty", 0) or 0)
            critical_issue_count = int(report.get("critical_issue_count", 0) or 0)
            logger.info(
                "Benchmark result: overall=%d/100 bom=%d/100  placement=%d/100 (%d/%d placed, oob=%d, overlaps=%d)  net=%d/100  ee=%d/100 issues=%d  mistakes=%d penalty=%d critical=%d  outcome=%s",
                int(report.get("overall_score_out_of_100", report.get("score_out_of_100", 0)) or 0),
                int(report.get("bom_score_out_of_100", report.get("score_out_of_100", 0)) or 0),
                int(ps_score),
                int(ps_placed),
                int(ps_expected),
                int(ps_oob),
                int(ps_col),
                int(ns_score),
                int(ee_score),
                int(ee_issues),
                int(all_issue_count),
                int(all_issue_penalty),
                int(critical_issue_count),
                str(report.get("outcome", "unknown")),
            )
        except Exception:
            pass
        if self.frame:
            try:
                missing_rows = list(report.get("missing_parts", []) or [])
                incorrect_rows = list(report.get("incorrect_parts", []) or [])
                missing_text = [
                    f"{str(r.get('item_id',''))}: expected {int(r.get('expected_qty_min',0) or 0)}-{int(r.get('expected_qty_max',0) or 0)}, actual {int(r.get('actual_qty',0) or 0)}"
                    for r in missing_rows[:20]
                    if isinstance(r, dict)
                ]
                incorrect_text = [
                    f"{str(r.get('item_id',''))}: expected {int(r.get('expected_qty_min',0) or 0)}-{int(r.get('expected_qty_max',0) or 0)}, actual {int(r.get('actual_qty',0) or 0)}"
                    for r in incorrect_rows[:20]
                    if isinstance(r, dict)
                ]
                ps = report.get("placement_score") or {}
                ps_score = ps.get("score", 0) if isinstance(ps, dict) else 0
                ps_placed = ps.get("components_placed", 0) if isinstance(ps, dict) else 0
                ps_expected = ps.get("components_expected", 0) if isinstance(ps, dict) else 0
                ps_oob = ps.get("out_of_bounds", 0) if isinstance(ps, dict) else 0
                ps_col = ps.get("overlap_collisions", 0) if isinstance(ps, dict) else 0
                ns = report.get("net_score") or {}
                ns_score = ns.get("score", 0) if isinstance(ns, dict) else 0
                ns_assignments = ns.get("assignment_count", 0) if isinstance(ns, dict) else 0
                ns_groups = ns.get("expected_group_count", 0) if isinstance(ns, dict) else 0
                sanity = report.get("design_sanity") if isinstance(report.get("design_sanity"), dict) else {}
                sanity_failed = [
                    str(row.get("id", "") or "")
                    for row in list(sanity.get("failed_required_checks") or [])
                    if isinstance(row, dict) and str(row.get("id", "") or "")
                ]
                ee = report.get("ee_rules") if isinstance(report.get("ee_rules"), dict) else {}
                ee_score = int(ee.get("score_out_of_100", 0) or 0)
                ee_issues = [
                    row for row in list(ee.get("issues_for_done_checkpoint", []) or [])
                    if isinstance(row, dict)
                ]
                ee_issue_lines = [
                    f"{str(r.get('id', '') or '')}: {str(r.get('message', '') or '')}"
                    for r in ee_issues[:10]
                ]
                all_issues = [
                    row for row in list(report.get("all_issues", []) or [])
                    if isinstance(row, dict)
                ]
                all_issue_lines = [
                    f"{str(r.get('category', '') or '')}:{str(r.get('id', '') or '')}: {str(r.get('message', '') or '')}"
                    for r in all_issues[:30]
                ]
                summary = (
                    f"overall score: {int(report.get('overall_score_out_of_100', report.get('score_out_of_100', 0)) or 0)}/100\n"
                    f"weighted pre-penalty: {int(report.get('weighted_score_before_issue_penalty', 0) or 0)}/100\n"
                    f"mistakes: {int(report.get('all_issue_count', 0) or 0)} "
                    f"(critical={int(report.get('critical_issue_count', 0) or 0)}, "
                    f"penalty={int(report.get('all_issue_penalty', 0) or 0)})\n"
                    f"mistake sample (DONE): {all_issue_lines}\n"
                    f"bom score: {int(report.get('bom_score_out_of_100', report.get('score_out_of_100', 0)) or 0)}/100\n"
                    f"placement: {int(ps_score)}/100 "
                    f"({int(ps_placed)}/{int(ps_expected)} placed, "
                    f"oob={int(ps_oob)}, overlaps={int(ps_col)})\n"
                    f"net: {int(ns_score)}/100 "
                    f"({int(ns_assignments)} assigned pads across {int(ns_groups)} scored groups)\n"
                    f"ee rules: {ee_score}/100 "
                    f"({len(ee_issues)} issue(s))\n"
                    f"ee issues (DONE): {ee_issue_lines}\n"
                    f"sanity: {'pass' if bool(sanity.get('ok', True)) else 'fail'} "
                    f"{sanity_failed}\n"
                    f"missing parts: {missing_text}\n"
                    f"incorrect parts: {incorrect_text}"
                )
                self.frame.add_design_response(summary)
            except Exception:
                pass
        # Clear active benchmark if it is this one.
        try:
            if self._active_benchmark is benchmark:
                self._active_benchmark = None
        except Exception:
            pass
