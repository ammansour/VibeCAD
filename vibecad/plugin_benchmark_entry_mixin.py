"""Benchmark entrypoint and preflight helpers for VibeCAD plugin."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .config.settings import default_settings_path

try:
    import wx

    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False

logger = logging.getLogger("vibecad")


class BenchmarkEntryMixin:
    """Start benchmark runs and manage early benchmark diagnostics."""

    def _on_run_benchmark(self):
        """Run a fixed end-to-end benchmark scenario with structured diagnostics."""
        if not WX_AVAILABLE or self.frame is None:
            return
        if self._active_benchmark and not bool(self._active_benchmark.get("finished")):
            self.frame.add_design_response("⚠️ Benchmark already running.")
            return
        if self._agent_loop and self._agent_loop.is_running:
            self.frame.add_design_response("⚠️ Stop/pause the current run before starting a benchmark.")
            return

        llm_model = ""
        llm_api_base = ""
        try:
            if self.llm_client is not None:
                cfg = getattr(self.llm_client, "config", None)
                llm_model = str(getattr(cfg, "model", "") or "")
                llm_api_base = str(getattr(cfg, "api_base", "") or "")
        except Exception:
            pass
        bench = {
            "id": f"uno_r3_{int(time.time())}",
            "scenario": "Arduino Uno R3 v4",
            "prompt": (
                "Build a complete Arduino Uno R3-compatible design using the v4 workflow. "
                "Resolve real parts from available evidence, include support passives/protection/clock/power, "
                "and avoid hardcoded board templates."
            ),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "start_ts": time.time(),
            "state_transitions": [],
            "phase_scores": [],
            "pending_phase_scores": [],
            "token_usage_summary": {},
            "ui_responses_tail": [],
            "auto_clarify_attempts": 0,
            "auto_clarify_budget": 12,
            "bom_only_mode": False,
            # Strict benchmark profile: fail early on weak/under-specified runs.
            "force_proceed_all_phases": False,
            # Temporary debug mode: evaluate fail-fast gates but warn instead of stopping.
            "fail_fast_warn_only": True,
            "strict_semantic_gate_before_geom": True,
            "strict_semantic_gate_include_secondary": True,
            "strict_bom_oracle_before_geom": True,
            "ignore_placement_drc_in_score": False,
            "strict_require_schematic": True,
            "strict_require_ground_plane": True,
            "strict_require_clock_placement": True,
            # Generalizable benchmark check profile. Keep this data-driven so
            # audits can adapt without embedding board-specific literals.
            "generalizable_checks_profile": {
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
            },
            # Keep benchmark feedback focused on SPEC/PLACE readiness before NET is implemented.
            "defer_net_dependent_ee_until_routed": True,
            "defer_net_issue_reporting_until_routed": True,
            "llm_model": llm_model,
            "llm_api_base": llm_api_base,
            "finished": False,
            "final_report_path": "",
        }

        preflight = self._benchmark_empty_project_payload(reload_data=True)
        if isinstance(preflight, dict):
            bench["preflight_warning"] = preflight

        self._active_benchmark = bench
        try:
            owner_mod = sys.modules.get(type(self).__module__)
            owner_file = str(getattr(owner_mod, "__file__", __file__) or __file__)
            self.frame.add_design_response("🧪 Running benchmark: Arduino Uno R3 (SPEC + PLACE + NET).")
            self.frame.add_design_response(
                "ℹ️ Benchmark runtime paths:\n"
                f"module: {str(Path(owner_file).resolve())}\n"
                f"settings: {str(default_settings_path())}\n"
                f"debug_dir: {str(self._benchmark_debug_dir())}"
            )
            if isinstance(preflight, dict):
                details = preflight.get("details") if isinstance(preflight.get("details"), dict) else {}
                self.frame.add_design_response(
                    "⚠️ Benchmark preflight: project appears empty at start, continuing with from-scratch synthesis.\n"
                    f"pcb refs={int(details.get('pcb_ref_count', 0) or 0)}, "
                    f"schematic components={int(details.get('schematic_component_count', 0) or 0)}."
                )
        except Exception:
            pass
        self._start_agent_loop(str(bench["prompt"]), benchmark=bench)

    def _benchmark_empty_project_payload(self, *, reload_data: bool = False) -> Optional[Dict[str, Any]]:
        """Detect clearly-empty benchmark projects and return a fail payload."""
        if reload_data:
            try:
                self._load_pcb_data()
            except Exception:
                logger.debug("Benchmark preflight: PCB load failed", exc_info=True)
            try:
                self._load_schematic_data()
            except Exception:
                logger.debug("Benchmark preflight: schematic load failed", exc_info=True)

        schematic = self._benchmark_schematic_audit()
        board_snapshot = self._benchmark_board_placement_snapshot()
        pcb_ref_count = len(board_snapshot) if isinstance(board_snapshot, dict) else 0
        sch_component_count = int(schematic.get("component_count", 0) or 0)

        pcb_outline_count = 0
        pcb_net_count = 0
        if self.pcb_data is not None:
            try:
                pcb_outline_count = int(getattr(self.pcb_data, "board_outline_element_count", 0) or 0)
            except Exception:
                pcb_outline_count = 0
            try:
                pcb_net_count = len(list(getattr(self.pcb_data, "nets", []) or []))
            except Exception:
                pcb_net_count = 0

        project_looks_empty = (pcb_ref_count <= 0) and (sch_component_count <= 0)
        if not project_looks_empty:
            return None

        return {
            "gate": "BENCHMARK.preflight_empty_project",
            "message": "Benchmark preflight detected empty project start (no PCB footprints and no schematic components)",
            "details": {
                "pcb_ref_count": int(pcb_ref_count),
                "pcb_net_count": int(pcb_net_count),
                "pcb_outline_element_count": int(pcb_outline_count),
                "schematic_loaded": bool(schematic.get("loaded", False)),
                "schematic_component_count": int(sch_component_count),
                "schematic_wire_count": int(schematic.get("wire_count", 0) or 0),
                "schematic_net_label_count": int(schematic.get("net_label_count", 0) or 0),
                "companion_path": str(schematic.get("companion_path", "") or ""),
                "companion_exists": bool(schematic.get("companion_exists", False)),
                "schematic_reason": str(schematic.get("reason", "") or ""),
            },
            "bounce_to": "SPEC",
        }

    def _benchmark_emit_preflight_abort(self, benchmark: Dict[str, Any], payload: Dict[str, Any]) -> None:
        """Show a clear chat message and persist a tiny benchmark report for preflight aborts."""
        out_path = None
        report = {
            "benchmark_id": str(benchmark.get("id", "") or ""),
            "scenario": str(benchmark.get("scenario", "") or ""),
            "prompt": str(benchmark.get("prompt", "") or ""),
            "started_at": str(benchmark.get("started_at", "") or ""),
            "duration_s": 0.0,
            "outcome": "preflight_abort",
            "terminal_agent_state": "NOT_STARTED",
            "score_out_of_100": 0,
            "overall_score_out_of_100": 0,
            "weighted_score_before_issue_penalty": 0,
            "all_issue_count": 1,
            "critical_issue_count": 1,
            "all_issue_penalty": 0,
            "all_issue_penalty_raw": 0,
            "all_issues": [
                {
                    "category": "preflight",
                    "id": str(payload.get("gate", "") or "preflight_abort"),
                    "message": str(payload.get("message", "") or "Benchmark preflight aborted"),
                    "stage": "SPEC",
                    "critical": True,
                    "score_penalty": 0,
                    "details": payload.get("details", {}),
                }
            ],
            "stage_statuses": {
                "SPEC": "blocked",
                "RESOLVE": "blocked",
                "IMPORT": "blocked",
                "GEOM": "blocked",
                "NET": "blocked",
                "BIND": "blocked",
                "DRC": "blocked",
            },
            "primary_failure": payload,
            "benchmark_fail_fast": payload,
            "phase_scores": [],
            "token_usage_summary": {},
            "state_transitions": [],
            "action_counts": {},
            "failed_actions": [],
            "gate_failures": [],
            "unresolved_roles": [],
            "critical_unresolved_roles": [],
            "history_len": 0,
        }
        try:
            out_path = self._benchmark_report_path(benchmark)
            out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.exception("Failed writing preflight benchmark report")

        details = payload.get("details") if isinstance(payload.get("details"), dict) else {}
        try:
            if self.frame:
                self.frame.add_design_response("🧪 Running benchmark: Arduino Uno R3 (SPEC + PLACE + NET).")
                self.frame.add_design_response(
                    "❌ Benchmark aborted in preflight: this project is empty for benchmark scoring.\n"
                    f"pcb refs={int(details.get('pcb_ref_count', 0) or 0)}, "
                    f"schematic components={int(details.get('schematic_component_count', 0) or 0)}.\n"
                    "Load a populated PCB/schematic and run benchmark again."
                )
                if out_path is not None:
                    self.frame.add_design_response(f"📝 Preflight report saved: {str(out_path)}")
        except Exception:
            pass

    def _benchmark_auto_clarify_reply(self, benchmark: Optional[Dict[str, Any]]) -> Optional[str]:
        """Return a benchmark-only clarification reply, or None if budget exhausted."""
        if not isinstance(benchmark, dict):
            return None
        attempts = int(benchmark.get("auto_clarify_attempts", 0) or 0)
        budget = int(benchmark.get("auto_clarify_budget", 0) or 0)
        force_proceed = bool(benchmark.get("force_proceed_all_phases", False))
        if attempts >= budget and (not force_proceed):
            return None
        benchmark["auto_clarify_attempts"] = attempts + 1
        # Keep this narrow and generic: use built-in KiCad-compatible substitutes.
        return json.dumps(
            {
                "prefer_builtin_generic": True,
                "allow_generic_substitutes": True,
                "proceed_with_defaults": True,
                "do_not_block_on_clarification": True,
            },
            separators=(",", ":"),
        )
