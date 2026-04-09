from __future__ import annotations

import json
import itertools
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import SubAgent, SubAgentResult

logger = logging.getLogger(__name__)

_NET_SKIP_NAMES = frozenset(
    {"", "NC", "N/C", "NO_CONNECT", "UNCONNECTED", "NOT_CONNECTED", "NO_NET", "NONET", "NONE"}
)

_SYSTEM_PROMPT = """You are the NET subagent for a PCB design tool.

Your job is to assign PCB pad nets for an already-placed board using the supplied
manifest, support-BOM hints, and live board pad list.

Return ONLY strict JSON:
{
  "assistant_message": "short status",
  "assignments": [
    {"ref": "U1", "pad": "1", "net": "NET_A"}
  ]
}

Rules:
- Use only refs and pad identifiers that appear in BOARD_COMPONENTS.
- Prefer the manifest pin nets when they are available.
- When BOARD_COMPONENTS provides support_candidates for a part, treat those as the
  authoritative narrowed candidate set for that secondary component and prefer
  their owner/net hints over free-form guessing.
- Add support-part assignments only when the circuit intent is clear from the
  goal, support hints, nearby primaries, and common application-circuit practice.
- If uncertain, omit the assignment instead of guessing.
- Every pad may have only one net assignment.
- Reuse stable schematic-like net names from the supplied data when possible.
- Do not output prose outside the JSON object.
"""


class NetAssignAgent(SubAgent):
    NAME = "net"
    SYSTEM_PROMPT = _SYSTEM_PROMPT

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        self.name = "NET"
        self.description = "Assigns pad nets from manifest/spec intent and then autoroutes"
        self._seed_skip_logged: set[Tuple[str, str, str]] = set()

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        manifest = context.get("manifest") or context.get("artifacts", {}).get("manifest", {})
        spec_debug = context.get("spec_debug") or context.get("artifacts", {}).get("spec_debug", {})
        placement_plan = context.get("placement_plan") or context.get("artifacts", {}).get("placement_plan", {})
        quality_constraints = context.get("quality_constraints") or context.get("artifacts", {}).get("quality_constraints", {})
        board_snapshot = board_snapshot or {}

        if board_snapshot.get("routing_attempted"):
            existing_plan = context.get("net_plan")
            if not isinstance(existing_plan, dict):
                existing_plan = context.get("artifacts", {}).get("net_plan", {})
            if not isinstance(existing_plan, dict):
                existing_plan = {}
            preserved_plan = dict(existing_plan) if existing_plan else {"assignments": []}
            preserved_plan.setdefault("assignments", [])
            return SubAgentResult(
                message="NET phase complete: Autorouting has already been executed.",
                confidence=1.0,
                phase_complete=True,
                artifacts={"net_plan": preserved_plan}
            )

        board_components = self._board_components(
            board_snapshot=board_snapshot,
            manifest=manifest if isinstance(manifest, dict) else {},
            placement_plan=placement_plan if isinstance(placement_plan, dict) else {},
        )
        if not board_components:
            return SubAgentResult(
                message="NET skipped: no placed footprints with visible pad data were found.",
                confidence=0.2,
                phase_complete=True,
                thinking="board_components=0",
                artifacts={
                    "net_plan": {
                        "assignment_count": 0,
                        "unique_net_count": 0,
                        "assignments": [],
                        "warnings": ["no_board_components"],
                    }
                },
            )

        board_index = self._board_index(board_components)
        manifest_dict = manifest if isinstance(manifest, dict) else {}
        net_aliases = self._manifest_pin_net_alias_map(manifest_dict)
        llm_alias_warning = ""
        if self._llm_available():
            try:
                llm_aliases = self._augment_net_aliases_with_llm(
                    manifest=manifest_dict,
                    base_aliases=net_aliases,
                )
                if llm_aliases:
                    net_aliases.update(llm_aliases)
            except Exception as e:
                llm_alias_warning = f"llm_net_alias_failed:{e}"
                logger.warning("NET LLM alias inference failed: %s", e)
        seeded_assignments = self._seed_assignments_from_manifest(
            manifest_dict,
            board_index,
            net_aliases=net_aliases,
        )
        deterministic_assignments = self._deterministic_support_assignments(
            manifest=manifest_dict,
            board_index=board_index,
            base_assignments=seeded_assignments,
            net_aliases=net_aliases,
        )
        deterministic_assignment_count = len(deterministic_assignments)
        if deterministic_assignments:
            seeded_assignments = list(seeded_assignments) + list(deterministic_assignments)
        else:
            seeded_assignments = list(seeded_assignments)

        llm_assignments: List[Dict[str, str]] = []
        llm_message = ""
        llm_warning = ""
        if self._llm_available():
            try:
                llm_assignments, llm_message = self._plan_support_assignments_with_llm(
                    goal=goal,
                    manifest=manifest if isinstance(manifest, dict) else {},
                    spec_debug=spec_debug if isinstance(spec_debug, dict) else {},
                    placement_plan=placement_plan if isinstance(placement_plan, dict) else {},
                    board_components=board_components,
                    seeded_assignments=seeded_assignments,
                    quality_constraints=quality_constraints if isinstance(quality_constraints, dict) else {},
                )
            except Exception as e:
                llm_warning = f"llm_support_assignment_failed:{e}"
                logger.warning("NET LLM completion failed: %s", e)

        assignments, stats = self._merge_assignments(
            seeded_assignments=seeded_assignments,
            llm_assignments=llm_assignments,
            board_index=board_index,
            net_aliases=net_aliases,
        )
        coverage_before_topology = self._coverage_summary(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
        )
        bridge_lint_before_topology = self._bridge_connectivity_lint(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
            net_aliases=net_aliases,
        )

        topology_message = ""
        topology_warning = ""
        topology_assignments_added = 0
        topology_reconcile_ran = False
        final_check_message = ""
        final_check_warning = ""
        final_check_assignments_added = 0
        final_check_ran = False
        topology_blocker_count = (
            int(coverage_before_topology.get("refs_without_nets_count", 0) or 0)
            + int(coverage_before_topology.get("partial_refs_count", 0) or 0)
            + int(bridge_lint_before_topology.get("issue_count", 0) or 0)
        )
        if self._llm_available() and topology_blocker_count > 0:
            topology_reconcile_ran = True
            try:
                topology_assignments, topology_message = self._plan_topology_reconcile_with_llm(
                    goal=goal,
                    manifest=manifest_dict,
                    board_components=board_components,
                    board_index=board_index,
                    placement_plan=placement_plan if isinstance(placement_plan, dict) else {},
                    current_assignments=assignments,
                    coverage=coverage_before_topology,
                    bridge_lint=bridge_lint_before_topology,
                    net_aliases=net_aliases,
                    quality_constraints=quality_constraints if isinstance(quality_constraints, dict) else {},
                )
            except Exception as e:
                topology_assignments = []
                topology_warning = f"llm_topology_reconcile_failed:{e}"
                logger.warning("NET topology reconcile failed: %s", e)
            if topology_assignments:
                assignments, topo_stats = self._merge_assignments(
                    seeded_assignments=assignments,
                    llm_assignments=topology_assignments,
                    board_index=board_index,
                    net_aliases=net_aliases,
                )
                topology_assignments_added = int(topo_stats.get("llm_assignment_count", 0) or 0)
                stats["llm_assignment_count"] = int(stats.get("llm_assignment_count", 0) or 0) + topology_assignments_added
                stats["warnings"] = list(stats.get("warnings", [])) + list(topo_stats.get("warnings", []))

        coverage_after_topology = self._coverage_summary(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
        )
        bridge_lint_after_topology = self._bridge_connectivity_lint(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
            net_aliases=net_aliases,
        )
        unconnected_audit_before_final = self._unconnected_pad_audit_rows(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
            net_aliases=net_aliases,
        )
        required_unconnected_before_final = [
            row for row in unconnected_audit_before_final
            if str(row.get("necessity", "") or "").lower() == "required"
        ]
        if self._llm_available() and required_unconnected_before_final:
            final_check_ran = True
            try:
                final_check_assignments, final_check_message = self._plan_topology_reconcile_with_llm(
                    goal=goal,
                    manifest=manifest_dict,
                    board_components=board_components,
                    board_index=board_index,
                    placement_plan=placement_plan if isinstance(placement_plan, dict) else {},
                    current_assignments=assignments,
                    coverage=coverage_after_topology,
                    bridge_lint=bridge_lint_after_topology,
                    net_aliases=net_aliases,
                    unconnected_audit_rows=unconnected_audit_before_final,
                    focus_label="FINAL_CHECK",
                    quality_constraints=quality_constraints if isinstance(quality_constraints, dict) else {},
                )
            except Exception as e:
                final_check_assignments = []
                final_check_warning = f"llm_final_check_failed:{e}"
                logger.warning("NET final unconnected check failed: %s", e)
            if final_check_assignments:
                assignments, final_stats = self._merge_assignments(
                    seeded_assignments=assignments,
                    llm_assignments=final_check_assignments,
                    board_index=board_index,
                    net_aliases=net_aliases,
                )
                final_check_assignments_added = int(final_stats.get("llm_assignment_count", 0) or 0)
                stats["llm_assignment_count"] = int(stats.get("llm_assignment_count", 0) or 0) + final_check_assignments_added
                stats["warnings"] = list(stats.get("warnings", [])) + list(final_stats.get("warnings", []))

        support_refs_without_nets = self._support_refs_without_nets(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
        )
        bridge_lint = self._bridge_connectivity_lint(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
            net_aliases=net_aliases,
        )
        existing_assigned_pad_count = sum(
            1
            for component in board_components
            for net_name in dict(component.get("pad_nets") or {}).values()
            if self._sanitize_net_name(net_name)
        )
        has_routable_nets = bool(assignments) or existing_assigned_pad_count > 0 or int(board_snapshot.get("nets_assigned", 0) or 0) > 0

        warnings: List[str] = []
        if llm_warning:
            warnings.append(llm_warning)
        if llm_alias_warning:
            warnings.append(llm_alias_warning)
        if topology_warning:
            warnings.append(topology_warning)
        if final_check_warning:
            warnings.append(final_check_warning)
        warnings.extend(stats.get("warnings", []))

        unique_nets = sorted({str(item.get("net", "") or "") for item in assignments if item.get("net")})
        coverage = self._coverage_summary(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
        )
        unconnected_audit = self._unconnected_pad_audit_rows(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
            net_aliases=net_aliases,
        )
        unconnected_required = [
            row for row in unconnected_audit
            if str(row.get("necessity", "") or "").lower() == "required"
        ]
        unconnected_optional = [
            row for row in unconnected_audit
            if str(row.get("necessity", "") or "").lower() != "required"
        ]
        routing_ready, routing_blocker = self._routing_readiness(
            manifest=manifest_dict,
            board_index=board_index,
            assignments=assignments,
            coverage=coverage,
            bridge_lint=bridge_lint,
        )
        actions = self._build_actions(
            assignments=assignments,
            routing_attempted=bool(board_snapshot.get("routing_attempted", False)),
            include_autoroute=has_routable_nets and routing_ready,
        )
        autoroute_queued = any(
            str(getattr(getattr(action, "action_type", None), "name", "") or "") == "AUTOROUTE_BOARD"
            for action in actions
        )
        artifacts = {
            "net_plan": {
                "assignment_count": len(assignments),
                "seed_assignment_count": int(stats.get("seed_assignment_count", 0) or 0),
                "deterministic_assignment_count": int(deterministic_assignment_count),
                "llm_assignment_count": int(stats.get("llm_assignment_count", 0) or 0),
                "topology_reconcile_ran": bool(topology_reconcile_ran),
                "topology_added_assignment_count": int(topology_assignments_added),
                "final_check_ran": bool(final_check_ran),
                "final_check_added_assignment_count": int(final_check_assignments_added),
                "net_alias_count": len(net_aliases),
                "net_aliases_sample": {
                    key: net_aliases[key]
                    for key in sorted(net_aliases.keys())[:40]
                },
                "unique_net_count": len(unique_nets),
                "unique_nets": unique_nets,
                "covered_refs": sorted({str(item.get("ref", "") or "") for item in assignments if item.get("ref")}),
                "coverage": coverage,
                "coverage_before_topology": coverage_before_topology,
                "bridge_lint_before_topology": bridge_lint_before_topology,
                "routing_ready": routing_ready,
                "routing_blocker": routing_blocker,
                "unconnected_audit_count": len(unconnected_audit),
                "unconnected_required_count": len(unconnected_required),
                "unconnected_optional_count": len(unconnected_optional),
                "unconnected_audit_sample": unconnected_audit[:120],
                "support_refs_without_nets_count": len(support_refs_without_nets),
                "support_refs_without_nets_sample": support_refs_without_nets[:40],
                "bridge_lint": bridge_lint,
                "assignments": assignments,
                "warnings": warnings,
            }
        }

        message = llm_message.strip() if llm_message else ""
        if topology_message:
            message = (f"{message} | " if message else "") + f"[TOPOLOGY] {topology_message.strip()}"
        if final_check_message:
            message = (f"{message} | " if message else "") + f"[FINAL_CHECK] {final_check_message.strip()}"
        if not message:
            if assignments:
                if routing_ready:
                    message = (
                        f"NET ready: prepared {len(assignments)} pad assignments "
                        f"across {len(unique_nets)} nets and queued Freerouting."
                    )
                else:
                    message = (
                        f"NET partial: prepared {len(assignments)} pad assignments "
                        f"across {len(unique_nets)} nets, but deferred Freerouting."
                    )
            elif has_routable_nets:
                if routing_ready:
                    message = "NET ready: existing pad nets were detected, so Freerouting was queued without new assignments."
                else:
                    message = "NET partial: existing pad nets were detected, but Freerouting was deferred until coverage improves."
            else:
                message = "NET needs another pass: no valid routable net assignments could be inferred yet."
        if has_routable_nets:
            zero_ref_count = int(coverage.get("refs_without_nets_count", 0) or 0)
            partial_ref_count = int(coverage.get("partial_refs_count", 0) or 0)
            if zero_ref_count or partial_ref_count:
                message += (
                    f" Coverage note: {zero_ref_count} ref(s) still have no netted required pads and "
                    f"{partial_ref_count} ref(s) remain partial on required electrical pads because "
                    "manifest pin mapping or support-part intent was incomplete."
                )
        if has_routable_nets and (not routing_ready) and routing_blocker:
            message += f" Routing deferred: {routing_blocker}."
        if support_refs_without_nets:
            message += (
                f" Support note: {len(support_refs_without_nets)} support ref(s) still have no netted pads "
                f"(sample: {', '.join(support_refs_without_nets[:6])})."
            )
        if int(bridge_lint.get("issue_count", 0) or 0) > 0:
            refs_sample = [
                str(row.get("ref", "") or "")
                for row in list(bridge_lint.get("issues_sample") or [])
                if isinstance(row, dict) and str(row.get("ref", "") or "").strip()
            ]
            if refs_sample:
                message += (
                    f" Bridge-lint note: {int(bridge_lint.get('issue_count', 0) or 0)} issue(s) "
                    f"(sample refs: {', '.join(refs_sample[:6])})."
                )

        confidence = 0.85 if routing_ready else (0.6 if has_routable_nets else 0.35)
        thinking = (
            f"board_components={len(board_components)} "
            f"seed={int(stats.get('seed_assignment_count', 0) or 0)} "
            f"deterministic={int(deterministic_assignment_count)} "
            f"llm={int(stats.get('llm_assignment_count', 0) or 0)} "
            f"topology_added={topology_assignments_added} "
            f"final_added={final_check_assignments_added} "
            f"kept={len(assignments)} "
            f"live_assigned={existing_assigned_pad_count} "
            f"unique_nets={len(unique_nets)} "
            f"unconnected_required={len(unconnected_required)} "
            f"routing_ready={int(routing_ready)}"
        )
        logger.info(
            "NET: %d board refs, %d assignments across %d nets, live_assigned=%d, zero_refs=%d, partial_refs=%d, unconnected_required=%d, routing_ready=%s, autoroute=%s",
            len(board_components),
            len(assignments),
            len(unique_nets),
            existing_assigned_pad_count,
            int(coverage.get("refs_without_nets_count", 0) or 0),
            int(coverage.get("partial_refs_count", 0) or 0),
            len(unconnected_required),
            "yes" if routing_ready else "no",
            "queued" if autoroute_queued else "skipped",
        )
        return SubAgentResult(
            message=message,
            actions=actions,
            confidence=confidence,
            phase_complete=(routing_ready and not autoroute_queued) or (not actions),
            thinking=thinking,
            artifacts=artifacts,
        )

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))

    @staticmethod
    def _sanitize_net_name(raw: Any) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"[^A-Za-z0-9_+\-./#]", "_", text)
        text = re.sub(r"_+", "_", text)
        text = text.strip("_")
        if text.upper() in _NET_SKIP_NAMES:
            return ""
        return text

    @staticmethod
    def _apply_net_alias(net_name: str, net_aliases: Optional[Dict[str, str]] = None) -> str:
        if not net_name:
            return ""
        if not isinstance(net_aliases, dict) or not net_aliases:
            return net_name
        return str(net_aliases.get(str(net_name).upper(), net_name))

    @staticmethod
    def _net_name_key(raw: Any) -> str:
        text = str(raw or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    @classmethod
    def _net_name_is_ground(cls, net_name: Any) -> bool:
        key = cls._net_name_key(net_name)
        if not key:
            return False
        if key in {"gnd", "agnd", "dgnd", "pgnd", "sgnd", "earth", "chassis"}:
            return True
        tokens = [tok for tok in key.split("_") if tok]
        return "gnd" in tokens

    @classmethod
    def _net_name_is_power(cls, net_name: Any) -> bool:
        key = cls._net_name_key(net_name)
        if not key or cls._net_name_is_ground(key):
            return False
        tokens = [tok for tok in key.split("_") if tok]
        if any(tok in {"vcc", "vdd", "vss", "vref", "vin", "avcc", "dvcc", "iovcc", "iovdd", "vio"} for tok in tokens):
            return True
        if key in {"5v", "5v0", "3v3", "3v30", "plus5v", "plus3v3"}:
            return True
        if re.match(r"^(?:plus)?\d+v\d*$", key):
            return True
        if re.match(r"^v\d+(?:_\d+)?$", key):
            return True
        return False

    def _known_net_names(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        nets: set[str] = set()
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            for pin in list(part.get("pins") or []):
                if not isinstance(pin, dict):
                    continue
                net = self._sanitize_net_name(pin.get("net"))
                net = self._apply_net_alias(net, net_aliases)
                if net:
                    nets.add(net)
        for item in board_index.values():
            live_nets = dict(item.get("pad_nets") or {}) if isinstance(item.get("pad_nets"), dict) else {}
            for net_name in live_nets.values():
                net = self._sanitize_net_name(net_name)
                net = self._apply_net_alias(net, net_aliases)
                if net:
                    nets.add(net)
        for row in assignments:
            net = self._sanitize_net_name(row.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if net:
                nets.add(net)
        return sorted(nets)

    def _support_part_text(self, part: Dict[str, Any]) -> str:
        chunks: List[str] = []
        for key in ("ref", "mpn", "value", "description", "footprint"):
            chunks.append(str(part.get(key, "") or ""))
        for sec in list(part.get("support_candidates") or []):
            if not isinstance(sec, dict):
                continue
            for key in ("family", "implementation", "type", "value", "function", "net_hint", "footprint"):
                chunks.append(str(sec.get(key, "") or ""))
            for net_name in list(sec.get("source_nets") or []):
                chunks.append(str(net_name or ""))
        text = " ".join(chunks).lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _paired_net_candidates(self, net_name: str, *, known_nets_upper: set[str]) -> List[str]:
        net = self._sanitize_net_name(net_name)
        if not net:
            return []
        candidates: List[str] = []
        upper = net.upper()
        if upper.endswith("_P"):
            candidates.append(net[:-2] + "_N")
        if upper.endswith("_N"):
            candidates.append(net[:-2] + "_P")
        if upper.endswith("+"):
            candidates.append(net[:-1] + "-")
        if upper.endswith("-"):
            candidates.append(net[:-1] + "+")
        if "D+" in upper:
            candidates.append(re.sub(r"D\+", "D-", net, flags=re.I))
        if "D-" in upper:
            candidates.append(re.sub(r"D-", "D+", net, flags=re.I))
        out: List[str] = []
        seen: set[str] = set()
        for cand in candidates:
            normalized = self._sanitize_net_name(cand)
            if not normalized:
                continue
            up = normalized.upper()
            if up not in known_nets_upper:
                continue
            if up in seen:
                continue
            seen.add(up)
            out.append(normalized)
        return out

    def _support_signal_net_candidates(
        self,
        *,
        part: Dict[str, Any],
        net_aliases: Optional[Dict[str, str]],
        known_nets: Sequence[str],
    ) -> List[str]:
        known_nets_upper = {str(n).upper() for n in known_nets if str(n).strip()}
        nets: List[str] = []

        def _add(net_name: Any) -> None:
            net = self._sanitize_net_name(net_name)
            net = self._apply_net_alias(net, net_aliases)
            if not net:
                return
            if net not in nets:
                nets.append(net)

        for pin in list(part.get("pins") or []):
            if not isinstance(pin, dict):
                continue
            _add(pin.get("net"))

        for sec in list(part.get("support_candidates") or []):
            if not isinstance(sec, dict):
                continue
            _add(sec.get("net_hint"))
            for net_name in list(sec.get("source_nets") or []):
                _add(net_name)

        if len([n for n in nets if not self._net_name_is_ground(n)]) == 1:
            base_signal = next((n for n in nets if not self._net_name_is_ground(n)), "")
            for pair in self._paired_net_candidates(base_signal, known_nets_upper=known_nets_upper):
                if pair not in nets:
                    nets.append(pair)

        # Prefer known board/manifest net names first for stability.
        nets.sort(
            key=lambda n: (
                0 if str(n).upper() in known_nets_upper else 1,
                0 if self._net_name_is_ground(n) else 1,
                n.upper(),
            )
        )
        return nets

    def _deterministic_support_assignments(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        base_assignments: Sequence[Dict[str, str]],
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Deterministically net unresolved support parts from support_candidates
        without board-specific hardcoding.
        """
        known_nets = self._known_net_names(
            manifest=manifest,
            board_index=board_index,
            assignments=base_assignments,
            net_aliases=net_aliases,
        )
        default_ground = next((n for n in known_nets if self._net_name_is_ground(n)), "")

        assigned_lookup: Dict[Tuple[str, str], str] = {}
        for row in base_assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if ref and pad and net:
                assigned_lookup[(ref, pad)] = net
        for ref, board_item in board_index.items():
            live_map = dict(board_item.get("pad_nets") or {}) if isinstance(board_item.get("pad_nets"), dict) else {}
            for pad, net_name in live_map.items():
                pad_name = self._normalize_pad(pad)
                net = self._sanitize_net_name(net_name)
                net = self._apply_net_alias(net, net_aliases)
                if pad_name and net:
                    assigned_lookup[(ref, pad_name)] = net

        out: List[Dict[str, str]] = []
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = self._normalize_ref(part.get("ref"))
            if not ref or ref not in board_index:
                continue
            support_candidates = [
                row for row in list(part.get("support_candidates") or [])
                if isinstance(row, dict)
            ]
            if not support_candidates:
                continue

            pins = [row for row in list(part.get("pins") or []) if isinstance(row, dict)]
            has_manifest_pin_nets = any(self._sanitize_net_name(pin.get("net")) for pin in pins)
            if has_manifest_pin_nets:
                continue

            board_item = board_index.get(ref, {})
            pads = sorted(
                [self._normalize_pad(v) for v in list(board_item.get("pads") or []) if self._normalize_pad(v)],
                key=self._pad_sort_key,
            )
            if len(pads) < 2:
                continue

            unassigned_pads = [pad for pad in pads if (ref, pad) not in assigned_lookup]
            if not unassigned_pads:
                continue

            support_text = self._support_part_text(part)
            is_cap_like = (
                ref.startswith("C")
                or "capacitor" in support_text
                or "decoupling" in support_text
                or "stability" in support_text
                or "bulk" in support_text
            )
            is_clock_like = any(tok in support_text for tok in ("clock", "xtal", "crystal", "resonator"))
            is_protection_like = any(
                tok in support_text
                for tok in ("esd", "tvs", "varistor", "clamp", "suppressor", "protection")
            )

            candidate_nets = self._support_signal_net_candidates(
                part=part,
                net_aliases=net_aliases,
                known_nets=known_nets,
            )
            ground_net = next((n for n in candidate_nets if self._net_name_is_ground(n)), default_ground)
            signal_nets = [n for n in candidate_nets if not self._net_name_is_ground(n)]

            pattern: List[str] = []
            if len(unassigned_pads) == 2:
                if signal_nets and ground_net and (is_cap_like or is_clock_like or is_protection_like):
                    pattern = [signal_nets[0], ground_net]
                elif len(signal_nets) >= 2:
                    pattern = [signal_nets[0], signal_nets[1]]
                elif signal_nets and ground_net:
                    pattern = [signal_nets[0], ground_net]
            else:
                if is_protection_like and signal_nets and ground_net:
                    pattern = list(signal_nets[:2]) + [ground_net]
                elif is_cap_like and signal_nets and ground_net:
                    pattern = [signal_nets[0], ground_net]
                elif len(signal_nets) >= 2:
                    pattern = list(signal_nets[: min(len(signal_nets), 3)])

            if not pattern:
                continue

            for idx, pad in enumerate(unassigned_pads):
                net_name = self._sanitize_net_name(pattern[idx % len(pattern)])
                net_name = self._apply_net_alias(net_name, net_aliases)
                if not net_name:
                    continue
                key = (ref, pad)
                if key in assigned_lookup:
                    continue
                assigned_lookup[key] = net_name
                out.append({"ref": ref, "pad": pad, "net": net_name})

        deduped: List[Dict[str, str]] = []
        seen: set[Tuple[str, str, str]] = set()
        for row in out:
            key = (row["ref"], row["pad"], row["net"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _manifest_pin_rows(self, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = self._normalize_ref(part.get("ref"))
            if not ref:
                continue
            is_connector = ref.startswith("J")
            for pin in list(part.get("pins") or []):
                if not isinstance(pin, dict):
                    continue
                net = self._sanitize_net_name(pin.get("net"))
                if not net:
                    continue
                pin_name = str(
                    pin.get("name")
                    or pin.get("pin_name")
                    or pin.get("label")
                    or ""
                ).strip()
                pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin") or pin.get("number"))
                rows.append(
                    {
                        "ref": ref,
                        "is_connector": is_connector,
                        "net": net.upper(),
                        "pad": pad,
                        "pin_name": pin_name,
                    }
                )
        return rows

    @staticmethod
    def _alias_terminal(net_name: str, aliases: Dict[str, str]) -> str:
        current = str(net_name or "").strip().upper()
        seen: set[str] = set()
        while current and current in aliases and current not in seen:
            seen.add(current)
            current = str(aliases.get(current, "") or "").strip().upper()
        return current

    def _augment_net_aliases_with_llm(
        self,
        *,
        manifest: Dict[str, Any],
        base_aliases: Dict[str, str],
    ) -> Dict[str, str]:
        """
        LLM fallback for unresolved functional connector-singleton aliases.

        This is intentionally constrained and validated: it only permits
        aliases from connector-only function nets (e.g. SCL/SDA/MISO) to
        existing manifest net names that have internal (non-connector) refs.
        """
        pin_rows = self._manifest_pin_rows(manifest)
        if not pin_rows:
            return {}

        nets_upper = sorted({str(row.get("net", "") or "").upper() for row in pin_rows if row.get("net")})
        if not nets_upper:
            return {}
        nets_lookup = set(nets_upper)

        refs_by_net: Dict[str, set[str]] = {}
        internal_ref_count_by_net: Dict[str, int] = {}
        for row in pin_rows:
            net = str(row.get("net", "") or "").upper()
            ref = self._normalize_ref(row.get("ref"))
            if not net or not ref:
                continue
            refs_by_net.setdefault(net, set()).add(ref)
        for net, refs in refs_by_net.items():
            internal_ref_count_by_net[net] = sum(1 for ref in refs if not str(ref).startswith("J"))

        base_aliases_upper = {
            str(k or "").strip().upper(): str(v or "").strip().upper()
            for k, v in dict(base_aliases or {}).items()
            if str(k or "").strip() and str(v or "").strip()
        }

        func_tokens = ("SDA", "SCL", "RX", "TX", "MISO", "MOSI", "SCK", "CLK")
        source_candidates: List[str] = []
        for token in func_tokens:
            net = str(token).upper()
            if net not in nets_lookup:
                continue
            if net in base_aliases_upper:
                continue
            refs = refs_by_net.get(net, set())
            if not refs:
                continue
            if any(not str(ref).startswith("J") for ref in refs):
                continue
            source_candidates.append(net)

        if not source_candidates:
            return {}

        usage_rows = []
        for net in nets_upper:
            refs = sorted(refs_by_net.get(net, set()))
            usage_rows.append(
                {
                    "net": net,
                    "connector_refs": [ref for ref in refs if str(ref).startswith("J")],
                    "internal_refs": [ref for ref in refs if not str(ref).startswith("J")],
                    "pin_count": sum(1 for row in pin_rows if str(row.get("net", "")).upper() == net),
                }
            )

        prompt = (
            "Infer net aliases for connector-only functional nets.\n"
            "Return strict JSON: {\"assistant_message\":\"...\",\"aliases\":[{\"source\":\"SCL\",\"target\":\"A5\",\"confidence\":0.0}]}\n\n"
            "Rules:\n"
            "- Only map from SOURCE_CANDIDATES_JSON.\n"
            "- Target must exist in NET_USAGE_JSON.\n"
            "- Prefer targets with internal_refs (non-connectors).\n"
            "- Do not output uncertain aliases.\n"
            "- Do not output transitive chains when a direct target exists.\n\n"
            "SOURCE_CANDIDATES_JSON:\n"
            + self._compact_json(source_candidates)
            + "\n\nNET_USAGE_JSON:\n"
            + self._compact_json(usage_rows[:320])
            + "\n\nPIN_ROWS_JSON:\n"
            + self._compact_json(pin_rows[:520])
            + "\n\nEXISTING_ALIASES_JSON:\n"
            + self._compact_json(base_aliases_upper)
        )

        raw = self._llm_chat(prompt, system_prompt=self.SYSTEM_PROMPT)
        obj = self._parse_json_object(raw)
        if not isinstance(obj, dict):
            return {}
        proposals = obj.get("aliases")
        if not isinstance(proposals, list):
            return {}

        merged_aliases = dict(base_aliases_upper)
        out: Dict[str, str] = {}
        source_lookup = set(source_candidates)
        for item in proposals:
            if not isinstance(item, dict):
                continue
            source = self._sanitize_net_name(item.get("source")).upper()
            target = self._sanitize_net_name(item.get("target")).upper()
            if not source or not target:
                continue
            if source not in source_lookup:
                continue
            if target not in nets_lookup:
                continue
            if source == target:
                continue
            terminal_target = self._alias_terminal(target, merged_aliases)
            if not terminal_target or terminal_target not in nets_lookup:
                continue
            if terminal_target == source:
                continue
            if int(internal_ref_count_by_net.get(terminal_target, 0) or 0) <= 0:
                continue
            merged_aliases[source] = terminal_target
            out[source] = terminal_target
        return out

    def _manifest_pin_net_alias_map(self, manifest: Dict[str, Any]) -> Dict[str, str]:
        """Build deterministic pin-function aliases (e.g. A4_SDA -> A4) when both
        forms appear in the same manifest pin-net vocabulary."""
        nets_upper: set[str] = set()
        net_pin_counts: Dict[str, int] = {}
        pin_rows: List[Tuple[str, str]] = []
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            for pin in list(part.get("pins") or []):
                if not isinstance(pin, dict):
                    continue
                net = self._sanitize_net_name(pin.get("net"))
                if net:
                    net_upper = net.upper()
                    nets_upper.add(net_upper)
                    net_pin_counts[net_upper] = int(net_pin_counts.get(net_upper, 0) or 0) + 1
                    pin_name = str(
                        pin.get("name")
                        or pin.get("pin_name")
                        or pin.get("label")
                        or ""
                    ).strip().upper()
                    pin_rows.append((net_upper, pin_name))

        aliases: Dict[str, str] = {}
        func_tokens = ("SDA", "SCL", "RX", "TX", "MISO", "MOSI", "SCK", "CLK")
        alias_pattern = re.compile(r"^(A\d+|D\d+)[_\-./]+(SDA|SCL|RX|TX|MISO|MOSI|SCK|CLK)$")
        positional_pattern = re.compile(r"^(A\d+|D\d+)(?:[_\-./]+(SDA|SCL|RX|TX|MISO|MOSI|SCK|CLK))?$")
        for net_upper in sorted(nets_upper):
            m = alias_pattern.match(net_upper)
            if not m:
                continue
            base = m.group(1)
            if base in nets_upper:
                aliases[net_upper] = base

        # Derive function -> positional net aliases from symbol pin names when
        # there is an unambiguous match (e.g., pin name "D0/RX" paired with net
        # names "RX" and "D0_RX" in the same manifest vocabulary)
        function_candidates: Dict[str, set[str]] = {name: set() for name in func_tokens}
        pin_name_token_pattern = re.compile(r"\b(A\d+|D\d+|SDA|SCL|RX|RXD|TX|TXD|MISO|MOSI|SCK|SCLK|CLK)\b")
        pin_name_positional_hint_pattern = re.compile(r"\b(A\d+|D\d+|P[A-Z]\d+)\b")

        def _normalize_function_token(token: str) -> str:
            t = str(token or "").strip().upper()
            if t == "RXD":
                return "RX"
            if t == "TXD":
                return "TX"
            if t == "SCLK":
                return "SCK"
            return t

        def _pin_positional_hints(pin_name_upper: str) -> set[str]:
            hints: set[str] = set()
            for token in pin_name_positional_hint_pattern.findall(str(pin_name_upper or "")):
                tok = str(token or "").strip().upper()
                if not tok:
                    continue
                # Common MCU port naming (e.g. PD0, PA4) can carry positional
                # board net intent without hardcoded board-specific maps.
                m_port = re.match(r"^P([A-Z])(\d+)$", tok)
                if m_port:
                    port = str(m_port.group(1) or "")
                    index = str(m_port.group(2) or "")
                    if port in {"A", "D"}:
                        hints.add(f"{port}{index}")
                    continue
                hints.add(tok)
            return hints

        def _pin_index_prefixes(pin_name_upper: str) -> Dict[str, set[str]]:
            out: Dict[str, set[str]] = {}
            for prefix, idx in re.findall(r"\b([A-Z]+)(\d+)\b", str(pin_name_upper or "")):
                pref = str(prefix or "").strip().upper()
                if not pref:
                    continue
                norm_idx = str(int(idx))
                out.setdefault(norm_idx, set()).add(pref)
            return out

        for net_upper, pin_name_upper in pin_rows:
            positional_match = positional_pattern.match(net_upper)
            if positional_match:
                fn = positional_match.group(2)
                if fn:
                    function_candidates.setdefault(fn, set()).add(net_upper)

            if not pin_name_upper:
                continue
            pin_tokens = {
                _normalize_function_token(tok)
                for tok in pin_name_token_pattern.findall(pin_name_upper)
            }
            net_fn = _normalize_function_token(net_upper)
            if net_fn in func_tokens:
                pin_tokens.add(net_fn)
            positional_hints = {
                hint for hint in _pin_positional_hints(pin_name_upper)
                if hint in nets_upper
            }
            index_prefixes = _pin_index_prefixes(pin_name_upper)
            m_port_net = re.match(r"^P([A-Z])(\d+)$", net_upper)
            if m_port_net:
                port_letter = str(m_port_net.group(1) or "").upper()
                port_index = str(int(m_port_net.group(2)))
                if port_letter not in {"A", "D"}:
                    analog_target = f"A{port_index}"
                    prefixes_same_index = index_prefixes.get(port_index, set())
                    has_cross_label = any(pref != f"P{port_letter}" for pref in prefixes_same_index)
                    has_function_hint = bool(pin_tokens & set(func_tokens))
                    if analog_target in nets_upper and (has_cross_label or has_function_hint):
                        aliases.setdefault(net_upper, analog_target)
            if not pin_tokens:
                continue
            if positional_match:
                for fn in pin_tokens:
                    if fn in function_candidates and fn in func_tokens and fn != net_upper:
                        function_candidates[fn].add(net_upper)
            if positional_hints:
                for fn in pin_tokens:
                    if fn not in function_candidates or fn not in func_tokens:
                        continue
                    for pos in positional_hints:
                        if pos != fn:
                            function_candidates[fn].add(pos)

        for fn in func_tokens:
            if fn not in nets_upper:
                continue
            candidates = sorted(
                n for n in function_candidates.get(fn, set()) if n in nets_upper and n != fn
            )
            if not candidates:
                continue
            explicit = [
                n
                for n in candidates
                if re.search(rf"[_\-./]+{re.escape(fn)}$", n)
            ]
            chosen = ""
            if len(explicit) == 1:
                chosen = explicit[0]
            elif len(candidates) == 1:
                chosen = candidates[0]
            if chosen:
                aliases.setdefault(fn, chosen)

        # Also fold prefixed functional nets (e.g. I2C_SDA) onto the selected
        # function alias target when one exists.
        prefixed_function_pattern = re.compile(
            r"^[A-Z0-9]+[_\-./]+(SDA|SCL|RX|TX|MISO|MOSI|SCK|CLK)$"
        )
        for net_upper in sorted(nets_upper):
            m = prefixed_function_pattern.match(net_upper)
            if not m:
                continue
            fn = m.group(1)
            target = aliases.get(fn, "")
            if target and target != net_upper:
                aliases.setdefault(net_upper, target)

        # Generic logic-reference folding: map IO reference-style singleton nets
        # to the manifest's dominant digital logic rail when evidence is clear.
        logic_supply_pin_pattern = re.compile(
            r"\b(VCC|VDD|AVCC|DVCC|UVCC|VCCIO|VDDIO|IOVCC|IOVDD|VIO)\b"
        )

        def _is_logic_supply_candidate(net_upper: str) -> bool:
            name = str(net_upper or "").strip().upper()
            if not name:
                return False
            if "GND" in name:
                return False
            if re.search(r"\b(VIN|VBAT|BATT|BAT|RAW|VUSB|VBUS|BOOST|SW|LX|PH)\b", name):
                return False
            if re.match(r"^\+?\d+(?:[._]\d+)?V(?:\d+)?$", name):
                return True
            if re.match(r"^V\d+(?:[._]\d+)?$", name):
                return True
            return bool(
                re.match(r"^(VCC|VDD|AVCC|DVCC|UVCC|VCCIO|VDDIO|IOVCC|IOVDD|VIO)$", name)
            )

        logic_votes: Dict[str, int] = {}
        for net_upper, pin_name_upper in pin_rows:
            if not pin_name_upper:
                continue
            if not logic_supply_pin_pattern.search(pin_name_upper):
                continue
            if not _is_logic_supply_candidate(net_upper):
                continue
            logic_votes[net_upper] = int(logic_votes.get(net_upper, 0) or 0) + 1

        dominant_logic_rail = ""
        if logic_votes:
            ranked = sorted(
                logic_votes.items(),
                key=lambda item: (
                    -int(item[1]),
                    -int(net_pin_counts.get(item[0], 0) or 0),
                    str(item[0]),
                ),
            )
            top_vote = int(ranked[0][1])
            tied = [name for name, vote in ranked if int(vote) == top_vote]
            if len(tied) == 1:
                dominant_logic_rail = str(tied[0])

        if dominant_logic_rail:
            for net_upper in sorted(nets_upper):
                if net_upper == dominant_logic_rail:
                    continue
                if int(net_pin_counts.get(net_upper, 0) or 0) > 2:
                    continue
                tokenized = [
                    tok
                    for tok in re.sub(r"[^A-Z0-9]+", "_", net_upper).strip("_").split("_")
                    if tok
                ]
                has_ref = net_upper.endswith("REF") or any(tok == "REF" or tok.endswith("REF") for tok in tokenized)
                has_io = ("IOREF" in net_upper) or any(
                    tok in {"IO", "VIO", "IOVCC", "IOVDD", "VCCIO", "VDDIO"} for tok in tokenized
                )
                if not (has_ref and has_io):
                    continue
                aliases.setdefault(net_upper, dominant_logic_rail)
        return aliases

    @staticmethod
    def _normalize_ref(raw: Any) -> str:
        return str(raw or "").strip().upper()

    @staticmethod
    def _normalize_pad(raw: Any) -> str:
        return str(raw or "").strip()

    @staticmethod
    def _pad_sort_key(pad: str) -> Tuple[int, int, str]:
        text = str(pad or "").strip()
        if re.match(r"^\d+$", text):
            return (0, int(text), text)
        m_alpha_num = re.match(r"^([A-Za-z]+)(\d+)$", text)
        if m_alpha_num:
            return (1, int(m_alpha_num.group(2)), text.upper())
        return (2, 0, text.upper())

    @staticmethod
    def _normalize_pin_token(raw: Any) -> str:
        token = re.sub(r"[^a-z0-9]+", "", str(raw or "").strip().lower())
        if not token:
            return ""
        # Collapse common non-electrical suffixes used by symbols/footprints.
        for suffix in ("thermalpad", "exposedpad", "tab", "pad", "pin"):
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                break
        return token

    def _resolve_pin_name_to_pad_fuzzy(self, board_item: Dict[str, Any], pin_name: str) -> str:
        pin_map = board_item.get("pin_name_to_pad", {}) if isinstance(board_item.get("pin_name_to_pad"), dict) else {}
        if not pin_map:
            return ""
        name = str(pin_name or "").strip()
        if not name:
            return ""
        exact = pin_map.get(name)
        if exact:
            return self._normalize_pad(exact)
        name_l = name.lower()
        for k, v in pin_map.items():
            if str(k).lower() == name_l:
                return self._normalize_pad(v)
        norm_name = self._normalize_pin_token(name)
        if not norm_name:
            return ""
        for k, v in pin_map.items():
            if self._normalize_pin_token(k) == norm_name:
                return self._normalize_pad(v)
        return ""

    def _project_missing_manifest_nets_to_board_pads(
        self,
        *,
        board_item: Dict[str, Any],
        pin_rows: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Deterministically project manifest net intent onto the actual footprint pads
        when manifest pin numbers don't exist on the loaded footprint variant.

        This is intentionally generic: it only triggers when there are missing
        manifest pads and the count of unique expected nets matches the physical
        pad count, so we can map one net per pad without hardcoded part rules.
        """
        board_pads = sorted(
            {
                self._normalize_pad(pad)
                for pad in list(board_item.get("pad_lookup") or set())
                if self._normalize_pad(pad)
            },
            key=self._pad_sort_key,
        )
        if not board_pads:
            return {}
        board_pad_lookup = set(board_pads)

        rows = [dict(row) for row in list(pin_rows or []) if isinstance(row, dict)]
        missing_rows = [
            row
            for row in rows
            if (not bool(row.get("on_footprint"))) and self._sanitize_net_name(row.get("net"))
        ]
        if not missing_rows:
            return {}

        ordered_unique_nets: List[str] = []
        seen_nets: set[str] = set()
        for row in rows:
            net = self._sanitize_net_name(row.get("net"))
            if not net or net in seen_nets:
                continue
            seen_nets.add(net)
            ordered_unique_nets.append(net)

        pad_count = len(board_pads)
        if pad_count < 2 or pad_count > 6:
            return {}
        if len(ordered_unique_nets) != pad_count:
            return {}

        direct_nets_present = {
            self._sanitize_net_name(row.get("net"))
            for row in rows
            if bool(row.get("on_footprint")) and self._sanitize_net_name(row.get("net"))
        }
        missing_nets = set(ordered_unique_nets) - set(direct_nets_present)
        if not missing_nets:
            return {}

        evidence_scores: Dict[Tuple[str, str], int] = {}
        for row in rows:
            net = self._sanitize_net_name(row.get("net"))
            if not net or net not in seen_nets:
                continue
            pad = self._normalize_pad(row.get("pad"))
            if bool(row.get("on_footprint")) and pad in board_pad_lookup:
                evidence_scores[(pad, net)] = int(evidence_scores.get((pad, net), 0) or 0) + 4
                continue
            pin_name = str(row.get("pin_name", "") or "").strip()
            if pin_name:
                hinted_pad = self._resolve_pin_name_to_pad_fuzzy(board_item, pin_name)
                if hinted_pad and hinted_pad in board_pad_lookup:
                    evidence_scores[(hinted_pad, net)] = int(evidence_scores.get((hinted_pad, net), 0) or 0) + 2

        live_pad_nets = (
            dict(board_item.get("pad_nets") or {})
            if isinstance(board_item.get("pad_nets"), dict)
            else {}
        )
        for pad in board_pads:
            live_net = self._sanitize_net_name(live_pad_nets.get(pad))
            if live_net and live_net in seen_nets:
                evidence_scores[(pad, live_net)] = int(evidence_scores.get((pad, live_net), 0) or 0) + 6

        if not evidence_scores:
            return {}

        best_perm: Optional[Tuple[str, ...]] = None
        best_score: Optional[int] = None
        for perm in itertools.permutations(ordered_unique_nets, pad_count):
            score = 0
            for pad, net in zip(board_pads, perm):
                score += int(evidence_scores.get((pad, net), 0) or 0)
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm

        if best_perm is None:
            return {}
        return {pad: net for pad, net in zip(board_pads, best_perm)}

    def _log_seed_skip_once(self, *, reason: str, ref: str, pad: str, pin_name: str, pads: Sequence[str]) -> None:
        key = (str(reason or ""), str(ref or ""), str(pad or pin_name or ""))
        if key in self._seed_skip_logged:
            return
        self._seed_skip_logged.add(key)
        logger.info(
            "SEED SKIP (nonfatal) %s: pad '%s' (pin '%s') not in board pads %s",
            ref,
            pad,
            pin_name or "?",
            sorted(set(str(p) for p in list(pads or []) if str(p)))[:8],
        )

    def _explicit_no_connect_pads(self, part: Dict[str, Any]) -> set[str]:
        out: set[str] = set()
        pins = part.get("pins")
        if not isinstance(pins, list):
            return out
        for pin in pins:
            if not isinstance(pin, dict):
                continue
            pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin"))
            raw_net = str(pin.get("net", "") or "").strip().upper()
            if pad and raw_net in _NET_SKIP_NAMES:
                out.add(pad)
        return out

    def _use_all_physical_pads_for_coverage(self, part: Dict[str, Any]) -> bool:
        ref = self._normalize_ref(part.get("ref"))
        footprint = str(part.get("footprint", "") or "").lower()
        if not ref.startswith("J"):
            return False
        if isinstance(part.get("support_candidates"), list) and part.get("support_candidates"):
            return False
        return "pinheader" in footprint or "pinsocket" in footprint

    def _requires_full_pad_coverage(self, part: Dict[str, Any]) -> bool:
        pins = part.get("pins")
        if isinstance(pins, list) and pins:
            return False
        ref = self._normalize_ref(part.get("ref"))
        footprint = str(part.get("footprint", "") or "").lower()
        # Only apply full physical-pad coverage fallback where the manifest
        # naturally models connectivity by exposed pads (headers/connectors).
        if ref.startswith("J"):
            return True
        if "pinheader" in footprint or "pinsocket" in footprint:
            return True
        return False

    def _board_components(
        self,
        *,
        board_snapshot: Dict[str, Any],
        manifest: Dict[str, Any],
        placement_plan: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        raw_components = list(board_snapshot.get("components") or [])
        if not raw_components:
            return []

        allowed_refs = {
            self._normalize_ref(part.get("ref"))
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and str(part.get("ref", "") or "").strip()
        }
        if isinstance(placement_plan, dict):
            allowed_refs.update(self._normalize_ref(ref) for ref in placement_plan.keys() if str(ref or "").strip())

        out: List[Dict[str, Any]] = []
        for item in raw_components:
            if not isinstance(item, dict):
                continue
            ref = self._normalize_ref(item.get("reference") or item.get("ref"))
            if not ref:
                continue
            if allowed_refs and ref not in allowed_refs:
                continue

            pads = [self._normalize_pad(p) for p in list(item.get("pads") or []) if self._normalize_pad(p)]
            if not pads:
                pad_count = int(item.get("pads_count", 0) or 0)
                if 0 < pad_count <= 64:
                    pads = [str(i) for i in range(1, pad_count + 1)]

            pad_nets_raw = item.get("pad_nets") if isinstance(item.get("pad_nets"), dict) else {}
            pad_nets = {
                self._normalize_pad(pad): self._sanitize_net_name(net)
                for pad, net in pad_nets_raw.items()
                if self._normalize_pad(pad)
            }
            out.append(
                {
                    "ref": ref,
                    "footprint": str(item.get("footprint", "") or ""),
                    "value": str(item.get("value", "") or ""),
                    "pads": pads,
                    "pad_nets": pad_nets,
                    "pin_name_to_pad": dict(item.get("pin_name_to_pad") or {}) if isinstance(item.get("pin_name_to_pad"), dict) else {},
                    "x": float(item.get("x", 0.0) or 0.0),
                    "y": float(item.get("y", 0.0) or 0.0),
                }
            )
        out.sort(key=lambda row: row["ref"])
        return out

    def _board_index(self, board_components: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for item in board_components:
            ref = self._normalize_ref(item.get("ref"))
            if not ref:
                continue
            pads = [self._normalize_pad(p) for p in list(item.get("pads") or []) if self._normalize_pad(p)]
            index[ref] = {
                "ref": ref,
                "pads": pads,
                "pad_lookup": set(pads),
                "pad_nets": dict(item.get("pad_nets") or {}) if isinstance(item.get("pad_nets"), dict) else {},
                "pin_name_to_pad": dict(item.get("pin_name_to_pad") or {}) if isinstance(item.get("pin_name_to_pad"), dict) else {},
                "x": float(item.get("x", 0.0) or 0.0),
                "y": float(item.get("y", 0.0) or 0.0),
                "footprint": str(item.get("footprint", "") or ""),
                "value": str(item.get("value", "") or ""),
            }
        return index

    def _seed_assignments_from_manifest(
        self,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        *,
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        seeded: List[Dict[str, str]] = []
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = self._normalize_ref(part.get("ref"))
            if not ref or ref not in board_index:
                continue
            board_item = board_index[ref]
            pads = board_item.get("pad_lookup", set())
            if not pads:
                continue

            pin_rows: List[Dict[str, Any]] = []
            seeded_by_pad: Dict[str, str] = {}
            for pin in list(part.get("pins") or []):
                if not isinstance(pin, dict):
                    continue
                raw_pad = self._normalize_pad(
                    pin.get("num") or pin.get("pad") or pin.get("pin") or pin.get("number")
                )
                pin_name = str(
                    pin.get("name")
                    or pin.get("pin_name")
                    or pin.get("label")
                    or ""
                ).strip()
                pad = raw_pad

                # Fallback: resolve pin name → pad number via board footprint data
                if not pad or pad not in pads:
                    name = str(pin_name or raw_pad).strip()
                    if name:
                        resolved_pad = self._resolve_pin_name_to_pad_fuzzy(board_item, name)
                        if resolved_pad:
                            pad = resolved_pad

                net = self._sanitize_net_name(pin.get("net"))
                net = self._apply_net_alias(net, net_aliases)
                on_footprint = bool(pad and pad in pads)
                pin_rows.append(
                    {
                        "pad": pad,
                        "raw_pad": raw_pad,
                        "pin_name": pin_name,
                        "net": net,
                        "on_footprint": on_footprint,
                    }
                )

                if not net:
                    if not raw_pad and not pad:
                        key = ("missing_pad_key", ref, str(pin.get("name", "") or "?"))
                        if key not in self._seed_skip_logged:
                            self._seed_skip_logged.add(key)
                            logger.warning(
                                "SEED SKIP %s: pin %s has no pad key. Pin keys: %s",
                                ref, pin.get("name", "?"), list(pin.keys())
                            )
                    continue

                if not pad:
                    key = ("missing_pad_key", ref, str(pin.get("name", "") or "?"))
                    if key not in self._seed_skip_logged:
                        self._seed_skip_logged.add(key)
                        logger.warning(
                            "SEED SKIP %s: pin %s has no pad key. Pin keys: %s",
                            ref, pin.get("name", "?"), list(pin.keys())
                        )
                    continue
                if pad not in pads:
                    self._log_seed_skip_once(
                        reason="pad_not_on_footprint",
                        ref=ref,
                        pad=pad,
                        pin_name=pin_name,
                        pads=sorted(pads),
                    )
                    continue

                existing_net = self._sanitize_net_name((board_item.get("pad_nets") or {}).get(pad))
                if existing_net and existing_net == net:
                    continue
                if pad not in seeded_by_pad:
                    seeded_by_pad[pad] = net
                elif seeded_by_pad.get(pad) != net:
                    self._log_seed_skip_once(
                        reason="seed_conflict_same_pad",
                        ref=ref,
                        pad=pad,
                        pin_name=pin_name,
                        pads=sorted(pads),
                    )

            projected_by_pad = self._project_missing_manifest_nets_to_board_pads(
                board_item=board_item,
                pin_rows=pin_rows,
            )
            if projected_by_pad:
                live_pad_nets = (
                    dict(board_item.get("pad_nets") or {})
                    if isinstance(board_item.get("pad_nets"), dict)
                    else {}
                )
                for pad, net in sorted(projected_by_pad.items(), key=lambda item: self._pad_sort_key(item[0])):
                    live_net = self._sanitize_net_name(live_pad_nets.get(pad))
                    if live_net and live_net != net:
                        continue
                    seeded_by_pad[pad] = net
                logger.info(
                    "SEED PROJECTION %s: remapped manifest nets to board pads %s",
                    ref,
                    {
                        pad: seeded_by_pad.get(pad, "")
                        for pad in sorted(projected_by_pad.keys(), key=self._pad_sort_key)
                    },
                )

            for pad in sorted(seeded_by_pad.keys(), key=self._pad_sort_key):
                seeded.append({"ref": ref, "pad": pad, "net": seeded_by_pad[pad]})
        return seeded

    def _plan_support_assignments_with_llm(
        self,
        *,
        goal: str,
        manifest: Dict[str, Any],
        spec_debug: Dict[str, Any],
        placement_plan: Dict[str, Any],
        board_components: Sequence[Dict[str, Any]],
        seeded_assignments: Sequence[Dict[str, str]],
        quality_constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        primary_refs = {
            self._normalize_ref(part.get("ref"))
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and isinstance(part.get("pins"), list) and part.get("pins")
        }
        compact_parts: List[Dict[str, Any]] = []
        manifest_by_ref: Dict[str, Dict[str, Any]] = {}
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = self._normalize_ref(part.get("ref"))
            if not ref:
                continue
            manifest_by_ref[ref] = part
            compact_parts.append(
                {
                    "ref": ref,
                    "mpn": str(part.get("mpn", "") or ""),
                    "footprint": str(part.get("footprint", "") or ""),
                    "pins": [
                        {
                            "pad": self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin")),
                            "name": str(pin.get("name", "") or ""),
                            "net": self._sanitize_net_name(pin.get("net")),
                        }
                            for pin in list(part.get("pins") or [])
                            if isinstance(pin, dict)
                        ],
                    "support_candidates": [
                        {
                            "family": str(sec.get("family", "") or ""),
                            "implementation": str(sec.get("implementation", "") or ""),
                            "type": str(sec.get("type", "") or ""),
                            "value": str(sec.get("value", "") or ""),
                            "function": str(sec.get("function", "") or ""),
                            "net_hint": self._sanitize_net_name(sec.get("net_hint")),
                            "shared": bool(sec.get("shared", False)),
                            "source_refs": [self._normalize_ref(v) for v in list(sec.get("source_refs") or []) if self._normalize_ref(v)],
                            "source_nets": [self._sanitize_net_name(v) for v in list(sec.get("source_nets") or []) if self._sanitize_net_name(v)],
                        }
                        for sec in list(part.get("support_candidates") or [])
                        if isinstance(sec, dict)
                    ],
                }
            )

        placement_lookup = {
            self._normalize_ref(ref): dict(pos)
            for ref, pos in placement_plan.items()
            if isinstance(ref, str) and isinstance(pos, dict)
        }

        def nearest_primary_refs(component: Dict[str, Any], limit: int = 4) -> List[str]:
            ref = self._normalize_ref(component.get("ref"))
            if ref in primary_refs:
                return []
            x = float(component.get("x", 0.0) or 0.0)
            y = float(component.get("y", 0.0) or 0.0)
            distances: List[Tuple[float, str]] = []
            for other in board_components:
                other_ref = self._normalize_ref(other.get("ref"))
                if other_ref == ref or other_ref not in primary_refs:
                    continue
                ox = float(other.get("x", 0.0) or 0.0)
                oy = float(other.get("y", 0.0) or 0.0)
                distances.append((math.hypot(x - ox, y - oy), other_ref))
            distances.sort(key=lambda row: (row[0], row[1]))
            return [ref_name for _dist, ref_name in distances[:limit]]

        compact_board: List[Dict[str, Any]] = []
        for component in board_components:
            ref = self._normalize_ref(component.get("ref"))
            pos = placement_lookup.get(ref, {})
            manifest_part = manifest_by_ref.get(ref, {})
            compact_board.append(
                {
                    "ref": ref,
                    "footprint": str(component.get("footprint", "") or ""),
                    "value": str(component.get("value", "") or ""),
                    "pads": list(component.get("pads") or []),
                    "current_pad_nets": dict(component.get("pad_nets") or {}),
                    "placement": {
                        "x": round(float(component.get("x", 0.0) or 0.0), 3),
                        "y": round(float(component.get("y", 0.0) or 0.0), 3),
                        "edge": pos.get("edge"),
                        "zone": pos.get("zone"),
                        "category": pos.get("category"),
                    },
                    "nearest_primaries": nearest_primary_refs(component),
                    "support_candidates": [
                        {
                            "family": str(sec.get("family", "") or ""),
                            "implementation": str(sec.get("implementation", "") or ""),
                            "type": str(sec.get("type", "") or ""),
                            "value": str(sec.get("value", "") or ""),
                            "function": str(sec.get("function", "") or ""),
                            "net_hint": self._sanitize_net_name(sec.get("net_hint")),
                            "shared": bool(sec.get("shared", False)),
                            "source_refs": [self._normalize_ref(v) for v in list(sec.get("source_refs") or []) if self._normalize_ref(v)],
                            "source_nets": [self._sanitize_net_name(v) for v in list(sec.get("source_nets") or []) if self._sanitize_net_name(v)],
                            "source_functions": [str(v or "") for v in list(sec.get("source_functions") or []) if str(v or "").strip()],
                        }
                        for sec in list(manifest_part.get("support_candidates") or [])
                        if isinstance(sec, dict)
                    ],
                }
            )

        support_hints = [
            {
                "owner_ref": self._normalize_ref(row.get("ref")),
                "role": str(row.get("role", "") or ""),
                "secondaries": [
                    {
                        "family": str(sec.get("family", "") or ""),
                        "implementation": str(sec.get("implementation", "") or ""),
                        "type": str(sec.get("type", "") or ""),
                        "value": str(sec.get("value", "") or ""),
                        "qty": int(sec.get("qty", 1) or 1),
                        "function": str(sec.get("function", "") or ""),
                        "net_hint": self._sanitize_net_name(sec.get("net_hint")),
                        "shared": bool(sec.get("shared", False)),
                        "ref_prefix": str(sec.get("ref_prefix", "") or ""),
                        "footprint": str(sec.get("footprint", "") or ""),
                    }
                    for sec in list(row.get("secondaries") or [])
                    if isinstance(sec, dict)
                ],
            }
            for row in list(spec_debug.get("datasheet_secondaries") or [])
            if isinstance(row, dict) and self._normalize_ref(row.get("ref"))
        ]

        board_level_hints = [
            {
                "family": str(row.get("family", "") or ""),
                "implementation": str(row.get("implementation", "") or ""),
                "type": str(row.get("type", "") or ""),
                "value": str(row.get("value", "") or ""),
                "qty": int(row.get("qty", 1) or 1),
                "function": str(row.get("function", "") or ""),
                "net_hint": self._sanitize_net_name(row.get("net_hint")),
                "shared": bool(row.get("shared", False)),
                "ref_prefix": str(row.get("ref_prefix", "") or ""),
                "footprint": str(row.get("footprint", "") or ""),
                "source_refs": [self._normalize_ref(v) for v in list(row.get("source_refs") or []) if self._normalize_ref(v)],
                "source_nets": [self._sanitize_net_name(v) for v in list(row.get("source_nets") or []) if self._sanitize_net_name(v)],
                "source_functions": [str(v or "") for v in list(row.get("source_functions") or []) if str(v or "").strip()],
            }
            for row in list(spec_debug.get("board_secondaries") or [])
            if isinstance(row, dict)
        ]

        hinted_owner_refs = sorted(
            {
                self._normalize_ref(row.get("owner_ref"))
                for row in support_hints
                if isinstance(row, dict)
                and self._normalize_ref(row.get("owner_ref"))
                and isinstance(row.get("secondaries"), list)
                and bool(row.get("secondaries"))
            }
        )
        all_normalized: List[Dict[str, str]] = []
        all_messages: List[str] = []

        def _normalize_assignments(assignments: Any) -> List[Dict[str, str]]:
            if not isinstance(assignments, list):
                return []
            normalized: List[Dict[str, str]] = []
            for item in assignments:
                if not isinstance(item, dict):
                    continue
                ref = self._normalize_ref(item.get("ref"))
                pad = self._normalize_pad(item.get("pad"))
                pin_ref = str(item.get("pin_ref", "") or "").strip()
                if (not ref or not pad) and pin_ref:
                    match = re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", pin_ref)
                    if match:
                        ref = self._normalize_ref(match.group(1))
                        pad = self._normalize_pad(match.group(2))
                net = self._sanitize_net_name(item.get("net") or item.get("net_name") or item.get("name"))
                if not ref or not pad or not net:
                    continue
                normalized.append({"ref": ref, "pad": pad, "net": net})
            return normalized

        def _run_llm_pass(label: str, focus_text: str, target_reminder: str) -> Tuple[List[Dict[str, str]], str]:
            qc = quality_constraints if isinstance(quality_constraints, dict) else {}
            prompt = (
                f"USER_GOAL: {str(goal or '').strip()}\n\n"
                f"{focus_text}\n\n"
                "PRIMARY_MANIFEST_WITH_PINS_JSON:\n"
                + self._compact_json(compact_parts)
                + "\n\nBOARD_COMPONENTS_JSON:\n"
                + self._compact_json(compact_board)
                + "\n\nSEEDED_ASSIGNMENTS_JSON:\n"
                + self._compact_json(list(seeded_assignments))
                + "\n\nSUPPORT_HINTS_BY_OWNER_JSON:\n"
                + self._compact_json(support_hints)
                + "\n\nBOARD_LEVEL_SUPPORT_HINTS_JSON:\n"
                + self._compact_json(board_level_hints)
                + "\n\nQUALITY_CONSTRAINTS_JSON:\n"
                + self._compact_json(qc)
                + (
                    "\n\nReturn ADDITIONAL pad assignments only for pads not already covered by "
                    "SEEDED_ASSIGNMENTS_JSON.\nFocus ONLY on "
                    + target_reminder
                    + " during this step."
                )
            )

            try:
                raw = self._llm_chat(prompt, system_prompt=self.SYSTEM_PROMPT)
                obj = self._parse_json_object(raw)
            except Exception as e:
                logger.warning("NET: llm_chat failed for focus %s: %s", label, e)
                return [], ""

            if not isinstance(obj, dict):
                return [], ""

            assistant_message = str(obj.get("assistant_message", "") or "").strip()
            normalized = _normalize_assignments(obj.get("assignments"))
            return normalized, (f"[{label}] {assistant_message}" if assistant_message else "")

        global_assignments, global_msg = _run_llm_pass(
            "GLOBAL",
            "CURRENT_FOCUS: Provide NET assignments for support parts and unresolved board-level connections across the full board.",
            "all unresolved support and board-level parts",
        )
        all_normalized.extend(global_assignments)
        if global_msg:
            all_messages.append(global_msg)

        # Keep support-assignment prompting compact: one broad pass and at most one
        # focused follow-up to avoid excessive context/passes.
        fallback_refs: List[str] = []
        if hinted_owner_refs:
            assigned_refs = {
                self._normalize_ref(row.get("ref"))
                for row in all_normalized
                if isinstance(row, dict)
            }
            fallback_refs = [ref for ref in hinted_owner_refs if ref not in assigned_refs][:6]

        if fallback_refs:
            refs_text = ", ".join(fallback_refs)
            focus_text = (
                "CURRENT_FOCUS: Provide NET assignments only for unresolved support-owner areas. "
                f"Prioritize these owner refs and their attached support parts: {refs_text}."
            )
            target_rows, target_msg = _run_llm_pass(
                "FOCUSED",
                focus_text,
                f"support parts near owner refs {refs_text}",
            )
            all_normalized.extend(target_rows)
            if target_msg:
                all_messages.append(target_msg)
        elif not all_normalized:
            focus_text = (
                "CURRENT_FOCUS: Provide NET assignments for unresolved board-level support parts "
                "(connectors and discrete support components) only."
            )
            target_rows, target_msg = _run_llm_pass(
                "BOARD_LEVEL",
                focus_text,
                "general unresolved board-level support parts",
            )
            all_normalized.extend(target_rows)
            if target_msg:
                all_messages.append(target_msg)

        # Deduplicate exact repeats while preserving proposal order for downstream
        # merge/conflict diagnostics.
        deduped: List[Dict[str, str]] = []
        seen_rows: set[Tuple[str, str, str]] = set()
        for row in all_normalized:
            key = (row["ref"], row["pad"], row["net"])
            if key in seen_rows:
                continue
            seen_rows.add(key)
            deduped.append(row)

        return deduped, " | ".join(msg for msg in all_messages if msg)

    def _parse_json_object(self, raw_text: str) -> Optional[Dict[str, Any]]:
        try:
            from ..design_actions import sanitize_llm_json_text
        except Exception:
            sanitize_llm_json_text = lambda value: str(value or "")
        cleaned = sanitize_llm_json_text(raw_text)
        obj_text = self._extract_json_object(cleaned) or cleaned
        try:
            obj = json.loads(obj_text)
        except Exception as e:
            logger.warning("NET: failed to parse LLM JSON: %s", e)
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def _merge_assignments(
        self,
        *,
        seeded_assignments: Sequence[Dict[str, str]],
        llm_assignments: Sequence[Dict[str, str]],
        board_index: Dict[str, Dict[str, Any]],
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        out: List[Dict[str, str]] = []
        seen: Dict[Tuple[str, str], str] = {}
        warnings: List[str] = []

        def add_one(item: Dict[str, str], *, source: str) -> None:
            ref = self._normalize_ref(item.get("ref"))
            pad = self._normalize_pad(item.get("pad"))
            net = self._sanitize_net_name(item.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if not ref or not pad or not net:
                return
            board_item = board_index.get(ref)
            if board_item is None:
                warnings.append(f"{source}:unknown_ref:{ref}")
                return
            if pad not in set(board_item.get("pad_lookup", set())):
                warnings.append(f"{source}:unknown_pad:{ref}/{pad}")
                return
            existing_live_net = self._sanitize_net_name((board_item.get("pad_nets") or {}).get(pad))
            if existing_live_net and existing_live_net == net:
                return

            key = (ref, pad)
            if key in seen:
                if seen[key] != net:
                    warnings.append(f"{source}:conflict:{ref}/{pad}:{seen[key]}->{net}")
                return
            seen[key] = net
            out.append({"ref": ref, "pad": pad, "net": net})

        for item in seeded_assignments:
            add_one(item, source="seed")
        seed_count = len(out)
        for item in llm_assignments:
            add_one(item, source="llm")

        out.sort(key=lambda row: (row["net"], row["ref"], row["pad"]))
        return out, {
            "seed_assignment_count": seed_count,
            "llm_assignment_count": max(0, len(out) - seed_count),
            "warnings": warnings[:40],
        }

    def _normalize_llm_assignments(self, assignments: Any) -> List[Dict[str, str]]:
        if not isinstance(assignments, list):
            return []
        normalized: List[Dict[str, str]] = []
        for item in assignments:
            if not isinstance(item, dict):
                continue
            ref = self._normalize_ref(item.get("ref"))
            pad = self._normalize_pad(item.get("pad"))
            pin_ref = str(item.get("pin_ref", "") or "").strip()
            if (not ref or not pad) and pin_ref:
                match = re.match(r"^\s*([A-Za-z]+\d+)\s*[-/:]\s*([A-Za-z0-9]+)\s*$", pin_ref)
                if match:
                    ref = self._normalize_ref(match.group(1))
                    pad = self._normalize_pad(match.group(2))
            net = self._sanitize_net_name(item.get("net") or item.get("net_name") or item.get("name"))
            if not ref or not pad or not net:
                continue
            normalized.append({"ref": ref, "pad": pad, "net": net})
        return normalized

    def _unassigned_required_pad_rows(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        required_pads_by_ref = self._required_pads_by_ref(manifest=manifest, board_index=board_index)
        assignment_lookup: Dict[Tuple[str, str], str] = {}
        for row in assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if ref and pad and net:
                assignment_lookup[(ref, pad)] = net

        expected_pin_net: Dict[Tuple[str, str], str] = {}
        expected_pin_name: Dict[Tuple[str, str], str] = {}
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = self._normalize_ref(part.get("ref"))
            if not ref:
                continue
            for pin in list(part.get("pins") or []):
                if not isinstance(pin, dict):
                    continue
                pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin"))
                if not pad:
                    continue
                expected_net = self._sanitize_net_name(pin.get("net"))
                expected_net = self._apply_net_alias(expected_net, net_aliases)
                pin_name = str(
                    pin.get("name")
                    or pin.get("pin_name")
                    or pin.get("label")
                    or ""
                ).strip()
                if expected_net:
                    expected_pin_net[(ref, pad)] = expected_net
                if pin_name:
                    expected_pin_name[(ref, pad)] = pin_name

        unresolved_rows: List[Dict[str, str]] = []
        for ref, required_pads in sorted(required_pads_by_ref.items()):
            board_item = board_index.get(ref)
            if board_item is None:
                continue
            live_pad_nets = (
                dict(board_item.get("pad_nets") or {})
                if isinstance(board_item.get("pad_nets"), dict)
                else {}
            )
            for pad in required_pads:
                live_net = self._sanitize_net_name(live_pad_nets.get(pad))
                live_net = self._apply_net_alias(live_net, net_aliases)
                assigned_net = self._sanitize_net_name(assignment_lookup.get((ref, pad)))
                assigned_net = self._apply_net_alias(assigned_net, net_aliases)
                if live_net or assigned_net:
                    continue
                row: Dict[str, str] = {"ref": ref, "pad": pad}
                expected_net = expected_pin_net.get((ref, pad), "")
                pin_name = expected_pin_name.get((ref, pad), "")
                if expected_net:
                    row["expected_net"] = expected_net
                if pin_name:
                    row["pin_name"] = pin_name
                unresolved_rows.append(row)
        return unresolved_rows

    def _unconnected_pad_audit_rows(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        required_pads_by_ref = self._required_pads_by_ref(manifest=manifest, board_index=board_index)
        assignment_lookup: Dict[Tuple[str, str], str] = {}
        for row in assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if ref and pad and net:
                assignment_lookup[(ref, pad)] = net

        manifest_by_ref = {
            self._normalize_ref(part.get("ref")): part
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and self._normalize_ref(part.get("ref"))
        }
        expected_pin_net: Dict[Tuple[str, str], str] = {}
        expected_pin_name: Dict[Tuple[str, str], str] = {}
        explicit_nc_by_ref: Dict[str, set[str]] = {}
        for ref, part in manifest_by_ref.items():
            explicit_nc_by_ref[ref] = self._explicit_no_connect_pads(part)
            for pin in [row for row in list(part.get("pins") or []) if isinstance(row, dict)]:
                pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin"))
                if not pad:
                    continue
                expected_net = self._sanitize_net_name(pin.get("net"))
                expected_net = self._apply_net_alias(expected_net, net_aliases)
                pin_name = str(
                    pin.get("name")
                    or pin.get("pin_name")
                    or pin.get("label")
                    or ""
                ).strip()
                if expected_net:
                    expected_pin_net[(ref, pad)] = expected_net
                if pin_name:
                    expected_pin_name[(ref, pad)] = pin_name

        rows: List[Dict[str, Any]] = []
        for ref, board_item in sorted(board_index.items()):
            part = manifest_by_ref.get(ref, {})
            board_pads = [
                self._normalize_pad(pad)
                for pad in list(board_item.get("pads") or [])
                if self._normalize_pad(pad)
            ]
            if not board_pads:
                continue
            board_pads_sorted = sorted(board_pads, key=self._pad_sort_key)
            board_pad_lookup = set(board_pads_sorted)
            required_lookup = set(required_pads_by_ref.get(ref, []))
            explicit_nc_lookup = set(explicit_nc_by_ref.get(ref, set()))
            live_pad_nets = (
                dict(board_item.get("pad_nets") or {})
                if isinstance(board_item.get("pad_nets"), dict)
                else {}
            )
            ref_known_nets: set[str] = set()
            for pad in board_pads_sorted:
                live_net = self._sanitize_net_name(live_pad_nets.get(pad))
                live_net = self._apply_net_alias(live_net, net_aliases)
                assigned_net = self._sanitize_net_name(assignment_lookup.get((ref, pad)))
                assigned_net = self._apply_net_alias(assigned_net, net_aliases)
                known_net = live_net or assigned_net
                if known_net:
                    ref_known_nets.add(known_net)

            pins = [row for row in list(part.get("pins") or []) if isinstance(row, dict)] if isinstance(part, dict) else []
            support_candidates = (
                [row for row in list(part.get("support_candidates") or []) if isinstance(row, dict)]
                if isinstance(part, dict)
                else []
            )
            part_text = (
                " ".join(
                    str(part.get(key, "") or "")
                    for key in ("ref", "footprint", "value", "description", "mpn")
                ).lower()
                if isinstance(part, dict)
                else ""
            )
            is_bridge_like = (
                len(board_pads_sorted) == 2
                and (not ref.startswith("J"))
                and (
                    ref.startswith(("D", "F", "R", "L", "FB", "SW"))
                    or bool(re.search(r"\b(fuse|switch|button|diode|resistor|bead|ferrite|inductor|jumper)\b", part_text))
                    or bool(support_candidates)
                )
            )

            missing_manifest_pads_seen: set[str] = set()
            for pin in pins:
                pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin"))
                if not pad or pad in board_pad_lookup or pad in missing_manifest_pads_seen:
                    continue
                expected_net = self._sanitize_net_name(pin.get("net"))
                expected_net = self._apply_net_alias(expected_net, net_aliases)
                if not expected_net:
                    continue
                if expected_net in ref_known_nets:
                    continue
                missing_manifest_pads_seen.add(pad)
                row: Dict[str, Any] = {
                    "ref": ref,
                    "pad": pad,
                    "necessity": "required",
                    "reason": "manifest_pad_missing_on_footprint",
                    "on_footprint": False,
                }
                pin_name = str(
                    pin.get("name")
                    or pin.get("pin_name")
                    or pin.get("label")
                    or ""
                ).strip()
                if pin_name:
                    row["pin_name"] = pin_name
                row["expected_net"] = expected_net
                rows.append(row)

            for pad in board_pads_sorted:
                live_net = self._sanitize_net_name(live_pad_nets.get(pad))
                live_net = self._apply_net_alias(live_net, net_aliases)
                assigned_net = self._sanitize_net_name(assignment_lookup.get((ref, pad)))
                assigned_net = self._apply_net_alias(assigned_net, net_aliases)
                if live_net or assigned_net:
                    continue

                necessity = "optional"
                reason = "non_required_pad"
                if pad in explicit_nc_lookup:
                    necessity = "optional"
                    reason = "explicit_no_connect"
                elif pad in required_lookup:
                    necessity = "required"
                    reason = "manifest_required_pad"
                elif is_bridge_like:
                    necessity = "required"
                    reason = "bridge_component_requires_two_nets"
                elif ref.startswith("J"):
                    necessity = "optional"
                    reason = "connector_exposed_pad"
                elif len(board_pads_sorted) <= 1:
                    necessity = "optional"
                    reason = "single_pad_component"

                row = {
                    "ref": ref,
                    "pad": pad,
                    "necessity": necessity,
                    "reason": reason,
                    "on_footprint": True,
                }
                expected_net = expected_pin_net.get((ref, pad), "")
                if expected_net:
                    row["expected_net"] = expected_net
                pin_name = expected_pin_name.get((ref, pad), "")
                if pin_name:
                    row["pin_name"] = pin_name
                rows.append(row)

        rows.sort(
            key=lambda row: (
                0 if str(row.get("necessity", "") or "").lower() == "required" else 1,
                str(row.get("reason", "") or ""),
                str(row.get("ref", "") or ""),
                self._pad_sort_key(str(row.get("pad", "") or "")),
            )
        )
        return rows

    def _plan_topology_reconcile_with_llm(
        self,
        *,
        goal: str,
        manifest: Dict[str, Any],
        board_components: Sequence[Dict[str, Any]],
        board_index: Dict[str, Dict[str, Any]],
        placement_plan: Dict[str, Any],
        current_assignments: Sequence[Dict[str, str]],
        coverage: Dict[str, Any],
        bridge_lint: Dict[str, Any],
        net_aliases: Optional[Dict[str, str]] = None,
        unconnected_audit_rows: Optional[Sequence[Dict[str, Any]]] = None,
        focus_label: str = "TOPOLOGY",
        quality_constraints: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, str]], str]:
        unresolved_required_rows = self._unassigned_required_pad_rows(
            manifest=manifest,
            board_index=board_index,
            assignments=current_assignments,
            net_aliases=net_aliases,
        )
        audit_rows = [
            dict(row)
            for row in list(unconnected_audit_rows or [])
            if isinstance(row, dict)
        ]
        if not audit_rows:
            audit_rows = self._unconnected_pad_audit_rows(
                manifest=manifest,
                board_index=board_index,
                assignments=current_assignments,
                net_aliases=net_aliases,
            )
        required_audit_rows = [
            row
            for row in audit_rows
            if str(row.get("necessity", "") or "").lower() == "required"
        ]
        bridge_issues = [
            row
            for row in list(bridge_lint.get("issues_sample") or [])
            if isinstance(row, dict)
        ]
        if not required_audit_rows and not unresolved_required_rows and not bridge_issues:
            return [], ""

        manifest_by_ref = {
            self._normalize_ref(part.get("ref")): part
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and self._normalize_ref(part.get("ref"))
        }
        placement_lookup = {
            self._normalize_ref(ref): dict(pos)
            for ref, pos in placement_plan.items()
            if isinstance(ref, str) and isinstance(pos, dict)
        }

        compact_board: List[Dict[str, Any]] = []
        current_net_names: set[str] = set()
        for component in board_components:
            ref = self._normalize_ref(component.get("ref"))
            if not ref:
                continue
            pos = placement_lookup.get(ref, {})
            part = manifest_by_ref.get(ref, {})
            live_pad_nets = (
                dict(component.get("pad_nets") or {})
                if isinstance(component.get("pad_nets"), dict)
                else {}
            )
            for net_name in live_pad_nets.values():
                net = self._sanitize_net_name(net_name)
                net = self._apply_net_alias(net, net_aliases)
                if net:
                    current_net_names.add(net)

            compact_board.append(
                {
                    "ref": ref,
                    "pads": [self._normalize_pad(v) for v in list(component.get("pads") or []) if self._normalize_pad(v)],
                    "current_pad_nets": live_pad_nets,
                    "placement": {
                        "x": round(float(component.get("x", 0.0) or 0.0), 3),
                        "y": round(float(component.get("y", 0.0) or 0.0), 3),
                        "edge": pos.get("edge"),
                        "category": pos.get("category"),
                    },
                    "pins": [
                        {
                            "pad": self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin")),
                            "name": str(pin.get("name", "") or ""),
                            "net": self._apply_net_alias(
                                self._sanitize_net_name(pin.get("net")),
                                net_aliases,
                            ),
                        }
                        for pin in list(part.get("pins") or [])
                        if isinstance(pin, dict)
                    ],
                    "support_candidates": [
                        {
                            "family": str(row.get("family", "") or ""),
                            "function": str(row.get("function", "") or ""),
                            "net_hint": self._apply_net_alias(
                                self._sanitize_net_name(row.get("net_hint")),
                                net_aliases,
                            ),
                            "source_nets": [
                                self._apply_net_alias(self._sanitize_net_name(v), net_aliases)
                                for v in list(row.get("source_nets") or [])
                                if self._sanitize_net_name(v)
                            ],
                        }
                        for row in list(part.get("support_candidates") or [])
                        if isinstance(row, dict)
                    ],
                }
            )

        normalized_current_assignments: List[Dict[str, str]] = []
        existing_pad_keys: set[Tuple[str, str]] = set()
        for row in current_assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if not ref or not pad or not net:
                continue
            normalized_current_assignments.append({"ref": ref, "pad": pad, "net": net})
            existing_pad_keys.add((ref, pad))
            current_net_names.add(net)

        for ref, board_item in board_index.items():
            live_pad_nets = (
                dict(board_item.get("pad_nets") or {})
                if isinstance(board_item.get("pad_nets"), dict)
                else {}
            )
            for pad, net_name in live_pad_nets.items():
                pad_name = self._normalize_pad(pad)
                net = self._sanitize_net_name(net_name)
                net = self._apply_net_alias(net, net_aliases)
                if not pad_name or not net:
                    continue
                existing_pad_keys.add((ref, pad_name))

        alias_subset = (
            {
                key: net_aliases[key]
                for key in sorted(net_aliases.keys())[:120]
            }
            if isinstance(net_aliases, dict)
            else {}
        )
        focus_name = str(focus_label or "TOPOLOGY").strip().upper() or "TOPOLOGY"
        if focus_name == "FINAL_CHECK":
            focus_line = (
                "CURRENT_FOCUS: Final unconnected-pad check after topology reconcile across the full board."
            )
        else:
            focus_line = "CURRENT_FOCUS: Topology reconciliation across the full board."

        prompt = (
            f"USER_GOAL: {str(goal or '').strip()}\n\n"
            + focus_line
            + "\nReturn ONLY ADDITIONAL assignments that improve unresolved board connectivity.\n"
            "Do not modify existing pad assignments.\n"
            "Use canonical net names already present when equivalent.\n"
            "Prioritize UNCONNECTED rows where necessity='required'.\n\n"
            "BOARD_COMPONENTS_JSON:\n"
            + self._compact_json(compact_board)
            + "\n\nCURRENT_ASSIGNMENTS_JSON:\n"
            + self._compact_json(normalized_current_assignments)
            + "\n\nUNASSIGNED_REQUIRED_PADS_JSON:\n"
            + self._compact_json(unresolved_required_rows[:320])
            + "\n\nUNCONNECTED_PAD_AUDIT_JSON:\n"
            + self._compact_json(audit_rows)
            + "\n\nREQUIRED_UNCONNECTED_PAD_AUDIT_JSON:\n"
            + self._compact_json(required_audit_rows)
            + "\n\nBRIDGE_LINT_ISSUES_JSON:\n"
            + self._compact_json(bridge_issues[:160])
            + "\n\nCURRENT_NET_NAMES_JSON:\n"
            + self._compact_json(sorted(current_net_names))
            + "\n\nCANONICAL_NET_ALIASES_JSON:\n"
            + self._compact_json(alias_subset)
            + "\n\nQUALITY_CONSTRAINTS_JSON:\n"
            + self._compact_json(quality_constraints if isinstance(quality_constraints, dict) else {})
            + "\n\nCOVERAGE_SUMMARY_JSON:\n"
            + self._compact_json(
                {
                    "refs_without_nets_count": int(coverage.get("refs_without_nets_count", 0) or 0),
                    "partial_refs_count": int(coverage.get("partial_refs_count", 0) or 0),
                    "unassigned_required_pads_count": int(coverage.get("unassigned_required_pads_count", 0) or 0),
                    "bridge_issue_count": int(bridge_lint.get("issue_count", 0) or 0),
                    "unconnected_audit_count": len(audit_rows),
                    "unconnected_required_count": len(required_audit_rows),
                }
            )
            + "\n\nRules:\n"
            "- Use only refs/pads that exist on BOARD_COMPONENTS_JSON footprints.\n"
            "- Assign each pad to at most one net.\n"
            "- Never propose rows where on_footprint is false.\n"
            "- Treat necessity='required' as highest priority.\n"
            "- For necessity='optional' rows, only assign when circuit intent is strong and clear.\n"
            "- For reasons explicit_no_connect, single_pad_component, and connector_exposed_pad, default to no assignment.\n"
            "- Use stable existing net names when logically equivalent.\n"
            "- If uncertain, omit the assignment.\n"
            "- Return strict JSON with assistant_message and assignments only."
        )

        try:
            raw = self._llm_chat(prompt, system_prompt=self.SYSTEM_PROMPT)
            obj = self._parse_json_object(raw)
        except Exception as e:
            logger.warning("NET: topology llm_chat failed: %s", e)
            return [], ""
        if not isinstance(obj, dict):
            return [], ""

        assistant_message = str(obj.get("assistant_message", "") or "").strip()
        proposed_rows = self._normalize_llm_assignments(obj.get("assignments"))
        out: List[Dict[str, str]] = []
        seen_keys: set[Tuple[str, str, str]] = set()
        for row in proposed_rows:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            net = self._apply_net_alias(net, net_aliases)
            if not ref or not pad or not net:
                continue
            if (ref, pad) in existing_pad_keys:
                continue
            key = (ref, pad, net)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out.append({"ref": ref, "pad": pad, "net": net})
        return out, assistant_message

    def _required_pads_by_ref(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        manifest_by_ref = {
            self._normalize_ref(part.get("ref")): part
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and self._normalize_ref(part.get("ref"))
        }
        required: Dict[str, List[str]] = {}
        for ref, board_item in sorted(board_index.items()):
            part = manifest_by_ref.get(ref)
            if not isinstance(part, dict):
                continue
            board_pads = [self._normalize_pad(pad) for pad in list(board_item.get("pads") or []) if self._normalize_pad(pad)]
            if not board_pads:
                continue
            board_pad_lookup = set(board_pads)

            required_pads: List[str] = []
            pins = part.get("pins")
            if self._use_all_physical_pads_for_coverage(part):
                nc_pads = self._explicit_no_connect_pads(part)
                required_pads.extend(pad for pad in board_pads if pad not in nc_pads)
            elif isinstance(pins, list) and pins:
                for pin in pins:
                    if not isinstance(pin, dict):
                        continue
                    pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin"))
                    expected_net = self._sanitize_net_name(pin.get("net"))
                    if not pad or not expected_net or pad not in board_pad_lookup:
                        continue
                    required_pads.append(pad)
            if not required_pads and self._requires_full_pad_coverage(part):
                required_pads.extend(board_pads)

            if not required_pads:
                continue

            ordered_unique: List[str] = []
            seen: set[str] = set()
            for pad in required_pads:
                if pad in seen:
                    continue
                seen.add(pad)
                ordered_unique.append(pad)
            required[ref] = ordered_unique
        return required

    def _support_refs_without_nets(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
    ) -> List[str]:
        assignment_lookup: Dict[Tuple[str, str], str] = {}
        for row in assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            if ref and pad and net:
                assignment_lookup[(ref, pad)] = net

        out: List[str] = []
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            ref = self._normalize_ref(part.get("ref"))
            if not ref or ref.startswith("J"):
                continue
            support_candidates = [
                row for row in list(part.get("support_candidates") or []) if isinstance(row, dict)
            ]
            if not support_candidates:
                continue
            board_item = board_index.get(ref)
            if board_item is None:
                continue
            pads = [
                self._normalize_pad(pad)
                for pad in list(board_item.get("pads") or [])
                if self._normalize_pad(pad)
            ]
            if len(pads) < 2:
                continue
            live_pad_nets = (
                dict(board_item.get("pad_nets") or {})
                if isinstance(board_item.get("pad_nets"), dict)
                else {}
            )
            assigned = False
            for pad in pads:
                net_name = self._sanitize_net_name(live_pad_nets.get(pad))
                if not net_name:
                    net_name = self._sanitize_net_name(assignment_lookup.get((ref, pad)))
                if net_name:
                    assigned = True
                    break
            if not assigned:
                out.append(ref)
        return sorted(set(out))

    def _canonical_net_name_for_lint(
        self,
        net_name: Any,
        *,
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> str:
        net = self._sanitize_net_name(net_name)
        net = self._apply_net_alias(net, net_aliases)
        return str(net or "").strip().upper()

    def _bridge_connectivity_lint(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
        net_aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        assignment_lookup: Dict[Tuple[str, str], str] = {}
        for row in assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            if not ref or not pad:
                continue
            net = self._canonical_net_name_for_lint(row.get("net"), net_aliases=net_aliases)
            if net:
                assignment_lookup[(ref, pad)] = net

        # Build per-net ref fanout from both live board nets and proposed assignments.
        net_refs: Dict[str, set[str]] = {}
        per_ref_pad_net: Dict[Tuple[str, str], str] = {}
        for ref, board_item in board_index.items():
            pads = [self._normalize_pad(v) for v in list(board_item.get("pads") or []) if self._normalize_pad(v)]
            live_pad_nets = (
                dict(board_item.get("pad_nets") or {})
                if isinstance(board_item.get("pad_nets"), dict)
                else {}
            )
            for pad in pads:
                live_net = self._canonical_net_name_for_lint(live_pad_nets.get(pad), net_aliases=net_aliases)
                net = live_net or assignment_lookup.get((ref, pad), "")
                if not net:
                    continue
                per_ref_pad_net[(ref, pad)] = net
                net_refs.setdefault(net, set()).add(ref)

        manifest_by_ref = {
            self._normalize_ref(part.get("ref")): part
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and self._normalize_ref(part.get("ref"))
        }
        issues: List[Dict[str, Any]] = []
        candidate_count = 0
        for ref, board_item in sorted(board_index.items()):
            part = manifest_by_ref.get(ref)
            if not isinstance(part, dict):
                continue
            if ref.startswith("J"):
                continue
            pads = [self._normalize_pad(v) for v in list(board_item.get("pads") or []) if self._normalize_pad(v)]
            if len(pads) != 2:
                continue

            support_candidates = [row for row in list(part.get("support_candidates") or []) if isinstance(row, dict)]
            pins = [row for row in list(part.get("pins") or []) if isinstance(row, dict)]
            if not support_candidates and not pins and not ref.startswith(("D", "F", "R", "L", "FB")):
                continue
            candidate_count += 1

            nets = [per_ref_pad_net.get((ref, pad), "") for pad in pads]
            for idx, pad in enumerate(pads):
                net = nets[idx]
                if not net:
                    issues.append(
                        {
                            "ref": ref,
                            "pad": pad,
                            "issue": "missing_pad_net",
                        }
                    )
                    continue
                peer_refs = sorted(set(net_refs.get(net, set())) - {ref})
                if not peer_refs:
                    issues.append(
                        {
                            "ref": ref,
                            "pad": pad,
                            "net": net,
                            "issue": "orphan_side_net",
                        }
                    )
            if nets[0] and nets[1] and nets[0] == nets[1]:
                issues.append(
                    {
                        "ref": ref,
                        "pads": [pads[0], pads[1]],
                        "net": nets[0],
                        "issue": "both_pads_same_net",
                    }
                )

        issue_by_kind: Dict[str, int] = {}
        for row in issues:
            kind = str(row.get("issue", "") or "unknown")
            issue_by_kind[kind] = int(issue_by_kind.get(kind, 0) or 0) + 1

        return {
            "candidate_ref_count": candidate_count,
            "issue_count": len(issues),
            "issue_counts_by_kind": issue_by_kind,
            "issues_sample": issues[:40],
        }

    def _coverage_summary(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
    ) -> Dict[str, Any]:
        assignment_lookup: Dict[Tuple[str, str], str] = {}
        for row in assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            if ref and pad and net:
                assignment_lookup[(ref, pad)] = net

        required_pads_by_ref = self._required_pads_by_ref(manifest=manifest, board_index=board_index)
        refs_without_nets: List[Dict[str, Any]] = []
        partial_refs: List[Dict[str, Any]] = []
        total_pads = 0
        assigned_pads = 0
        routable_refs_total = 0
        routable_refs_with_any_nets = 0
        routable_refs_partial = 0
        routable_refs_without_nets = 0

        ignored_refs = max(0, len(board_index) - len(required_pads_by_ref))
        ignored_pads = 0

        for ref, pads in sorted(required_pads_by_ref.items()):
            board_item = board_index.get(ref)
            if board_item is None:
                continue
            if not pads:
                continue
            missing: List[str] = []
            assigned_here = 0
            live_pad_nets = dict(board_item.get("pad_nets") or {}) if isinstance(board_item.get("pad_nets"), dict) else {}
            for pad in pads:
                net_name = self._sanitize_net_name(live_pad_nets.get(pad))
                if not net_name:
                    net_name = self._sanitize_net_name(assignment_lookup.get((ref, pad)))
                if net_name:
                    assigned_here += 1
                else:
                    missing.append(pad)
            total_pads += len(pads)
            assigned_pads += assigned_here
            if len(pads) >= 2:
                routable_refs_total += 1
                if assigned_here > 0:
                    routable_refs_with_any_nets += 1
            if assigned_here == 0:
                refs_without_nets.append(
                    {
                        "ref": ref,
                        "pad_count": len(pads),
                        "unassigned_pads_sample": missing[:12],
                    }
                )
                if len(pads) >= 2:
                    routable_refs_without_nets += 1
            elif missing:
                partial_refs.append(
                    {
                        "ref": ref,
                        "pad_count": len(pads),
                        "assigned_pad_count": assigned_here,
                        "unassigned_pads_sample": missing[:12],
                    }
                )
                if len(pads) >= 2:
                    routable_refs_partial += 1

        for ref, board_item in sorted(board_index.items()):
            required_here = set(required_pads_by_ref.get(ref) or [])
            board_pads = [self._normalize_pad(pad) for pad in list(board_item.get("pads") or []) if self._normalize_pad(pad)]
            if not board_pads:
                continue
            ignored_pads += max(0, len(board_pads) - len(required_here))

        return {
            "coverage_basis": "required_pads",
            "total_refs": len(required_pads_by_ref),
            "total_pads": total_pads,
            "assigned_pads": assigned_pads,
            "unassigned_required_pads_count": max(0, int(total_pads) - int(assigned_pads)),
            "coverage_ratio": round(float(assigned_pads) / float(max(total_pads, 1)), 4),
            "routable_refs_total": routable_refs_total,
            "routable_refs_with_any_nets": routable_refs_with_any_nets,
            "routable_ref_coverage_ratio": round(float(routable_refs_with_any_nets) / float(max(routable_refs_total, 1)), 4),
            "routable_refs_without_nets_count": routable_refs_without_nets,
            "routable_refs_partial_count": routable_refs_partial,
            "refs_without_nets_count": len(refs_without_nets),
            "partial_refs_count": len(partial_refs),
            "refs_without_nets_sample": refs_without_nets[:20],
            "partial_refs_sample": partial_refs[:20],
            "ignored_refs_count": ignored_refs,
            "ignored_pads_count": ignored_pads,
        }

    def _routing_readiness(
        self,
        *,
        manifest: Dict[str, Any],
        board_index: Dict[str, Dict[str, Any]],
        assignments: Sequence[Dict[str, str]],
        coverage: Dict[str, Any],
        bridge_lint: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        _ = bridge_lint
        assignment_lookup: Dict[Tuple[str, str], str] = {}
        for row in assignments:
            ref = self._normalize_ref(row.get("ref"))
            pad = self._normalize_pad(row.get("pad"))
            net = self._sanitize_net_name(row.get("net"))
            if ref and pad and net:
                assignment_lookup[(ref, pad)] = net

        def assigned_net(ref: str, pad: str) -> str:
            board_item = board_index.get(ref)
            if not board_item:
                return ""
            live_pad_nets = board_item.get("pad_nets") or {}
            net_name = self._sanitize_net_name(live_pad_nets.get(pad))
            if net_name:
                return net_name
            return self._sanitize_net_name(assignment_lookup.get((ref, pad)))

        primary_missing: List[str] = []
        for part in list(manifest.get("parts") or []):
            if not isinstance(part, dict):
                continue
            pins = part.get("pins")
            if not isinstance(pins, list) or not pins:
                continue
            ref = self._normalize_ref(part.get("ref"))
            board_item = board_index.get(ref)
            if not ref or board_item is None:
                continue
            missing_pads: List[str] = []
            for pin in pins:
                if not isinstance(pin, dict):
                    continue
                pad = self._normalize_pad(pin.get("num") or pin.get("pad") or pin.get("pin"))
                expected_net = self._sanitize_net_name(pin.get("net"))
                if not pad or not expected_net:
                    continue
                if pad not in set(board_item.get("pad_lookup", set())):
                    continue
                if not assigned_net(ref, pad):
                    missing_pads.append(pad)
            if missing_pads:
                primary_missing.append(f"{ref}({','.join(missing_pads[:4])})")

        if primary_missing:
            return False, f"primary refs still have unassigned manifest pins: {', '.join(primary_missing[:4])}"

        unresolved_support_refs: List[str] = []
        manifest_by_ref = {
            self._normalize_ref(part.get("ref")): part
            for part in list(manifest.get("parts") or [])
            if isinstance(part, dict) and self._normalize_ref(part.get("ref"))
        }
        required_pads_by_ref = self._required_pads_by_ref(
            manifest=manifest,
            board_index=board_index,
        )
        for ref, required_pads in sorted(required_pads_by_ref.items()):
            part = manifest_by_ref.get(ref)
            if not isinstance(part, dict):
                continue
            if isinstance(part.get("pins"), list) and part.get("pins"):
                continue
            if len(required_pads) < 2:
                continue
            assigned_here = sum(1 for pad in required_pads if assigned_net(ref, pad))
            if assigned_here == 0:
                unresolved_support_refs.append(ref)

        if unresolved_support_refs:
            return False, f"support refs still have no netted pads: {', '.join(unresolved_support_refs[:6])}"

        if int(coverage.get("partial_refs_count", 0) or 0) > 0:
            return False, f"{int(coverage.get('partial_refs_count', 0) or 0)} ref(s) still have partially netted required pads"

        pad_coverage = float(coverage.get("coverage_ratio", 0.0) or 0.0)
        if pad_coverage < 1.0:
            return False, f"only {pad_coverage:.0%} of required electrical pads have nets"

        routable_refs_total = int(coverage.get("routable_refs_total", 0) or 0)
        routable_refs_without_nets = int(coverage.get("routable_refs_without_nets_count", 0) or 0)
        if routable_refs_total > 0:
            allowed_without_nets = 0
            if routable_refs_without_nets > allowed_without_nets:
                return (
                    False,
                    f"{routable_refs_without_nets}/{routable_refs_total} multi-pad refs still have no netted pads",
                )
        return True, ""

    def _build_actions(
        self,
        *,
        assignments: Sequence[Dict[str, str]],
        routing_attempted: bool,
        include_autoroute: bool,
        chunk_size: int = 96,
    ) -> List[object]:
        try:
            from ..design_actions import DesignAction, DesignActionType
        except Exception as e:
            logger.error("NET: cannot import DesignAction: %s", e)
            return []

        actions: List[object] = []
        assignment_rows = list(assignments)
        if assignment_rows:
            total_chunks = max(1, math.ceil(len(assignment_rows) / max(chunk_size, 1)))
            for idx in range(total_chunks):
                chunk = assignment_rows[idx * chunk_size:(idx + 1) * chunk_size]
                nets = sorted({row["net"] for row in chunk if row.get("net")})
                actions.append(
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

        if include_autoroute and (not routing_attempted or bool(assignments)):
            actions.append(
                DesignAction(
                    action_type=DesignActionType.AUTOROUTE_BOARD,
                    description="Autoroute the assigned nets with Freerouting",
                    parameters={"router": "freerouting"},
                    requires_approval=False,
                )
            )
        return actions
