# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UNIVERSAL PLUGIN — NO BOARD-SPECIFIC HARDCODING IN THIS FILE      ║
# ║  Prompts must use only goal_str / context variables.               ║
# ║  Never embed specific MPNs, part names, board names, or            ║
# ║  design-specific quantities in prompt strings or system prompts.   ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
ComponentCheckAgent — SPEC agent (v1 3-step design).

Step 1  (first_pass)     — Infer ALL primary parts (ICs, connectors, regulators)
                           with MPN + KiCad footprint string.
Step 2  (datasheet_pass) — Per-part, open a FRESH LLM context to extract pinout
                           and recommend secondary support parts (values + types).
Step 3  (second_pass)    — Consolidate primaries + datasheet findings into the
                           final manifest (secondary parts, net group assignment).

Output manifest schema:
{
  "parts": [
    {
      "ref":       "U1",
      "mpn":       "ATMEGA328P-PU",
      "footprint": "Package_DIP:DIP-28_W7.62",
      "pins": [
        { "num": 7, "name": "VCC", "net": "VCC" },
        { "num": 8, "name": "GND", "net": "GND" }
      ]
    }
  ]
}

Backward-compat: also writes design_spec_draft.roles so existing benchmark
scoring (which looks for design_spec_draft) keeps working.
"""

from __future__ import annotations

import json
import logging
import os
import re
import concurrent.futures
import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import SubAgent, SubAgentResult
from ...llm.digikey_client import DigiKeyClient

logger = logging.getLogger(__name__)

# ── Net-keyword blocklist for non-IC alternates (prevents oracle false positives) ──
_NON_IC_NET_BLOCKLIST = frozenset({"dc", "input", "barrel", "jack", "vin", "power"})
_NET_SKIP_NAMES = frozenset(
    {"", "NC", "N/C", "NO_CONNECT", "UNCONNECTED", "NOT_CONNECTED", "NO_NET", "NONET", "NONE"}
)


def _normalize_value(val: str) -> str:
    """Canonicalize any passive component value string to a compact canonical form.

    Handles resistors (Ω / ohm / R suffix), capacitors (F), and inductors (H).
    Strips whitespace, normalises SI prefix aliases (μ→u, K→k, Meg→M), removes
    unit suffixes for resistors, rescales into the 1–999 window (e.g. 0.1uF→100nF,
    1000K→1M), and returns canonical strings like "100nF", "47uF", "1M", "10k",
    "22pF", "4.7k", "100".
    """
    if not val:
        return val
    v = val.strip()

    # ── Text normalisations before parsing ──────────────────────────────────
    v = re.sub(r'(?<=[0-9\s.])[Mm]eg\b', 'M', v)   # Meg → M
    v = v.replace(',', '')                           # thousands separator
    v = v.replace('μ', 'u').replace('µ', 'u')       # μ / µ → u
    v = re.sub(r'(?<=[0-9\s.])K\b', 'k', v)         # K (capital) → k

    # ── Parse  <number> <prefix> <unit> ─────────────────────────────────────
    _SI: dict[str, float] = {
        'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
        '':  1.0,
        'k': 1e3,   'M': 1e6,  'G': 1e9,
    }

    _m = re.fullmatch(
        r'([0-9]+(?:\.[0-9]+)?)'       # number
        r'\s*([pnumkMG]?)\s*'          # SI prefix (optional)
        r'(Ω|ohms?|[FHR])?'           # unit (optional)
        r'\s*',
        v, re.IGNORECASE,
    )
    if not _m:
        return val.strip()

    num_str, prefix_raw, unit_raw = _m.group(1), _m.group(2), (_m.group(3) or '')

    prefix = prefix_raw if prefix_raw in _SI else prefix_raw.lower()
    if prefix not in _SI:
        return val.strip()

    u = unit_raw.lower()
    if u in ('', 'r', 'ω', 'ohm', 'ohms'):
        unit_out = ''       # resistor: no unit suffix
    elif u == 'f':
        unit_out = 'F'
    elif u == 'h':
        unit_out = 'H'
    else:
        unit_out = ''

    try:
        number = float(num_str)
    except ValueError:
        return val.strip()

    # ── Rescale: pick the prefix that puts the number in [1, 999] ───────────
    # Walking largest-to-smallest, take the first prefix where 1 ≤ scaled < 1000.
    # This avoids expanding canonical "4.7k" → "4700" and prevents FP noise
    # from jumping a prefix tier (e.g. 0.1uF → 100nF, not 100000pF).
    base = number * _SI[prefix]
    ordered = ['G', 'M', 'k', '', 'm', 'u', 'n', 'p']
    best_prefix = prefix
    best_num: float = number
    for p in ordered:
        scaled = round(base / _SI[p], 9)  # 9 dp removes IEEE-754 noise
        if 1.0 <= scaled < 1000.0:
            best_prefix = p
            best_num = scaled
            break
    else:
        # Nothing in window (e.g. sub-pico or supra-giga) – keep original
        best_prefix = prefix
        best_num = number

    # ── Format number: strip trailing zeros/dot via g notation ──────────────
    num_out = f"{best_num:.6g}"

    return f"{num_out}{best_prefix}{unit_out}"


# ── System prompts (one per step, each gets a FRESH LLM context) ───────────

_STEP1_SYSTEM = """You are an expert electronics BOM engineer.
Your task: list ALL board-defining primary parts (ICs, connectors, voltage regulators,
crystals/resonators, power-path protection devices, major switches, and other explicit
functional blocks) required for the requested board design.
Do NOT include passives (resistors, capacitors, LEDs) yet — those are derived from datasheets.

Return ONLY strict JSON (no markdown fences):
{
  "primary_parts": [
    {
      "ref": "U1",
      "role": "<human-readable role description>",
      "mpn": "<manufacturer part number>",
      "footprint": "<KiCad standard-library footprint string>",
      "qty": 1
    }
  ]
}

Rules:
- Be exhaustive: include every IC, connector, regulator, crystal, resonator, polyfuse,
  protection diode, MOSFET, op-amp, and switch required by the design.
- PRIMARY OWNERSHIP RULE: step 1 is for standalone architectural parts that would normally
  appear as their own named schematic/BOM items from the board description alone.
  Do NOT emit tiny local support parts that are better inferred from application circuits
  in step 2, such as per-line clamp diodes, steering diodes, reset helper diodes, or
  local ESD cells, unless the available evidence clearly identifies a distinct standalone
  protection device or integrated protection array as part of the board architecture.
- Do not consolidate distinct physical primary parts just because they share the same
    MPN, package, value, or part family. If the circuit needs two separate discrete diodes,
    crystals, connectors, switches, or other primaries for different roles or different nets,
    emit two separate primary entries with different refs.
- Treat repeated discrete signal-protection or steering diodes as separate primary parts
    whenever the board architecture uses them as separate physical components, even if they
    would resolve to the same diode MPN.
- Do not collapse two or more separately implied line-protection diodes or clamps into a
    single integrated protection array unless the available evidence clearly points to that
    array as the intended physical implementation.
- Use real, specific MPNs — not generic descriptions.
- Use real KiCad standard-library footprint identifiers.
- qty must be an integer >= 1.
- CONNECTOR RULE: Parts with J refs MUST use connector, header, or socket MPNs only.
  Never assign polyfuse, PTC thermistor, fuse, or current-limiting device MPNs to J refs.
- For shielded external connectors, do NOT emit shield-termination passives here.
  Those support parts belong in step 2. Step 1 should only emit the connector itself
  as the primary part.
"""

_STEP1_REVIEW_SYSTEM = """You are reviewing a candidate list of board-defining primary parts.
Your task: correct the list so it is internally coherent before datasheet extraction.

Return ONLY strict JSON (no markdown fences):
{
  "primary_parts": [
    {
      "ref": "U1",
      "role": "<human-readable role description>",
      "mpn": "<manufacturer part number>",
      "footprint": "<KiCad standard-library footprint string>",
      "qty": 1
    }
  ]
}

Rules:
- Preserve the original list where it is already coherent; make the minimum necessary edits.
- Keep only board-defining primary parts. Do not add passives or tiny support parts.
- Correct misclassified or duplicated power-path/protection primaries when two entries
  are pretending to solve the same architectural function without clear evidence.
- Ensure the connector inventory is internally coherent. Programming headers, expansion
  headers, power headers, analog/digital headers, USB connectors, and barrel jacks are
  distinct connector roles and must not be substituted for one another.
- Keep connector roles aligned with footprint geometry and pin count. A 2x3 programming
  header is not a substitute for a long 1x6/1x8/1x10 expansion header, and vice versa.
- Keep timing parts coherent: do not casually replace a crystal with a resonator or emit
  duplicate primary timing parts unless the board architecture clearly needs both.
- Keep reverse-polarity, power-entry, and major protection devices realistic for the role.
- Use real MPNs and real KiCad standard-library footprints.
- qty must be an integer >= 1.
"""

_STEP2_SYSTEM = """You are a datasheet extraction specialist for electronics components.
Given a component's role and MPN, you will:
1. Enumerate EVERY SINGLE PIN (number + name + net hint) physically present on the part. For microcontrollers and connectors, you MUST include all generalized I/O, analog pins, data pins, etc. Do NOT omit pins just because they are unused in a "typical minimal circuit"!
2. List every SUPPORT PART this component requires, with real quantities, using your
   knowledge of its typical application circuit and datasheet.

Return ONLY strict JSON (no markdown fences):
{
  "pins": [
    { "num": 1, "name": "<pin name>", "net": "<net hint>" }
  ],
  "secondaries": [
    {
            "family": "<support family, e.g. decoupling_cap | bulk_cap | clock_load | usb_data_protection | indicator_led | reset_network | shield_termination>",
            "implementation": "<implementation style within that family, e.g. ceramic_cap | electrolytic_cap | discrete_clamp_diodes | tvs_array | crystal_load_caps | discrete_led>",
      "type": "<part type, e.g. capacitor, resistor, LED, ferrite bead, zener diode>",
      "value": "<value, e.g. 100nF, 22pF, 10k, 1M, 3.3V>",
      "qty": <integer — exact count needed FOR THIS COMPONENT based on its application circuit>,
      "function": "<one-line plain-English description of why this part is needed>",
      "net_hint": "<signal or rail it connects to, e.g. VCC, RESET, D+, D->",
      "shared": <true if this is a board-level resource shared across all ICs
                  (e.g. a single bulk cap for a power rail); false if each
                  instance of THIS component independently needs its own copies>
    }
  ]
}

CRITICAL RULES:
- family and implementation are REQUIRED.
- family says WHAT support need this belongs to.
- implementation says HOW that support need is implemented.
- If you choose an implementation for a family, keep that implementation consistent.
    Examples:
        • family=clock_source → implementation must stay crystal OR resonator, not both for the same role
        • family=usb_data_protection → implementation must stay discrete_clamp_diodes OR tvs_array, not both
        • family=indicator_led → implementation should be discrete_led
- qty is REQUIRED. It must equal the number of PHYSICAL PARTS to solder onto the board,
  NOT the number of pins or connections.
  Correct examples:
    • A crystal oscillator → qty 1 (one physical crystal, even though it has XTAL1+XTAL2 pins)
    • USB D+/D− Zener clamps → qty 2 (two separate diode parts, one per data line)
    • Reset pull-up resistor → qty 1 (one resistor part)
    • Decoupling cap on a VCC pin → qty 1 per DISTINCT VCC supply rail
      (NOT one per VCC pin — a chip with 3 VCC pins on the same rail still gets 1 cap)
- shared=true: one instance exists at board level, shared by all ICs on that rail.
  Use for: bulk supply caps, board-level EMI filters. qty = the one board-level count.
    If two shared parts belong on DIFFERENT rails or nodes (e.g. regulator input bulk cap on VIN
    and regulator output bulk cap on +5V), emit them as SEPARATE entries with distinct `net_hint`s.
- shared=true is NOT limited to power rails. Use it for any single board-wide node that multiple
    components connect to but that is implemented with one physical support network on the board,
    such as RESET, RESET#, ENABLE, BOOT, HWB, shield-to-ground EMI parts, or other global control nets.
- For support parts on a shared control net, set shared=true when the board normally has one physical
    pull-up, pull-down, clamp, filter, or similar network for that node and other components merely attach
    to the same node. Do not emit separate non-shared copies just because multiple components mention the same net.
- shared=false (default): each instance of THIS component independently needs its own
  copies. Counts will be summed across all board components at assembly time.
- Use shared=false only when THIS component really needs its own dedicated local copy that would still
    exist as a separate physical part even if another component connects to the same named net.
- Discrete protection devices are almost never board-wide shared resources. For signal
    clamp diodes, steering diodes, TVS arrays, varistors, or ESD suppressors, use
    shared=false unless the evidence clearly points to one specific multi-line device
    protecting the whole interface as a single physical part.
- For shielded external connectors or connectors with exposed shield/chassis pins,
  include any typical shield-termination support network required by the application
  circuit. If the shield is not tied directly to ground, this commonly means separate
  shield/chassis-to-ground passives such as a high-value bleed resistor and/or a small
  capacitor. Emit those as step-2 secondaries, not primaries, using a stable net hint
  such as SHIELD or CHASSIS.
- For generic passive headers, socket headers, board-to-board expansion headers,
  ICSP/programming headers, jumpers, and simple momentary switches, be conservative:
  these parts usually expose or short existing nets rather than introducing new local
  support circuitry.
- Exposed pins by themselves do NOT justify adding per-pin ESD/TVS/clamp devices,
  local bulk capacitors, local decoupling caps, pull-ups, pull-downs, or reset helper
  networks to a generic header or switch.
- Only emit support parts for a generic header or simple switch when the evidence is
  explicit and specific: the connector datasheet/application circuit itself requires it,
  the role is a well-defined external interface that normally includes a dedicated
  support device as part of that interface, or the board context clearly identifies the
  header/switch as the owner of a distinct physical support network.
- In particular, shield/expansion headers and ICSP headers normally contribute zero new
  ESD arrays, zero new rail bulk/decoupling parts, and zero new reset-protection diodes
  on their own. Do not infer one support part per exposed signal.
- If a support part more naturally belongs to the attached IC, regulator, USB connector,
  or other primary interface owner, do not duplicate it under the passive header or
  switch. When ownership is ambiguous, omit the secondary instead of guessing.
- For regulators and power-entry/power-path parts, distinguish local stability caps from
  board-level bulk reservoir caps. Small ceramic caps that satisfy regulator stability
  or high-frequency decoupling do NOT replace larger shared bulk storage on the raw input
  rail and/or the generated main output rail.
- If a regulator or power-path stage sits between two distinct power nodes, keep support
  parts for those nodes separate. For example, an input-side reservoir on VIN and an
  output-side reservoir on +5V are different physical support needs and should be emitted
  as separate entries with distinct net hints rather than collapsed into one generic cap.
- Do not silently downgrade a clearly board-level bulk reservoir into only a small ceramic
  decoupling/stability capacitor unless the evidence specifically says no larger reservoir
  is needed.
- OWNERSHIP RULE: step 2 is the authoritative place for local support/protection parts
  implied by application circuits, including per-line clamp diodes, steering diodes,
  reset helper diodes, local ESD suppressors, and similar small discrete protection
  parts that are not explicit board-level standalone devices.
- DO NOT emit power-path diodes as secondaries: reverse-polarity diodes, rectifier
  diodes, Schottky power diodes, polyfuses, or PTC resettable fuses. Those are always
  primary parts with their own ref. Signal-level clamping diodes on logic/interface
  lines ARE allowed as secondaries.
- DO NOT emit crystals, resonators, oscillators, clock generators, or other primary timing
    components as secondaries. Those are primary parts with their own ref.
- Include every pin, including all power, ground, analog, digital, and general purpose I/O pins, even if not strictly required for standard minimal operation. DO NOT leave a pin out of the `pins` list.
- **SHARED BUS RULE:** For internal board-level communication buses (like UART/I2C/SPI connecting microcontrollers to bridge/interface chips), AND for microcontrollers connecting to shield/expansion headers, you MUST explicitly emit those pins for BOTH connected components and assign them matching `net` hints (e.g. `D0`, `A0`, `Serial_RX`, `SCK`). Do not omit them assuming they are "handled internally" or leave them out of the pin list, otherwise the netlist generator will leave them unconnected.
- Only output the JSON object. No prose.
"""

_STEP3_SYSTEM = """You are a KiCad BOM assembler and experienced PCB design engineer.
Given authoritative primary parts and an authoritative board-level secondary BOM,
produce the COMPLETE board manifest.

CRITICAL RULE — PRIMARY PARTS ARE MANDATORY:
Every single primary part listed in the PRIMARY PARTS section MUST appear in your output
using the EXACT ref, EXACT mpn, EXACT footprint, and EXACT qty provided. Do NOT rename,
renumber, merge, consolidate, or omit ANY primary part.

SECONDARY PARTS ARE ALSO MANDATORY:
The BOARD-LEVEL SECONDARY LIST is authoritative. Do not omit entries. If a secondary
entry has qty N, emit N separate physical parts with unique refs. Do not group them into
one manifest row and do not emit grouped refs like "C1, C2" or "R3/R4".

Rules:
- Copy each primary part's ref / mpn / footprint / qty EXACTLY as given. No changes.
- Emit one manifest item per PHYSICAL board component.
- Assign each secondary part a unique ref (C1, R1, D1, LED1, FB1, etc.).
  Do NOT reuse a ref already claimed by a primary part.
- Never output grouped refs, ranges, comma-separated refs, slash-separated refs, or
  any manifest item that represents more than one physical component.
- For passive parts, use a concrete value/description string as the mpn/query.
- For semiconductors and protection devices, the mpn/query must describe the device
  class, not just a bare electrical value. For example, use a diode/TVS descriptor,
  not just "5V".
- Use KiCad standard-library footprint identifiers when a footprint is provided.
- Do NOT include a "pins" field.

Before answering, internally check:
1. every primary part is present exactly once,
2. the total output part count equals primary_count + sum(secondary qty),
3. every ref is unique and names exactly one physical component.

Return ONLY strict JSON (no markdown fences):
{
  "parts": [
    {
      "ref": "U1",
      "mpn": "<mpn>",
      "footprint": "<footprint>",
      "qty": 1
    }
  ]
}
"""

_STEP2B_SYSTEM = """You are an expert PCB support-BOM reviewer.
Given:
1. the user's board goal,
2. the board's primary parts,
3. a provenance-rich generic aggregate of per-component secondary recommendations,
produce the SINGLE authoritative board-level support BOM.

The provenance aggregate is your baseline. Review it carefully and make the minimum
changes needed to arrive at a realistic board-level support BOM.

Return ONLY strict JSON (no markdown fences):
{
    "secondaries": [
        {
            "family": "<short stable support-need label>",
            "implementation": "<short implementation label>",
            "type": "<part type>",
            "value": "<normalized value>",
            "qty": <integer>,
            "function": "<one-line reason>",
            "net_hint": "<rail or signal hint>",
            "shared": <true or false>,
            "ref_prefix": "<preferred ref prefix such as C, R, D, LED, FB, RN, SW, J, X>",
            "footprint": "<KiCad standard-library footprint string>"
        }
    ]
}

CRITICAL RULES:
- This output is the final authoritative support BOM for the board.
- Start from the provenance aggregate as your baseline rather than inventing a new BOM.
- Preserve entries when the provenance shows distinct rails, distinct net hints, or
    distinct source parts that imply physically separate support parts.
- Only merge entries when they are truly the same physical support need.
- Do not reduce a non-shared aggregate below what the cited source parts still require.
- Do not increase quantities unless the provenance clearly implies distinct physical parts.
- Do not collapse multiple discrete protection devices into one shared entry just because
    they touch the same net. Clamp diodes, steering diodes, TVS arrays, varistors, and
    similar protection parts should keep the physical count implied by the provenance
    unless the evidence clearly identifies one specific shared device.
- Aggressively prune speculative support that is sourced only from passive headers,
    socket headers, ICSP/programming headers, jumpers, or simple reset/user switches.
    Those parts usually do not add their own ESD arrays, reset clamp diodes, local rail
    capacitors, or per-pin protection networks.
- Exposed header pins alone are not enough evidence for interface-protection parts.
    If a protection, bulk-cap, or decoupling entry is only justified by "these signals
    are accessible on a header", remove it unless a real interface standard or explicit
    application circuit calls for a dedicated device there.
- Prefer keeping support ownership with the actual active/interface-defining primary part
    instead of duplicating the same support network under passive headers or switches.
- Preserve distinct bulk reservoirs on distinct power rails. Do not merge or replace an
    input-rail bulk capacitor and an output-rail bulk capacitor with one generic small
    ceramic capacitor just because both belong to the same regulator or power-path area.
- For power architectures, keep local regulator-stability caps separate from larger
    board-level bulk storage. A 10uF ceramic stability cap does not automatically satisfy
    the role of a 47uF shared input/output reservoir.
- Remove implausible entries if they clearly do not fit the board architecture.
- Keep the chosen support architecture coherent. Do not casually switch between
    incompatible implementations unless the aggregate is clearly inconsistent.
- Do not include primary parts: connectors, regulators, crystals, resonators, oscillators,
  MOSFETs, op-amps, switches, polyfuses, reverse-polarity diodes, rectifier diodes,
  Schottky power diodes, or other major active components.
- Think in physical board-level parts, not per-pin counts.
- Provide a real KiCad standard-library footprint for every secondary whenever possible.
- Return only a single JSON object. No prose.
- Output the final list only. No prose.
"""


# ── Agent ──────────────────────────────────────────────────────────────────

class ComponentCheckAgent(SubAgent):
    NAME = "component_check"
    _STEP2_CACHE_VERSION = 8

    # Kept for backward compat — base class uses SYSTEM_PROMPT as default.
    SYSTEM_PROMPT = _STEP1_SYSTEM

    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        try:
            from ..design_agent import DesignActionType
            self.HANDLED_ACTION_TYPES = frozenset({
                DesignActionType.SEARCH_PART,
                DesignActionType.SEARCH_WEB,
                DesignActionType.LOOKUP_DATASHEET,
            })
        except Exception:
            self.HANDLED_ACTION_TYPES = frozenset()
        self._step2_cache_lock = threading.Lock()
        self._step2_cache_loaded = False
        self._step2_cache: Dict[str, Any] = {
            "version": self._STEP2_CACHE_VERSION,
            "step2": {},
            "digikey_excerpt": {},
        }

    # ── Public ────────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        context: Dict[str, Any],
        board_snapshot: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        """Run 3-step SPEC and return manifest + backward-compat roles."""
        benchmark_mode = bool(context.get("benchmark_mode"))
        goal_str = str(goal or "").strip()
        llm_model = ""
        try:
            cfg = getattr(self._llm_client, "config", None)
            llm_model = str(getattr(cfg, "model", "") or "")
        except Exception:
            llm_model = ""

        # ── DigiKey client (optional — only active when credentials are set) ──
        _dk: Optional[DigiKeyClient] = None
        try:
            from ...config import VibeCADSettings
            _sett = VibeCADSettings.load()
            if _sett.digikey_client_id and _sett.digikey_client_secret:
                _dk = DigiKeyClient(_sett.digikey_client_id, _sett.digikey_client_secret)
                logger.info("SPEC: DigiKey client initialised (real datasheet lookups enabled)")
            else:
                logger.debug("SPEC: DigiKey credentials not set — using LLM training memory for datasheets")
        except Exception as _e:
            logger.debug("SPEC: could not load settings for DigiKey: %s", _e)

        # ── Step 1: primary parts ──────────────────────────────────────
        logger.info("SPEC step 1: primary parts inference")
        step1_prompt = (
            f"USER GOAL: {goal_str}\n\n"
            + ("BENCHMARK MODE: be exhaustive and accurate.\n" if benchmark_mode else "")
            + "List ALL primary parts for this board. Return JSON only."
        )
        raw1 = self._llm_chat(step1_prompt, system_prompt=_STEP1_SYSTEM)
        primary_parts: List[Dict[str, Any]] = self._parse_list(raw1, "primary_parts")
        if primary_parts:
            # ── 1.1: Exact footprint check + Coherence Review Loop
            _lm = None
            try:
                from ...design.library_manager import LibraryManager
                _lm = LibraryManager()
            except Exception as e:
                logger.debug("SPEC: could not load LibraryManager for exact footprint review: %s", e)

            max_review_passes = 3
            for review_attempt in range(max_review_passes):
                errors = []
                if _lm:
                    for part in primary_parts:
                        fp = str(part.get("footprint", "") or "").strip()
                        if fp and not _lm._resolve_exact_local_footprint_path(fp):
                            suggestions = []
                            try:
                                parts_colon = fp.split(":", 1)
                                if len(parts_colon) == 2:
                                    term = f"{parts_colon[0]}:{parts_colon[1].split('_')[0]}"
                                else:
                                    term = fp
                                res = _lm.search_parts_sync(term)
                                for match in (res or [])[:15]:
                                    if getattr(match, "local_footprint_path", None) and getattr(match, "name", ""):
                                        if match.name not in suggestions:
                                            suggestions.append(match.name)
                            except Exception:
                                pass
                            
                            sug_str = f" Try one of these existing identical/similar footprints: {', '.join(suggestions[:8])}." if suggestions else ""
                            errors.append(f"Footprint '{fp}' for part '{part.get('ref')}' does not exist in standard KiCad libraries.{sug_str}")

                if not errors and review_attempt > 0:
                    break  # footprints resolved and list reviewed!

                review_prompt = (
                    f"USER GOAL: {goal_str}\n\n"
                    "CANDIDATE PRIMARY PARTS JSON:\n"
                    + self._compact_json(primary_parts)
                    + "\n\nReturn the corrected primary part list as strict JSON only."
                )
                if errors:
                    review_prompt += "\n\nCRITICAL ERRORS IN CANDIDATE LIST:\n"
                    for err in errors:
                        review_prompt += f"- {err}\n"
                    review_prompt += "\nYou MUST fix these footprint strings to exact KiCad library matches."

                try:
                    reviewed_primary_parts = self._parse_list(
                        self._llm_chat(review_prompt, system_prompt=_STEP1_REVIEW_SYSTEM),
                        "primary_parts",
                    )
                    if reviewed_primary_parts:
                        primary_parts = reviewed_primary_parts
                except Exception as e:
                    logger.warning("SPEC step 1 review failed loop %d, keeping current list: %s", review_attempt, e)
                    break

        logger.info("SPEC step 1: got %d primary parts", len(primary_parts))
        primary_parts, primary_sanity_fixes = self._enforce_primary_part_sanity(primary_parts)
        if primary_sanity_fixes:
            logger.warning(
                "SPEC step 1: applied %d deterministic primary sanity fix(es): %s",
                len(primary_sanity_fixes),
                "; ".join(primary_sanity_fixes[:8]),
            )

        # ── Step 2: per-part datasheet extraction (parallel, fresh context each) ─
        def _fetch_datasheet(part: Dict[str, Any]) -> Dict[str, Any]:
            ref  = str(part.get("ref",  "") or "")
            role = str(part.get("role", "") or "")
            mpn  = str(part.get("mpn",  "") or "")
            logger.info("SPEC step 2: datasheet pass for %s (%s)", ref, mpn)

            # ── Real datasheet via DigiKey API ─────────────────────────
            # Only worth fetching for ICs and active components (U/Q) — connectors,
            # passives, diodes etc. rarely have useful secondary-part data in their
            # datasheets and the extra HTTP call just adds latency.
            datasheet_snippet = self._get_digikey_datasheet_snippet(
                mpn=mpn,
                ref=ref,
                digikey_client=_dk,
            )

            ds_prompt = (
                f"Component: {role}\n"
                f"MPN: {mpn}\n"
                f"Ref: {ref}\n"
                f"Footprint: {str(part.get('footprint', '') or '')}\n"
                f"Board context: {goal_str}\n"
                f"Using your engineering knowledge, identify all functional support "
                f"needs for this component (decoupling caps, pull-up/down resistors, "
                f"series resistors, shield-termination passives, ESD parts, etc.).\n"
                "Be conservative for passive headers, sockets, ICSP/programming headers, "
                "jumpers, and simple switches: if the support need is not explicitly owned "
                "by that part, omit it rather than guessing.\n"
                "For regulators and power-entry/power-path parts, keep input-rail bulk, "
                "output-rail bulk, and local stability/decoupling capacitors distinct "
                "when they serve different physical roles.\n"
                + (datasheet_snippet + "\n" if datasheet_snippet else "")
                + "\nReturn JSON only."
            )
            cache_key = self._build_step2_cache_key(
                ref=ref,
                role=role,
                mpn=mpn,
                footprint=str(part.get("footprint", "") or ""),
                llm_model=llm_model,
            )
            cached = self._get_step2_cache_entry(cache_key)
            if isinstance(cached, dict):
                logger.info("SPEC step 2: cache hit for %s (%s)", ref, mpn)
                return {
                    "ref": ref,
                    "mpn": mpn,
                    "role": role,
                    "footprint": str(part.get("footprint", "") or ""),
                    "qty": int(part.get("qty", 1) or 1),
                    "pins": list(cached.get("pins") or []) if isinstance(cached.get("pins"), list) else [],
                    "secondaries": list(cached.get("secondaries") or []) if isinstance(cached.get("secondaries"), list) else [],
                }
            try:
                raw2 = self._llm_chat(ds_prompt, system_prompt=_STEP2_SYSTEM)
                ds_obj = self._parse_json_tolerant(raw2)
            except Exception as e:
                logger.warning("SPEC step 2: failed for %s (%s): %s", ref, mpn, e)
                ds_obj = {}
            pins, raw_secs = self._normalize_step2_payload(ds_obj)
            self._store_step2_cache_entry(
                cache_key,
                {
                    "pins": pins,
                    "secondaries": raw_secs,
                    "meta": {
                        "mpn": mpn,
                        "role": role,
                        "goal": goal_str,
                        "llm_model": llm_model,
                    },
                },
            )
            logger.info("SPEC step 2 secondaries for %s (%s): %s", ref, mpn, raw_secs)
            return {
                "ref":        ref,
                "mpn":        mpn,
                "role":       role,
                "footprint":  str(part.get("footprint", "") or ""),
                "qty":        int(part.get("qty", 1) or 1),
                "pins":       pins,
                "secondaries": raw_secs,
            }

        # Cap concurrency to avoid hammering rate limits; preserve original order.
        _max_workers = min(len(primary_parts), 6)
        datasheet_results: List[Dict[str, Any]] = [{}] * len(primary_parts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=_max_workers) as _pool:
            _futures = {_pool.submit(_fetch_datasheet, part): i for i, part in enumerate(primary_parts)}
            for _fut in concurrent.futures.as_completed(_futures):
                datasheet_results[_futures[_fut]] = _fut.result()

        # ── Step 2b: synthesize authoritative board-level secondaries ───
        logger.info("SPEC step 2b: authoritative board-level secondary synthesis")
        board_secondaries = self._synthesize_board_secondaries(
            goal_str,
            datasheet_results,
        )

        # ── Step 3: assemble full manifest ────────────────────────────
        logger.info("SPEC step 3: full manifest assembly")
        logger.info("SPEC step 3: %d board-level secondaries (aggregated), %d source parts",
                    len(board_secondaries), len(datasheet_results))

        primary_manifest_seed = [
            {
                "ref": str(ds.get("ref", "") or ""),
                "mpn": str(ds.get("mpn", "") or ""),
                "footprint": str(ds.get("footprint", "") or ""),
                "qty": int(ds.get("qty", 1) or 1),
            }
            for ds in datasheet_results
            if isinstance(ds, dict)
        ]
        step3_prompt_base = (
            f"USER GOAL: {goal_str}\n\n"
            "PRIMARY PARTS JSON:\n"
            + self._compact_json(primary_manifest_seed)
            + "\n\nBOARD-LEVEL SECONDARIES JSON:\n"
            + self._compact_json(board_secondaries)
        )
        _STEP3_RETRIES = 3
        manifest: Dict[str, Any] = {"parts": []}
        validation_errors: List[str] = []
        for _attempt in range(1 + _STEP3_RETRIES):
            retry_suffix = ""
            if validation_errors:
                retry_suffix = (
                    "\n\nPREVIOUS ATTEMPT FAILED VALIDATION:\n- "
                    + "\n- ".join(validation_errors[:8])
                    + "\n\nReturn the full corrected manifest from scratch."
                )
            raw3 = self._llm_chat(
                step3_prompt_base + "\n\nReturn the complete manifest as strict JSON only." + retry_suffix,
                system_prompt=_STEP3_SYSTEM,
            )
            manifest = self._parse_json_tolerant(raw3)
            validation_errors = self._validate_step3_manifest(
                manifest.get("parts"),
                primary_manifest_seed=primary_manifest_seed,
                board_secondaries=board_secondaries,
            )
            if not validation_errors:
                break
            if _attempt < _STEP3_RETRIES:
                logger.warning(
                    "SPEC step 3: attempt %d invalid manifest — retrying: %s",
                    _attempt + 1,
                    "; ".join(validation_errors[:3]),
                )
            else:
                logger.error(
                    "SPEC step 3: final manifest still invalid: %s",
                    "; ".join(validation_errors[:5]),
                )

        if not isinstance(manifest.get("parts"), list):
            manifest["parts"] = []

        # Guarantee: re-inject any step-1 primary that step 3 silently dropped.
        self._ensure_primaries_in_manifest(manifest["parts"], datasheet_results)
        self._merge_step2_pins(manifest["parts"], datasheet_results)
        self._attach_secondary_candidates_to_manifest(
            manifest["parts"],
            datasheet_results=datasheet_results,
            board_secondaries=board_secondaries,
        )
        self._backfill_secondary_manifest_metadata(manifest["parts"])
        manifest_net_aliases = self._harmonize_manifest_net_names(manifest["parts"])

        # ── Build backward-compat design_spec_draft.roles ─────────────
        roles = self._manifest_to_roles(manifest["parts"], datasheet_results)

        artifacts: Dict[str, Any] = {
            "manifest":          manifest,
            "design_spec_draft": {"roles": roles},
            "spec_debug": {
                "datasheet_primaries": [
                    {
                        "ref": str(ds.get("ref", "") or ""),
                        "mpn": str(ds.get("mpn", "") or ""),
                        "role": str(ds.get("role", "") or ""),
                        "footprint": str(ds.get("footprint", "") or ""),
                        "qty": int(ds.get("qty", 1) or 1),
                    }
                    for ds in datasheet_results
                    if isinstance(ds, dict)
                ],
                "datasheet_secondaries": [
                    {
                        "ref": str(ds.get("ref", "") or ""),
                        "mpn": str(ds.get("mpn", "") or ""),
                        "role": str(ds.get("role", "") or ""),
                        "secondaries": [
                            self._secondary_debug_row(s)
                            for s in list(ds.get("secondaries") or [])
                            if isinstance(s, dict)
                        ],
                    }
                    for ds in datasheet_results
                    if isinstance(ds, dict)
                ],
                "board_secondaries": [
                    self._secondary_debug_row(s)
                    for s in board_secondaries
                    if isinstance(s, dict)
                ],
                "manifest_net_alias_count": len(manifest_net_aliases),
                "manifest_net_aliases_sample": {
                    key: manifest_net_aliases[key]
                    for key in sorted(manifest_net_aliases.keys())[:40]
                },
            },
        }

        return SubAgentResult(
            message=(
                f"SPEC complete: {len(manifest['parts'])} parts, "
                f"{len(roles)} roles."
            ),
            actions=[],
            confidence=0.9 if manifest["parts"] else 0.4,
            phase_complete=True,
            thinking=(
                f"step1_primary={len(primary_parts)} "
                f"step2_ds={len(datasheet_results)} "
                f"step3_parts={len(manifest['parts'])}"
            ),
            artifacts=artifacts,
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _parse_list(self, raw: str, key: str) -> List[Dict[str, Any]]:
        """Parse JSON and return list at *key*, tolerating markdown fences."""
        obj = self._parse_json_tolerant(raw)
        lst = obj.get(key)
        if isinstance(lst, list):
            return [x for x in lst if isinstance(x, dict)]
        return []

    @staticmethod
    def _text_has_any_token(text: str, tokens: Tuple[str, ...]) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
        return any(tok in normalized for tok in tokens)

    def _fallback_connector_mpn_query(self, part: Dict[str, Any]) -> str:
        role = str(part.get("role", "") or "").strip()
        if role:
            return role
        footprint = str(part.get("footprint", "") or "").strip()
        if footprint:
            tail = footprint.split(":", 1)[-1]
            tail = re.sub(r"[_\\-]+", " ", tail).strip()
            if tail:
                return f"{tail} connector"
        ref = str(part.get("ref", "") or "").strip().upper()
        return f"{ref} connector" if ref else "connector"

    def _enforce_primary_part_sanity(
        self,
        primary_parts: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Apply deterministic guardrails to obvious ref/part-class mismatches."""
        fuse_tokens = ("fuse", "polyfuse", "ptc", "resettable", "msmf", "pptc")
        connector_tokens = ("connector", "header", "socket", "jack", "receptacle", "pinsocket", "pinheader")
        sanitized: List[Dict[str, Any]] = []
        fixes: List[str] = []

        for raw in primary_parts:
            if not isinstance(raw, dict):
                continue
            part = dict(raw)
            ref = str(part.get("ref", "") or "").strip().upper()
            ref_prefix = self._manifest_ref_prefix(ref)
            if ref_prefix == "J":
                mpn = str(part.get("mpn", "") or "").strip()
                role = str(part.get("role", "") or "").strip()
                footprint = str(part.get("footprint", "") or "").strip()
                if mpn and self._text_has_any_token(mpn, fuse_tokens):
                    blob = " ".join((role, footprint, ref)).strip()
                    if self._text_has_any_token(blob, connector_tokens):
                        replacement = self._fallback_connector_mpn_query(part)
                        if replacement and replacement != mpn:
                            part["mpn"] = replacement
                            fixes.append(f"{ref}: fuse-like connector MPN replaced")
            sanitized.append(part)
        return sanitized, fixes

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

    def _build_manifest_net_alias_map(self, parts: List[Dict[str, Any]]) -> Dict[str, str]:
        nets_upper: set[str] = set()
        net_pin_counts: Dict[str, int] = {}
        pin_rows: List[Tuple[str, str]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            for pin in list(part.get("pins") or []):
                if not isinstance(pin, dict):
                    continue
                net = self._sanitize_net_name(pin.get("net"))
                if not net:
                    continue
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
            explicit = [n for n in candidates if re.search(rf"[_\-./]+{re.escape(fn)}$", n)]
            chosen = ""
            if len(explicit) == 1:
                chosen = explicit[0]
            elif len(candidates) == 1:
                chosen = candidates[0]
            if chosen:
                aliases.setdefault(fn, chosen)

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
            return bool(re.match(r"^(VCC|VDD|AVCC|DVCC|UVCC|VCCIO|VDDIO|IOVCC|IOVDD|VIO)$", name))

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

    def _harmonize_manifest_net_names(self, parts: List[Dict[str, Any]]) -> Dict[str, str]:
        aliases = self._build_manifest_net_alias_map(parts)

        def _apply(raw: Any) -> str:
            net = self._sanitize_net_name(raw)
            if not net:
                return ""
            return str(aliases.get(net.upper(), net))

        for part in parts:
            if not isinstance(part, dict):
                continue
            pins = part.get("pins")
            if isinstance(pins, list):
                for pin in pins:
                    if not isinstance(pin, dict):
                        continue
                    net = _apply(pin.get("net"))
                    if net:
                        pin["net"] = net
                    elif "net" in pin:
                        pin.pop("net", None)

            candidates = part.get("support_candidates")
            if isinstance(candidates, list):
                for row in candidates:
                    if not isinstance(row, dict):
                        continue
                    net_hint = _apply(row.get("net_hint"))
                    if net_hint:
                        row["net_hint"] = net_hint
                    elif "net_hint" in row:
                        row.pop("net_hint", None)
                    if isinstance(row.get("source_nets"), list):
                        folded = []
                        seen: set[str] = set()
                        for value in row.get("source_nets"):
                            net_name = _apply(value)
                            if net_name and net_name not in seen:
                                seen.add(net_name)
                                folded.append(net_name)
                        row["source_nets"] = folded
        return aliases

    @staticmethod
    def _cache_file_path() -> Path:
        configured = str(os.environ.get("VIBECAD_SPEC_STEP2_CACHE", "") or "").strip()
        if configured:
            return Path(configured).expanduser()
        debug_dir = Path(__file__).resolve().parents[2] / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir / "spec_step2_cache.json"

    @staticmethod
    def _stable_hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()

    def _ensure_step2_cache_loaded(self) -> None:
        with self._step2_cache_lock:
            if self._step2_cache_loaded:
                return
            data = {
                "version": self._STEP2_CACHE_VERSION,
                "step2": {},
                "digikey_excerpt": {},
            }
            path = self._cache_file_path()
            try:
                if path.exists():
                    raw = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and int(raw.get("version", 0) or 0) == self._STEP2_CACHE_VERSION:
                        data["step2"] = raw.get("step2") if isinstance(raw.get("step2"), dict) else {}
                        data["digikey_excerpt"] = raw.get("digikey_excerpt") if isinstance(raw.get("digikey_excerpt"), dict) else {}
                    else:
                        logger.info("SPEC step 2 cache: ignoring incompatible cache format at %s", path)
            except Exception as e:
                logger.warning("SPEC step 2 cache: failed to load %s: %s", path, e)
            self._step2_cache = data
            self._step2_cache_loaded = True

    def _persist_step2_cache(self) -> None:
        path = self._cache_file_path()
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": self._STEP2_CACHE_VERSION,
            "step2": self._step2_cache.get("step2", {}),
            "digikey_excerpt": self._step2_cache.get("digikey_excerpt", {}),
        }
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)

    def _get_step2_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        self._ensure_step2_cache_loaded()
        with self._step2_cache_lock:
            entry = self._step2_cache.get("step2", {}).get(cache_key)
            return dict(entry) if isinstance(entry, dict) else None

    def _store_step2_cache_entry(self, cache_key: str, payload: Dict[str, Any]) -> None:
        if not cache_key or not isinstance(payload, dict):
            return
        self._ensure_step2_cache_loaded()
        with self._step2_cache_lock:
            bucket = self._step2_cache.setdefault("step2", {})
            if not isinstance(bucket, dict):
                bucket = {}
                self._step2_cache["step2"] = bucket
            bucket[cache_key] = {
                "pins": payload.get("pins") if isinstance(payload.get("pins"), list) else [],
                "secondaries": payload.get("secondaries") if isinstance(payload.get("secondaries"), list) else [],
                "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
                "updated_at": int(time.time()),
            }
            self._persist_step2_cache()

    def _get_digikey_excerpt_cache_entry(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not cache_key:
            return None
        self._ensure_step2_cache_loaded()
        with self._step2_cache_lock:
            entry = self._step2_cache.get("digikey_excerpt", {}).get(cache_key)
            return dict(entry) if isinstance(entry, dict) else None

    def _store_digikey_excerpt_cache_entry(self, cache_key: str, payload: Dict[str, Any]) -> None:
        if not cache_key or not isinstance(payload, dict):
            return
        self._ensure_step2_cache_loaded()
        with self._step2_cache_lock:
            bucket = self._step2_cache.setdefault("digikey_excerpt", {})
            if not isinstance(bucket, dict):
                bucket = {}
                self._step2_cache["digikey_excerpt"] = bucket
            bucket[cache_key] = {
                "excerpt": str(payload.get("excerpt", "") or ""),
                "status": str(payload.get("status", "") or ""),
                "updated_at": int(time.time()),
            }
            self._persist_step2_cache()

    def _build_step2_cache_key(
        self,
        *,
        ref: str,
        role: str,
        mpn: str,
        footprint: str,
        llm_model: str,
    ) -> str:
        material = json.dumps(
            {
                "v": self._STEP2_CACHE_VERSION,
                "mpn": self._bench_norm_text(mpn),
                "role": self._bench_norm_text(role),
                "ref_prefix": self._bench_norm_text(ref[:1]),
                "footprint": footprint,
                "llm_model": llm_model,
                "step2_system": self._stable_hash(_STEP2_SYSTEM),
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return self._stable_hash(material)

    @staticmethod
    def _bench_norm_text(text: str) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _get_digikey_datasheet_snippet(
        self,
        *,
        mpn: str,
        ref: str,
        digikey_client: Optional[DigiKeyClient],
    ) -> str:
        if digikey_client is None or not mpn or not ref.startswith(("U", "Q")):
            return ""
        cache_key = self._stable_hash(json.dumps({"v": 1, "mpn": mpn}, sort_keys=True))
        cached = self._get_digikey_excerpt_cache_entry(cache_key)
        if isinstance(cached, dict):
            status = str(cached.get("status", "") or "")
            excerpt = str(cached.get("excerpt", "") or "")
            logger.info("SPEC step 2: DigiKey cache hit for %s (%s)", ref, mpn)
            if status == "ok" and excerpt:
                return excerpt
            return ""
        excerpt = ""
        status = "missing"
        try:
            pdf_url = digikey_client.get_datasheet_url(mpn)
            if pdf_url:
                text = digikey_client.fetch_datasheet_text(pdf_url)
                if text.strip():
                    excerpt = (
                        f"\n\nDATASHEET EXCERPT (use to validate or correct your value "
                        f"choices — do not add new secondary types not in your reasoning):\n"
                        f"{text[:8000]}"
                    )
                    status = "ok"
                    logger.info("SPEC step 2: datasheet injected for %s (%d chars)", mpn, len(text))
                else:
                    status = "unreadable"
                    logger.info("SPEC step 2: datasheet PDF unreadable for %s (using training knowledge)", mpn)
            else:
                logger.info("SPEC step 2: no datasheet URL from DigiKey for %s", mpn)
        except Exception as e:
            status = "error"
            logger.warning("SPEC step 2: DigiKey lookup failed for %s: %s", mpn, e)
        self._store_digikey_excerpt_cache_entry(
            cache_key,
            {"excerpt": excerpt, "status": status},
        )
        return excerpt if status == "ok" else ""

    @staticmethod
    def _sanitize_ref_prefix(value: str) -> str:
        prefix = re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())
        return prefix[:6]

    @staticmethod
    def _secondary_match_key(item: Dict[str, Any]) -> str:
        return json.dumps(
            {
                "family": str(item.get("family", "") or ""),
                "implementation": str(item.get("implementation", "") or ""),
                "type": str(item.get("type", "") or ""),
                "value": str(item.get("value", "") or ""),
                "net_hint": str(item.get("net_hint", "") or ""),
                "ref_prefix": str(item.get("ref_prefix", "") or ""),
                "footprint": str(item.get("footprint", "") or ""),
            },
            sort_keys=True,
            ensure_ascii=True,
        )

    @staticmethod
    def _is_protection_secondary(item: Dict[str, Any]) -> bool:
        text = " ".join(
            str(item.get(key, "") or "")
            for key in ("family", "implementation", "type", "value", "function", "net_hint", "footprint")
        ).lower()
        if not any(tok in text for tok in ("diode", "zener", "tvs", "varistor", "esd", "clamp", "suppressor", "steering")):
            return False
        if any(tok in text for tok in ("reverse polarity", "rectifier", "schottky", "power rectifier", "power path")):
            return False
        return True

    def _sanitize_secondary_entry(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        stype = str(item.get("type", "") or "").strip()
        if not stype:
            return None
        try:
            qty = max(1, int(item.get("qty", 1) or 1))
        except (TypeError, ValueError):
            qty = 1
        value = str(item.get("value", "") or "").strip()
        sanitized: Dict[str, Any] = {
            "family": str(item.get("family", "") or "").strip(),
            "implementation": str(item.get("implementation", "") or "").strip(),
            "type": stype,
            "value": _normalize_value(value) if value else "",
            "qty": qty,
            "function": str(item.get("function", "") or "").strip(),
            "net_hint": str(item.get("net_hint", "") or "").strip(),
            "shared": bool(item.get("shared", False)),
        }
        ref_prefix = self._sanitize_ref_prefix(str(item.get("ref_prefix", "") or ""))
        if ref_prefix:
            sanitized["ref_prefix"] = ref_prefix
        footprint = str(item.get("footprint", "") or "").strip()
        if footprint:
            sanitized["footprint"] = footprint
        if self._is_protection_secondary(sanitized):
            sanitized["shared"] = False
            if not sanitized.get("ref_prefix"):
                type_text = str(sanitized.get("type", "") or "").lower()
                sanitized["ref_prefix"] = "RV" if "varistor" in type_text else "D"
        return sanitized

    def _secondary_debug_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "family": str(item.get("family", "") or ""),
            "implementation": str(item.get("implementation", "") or ""),
            "type": str(item.get("type", "") or ""),
            "value": str(item.get("value", "") or ""),
            "qty": int(item.get("qty", 1) or 1),
            "function": str(item.get("function", "") or ""),
            "net_hint": str(item.get("net_hint", "") or ""),
            "shared": bool(item.get("shared", False)),
            "ref_prefix": str(item.get("ref_prefix", "") or ""),
            "footprint": str(item.get("footprint", "") or ""),
        }
        if isinstance(item.get("source_refs"), list) and item.get("source_refs"):
            row["source_refs"] = [str(v or "") for v in item.get("source_refs") if str(v or "").strip()]
            row["source_count"] = int(item.get("source_count", len(row["source_refs"])) or len(row["source_refs"]))
        if isinstance(item.get("source_nets"), list) and item.get("source_nets"):
            row["source_nets"] = [str(v or "") for v in item.get("source_nets") if str(v or "").strip()]
        if isinstance(item.get("source_functions"), list) and item.get("source_functions"):
            row["source_functions"] = [str(v or "") for v in item.get("source_functions") if str(v or "").strip()]
        if "non_shared_source_count" in item:
            row["non_shared_source_count"] = int(item.get("non_shared_source_count", 0) or 0)
        if "shared_source_count" in item:
            row["shared_source_count"] = int(item.get("shared_source_count", 0) or 0)
        return row

    @staticmethod
    def _manifest_ref_prefix(ref: str) -> str:
        match = re.match(r"^\s*([A-Za-z]+)", str(ref or "").strip())
        return match.group(1).upper() if match else ""

    def _merge_secondary_provenance(
        self,
        item: Dict[str, Any],
        matches: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        enriched = dict(item)
        source_refs: List[str] = []
        source_nets: List[str] = []
        source_functions: List[str] = []
        longest_function = str(enriched.get("function", "") or "")
        non_shared_source_count = int(enriched.get("non_shared_source_count", 0) or 0)
        shared_source_count = int(enriched.get("shared_source_count", 0) or 0)

        for match in matches:
            if not isinstance(match, dict):
                continue
            function = str(match.get("function", "") or "").strip()
            if len(function) > len(longest_function):
                longest_function = function
            if not enriched.get("ref_prefix") and match.get("ref_prefix"):
                enriched["ref_prefix"] = str(match.get("ref_prefix", "") or "")
            if not enriched.get("footprint") and match.get("footprint"):
                enriched["footprint"] = str(match.get("footprint", "") or "")
            for ref in list(match.get("source_refs") or []):
                ref_s = str(ref or "").strip()
                if ref_s and ref_s not in source_refs:
                    source_refs.append(ref_s)
            for net in list(match.get("source_nets") or []):
                net_s = str(net or "").strip()
                if net_s and net_s not in source_nets:
                    source_nets.append(net_s)
            for fn in list(match.get("source_functions") or []):
                fn_s = str(fn or "").strip()
                if fn_s and fn_s not in source_functions:
                    source_functions.append(fn_s)
            if function and function not in source_functions:
                source_functions.append(function)
            non_shared_source_count = max(non_shared_source_count, int(match.get("non_shared_source_count", 0) or 0))
            shared_source_count = max(shared_source_count, int(match.get("shared_source_count", 0) or 0))

        if longest_function:
            enriched["function"] = longest_function
        if source_refs:
            enriched["source_refs"] = source_refs
            enriched["source_count"] = len(source_refs)
        if source_nets:
            enriched["source_nets"] = source_nets
        if source_functions:
            enriched["source_functions"] = source_functions[:8]
        if non_shared_source_count:
            enriched["non_shared_source_count"] = non_shared_source_count
        if shared_source_count:
            enriched["shared_source_count"] = shared_source_count
        return enriched

    def _enrich_secondaries_with_provenance(
        self,
        secondaries: List[Dict[str, Any]],
        fallback_secondaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        fallback_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for raw in fallback_secondaries:
            if not isinstance(raw, dict):
                continue
            sanitized = self._sanitize_secondary_entry(raw)
            if not sanitized:
                continue
            key = self._secondary_match_key(sanitized)
            fallback_by_key.setdefault(key, []).append(dict(raw))

        enriched: List[Dict[str, Any]] = []
        for item in secondaries:
            if not isinstance(item, dict):
                continue
            key = self._secondary_match_key(item)
            matches = fallback_by_key.get(key, [])
            if matches:
                enriched.append(self._merge_secondary_provenance(item, matches))
            else:
                enriched.append(dict(item))
        return enriched

    def _secondary_candidates_for_part(
        self,
        part: Dict[str, Any],
        board_secondaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ref = str(part.get("ref", "") or "").strip().upper()
        if not ref:
            return []
        mpn_value = _normalize_value(str(part.get("mpn", "") or "").strip())
        footprint = str(part.get("footprint", "") or "").strip()
        ref_prefix = self._manifest_ref_prefix(ref)
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for raw in board_secondaries:
            if not isinstance(raw, dict):
                continue
            entry = self._sanitize_secondary_entry(raw)
            if not entry:
                continue
            entry_value = _normalize_value(str(entry.get("value", "") or "").strip())
            if mpn_value and entry_value and mpn_value != entry_value:
                continue
            if mpn_value and not entry_value:
                continue
            entry_prefix = self._sanitize_ref_prefix(str(entry.get("ref_prefix", "") or ""))
            if entry_prefix and ref_prefix and not ref_prefix.startswith(entry_prefix):
                continue
            entry_footprint = str(entry.get("footprint", "") or "").strip()
            if footprint and entry_footprint and footprint != entry_footprint:
                continue
            candidate = self._secondary_debug_row(raw)
            key = json.dumps(candidate, sort_keys=True, ensure_ascii=True)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return candidates

    def _attach_secondary_candidates_to_manifest(
        self,
        parts: List[Dict[str, Any]],
        *,
        datasheet_results: List[Dict[str, Any]],
        board_secondaries: List[Dict[str, Any]],
    ) -> None:
        primary_refs = {
            str(ds.get("ref", "") or "").strip().upper()
            for ds in datasheet_results
            if isinstance(ds, dict) and str(ds.get("ref", "") or "").strip()
        }
        for part in parts:
            if not isinstance(part, dict):
                continue
            ref = str(part.get("ref", "") or "").strip().upper()
            if not ref or ref in primary_refs:
                continue
            candidates = self._secondary_candidates_for_part(part, board_secondaries)
            if candidates:
                part["support_candidates"] = candidates

    @staticmethod
    def _looks_like_numeric_value(text: str) -> bool:
        token = str(text or "").strip().lower()
        if not token:
            return False
        token = token.replace("µ", "u").replace("μ", "u")
        token = token.replace(" ", "")
        if re.search(r"\d", token) is None:
            return False
        if re.search(r"(?:pf|nf|uf|mf|f|ohm|r|k|m|v|hz)\b", token):
            return True
        if re.match(r"^\d+(?:\.\d+)?[rkmunp]$", token):
            return True
        return False

    def _backfill_secondary_manifest_metadata(
        self,
        parts: List[Dict[str, Any]],
    ) -> None:
        """
        Fill missing secondary value/description fields from support-candidate
        evidence so downstream checks can reason about part intent.
        """
        for part in parts:
            if not isinstance(part, dict):
                continue
            ref = str(part.get("ref", "") or "").strip().upper()
            if not ref:
                continue
            pins = [row for row in list(part.get("pins") or []) if isinstance(row, dict)]
            if pins:
                continue

            support_candidates = [
                row for row in list(part.get("support_candidates") or [])
                if isinstance(row, dict)
            ]
            if not support_candidates:
                continue

            ref_prefix = self._manifest_ref_prefix(ref)
            part_value_norm = _normalize_value(str(part.get("value", "") or "").strip())
            part_mpn_norm = _normalize_value(str(part.get("mpn", "") or "").strip())
            matched_values: List[str] = []
            matched_functions: List[str] = []

            for candidate in support_candidates:
                candidate_prefix = self._sanitize_ref_prefix(str(candidate.get("ref_prefix", "") or ""))
                if candidate_prefix and ref_prefix and not ref_prefix.startswith(candidate_prefix):
                    continue
                candidate_value = _normalize_value(str(candidate.get("value", "") or "").strip())
                if candidate_value:
                    matched_values.append(candidate_value)
                function = str(candidate.get("function", "") or "").strip()
                if function:
                    matched_functions.append(function)

            if not part_value_norm:
                unique_values = sorted({value for value in matched_values if value})
                chosen_value = ""
                if part_mpn_norm and part_mpn_norm in unique_values:
                    chosen_value = part_mpn_norm
                elif len(unique_values) == 1:
                    chosen_value = unique_values[0]
                elif part_mpn_norm and self._looks_like_numeric_value(part_mpn_norm):
                    chosen_value = part_mpn_norm
                if chosen_value:
                    part["value"] = chosen_value

            if not str(part.get("description", "") or "").strip() and matched_functions:
                longest = max((str(fn or "") for fn in matched_functions), key=len, default="")
                if longest:
                    part["description"] = longest

    def _reconcile_board_secondaries(
        self,
        secondaries: List[Dict[str, Any]],
        fallback_secondaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Preserve protection-device counts from provenance and explode discrete qtys.

        The LLM board-level synthesis is allowed to clean up passives, but it should
        not collapse separate physical protection parts into one shared entry.
        """
        result = [dict(item) for item in secondaries if isinstance(item, dict)]
        key_to_indexes: Dict[str, List[int]] = {}
        for idx, item in enumerate(result):
            key_to_indexes.setdefault(self._secondary_match_key(item), []).append(idx)

        for raw in fallback_secondaries:
            entry = self._sanitize_secondary_entry(raw)
            if not entry or not self._is_protection_secondary(entry):
                continue
            key = self._secondary_match_key(entry)
            existing_indexes = key_to_indexes.get(key, [])
            if existing_indexes:
                for idx in existing_indexes:
                    result[idx]["shared"] = False
                    if not result[idx].get("ref_prefix") and entry.get("ref_prefix"):
                        result[idx]["ref_prefix"] = entry["ref_prefix"]
                    if not result[idx].get("footprint") and entry.get("footprint"):
                        result[idx]["footprint"] = entry["footprint"]
            existing_qty = sum(int(result[i].get("qty", 1) or 1) for i in existing_indexes)
            needed_qty = int(entry.get("qty", 1) or 1)
            if existing_qty >= needed_qty:
                continue
            if existing_indexes:
                deficit = needed_qty - existing_qty
                result[existing_indexes[0]]["qty"] = int(result[existing_indexes[0]].get("qty", 1) or 1) + deficit
            else:
                result.append(dict(entry))
                key_to_indexes.setdefault(key, []).append(len(result) - 1)

        expanded: List[Dict[str, Any]] = []
        for item in result:
            if self._is_protection_secondary(item) and not bool(item.get("shared", False)):
                qty = max(1, int(item.get("qty", 1) or 1))
                for _ in range(qty):
                    clone = dict(item)
                    clone["qty"] = 1
                    expanded.append(clone)
                continue
            expanded.append(item)
        return self._enrich_secondaries_with_provenance(expanded, fallback_secondaries)

    def _normalize_step2_payload(self, ds_obj: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pins = [p for p in (ds_obj.get("pins") if isinstance(ds_obj.get("pins"), list) else []) if isinstance(p, dict)]
        raw_secs: List[Dict[str, Any]] = []
        for item in (ds_obj.get("secondaries") if isinstance(ds_obj.get("secondaries"), list) else []):
            sanitized = self._sanitize_secondary_entry(item)
            if sanitized:
                raw_secs.append(sanitized)
        return pins, raw_secs

    def _parse_json_tolerant(self, raw: str) -> Dict[str, Any]:
        """Extract and parse the outermost JSON object, fixing common quirks."""
        try:
            from ..design_agent import sanitize_llm_json_text
            _sanitize = sanitize_llm_json_text
        except Exception:
            _sanitize = lambda t: t  # noqa: E731

        def _fix(text: str) -> str:
            # Strip trailing commas before ] or }
            text = re.sub(r",\s*([}\]])", r"\1", text)
            # Insert missing commas: "value"\n  "key" or number/bool/null\n  "key"
            text = re.sub(
                r'("(?:[^"\\]|\\.)*"|true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
                r'(\s*\n\s*)(")',
                r'\1,\2\3',
                text,
            )
            # Insert missing commas: } or ] followed by { or [
            text = re.sub(r'([}\]])\s*\n(\s*)([{\[])', r'\1,\n\2\3', text)
            return text

        text = self._extract_json_object(raw) or raw
        text = _fix(text)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        cleaned = _sanitize(raw or "")
        text2 = self._extract_json_object(cleaned) or cleaned
        text2 = _fix(text2)
        try:
            obj = json.loads(text2)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logger.error(
                "component_check tolerant parse failed: %s | first80=%r",
                e, (raw or "")[:80],
            )
        return {}

    def _fallback_board_secondaries(
        self,
        ds_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generic fallback aggregation used only when board-level synthesis fails."""
        from collections import defaultdict

        groups: Dict[str, List] = defaultdict(list)

        for ds in ds_results:
            src_ref = str(ds.get("ref", "") or "")
            for s in (ds.get("secondaries") or []):
                entry = self._sanitize_secondary_entry(s)
                if not entry:
                    continue
                key = json.dumps(
                    {
                        "family": entry.get("family", ""),
                        "implementation": entry.get("implementation", ""),
                        "type": entry.get("type", ""),
                        "value": entry.get("value", ""),
                        "net_hint": entry.get("net_hint", ""),
                        "ref_prefix": entry.get("ref_prefix", ""),
                        "footprint": entry.get("footprint", ""),
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                groups[key].append((src_ref, entry))

        result: List[Dict[str, Any]] = []
        for emitters in groups.values():
            first = emitters[0][1]
            shared_qtys = [int(e[1].get("qty", 1) or 1) for e in emitters if bool(e[1].get("shared", False))]
            per_ic_max: Dict[str, int] = {}
            source_refs: List[str] = []
            source_nets: List[str] = []
            source_functions: List[str] = []
            for src_ref, entry in emitters:
                if src_ref and src_ref not in source_refs:
                    source_refs.append(src_ref)
                net_hint = str(entry.get("net_hint", "") or "").strip()
                if net_hint and net_hint not in source_nets:
                    source_nets.append(net_hint)
                function = str(entry.get("function", "") or "").strip()
                if function and function not in source_functions:
                    source_functions.append(function)
                if not bool(entry.get("shared", False)):
                    q = int(entry.get("qty", 1) or 1)
                    per_ic_max[src_ref] = max(per_ic_max.get(src_ref, 0), q)
            per_ic_qtys = list(per_ic_max.values())
            shared_total = max(shared_qtys)     if shared_qtys  else 0
            per_ic_total = sum(per_ic_qtys)     if per_ic_qtys  else 0
            qty = max(shared_total, per_ic_total)
            if qty <= 0:
                qty = 1

            aggregated = dict(first)
            aggregated["qty"] = qty
            aggregated["shared"] = bool(shared_qtys) and not bool(per_ic_max)
            if not aggregated.get("function"):
                aggregated["function"] = max(
                    (str(e[1].get("function", "") or "") for e in emitters),
                    key=len,
                    default="",
                )
            aggregated["source_refs"] = source_refs
            aggregated["source_count"] = len(source_refs)
            aggregated["source_nets"] = source_nets
            if source_functions:
                aggregated["source_functions"] = source_functions[:4]
            aggregated["non_shared_source_count"] = len(per_ic_max)
            aggregated["shared_source_count"] = len(shared_qtys)
            result.append(aggregated)
        return result

    def _synthesize_board_secondaries(
        self,
        goal_str: str,
        ds_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Use a fresh LLM context as the single source of truth for board secondaries."""
        primary_parts = [
            {
                "ref": str(ds.get("ref", "") or ""),
                "mpn": str(ds.get("mpn", "") or ""),
                "role": str(ds.get("role", "") or ""),
                "footprint": str(ds.get("footprint", "") or ""),
                "qty": int(ds.get("qty", 1) or 1),
            }
            for ds in ds_results
            if isinstance(ds, dict)
        ]
        fallback_secondaries = self._fallback_board_secondaries(ds_results)
        prompt = (
            f"USER GOAL: {goal_str}\n\n"
            "PRIMARY PARTS:\n"
            + self._compact_json(primary_parts)
            + "\n\nPROVENANCE AGGREGATE:\n"
            + self._compact_json(fallback_secondaries)
            + "\n\nReturn the final authoritative board-level support BOM as JSON."
        )
        try:
            raw = self._llm_chat(prompt, system_prompt=_STEP2B_SYSTEM)
            secondaries = [
                sanitized
                for sanitized in (self._sanitize_secondary_entry(item) for item in self._parse_list(raw, "secondaries"))
                if sanitized
            ]
            if secondaries:
                secondaries = self._reconcile_board_secondaries(secondaries, fallback_secondaries)
                logger.info(
                    "SPEC step 2b: synthesized %d authoritative board-level secondaries",
                    len(secondaries),
                )
                return secondaries
            logger.warning("SPEC step 2b: synthesis returned no secondaries, using generic fallback aggregate")
        except Exception as e:
            logger.warning("SPEC step 2b: board-level synthesis failed, using generic fallback aggregate: %s", e)
        return self._reconcile_board_secondaries(
            [
                sanitized
                for sanitized in (self._sanitize_secondary_entry(item) for item in fallback_secondaries)
                if sanitized
            ],
            fallback_secondaries,
        )

    def _ensure_primaries_in_manifest(
        self,
        parts: List[Dict[str, Any]],
        ds_results: List[Dict[str, Any]],
    ) -> None:
        """Re-inject any step-1 primary part that step 3 silently dropped."""
        existing_refs = {str(p.get("ref", "") or "").upper() for p in parts}
        existing_mpns = {str(p.get("mpn", "") or "").upper() for p in parts}
        for ds in ds_results:
            ref = str(ds.get("ref", "") or "")
            mpn = str(ds.get("mpn", "") or "")
            if ref.upper() in existing_refs or (mpn and mpn.upper() in existing_mpns):
                continue  # already present
            logger.warning("SPEC step3 dropped %s (%s) — re-injecting from step-1", ref, mpn)
            parts.append({
                "ref":       ref,
                "mpn":       mpn,
                "footprint": str(ds.get("footprint", "") or ""),
                "qty":       int(ds.get("qty", 1) or 1),
                "pins":      ds.get("pins") if isinstance(ds.get("pins"), list) else [],
            })

    def _validate_step3_manifest(
        self,
        parts: Any,
        *,
        primary_manifest_seed: List[Dict[str, Any]],
        board_secondaries: List[Dict[str, Any]],
    ) -> List[str]:
        if not isinstance(parts, list) or not parts:
            return ["manifest returned no parts list"]

        errors: List[str] = []
        dict_parts = [part for part in parts if isinstance(part, dict)]
        if len(dict_parts) != len(parts):
            errors.append("manifest contains non-object part entries")

        expected_secondary_total = sum(
            max(1, int(raw.get("qty", 1) or 1))
            for raw in board_secondaries
            if isinstance(raw, dict)
        )
        expected_total = len(primary_manifest_seed) + expected_secondary_total
        if len(dict_parts) != expected_total:
            errors.append(
                f"expected exactly {expected_total} physical parts "
                f"({len(primary_manifest_seed)} primaries + {expected_secondary_total} secondaries), "
                f"got {len(dict_parts)}"
            )

        seen_by_ref: Dict[str, Dict[str, Any]] = {}
        duplicate_refs: List[str] = []
        grouped_refs: List[str] = []
        missing_refs = 0
        for part in dict_parts:
            ref = str(part.get("ref", "") or "").strip()
            if not ref:
                missing_refs += 1
                continue
            if (
                any(ch in ref for ch in ",;/")
                or len(ref.split()) > 1
                or re.search(r"[A-Za-z]+\d+\s*-\s*[A-Za-z]*\d+", ref)
            ):
                grouped_refs.append(ref)
            ref_key = ref.upper()
            if ref_key in seen_by_ref:
                duplicate_refs.append(ref)
                continue
            seen_by_ref[ref_key] = part
        if missing_refs:
            errors.append(f"{missing_refs} part(s) are missing refs")
        if grouped_refs:
            errors.append("grouped refs are not allowed: " + ", ".join(grouped_refs[:6]))
        if duplicate_refs:
            errors.append("duplicate refs are not allowed: " + ", ".join(duplicate_refs[:6]))

        missing_primaries: List[str] = []
        changed_primaries: List[str] = []
        for seed in primary_manifest_seed:
            ref = str(seed.get("ref", "") or "").strip()
            if not ref:
                continue
            actual = seen_by_ref.get(ref.upper())
            if not actual:
                missing_primaries.append(ref)
                continue
            expected_mpn = str(seed.get("mpn", "") or "").strip()
            actual_mpn = str(actual.get("mpn", "") or "").strip()
            expected_footprint = str(seed.get("footprint", "") or "").strip()
            actual_footprint = str(actual.get("footprint", "") or "").strip()
            expected_qty = int(seed.get("qty", 1) or 1)
            actual_qty = int(actual.get("qty", 1) or 1)
            if actual_mpn != expected_mpn or actual_footprint != expected_footprint or actual_qty != expected_qty:
                changed_primaries.append(ref)
        if missing_primaries:
            errors.append("missing primary refs: " + ", ".join(missing_primaries[:10]))
        if changed_primaries:
            errors.append("primary parts were modified: " + ", ".join(changed_primaries[:10]))
        return errors

    def _merge_step2_pins(
        self,
        parts: List[Dict[str, Any]],
        ds_results: List[Dict[str, Any]],
    ) -> None:
        """Fill in missing pin lists on primary parts from step-2 data."""
        ds_by_ref = {r["ref"]: r for r in ds_results if r.get("ref")}
        for part in parts:
            ref = str(part.get("ref", "") or "")
            if not isinstance(part.get("pins"), list) or not part["pins"]:
                ds = ds_by_ref.get(ref)
                if ds and isinstance(ds.get("pins"), list) and ds["pins"]:
                    part["pins"] = ds["pins"]

    def _manifest_to_roles(
        self,
        parts: List[Dict[str, Any]],
        ds_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Convert manifest parts to the legacy role-list schema for benchmark compat."""
        ds_by_ref: Dict[str, str] = {}
        for ds in (ds_results or []):
            if isinstance(ds, dict) and ds.get("ref"):
                ds_by_ref[str(ds["ref"])] = str(ds.get("role", "") or "")
        roles = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            ref  = str(part.get("ref",       "") or "")
            mpn  = str(part.get("mpn",       "") or "")
            fp   = str(part.get("footprint", "") or "")
            pins = part.get("pins") if isinstance(part.get("pins"), list) else []
            # Gather unique net names and reuse the alternates slot for text-blob matching
            nets = list({str(p.get("net", "") or "") for p in pins if p.get("net")})
            # Only include role_desc for ICs and active components (U/Q refs);
            # passives, connectors, diodes, etc. don't benefit and it pollutes matching.
            role_desc = ds_by_ref.get(ref, "")
            if ref.startswith(("U", "Q")):
                alternates = ([role_desc] if role_desc else []) + nets[:5]
            else:
                # Strip nets whose names contain power/connector keywords to avoid
                # oracle false-positive matches (e.g. dc_jack double-counting).
                alternates = [
                    n for n in nets
                    if not any(kw in n.lower() for kw in _NON_IC_NET_BLOCKLIST)
                ][:5]
            roles.append({
                "role_id":    ref,
                "role_type":  mpn or ref,
                "quantity":   int(part.get("qty", 1) or 1),
                "critical":   True,
                "constraints": {
                    "part_query": mpn,
                    "package":    fp,
                },
                "alternates": alternates,
            })
        return roles
