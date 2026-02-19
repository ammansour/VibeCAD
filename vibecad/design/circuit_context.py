"""
Smart Circuit Context Builder for VibeCAD.

Builds a compact, query-relevant snapshot of the current board/schematic
state so the LLM can answer questions like "what value should R1 be?"
without dumping the entire netlist into the context window.

Strategy:
1. Extract a *full* lightweight index (ref → value, footprint, nets) — cheap.
2. Given a user query, identify which components / nets are mentioned.
3. Expand one hop: include directly connected neighbours.
4. Serialise only the relevant sub-graph as compact text.

The full index rarely exceeds a few KB even on 500-component boards,
so the LLM always sees the full component table plus a focused
neighbourhood expansion.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)

# Patterns for reference designators (R1, U3, C12, D7, etc.)
_REF_PATTERN = re.compile(r'\b([A-Z]{1,3}\d{1,4})\b')
# Patterns for net names users might mention
_NET_PATTERN = re.compile(r'\b(GND|VCC|VDD|V3V3|3V3|5V|12V|\+\d+V|VBUS|SDA|SCL|MOSI|MISO|SCK|CS|RST|EN|TX|RX|CLK|D[+-]?)\b', re.IGNORECASE)


@dataclass
class ComponentSummary:
    """Compact summary of one component."""
    reference: str
    value: str
    footprint: str
    library: str
    layer: str
    x: float
    y: float
    rotation: float
    pad_nets: Dict[str, str] = field(default_factory=dict)  # pad_number -> net_name

    def to_text(self, verbose: bool = False) -> str:
        nets_str = ", ".join(
            f"pad {p}→{n}" for p, n in sorted(self.pad_nets.items()) if n
        )
        pos = f"({self.x:.1f}, {self.y:.1f})"
        parts = [f"{self.reference}: {self.value} [{self.footprint}] @ {pos}"]
        if nets_str:
            parts.append(f"  nets: {nets_str}")
        return "\n".join(parts)


@dataclass
class NetSummary:
    """Compact summary of one net."""
    name: str
    number: int
    connected_pads: List[Tuple[str, str]] = field(default_factory=list)  # (ref, pad)
    track_count: int = 0

    def to_text(self) -> str:
        pads = ", ".join(f"{ref}.{pad}" for ref, pad in self.connected_pads)
        return f"Net '{self.name}' (#{self.number}): {pads} [{self.track_count} tracks]"


@dataclass
class CircuitSnapshot:
    """Full lightweight index of the current board state."""
    components: Dict[str, ComponentSummary] = field(default_factory=dict)
    nets: Dict[str, NetSummary] = field(default_factory=dict)
    net_by_number: Dict[int, str] = field(default_factory=dict)
    board_outline: bool = False
    track_count: int = 0
    via_count: int = 0
    layer_count: int = 0


class CircuitContextBuilder:
    """Builds smart, compact circuit context for LLM queries.

    Usage:
        builder = CircuitContextBuilder()
        snapshot = builder.build_snapshot(pcb_data, schematic_data)
        context_str = builder.build_context_for_query("what value should R1 be?", snapshot)
    """

    def build_snapshot(self, pcb_data=None, schematic_data=None) -> CircuitSnapshot:
        """Build a full lightweight index from the current board/schematic.

        This is cheap — just extracts references, values, and net connections.
        Call once after each board change and cache the result.
        """
        snap = CircuitSnapshot()

        if pcb_data:
            self._index_pcb(pcb_data, snap)
        if schematic_data:
            self._index_schematic(schematic_data, snap)

        return snap

    def _index_pcb(self, pcb_data, snap: CircuitSnapshot) -> None:
        """Index all components and nets from PCB data."""
        # Build net number → name map
        for net in pcb_data.nets:
            if net.name:
                snap.net_by_number[net.number] = net.name
                if net.name not in snap.nets:
                    snap.nets[net.name] = NetSummary(
                        name=net.name, number=net.number,
                    )

        # Index footprints
        for fp in pcb_data.footprints:
            ref = fp.reference
            if not ref:
                continue

            comp = ComponentSummary(
                reference=ref,
                value=fp.value,
                footprint=fp.footprint_name,
                library=fp.library,
                layer=fp.layer,
                x=fp.at.x,
                y=fp.at.y,
                rotation=fp.rotation,
            )

            # Map pads to nets
            for pad in fp.pads:
                net_name = pad.net_name or snap.net_by_number.get(
                    int(pad.net) if pad.net and str(pad.net).isdigit() else -1, ""
                )
                if net_name:
                    comp.pad_nets[pad.number] = net_name
                    # Register connection in net summary
                    if net_name in snap.nets:
                        snap.nets[net_name].connected_pads.append((ref, pad.number))

            snap.components[ref] = comp

        # Count tracks per net
        for track in pcb_data.tracks:
            net_name = snap.net_by_number.get(track.net, "")
            if net_name and net_name in snap.nets:
                snap.nets[net_name].track_count += 1

        snap.board_outline = pcb_data.has_board_outline
        snap.track_count = len(pcb_data.tracks)
        snap.via_count = len(pcb_data.vias)
        snap.layer_count = len(pcb_data.layers)

    def _index_schematic(self, sch_data, snap: CircuitSnapshot) -> None:
        """Merge schematic properties into existing snapshot.

        Schematic symbols have richer properties (datasheet field, description, etc.)
        that the PCB footprint doesn't carry.
        """
        for sym in sch_data.symbols:
            ref = sym.reference
            if not ref:
                continue
            if ref in snap.components:
                # Merge properties from schematic
                comp = snap.components[ref]
                if sym.value and (not comp.value or comp.value == ref):
                    comp.value = sym.value
                # Schematic symbols often have extra properties
                if hasattr(sym, 'properties') and sym.properties:
                    # Store useful properties that aren't already captured
                    for key in ('Datasheet', 'Description', 'Manufacturer', 'MPN'):
                        val = sym.properties.get(key, '')
                        if val and val not in ('~', ''):
                            comp.pad_nets[f'_prop_{key}'] = val
            else:
                # Component in schematic but not on PCB yet
                snap.components[ref] = ComponentSummary(
                    reference=ref,
                    value=sym.value,
                    footprint=sym.symbol_name,
                    library=sym.library,
                    layer="",
                    x=sym.at.x,
                    y=sym.at.y,
                    rotation=sym.rotation,
                )

    # ── Query-aware context building ─────────────────────────────

    def build_context_for_query(
        self,
        query: str,
        snapshot: CircuitSnapshot,
        *,
        max_components: int = 80,
        include_full_table: bool = True,
    ) -> str:
        """Build a compact context string tailored to the user's query.

        1. Always includes a short component table (ref, value, footprint).
        2. Identifies components/nets mentioned in the query.
        3. Expands one hop to show neighbours.
        4. Returns a focused "neighbourhood" section with full detail.
        """
        sections: List[str] = []

        # ── Section 1: Board overview ──
        sections.append(self._board_overview(snapshot))

        # ── Section 2: Compact component table ──
        if include_full_table:
            sections.append(self._component_table(snapshot, max_components))

        # ── Section 3: Focused neighbourhood ──
        mentioned_refs = self._extract_refs(query, snapshot)
        mentioned_nets = self._extract_nets(query, snapshot)

        if mentioned_refs or mentioned_nets:
            neighbourhood = self._expand_neighbourhood(
                mentioned_refs, mentioned_nets, snapshot,
            )
            sections.append(self._format_neighbourhood(neighbourhood, snapshot))

        return "\n\n".join(sections)

    def _board_overview(self, snap: CircuitSnapshot) -> str:
        return (
            f"## Board Overview\n"
            f"Components: {len(snap.components)} | "
            f"Nets: {len(snap.nets)} | "
            f"Tracks: {snap.track_count} | "
            f"Vias: {snap.via_count} | "
            f"Layers: {snap.layer_count} | "
            f"Outline: {'yes' if snap.board_outline else 'no'}"
        )

    def _component_table(self, snap: CircuitSnapshot, max_rows: int) -> str:
        """Short table: Ref | Value | Footprint — fits ~1 token per component."""
        lines = ["## Component Table", "Ref | Value | Footprint"]
        lines.append("--- | --- | ---")
        for i, (ref, comp) in enumerate(sorted(snap.components.items())):
            if i >= max_rows:
                lines.append(f"... and {len(snap.components) - max_rows} more")
                break
            lines.append(f"{ref} | {comp.value} | {comp.footprint}")
        return "\n".join(lines)

    def _extract_refs(self, query: str, snap: CircuitSnapshot) -> Set[str]:
        """Extract component references mentioned in the query."""
        found = set()
        for m in _REF_PATTERN.finditer(query.upper()):
            candidate = m.group(1)
            if candidate in snap.components:
                found.add(candidate)
        # Also match by value or partial name (e.g. "the resistor" → all R*)
        ql = query.lower()
        if 'resistor' in ql and not found:
            found.update(r for r in snap.components if r.startswith('R'))
        if 'capacitor' in ql and not found:
            found.update(r for r in snap.components if r.startswith('C'))
        if 'inductor' in ql and not found:
            found.update(r for r in snap.components if r.startswith('L'))
        if any(w in ql for w in ('led', 'diode')) and not found:
            found.update(r for r in snap.components if r.startswith('D'))
        if any(w in ql for w in ('ic', 'chip', 'mcu', 'microcontroller')) and not found:
            found.update(r for r in snap.components if r.startswith('U'))
        return found

    def _extract_nets(self, query: str, snap: CircuitSnapshot) -> Set[str]:
        """Extract net names mentioned in the query."""
        found = set()
        for m in _NET_PATTERN.finditer(query):
            candidate = m.group(1).upper()
            # Try case-insensitive match against known nets
            for net_name in snap.nets:
                if net_name.upper() == candidate or candidate in net_name.upper():
                    found.add(net_name)
        return found

    def _expand_neighbourhood(
        self,
        refs: Set[str],
        nets: Set[str],
        snap: CircuitSnapshot,
    ) -> Set[str]:
        """Expand to include all components connected to the mentioned ones."""
        expanded_refs = set(refs)

        # Get nets connected to the mentioned components
        target_nets = set(nets)
        for ref in refs:
            comp = snap.components.get(ref)
            if comp:
                target_nets.update(
                    n for n in comp.pad_nets.values()
                    if not n.startswith('_prop_')
                )

        # Get all components on those nets (one-hop expansion)
        for net_name in target_nets:
            net = snap.nets.get(net_name)
            if net:
                for connected_ref, _ in net.connected_pads:
                    expanded_refs.add(connected_ref)

        return expanded_refs

    def _format_neighbourhood(
        self, refs: Set[str], snap: CircuitSnapshot,
    ) -> str:
        """Format a detailed view of the neighbourhood."""
        lines = ["## Focused Context (mentioned components + neighbours)"]
        for ref in sorted(refs):
            comp = snap.components.get(ref)
            if comp:
                lines.append(comp.to_text(verbose=True))
                # Also list properties if available
                props = {
                    k.replace('_prop_', ''): v
                    for k, v in comp.pad_nets.items()
                    if k.startswith('_prop_')
                }
                if props:
                    lines.append(f"  properties: {props}")
        # Show relevant nets
        relevant_nets: Set[str] = set()
        for ref in refs:
            comp = snap.components.get(ref)
            if comp:
                relevant_nets.update(
                    n for n in comp.pad_nets.values()
                    if not n.startswith('_prop_')
                )
        if relevant_nets:
            lines.append("\n### Connected Nets")
            for net_name in sorted(relevant_nets):
                net = snap.nets.get(net_name)
                if net:
                    lines.append(net.to_text())
        return "\n".join(lines)

    # ── Utility: estimate token cost ─────────────────────────────

    def estimate_tokens(self, context_str: str) -> int:
        """Rough token estimate (1 token ≈ 4 chars for English text)."""
        return len(context_str) // 4
