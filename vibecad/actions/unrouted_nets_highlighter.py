"""
Unrouted nets highlighter.

Identifies and highlights unrouted nets and disconnected pins.
This is a highlight-only action (no actual fix, just visualization).
"""

from typing import Optional, List, Set, Tuple, Dict
from dataclasses import dataclass
import logging

from .base import (
    SuggestionGenerator,
    Suggestion,
    SuggestionStatus,
    GeometryChange,
)
from ..checks.base import Finding
from ..parsers.pcb_parser import PCBData, Net, Footprint, Pad, Track

logger = logging.getLogger(__name__)


@dataclass
class UnroutedConnection:
    """Represents an unrouted connection between two pads."""
    net_name: str
    net_code: int
    from_pad: Tuple[str, str]  # (footprint_ref, pad_number)
    to_pad: Tuple[str, str]
    from_position: Tuple[float, float]
    to_position: Tuple[float, float]
    

class UnroutedNetsHighlighter(SuggestionGenerator):
    """Highlighter for unrouted nets and disconnected pins.
    
    This generator identifies:
    - Nets with pads that are not connected by tracks
    - Disconnected pins that should be connected
    
    It creates HIGHLIGHT-only suggestions (no actual modifications).
    The user can use these highlights to manually route or investigate.
    
    Note: This is a simplified connectivity check. Full connectivity
    analysis would require flood-fill algorithms and zone analysis.
    """
    
    @property
    def generator_id(self) -> str:
        return "UNROUTED_NETS_HIGHLIGHT"
    
    @property
    def handles_rules(self) -> List[str]:
        return ["UNROUTED_NETS_001", "DISCONNECTED_PIN_001"]
    
    @property
    def description(self) -> str:
        return (
            "Highlights unrouted nets and disconnected pins. "
            "Does not modify the design, only shows visual indicators."
        )
    
    def can_generate(self, finding_rule_id: str, pcb_data: PCBData) -> bool:
        """Check if we can generate highlights."""
        if finding_rule_id not in self.handles_rules:
            return False
        
        return pcb_data is not None
    
    def generate(self, finding: Finding, pcb_data: PCBData) -> Optional[Suggestion]:
        """Generate a highlight suggestion for unrouted nets.
        
        Args:
            finding: The finding about unrouted nets
            pcb_data: Current PCB data
        
        Returns:
            A highlight-only Suggestion
        """
        if not self.can_generate(finding.rule_id, pcb_data):
            return None
        
        # Get net information from finding
        net_name = finding.net_name or finding.details.get('net_name', '')
        
        if not net_name:
            logger.warning("No net name in finding for highlight")
            return None
        
        # Find all pads on this net
        pads_on_net = self._find_pads_on_net(pcb_data, net_name)
        
        if len(pads_on_net) < 2:
            logger.info(f"Net {net_name} has fewer than 2 pads, no highlight needed")
            return None
        
        # Create the suggestion
        suggestion_id = self._generate_suggestion_id(finding.rule_id)
        
        # Create highlight changes for each pad
        geometry_changes = []
        
        for fp_ref, pad_num, x, y in pads_on_net:
            highlight = GeometryChange(
                change_type='highlight',
                layer='User.1',  # Highlight layer
                params={
                    'type': 'pad_highlight',
                    'footprint_ref': fp_ref,
                    'pad_number': pad_num,
                    'x': x,
                    'y': y,
                    'net_name': net_name,
                },
                description=f"Highlight {fp_ref} pad {pad_num} on net {net_name}"
            )
            geometry_changes.append(highlight)
        
        # Create ratsnest-style lines between unconnected pads
        for i in range(len(pads_on_net) - 1):
            fp1, pad1, x1, y1 = pads_on_net[i]
            fp2, pad2, x2, y2 = pads_on_net[i + 1]
            
            ratsnest = GeometryChange(
                change_type='highlight',
                layer='User.1',
                params={
                    'type': 'ratsnest_line',
                    'start_x': x1,
                    'start_y': y1,
                    'end_x': x2,
                    'end_y': y2,
                    'from_pad': f"{fp1}.{pad1}",
                    'to_pad': f"{fp2}.{pad2}",
                    'net_name': net_name,
                },
                description=f"Ratsnest: {fp1}.{pad1} → {fp2}.{pad2}"
            )
            geometry_changes.append(ratsnest)
        
        assumptions = [
            "This is a highlight-only suggestion (no modifications)",
            "Connectivity analysis is simplified (tracks only, no zones)",
            "Use KiCad's native DRC for complete connectivity check",
            f"Net {net_name} has {len(pads_on_net)} pads to connect",
        ]
        
        # Preview data for rendering
        preview_data = {
            'type': 'highlight_net',
            'net_name': net_name,
            'pads': [
                {'ref': fp_ref, 'pad': pad_num, 'x': x, 'y': y}
                for fp_ref, pad_num, x, y in pads_on_net
            ],
            'color': 'red',
            'label': f"Unrouted: {net_name}",
        }
        
        finding_context = {
            'original_finding': finding.to_dict(),
            'net_name': net_name,
            'pad_count': len(pads_on_net),
            'pads': [
                {'footprint': fp_ref, 'pad': pad_num, 'x': x, 'y': y}
                for fp_ref, pad_num, x, y in pads_on_net
            ],
        }
        
        return Suggestion(
            suggestion_id=suggestion_id,
            rule_id=finding.rule_id,
            title=f"Highlight Unrouted Net: {net_name}",
            description=(
                f"Highlight {len(pads_on_net)} pads on net '{net_name}' that may need routing. "
                "This is a visual indicator only - no changes will be made to the design."
            ),
            status=SuggestionStatus.PENDING,
            geometry_changes=geometry_changes,
            assumptions=assumptions,
            preview_data=preview_data,
            finding_context=finding_context,
        )
    
    def _find_pads_on_net(self, pcb_data: PCBData, 
                          net_name: str) -> List[Tuple[str, str, float, float]]:
        """Find all pads connected to a net.
        
        Args:
            pcb_data: Current PCB data
            net_name: Name of the net
        
        Returns:
            List of (footprint_ref, pad_number, x, y) tuples
        """
        pads = []
        
        for footprint in pcb_data.footprints:
            for pad in footprint.pads:
                if pad.net_name == net_name:
                    # Calculate absolute pad position
                    # (pad.at is relative to footprint)
                    abs_x = footprint.at.x + pad.at.x
                    abs_y = footprint.at.y + pad.at.y
                    pads.append((footprint.reference, pad.number, abs_x, abs_y))
        
        return pads
    
    def find_unrouted_nets(self, pcb_data: PCBData) -> List[str]:
        """Find all nets that may have unrouted connections.
        
        This is a simplified check that looks for nets with multiple
        pads but no tracks. A full check would need flood-fill.
        
        Args:
            pcb_data: Current PCB data
        
        Returns:
            List of potentially unrouted net names
        """
        # Count pads per net
        net_pad_counts: Dict[str, int] = {}
        
        for footprint in pcb_data.footprints:
            for pad in footprint.pads:
                if pad.net_name:
                    net_pad_counts[pad.net_name] = net_pad_counts.get(pad.net_name, 0) + 1
        
        # Count tracks per net
        net_track_counts: Dict[int, int] = {}
        
        for track in pcb_data.tracks:
            net_track_counts[track.net] = net_track_counts.get(track.net, 0) + 1
        
        # Build net number to name mapping
        net_num_to_name = {net.number: net.name for net in pcb_data.nets}
        
        # Find nets with multiple pads but no/few tracks
        unrouted = []
        
        for net_name, pad_count in net_pad_counts.items():
            if pad_count < 2:
                continue  # Single pad nets don't need routing
            
            # Find net number
            net_num = None
            for net in pcb_data.nets:
                if net.name == net_name:
                    net_num = net.number
                    break
            
            if net_num is None:
                continue
            
            track_count = net_track_counts.get(net_num, 0)
            
            # Heuristic: if tracks < pads - 1, probably unrouted
            # (minimum spanning tree needs N-1 edges for N nodes)
            if track_count < pad_count - 1:
                unrouted.append(net_name)
        
        return unrouted
    
    def generate_all_highlights(self, pcb_data: PCBData) -> List[Suggestion]:
        """Generate highlight suggestions for all potentially unrouted nets.
        
        Args:
            pcb_data: Current PCB data
        
        Returns:
            List of highlight Suggestions
        """
        suggestions = []
        
        unrouted_nets = self.find_unrouted_nets(pcb_data)
        
        for net_name in unrouted_nets:
            # Create a synthetic finding
            finding = Finding(
                rule_id="UNROUTED_NETS_001",
                severity="warning",
                message=f"Net {net_name} may have unrouted connections",
                net_name=net_name,
            )
            
            suggestion = self.generate(finding, pcb_data)
            if suggestion:
                suggestions.append(suggestion)
        
        return suggestions
