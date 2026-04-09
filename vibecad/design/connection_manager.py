"""
Connection Manager for drawing user-specified connections.

Handles:
- PCB track routing between nets/pads
- Schematic wire connections
- Net-to-net connections (e.g., "connect 5V to GND")

All connections require explicit user approval with visual preview.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# KiCad imports
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False

try:
    import eeschema
    EESCHEMA_AVAILABLE = True
except ImportError:
    EESCHEMA_AVAILABLE = False


class ConnectionType(Enum):
    """Type of connection to create."""
    PCB_TRACK = "pcb_track"
    PCB_VIA = "pcb_via"
    SCHEMATIC_WIRE = "schematic_wire"
    NET_TIE = "net_tie"


@dataclass
class ConnectionPoint:
    """A point to connect (pad, pin, or coordinate)."""
    x: float  # mm
    y: float  # mm
    layer: str = "F.Cu"
    net_name: Optional[str] = None
    component_ref: Optional[str] = None
    pad_number: Optional[str] = None
    pin_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x,
            'y': self.y,
            'layer': self.layer,
            'net_name': self.net_name,
            'component_ref': self.component_ref,
            'pad_number': self.pad_number,
            'pin_name': self.pin_name,
        }


@dataclass
class ConnectionRequest:
    """A user request to create a connection."""
    from_point: ConnectionPoint
    to_point: ConnectionPoint
    connection_type: ConnectionType = ConnectionType.PCB_TRACK
    
    # Track properties
    width_mm: float = 0.25  # Default track width
    layer: str = "F.Cu"
    via_size_mm: float = 0.8
    via_drill_mm: float = 0.4
    
    # For net-to-net connections
    from_net_name: Optional[str] = None
    to_net_name: Optional[str] = None
    
    # Routing hints
    allow_vias: bool = True
    prefer_direct: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_point': self.from_point.to_dict(),
            'to_point': self.to_point.to_dict(),
            'connection_type': self.connection_type.value,
            'width_mm': self.width_mm,
            'layer': self.layer,
            'from_net_name': self.from_net_name,
            'to_net_name': self.to_net_name,
        }


@dataclass
class ConnectionResult:
    """Result of a connection operation."""
    success: bool
    message: str
    request: ConnectionRequest
    tracks_created: int = 0
    vias_created: int = 0
    wires_created: int = 0
    error: Optional[str] = None
    undo_available: bool = True


@dataclass
class ConnectionPreview:
    """Preview data for a proposed connection."""
    request: ConnectionRequest
    path_points: List[Tuple[float, float]]  # List of (x, y) waypoints
    total_length_mm: float
    layer_changes: int  # Number of vias needed
    description: str
    warnings: List[str] = field(default_factory=list)
    
    def to_preview_string(self) -> str:
        """Generate human-readable preview."""
        lines = [
            "🔌 Connection Preview",
            "",
            f"From: {self._format_point(self.request.from_point)}",
            f"To: {self._format_point(self.request.to_point)}",
            "",
            f"Route length: {self.total_length_mm:.2f} mm",
            f"Track width: {self.request.width_mm} mm",
            f"Layer: {self.request.layer}",
        ]
        
        if self.layer_changes > 0:
            lines.append(f"Vias required: {self.layer_changes}")
        
        if self.warnings:
            lines.append("")
            lines.append("⚠️ Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        
        return "\n".join(lines)
    
    def _format_point(self, point: ConnectionPoint) -> str:
        if point.component_ref and point.pad_number:
            return f"{point.component_ref} pad {point.pad_number}"
        elif point.net_name:
            return f"Net '{point.net_name}'"
        else:
            return f"({point.x:.2f}, {point.y:.2f}) mm"


class ConnectionManager:
    """Manages connection drawing operations.
    
    Core principles:
    - All connections are previewed before execution
    - User must explicitly approve each connection
    - Connections are undoable via KiCad's undo system
    - Works in both PCB and schematic editors
    """
    
    def __init__(self):
        self._preview_callback = None
        self._progress_callback = None
    
    def set_preview_callback(self, callback):
        """Set callback for connection preview approval."""
        self._preview_callback = callback
    
    def parse_connection_request(self, 
                                  user_input: str,
                                  pcb_data: Any = None,
                                  schematic_data: Any = None) -> Optional[ConnectionRequest]:
        """Parse a natural language connection request.
        
        Examples:
        - "connect 5V to GND"
        - "draw a track from U1 pin 1 to R1 pin 2"
        - "route VCC to C1 positive"
        
        Args:
            user_input: Natural language request
            pcb_data: Current PCB data for context
            schematic_data: Current schematic data for context
        
        Returns:
            ConnectionRequest if parseable, None otherwise
        """
        # Normalize input
        text = user_input.lower().strip()
        
        # Extract connection keywords
        connect_patterns = [
            ("connect", "to"),
            ("draw", "to"),
            ("route", "to"),
            ("wire", "to"),
            ("link", "to"),
        ]
        
        from_part = None
        to_part = None
        
        for start_word, mid_word in connect_patterns:
            if start_word in text and mid_word in text:
                # Split on the pattern
                parts = text.split(start_word, 1)
                if len(parts) > 1:
                    remainder = parts[1].strip()
                    if mid_word in remainder:
                        from_to = remainder.split(mid_word, 1)
                        if len(from_to) == 2:
                            from_part = from_to[0].strip()
                            to_part = from_to[1].strip()
                            break
        
        if not from_part or not to_part:
            return None
        
        # Try to resolve the endpoints
        from_point = self._resolve_endpoint(from_part, pcb_data, schematic_data)
        to_point = self._resolve_endpoint(to_part, pcb_data, schematic_data)
        
        if not from_point or not to_point:
            return None
        
        return ConnectionRequest(
            from_point=from_point,
            to_point=to_point,
            from_net_name=from_point.net_name,
            to_net_name=to_point.net_name,
        )
    
    def _resolve_endpoint(self, 
                          text: str,
                          pcb_data: Any = None,
                          schematic_data: Any = None) -> Optional[ConnectionPoint]:
        """Resolve a text description to a ConnectionPoint."""
        text = text.strip()
        
        # Check if it's a net name
        if pcb_data:
            nets = getattr(pcb_data, 'nets', [])
            for net in nets:
                if net.name.lower() == text.lower():
                    # Find a pad on this net
                    for fp in getattr(pcb_data, 'footprints', []):
                        for pad in getattr(fp, 'pads', []):
                            if getattr(pad, 'net_name', '') == net.name:
                                return ConnectionPoint(
                                    x=fp.at.x + pad.at.x,
                                    y=fp.at.y + pad.at.y,
                                    net_name=net.name,
                                    component_ref=fp.reference,
                                    pad_number=pad.number,
                                )
        
        # Check for component.pad pattern (e.g., "U1.1", "R1 pin 2", "C1 positive")
        import re
        
        # Pattern: REFDES.PAD or REFDES pad PAD
        patterns = [
            r'(\w+)\.(\w+)',  # U1.1
            r'(\w+)\s+(?:pin|pad)\s+(\w+)',  # U1 pin 1
            r'(\w+)\s+(\w+)',  # C1 positive
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                ref = match.group(1).upper()
                pad_or_name = match.group(2)
                
                if pcb_data:
                    for fp in getattr(pcb_data, 'footprints', []):
                        if fp.reference.upper() == ref:
                            # Find matching pad
                            for pad in getattr(fp, 'pads', []):
                                if (str(pad.number) == pad_or_name or 
                                    getattr(pad, 'name', '').lower() == pad_or_name.lower()):
                                    return ConnectionPoint(
                                        x=fp.at.x + pad.at.x,
                                        y=fp.at.y + pad.at.y,
                                        net_name=getattr(pad, 'net_name', None),
                                        component_ref=ref,
                                        pad_number=str(pad.number),
                                    )
                break
        
        return None
    
    def create_preview(self, request: ConnectionRequest) -> ConnectionPreview:
        """Create a preview of the proposed connection.
        
        Args:
            request: The connection request
        
        Returns:
            ConnectionPreview with route visualization data
        """
        # Simple direct route for now
        # TODO: Implement actual path finding with obstacle avoidance
        
        from_p = request.from_point
        to_p = request.to_point
        
        path_points = [(from_p.x, from_p.y), (to_p.x, to_p.y)]
        
        # Calculate length
        dx = to_p.x - from_p.x
        dy = to_p.y - from_p.y
        length = math.sqrt(dx * dx + dy * dy)
        
        # Check for layer changes
        layer_changes = 0
        if from_p.layer != to_p.layer:
            layer_changes = 1
        
        # Generate warnings
        warnings = []
        
        if request.from_net_name and request.to_net_name:
            if request.from_net_name == request.to_net_name:
                warnings.append("Both endpoints are on the same net - already connected")
            else:
                warnings.append(f"This will short {request.from_net_name} to {request.to_net_name}")
        
        description = f"Draw {request.width_mm}mm track from {from_p.x:.1f},{from_p.y:.1f} to {to_p.x:.1f},{to_p.y:.1f}"
        
        return ConnectionPreview(
            request=request,
            path_points=path_points,
            total_length_mm=length,
            layer_changes=layer_changes,
            description=description,
            warnings=warnings,
        )
    
    def execute_connection(self, 
                           request: ConnectionRequest,
                           board: Any = None) -> ConnectionResult:
        """Execute a connection after user approval.
        
        Args:
            request: The approved connection request
            board: KiCad board object
        
        Returns:
            ConnectionResult with success status
        """
        if not PCBNEW_AVAILABLE:
            return ConnectionResult(
                success=False,
                message="pcbnew not available",
                request=request,
                error="KiCad's pcbnew module is not available",
            )
        
        import pcbnew
        
        if board is None:
            board = pcbnew.GetBoard()
        
        if board is None:
            return ConnectionResult(
                success=False,
                message="No board loaded",
                request=request,
                error="No PCB board is currently loaded",
            )
        
        try:
            # Use BOARD_COMMIT for undo support
            commit = None
            try:
                commit_cls = getattr(pcbnew, "BOARD_COMMIT", None)
                if callable(commit_cls):
                    commit = commit_cls(board)
            except Exception:
                pass
            
            tracks_created = 0
            vias_created = 0
            
            from_p = request.from_point
            to_p = request.to_point
            
            # Create track
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(pcbnew.VECTOR2I(
                pcbnew.FromMM(from_p.x),
                pcbnew.FromMM(from_p.y)
            ))
            track.SetEnd(pcbnew.VECTOR2I(
                pcbnew.FromMM(to_p.x),
                pcbnew.FromMM(to_p.y)
            ))
            track.SetWidth(pcbnew.FromMM(request.width_mm))
            
            # Set layer
            layer_id = board.GetLayerID(request.layer)
            track.SetLayer(layer_id)
            
            # Set net if known
            if request.from_net_name:
                net = board.FindNet(request.from_net_name)
                if net:
                    track.SetNet(net)
            
            # Add via if layer change needed
            if from_p.layer != to_p.layer and request.allow_vias:
                via = pcbnew.PCB_VIA(board)
                via.SetPosition(pcbnew.VECTOR2I(
                    pcbnew.FromMM(to_p.x),
                    pcbnew.FromMM(to_p.y)
                ))
                via.SetWidth(pcbnew.FromMM(request.via_size_mm))
                via.SetDrill(pcbnew.FromMM(request.via_drill_mm))
                
                if request.from_net_name:
                    net = board.FindNet(request.from_net_name)
                    if net:
                        via.SetNet(net)
                
                if hasattr(via, 'thisown'):
                    via.thisown = False
                if commit:
                    commit.Add(via)
                else:
                    board.Add(via)
                vias_created = 1
            
            # Add track
            if hasattr(track, 'thisown'):
                track.thisown = False
            if commit:
                commit.Add(track)
            else:
                board.Add(track)
            tracks_created = 1
            
            # Push undo commit
            if commit:
                try:
                    commit.Push("VibeCAD: Draw connection")
                except Exception:
                    pass
            
            pcbnew.Refresh()
            
            return ConnectionResult(
                success=True,
                message=f"Created connection with {tracks_created} track(s)",
                request=request,
                tracks_created=tracks_created,
                vias_created=vias_created,
                undo_available=(commit is not None),
            )
            
        except Exception as e:
            logger.exception(f"Connection failed: {e}")
            return ConnectionResult(
                success=False,
                message=f"Connection failed: {e}",
                request=request,
                error=str(e),
            )
    
    def find_pads_for_net(self, net_name: str, pcb_data: Any) -> List[ConnectionPoint]:
        """Find all pads connected to a net."""
        pads = []
        
        for fp in getattr(pcb_data, 'footprints', []):
            for pad in getattr(fp, 'pads', []):
                if getattr(pad, 'net_name', '') == net_name:
                    pads.append(ConnectionPoint(
                        x=fp.at.x + pad.at.x,
                        y=fp.at.y + pad.at.y,
                        net_name=net_name,
                        component_ref=fp.reference,
                        pad_number=str(pad.number),
                        layer=getattr(pad, 'layers', ['F.Cu'])[0] if hasattr(pad, 'layers') else 'F.Cu',
                    ))
        
        return pads
