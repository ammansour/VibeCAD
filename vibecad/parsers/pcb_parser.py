"""
Parser for KiCad PCB files (.kicad_pcb).

Extracts structured data from PCB files for deterministic rule checking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .sexpr import parse_sexpr, SExprNode, SExprParseError


@dataclass
class Point:
    """A 2D point in mm."""
    x: float
    y: float


@dataclass
class Line:
    """A line segment on the PCB."""
    start: Point
    end: Point
    layer: str
    width: float


@dataclass
class Arc:
    """An arc segment on the PCB."""
    start: Point
    mid: Point
    end: Point
    layer: str
    width: float


@dataclass
class Circle:
    """A circle on the PCB."""
    center: Point
    radius: float
    layer: str
    width: float


@dataclass
class Rect:
    """A rectangle on the PCB."""
    start: Point
    end: Point
    layer: str
    width: float


@dataclass
class Polygon:
    """A polygon on the PCB."""
    points: List[Point]
    layer: str
    width: float


@dataclass 
class Pad:
    """A pad on a footprint."""
    number: str
    pad_type: str  # 'thru_hole', 'smd', 'np_thru_hole', 'connect'
    shape: str  # 'circle', 'rect', 'oval', 'roundrect', 'trapezoid', 'custom'
    at: Point
    size: Tuple[float, float]
    layers: List[str]
    net: Optional[str] = None
    net_name: Optional[str] = None


@dataclass
class Footprint:
    """A footprint (component) on the PCB."""
    reference: str
    value: str
    library: str
    footprint_name: str
    at: Point
    rotation: float
    layer: str
    pads: List[Pad] = field(default_factory=list)
    

@dataclass
class Net:
    """A net in the design."""
    number: int
    name: str


@dataclass
class Track:
    """A track segment."""
    start: Point
    end: Point
    width: float
    layer: str
    net: int


@dataclass
class Via:
    """A via."""
    at: Point
    size: float
    drill: float
    layers: List[str]
    net: int


@dataclass
class Zone:
    """A copper zone (pour)."""
    net: int
    net_name: str
    layer: str
    points: List[Point]


@dataclass
class PCBData:
    """Structured data extracted from a PCB file."""
    version: int
    generator: str
    general: Dict[str, Any]
    layers: Dict[int, Tuple[str, str]]  # layer_num -> (name, type)
    setup: Dict[str, Any]
    nets: List[Net]
    footprints: List[Footprint]
    tracks: List[Track]
    vias: List[Via]
    zones: List[Zone]
    
    # Board outline elements (Edge.Cuts layer)
    board_outline_lines: List[Line] = field(default_factory=list)
    board_outline_arcs: List[Arc] = field(default_factory=list)
    board_outline_circles: List[Circle] = field(default_factory=list)
    board_outline_rects: List[Rect] = field(default_factory=list)
    board_outline_polygons: List[Polygon] = field(default_factory=list)
    
    @property
    def has_board_outline(self) -> bool:
        """Check if the PCB has any board outline defined."""
        return bool(
            self.board_outline_lines or 
            self.board_outline_arcs or 
            self.board_outline_circles or
            self.board_outline_rects or
            self.board_outline_polygons
        )
    
    @property
    def board_outline_element_count(self) -> int:
        """Count total number of board outline elements."""
        return (
            len(self.board_outline_lines) +
            len(self.board_outline_arcs) +
            len(self.board_outline_circles) +
            len(self.board_outline_rects) +
            len(self.board_outline_polygons)
        )
    
    def get_footprint_by_ref(self, reference: str) -> Optional[Footprint]:
        """Find a footprint by its reference designator."""
        for fp in self.footprints:
            if fp.reference == reference:
                return fp
        return None
    
    def get_net_by_name(self, name: str) -> Optional[Net]:
        """Find a net by name."""
        for net in self.nets:
            if net.name == name:
                return net
        return None


class PCBParser:
    """Parser for KiCad PCB files."""
    
    EDGE_CUTS_LAYER = "Edge.Cuts"
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.root: Optional[SExprNode] = None
    
    def parse(self) -> PCBData:
        """Parse the PCB file and return structured data."""
        content = self.filepath.read_text(encoding='utf-8')
        self.root = parse_sexpr(content)
        
        if self.root.name != 'kicad_pcb':
            raise PCBParseError(f"Expected 'kicad_pcb' root node, got '{self.root.name}'")
        
        return PCBData(
            version=self._parse_version(),
            generator=self._parse_generator(),
            general=self._parse_general(),
            layers=self._parse_layers(),
            setup=self._parse_setup(),
            nets=self._parse_nets(),
            footprints=self._parse_footprints(),
            tracks=self._parse_tracks(),
            vias=self._parse_vias(),
            zones=self._parse_zones(),
            board_outline_lines=self._parse_board_outline_lines(),
            board_outline_arcs=self._parse_board_outline_arcs(),
            board_outline_circles=self._parse_board_outline_circles(),
            board_outline_rects=self._parse_board_outline_rects(),
            board_outline_polygons=self._parse_board_outline_polygons(),
        )
    
    def _parse_version(self) -> int:
        """Parse the file version."""
        version_node = self.root.get_child('version')
        if version_node:
            return version_node.get_value(0, 0)
        return 0
    
    def _parse_generator(self) -> str:
        """Parse the generator string."""
        gen_node = self.root.get_child('generator')
        if gen_node:
            return gen_node.get_value(0, "")
        return ""
    
    def _parse_general(self) -> Dict[str, Any]:
        """Parse general section."""
        general = {}
        general_node = self.root.get_child('general')
        if general_node:
            thickness_node = general_node.get_child('thickness')
            if thickness_node:
                general['thickness'] = thickness_node.get_value(0)
        return general
    
    def _parse_layers(self) -> Dict[int, Tuple[str, str]]:
        """Parse layer definitions."""
        layers = {}
        layers_node = self.root.get_child('layers')
        if layers_node:
            for child in layers_node.children:
                if child.values:
                    layer_num = int(child.name) if child.name.isdigit() else -1
                    if layer_num >= 0 and len(child.values) >= 2:
                        layer_name = child.values[0]
                        layer_type = child.values[1]
                        layers[layer_num] = (layer_name, layer_type)
        return layers
    
    def _parse_setup(self) -> Dict[str, Any]:
        """Parse setup section."""
        setup = {}
        setup_node = self.root.get_child('setup')
        if setup_node:
            # Parse common setup values
            for name in ['pad_to_mask_clearance', 'solder_mask_min_width', 
                        'pad_to_paste_clearance', 'aux_axis_origin']:
                child = setup_node.get_child(name)
                if child:
                    setup[name] = child.get_value(0)
        return setup
    
    def _parse_nets(self) -> List[Net]:
        """Parse net definitions."""
        nets = []
        for net_node in self.root.get_children('net'):
            if len(net_node.values) >= 2:
                nets.append(Net(
                    number=int(net_node.values[0]),
                    name=str(net_node.values[1])
                ))
        return nets
    
    def _parse_point(self, node: SExprNode) -> Point:
        """Parse a point from an at/start/end/etc node."""
        return Point(
            x=float(node.get_value(0, 0)),
            y=float(node.get_value(1, 0))
        )
    
    def _parse_footprints(self) -> List[Footprint]:
        """Parse footprint definitions."""
        footprints = []
        for fp_node in self.root.get_children('footprint'):
            footprint_full = fp_node.get_value(0, "")
            # Split library:footprint_name
            if ':' in footprint_full:
                library, footprint_name = footprint_full.split(':', 1)
            else:
                library = ""
                footprint_name = footprint_full
            
            at_node = fp_node.get_child('at')
            at = self._parse_point(at_node) if at_node else Point(0, 0)
            rotation = float(at_node.get_value(2, 0)) if at_node else 0
            
            layer_node = fp_node.get_child('layer')
            layer = layer_node.get_value(0, "F.Cu") if layer_node else "F.Cu"
            
            # Get reference and value from properties
            reference = ""
            value = ""
            for prop in fp_node.get_children('property'):
                prop_name = prop.get_value(0, "")
                if prop_name == "Reference":
                    reference = prop.get_value(1, "")
                elif prop_name == "Value":
                    value = prop.get_value(1, "")
            
            # Fallback to fp_text for older format
            if not reference:
                for fp_text in fp_node.get_children('fp_text'):
                    text_type = fp_text.get_value(0, "")
                    if text_type == "reference":
                        reference = fp_text.get_value(1, "")
                    elif text_type == "value":
                        value = fp_text.get_value(1, "")
            
            # Parse pads
            pads = []
            for pad_node in fp_node.get_children('pad'):
                pad_number = str(pad_node.get_value(0, ""))
                pad_type = pad_node.get_value(1, "")
                shape = pad_node.get_value(2, "")
                
                pad_at_node = pad_node.get_child('at')
                pad_at = self._parse_point(pad_at_node) if pad_at_node else Point(0, 0)
                
                size_node = pad_node.get_child('size')
                size = (
                    float(size_node.get_value(0, 0)),
                    float(size_node.get_value(1, 0))
                ) if size_node else (0, 0)
                
                layers_node = pad_node.get_child('layers')
                pad_layers = list(layers_node.values) if layers_node else []
                
                net_node = pad_node.get_child('net')
                net = int(net_node.get_value(0, 0)) if net_node else None
                net_name = str(net_node.get_value(1, "")) if net_node and len(net_node.values) > 1 else None
                
                pads.append(Pad(
                    number=pad_number,
                    pad_type=pad_type,
                    shape=shape,
                    at=pad_at,
                    size=size,
                    layers=pad_layers,
                    net=net,
                    net_name=net_name
                ))
            
            footprints.append(Footprint(
                reference=reference,
                value=value,
                library=library,
                footprint_name=footprint_name,
                at=at,
                rotation=rotation,
                layer=layer,
                pads=pads
            ))
        
        return footprints
    
    def _parse_tracks(self) -> List[Track]:
        """Parse track (segment) definitions."""
        tracks = []
        for seg_node in self.root.get_children('segment'):
            start_node = seg_node.get_child('start')
            end_node = seg_node.get_child('end')
            width_node = seg_node.get_child('width')
            layer_node = seg_node.get_child('layer')
            net_node = seg_node.get_child('net')
            
            if start_node and end_node:
                tracks.append(Track(
                    start=self._parse_point(start_node),
                    end=self._parse_point(end_node),
                    width=float(width_node.get_value(0, 0)) if width_node else 0,
                    layer=layer_node.get_value(0, "") if layer_node else "",
                    net=int(net_node.get_value(0, 0)) if net_node else 0
                ))
        return tracks
    
    def _parse_vias(self) -> List[Via]:
        """Parse via definitions."""
        vias = []
        for via_node in self.root.get_children('via'):
            at_node = via_node.get_child('at')
            size_node = via_node.get_child('size')
            drill_node = via_node.get_child('drill')
            layers_node = via_node.get_child('layers')
            net_node = via_node.get_child('net')
            
            if at_node:
                vias.append(Via(
                    at=self._parse_point(at_node),
                    size=float(size_node.get_value(0, 0)) if size_node else 0,
                    drill=float(drill_node.get_value(0, 0)) if drill_node else 0,
                    layers=list(layers_node.values) if layers_node else [],
                    net=int(net_node.get_value(0, 0)) if net_node else 0
                ))
        return vias
    
    def _parse_zones(self) -> List[Zone]:
        """Parse zone (copper pour) definitions."""
        zones = []
        for zone_node in self.root.get_children('zone'):
            net_node = zone_node.get_child('net')
            net_name_node = zone_node.get_child('net_name')
            layer_node = zone_node.get_child('layer')
            
            # Parse polygon points from filled_polygon or polygon
            points = []
            polygon_node = zone_node.get_child('polygon')
            if polygon_node:
                pts_node = polygon_node.get_child('pts')
                if pts_node:
                    for xy_node in pts_node.get_children('xy'):
                        points.append(Point(
                            x=float(xy_node.get_value(0, 0)),
                            y=float(xy_node.get_value(1, 0))
                        ))
            
            zones.append(Zone(
                net=int(net_node.get_value(0, 0)) if net_node else 0,
                net_name=str(net_name_node.get_value(0, "")) if net_name_node else "",
                layer=layer_node.get_value(0, "") if layer_node else "",
                points=points
            ))
        
        return zones
    
    def _parse_board_outline_lines(self) -> List[Line]:
        """Parse line segments on the Edge.Cuts layer."""
        lines = []
        for line_node in self.root.get_children('gr_line'):
            layer_node = line_node.get_child('layer')
            if layer_node and layer_node.get_value(0) == self.EDGE_CUTS_LAYER:
                start_node = line_node.get_child('start')
                end_node = line_node.get_child('end')
                width_node = line_node.get_child('width')
                stroke_node = line_node.get_child('stroke')
                
                width = 0.0
                if width_node:
                    width = float(width_node.get_value(0, 0))
                elif stroke_node:
                    w_node = stroke_node.get_child('width')
                    if w_node:
                        width = float(w_node.get_value(0, 0))
                
                if start_node and end_node:
                    lines.append(Line(
                        start=self._parse_point(start_node),
                        end=self._parse_point(end_node),
                        layer=self.EDGE_CUTS_LAYER,
                        width=width
                    ))
        return lines
    
    def _parse_board_outline_arcs(self) -> List[Arc]:
        """Parse arc segments on the Edge.Cuts layer."""
        arcs = []
        for arc_node in self.root.get_children('gr_arc'):
            layer_node = arc_node.get_child('layer')
            if layer_node and layer_node.get_value(0) == self.EDGE_CUTS_LAYER:
                start_node = arc_node.get_child('start')
                mid_node = arc_node.get_child('mid')
                end_node = arc_node.get_child('end')
                width_node = arc_node.get_child('width')
                stroke_node = arc_node.get_child('stroke')
                
                width = 0.0
                if width_node:
                    width = float(width_node.get_value(0, 0))
                elif stroke_node:
                    w_node = stroke_node.get_child('width')
                    if w_node:
                        width = float(w_node.get_value(0, 0))
                
                if start_node and mid_node and end_node:
                    arcs.append(Arc(
                        start=self._parse_point(start_node),
                        mid=self._parse_point(mid_node),
                        end=self._parse_point(end_node),
                        layer=self.EDGE_CUTS_LAYER,
                        width=width
                    ))
        return arcs
    
    def _parse_board_outline_circles(self) -> List[Circle]:
        """Parse circles on the Edge.Cuts layer."""
        circles = []
        for circle_node in self.root.get_children('gr_circle'):
            layer_node = circle_node.get_child('layer')
            if layer_node and layer_node.get_value(0) == self.EDGE_CUTS_LAYER:
                center_node = circle_node.get_child('center')
                end_node = circle_node.get_child('end')
                width_node = circle_node.get_child('width')
                stroke_node = circle_node.get_child('stroke')
                
                width = 0.0
                if width_node:
                    width = float(width_node.get_value(0, 0))
                elif stroke_node:
                    w_node = stroke_node.get_child('width')
                    if w_node:
                        width = float(w_node.get_value(0, 0))
                
                if center_node and end_node:
                    center = self._parse_point(center_node)
                    end_point = self._parse_point(end_node)
                    # Calculate radius from center to end point
                    radius = ((end_point.x - center.x)**2 + (end_point.y - center.y)**2)**0.5
                    
                    circles.append(Circle(
                        center=center,
                        radius=radius,
                        layer=self.EDGE_CUTS_LAYER,
                        width=width
                    ))
        return circles
    
    def _parse_board_outline_rects(self) -> List[Rect]:
        """Parse rectangles on the Edge.Cuts layer."""
        rects = []
        for rect_node in self.root.get_children('gr_rect'):
            layer_node = rect_node.get_child('layer')
            if layer_node and layer_node.get_value(0) == self.EDGE_CUTS_LAYER:
                start_node = rect_node.get_child('start')
                end_node = rect_node.get_child('end')
                width_node = rect_node.get_child('width')
                stroke_node = rect_node.get_child('stroke')
                
                width = 0.0
                if width_node:
                    width = float(width_node.get_value(0, 0))
                elif stroke_node:
                    w_node = stroke_node.get_child('width')
                    if w_node:
                        width = float(w_node.get_value(0, 0))
                
                if start_node and end_node:
                    rects.append(Rect(
                        start=self._parse_point(start_node),
                        end=self._parse_point(end_node),
                        layer=self.EDGE_CUTS_LAYER,
                        width=width
                    ))
        return rects
    
    def _parse_board_outline_polygons(self) -> List[Polygon]:
        """Parse polygons on the Edge.Cuts layer."""
        polygons = []
        for poly_node in self.root.get_children('gr_poly'):
            layer_node = poly_node.get_child('layer')
            if layer_node and layer_node.get_value(0) == self.EDGE_CUTS_LAYER:
                pts_node = poly_node.get_child('pts')
                width_node = poly_node.get_child('width')
                stroke_node = poly_node.get_child('stroke')
                
                width = 0.0
                if width_node:
                    width = float(width_node.get_value(0, 0))
                elif stroke_node:
                    w_node = stroke_node.get_child('width')
                    if w_node:
                        width = float(w_node.get_value(0, 0))
                
                points = []
                if pts_node:
                    for xy_node in pts_node.get_children('xy'):
                        points.append(Point(
                            x=float(xy_node.get_value(0, 0)),
                            y=float(xy_node.get_value(1, 0))
                        ))
                
                if points:
                    polygons.append(Polygon(
                        points=points,
                        layer=self.EDGE_CUTS_LAYER,
                        width=width
                    ))
        return polygons


class PCBParseError(Exception):
    """Exception raised for PCB parsing errors."""
    pass
