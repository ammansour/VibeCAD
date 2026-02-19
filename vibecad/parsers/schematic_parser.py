"""
Parser for KiCad Schematic files (.kicad_sch).

Extracts structured data from schematic files for deterministic rule checking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from .sexpr import parse_sexpr, SExprNode, SExprParseError


@dataclass
class Point:
    """A 2D point."""
    x: float
    y: float


@dataclass
class SchematicPin:
    """A pin on a schematic symbol."""
    name: str
    number: str
    pin_type: str  # input, output, bidirectional, etc.
    at: Point


@dataclass
class SchematicSymbol:
    """A symbol instance in the schematic."""
    reference: str
    value: str
    library: str
    symbol_name: str
    at: Point
    rotation: float
    unit: int
    in_bom: bool
    on_board: bool
    uuid: str
    pins: List[SchematicPin] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class Wire:
    """A wire in the schematic."""
    start: Point
    end: Point
    uuid: str


@dataclass
class Junction:
    """A junction point in the schematic."""
    at: Point
    uuid: str


@dataclass
class Label:
    """A net label in the schematic."""
    text: str
    at: Point
    label_type: str  # local, global, hierarchical
    uuid: str


@dataclass
class PowerSymbol:
    """A power symbol (VCC, GND, etc.)."""
    reference: str
    value: str
    at: Point
    uuid: str


@dataclass
class NoConnect:
    """A no-connect marker."""
    at: Point
    uuid: str


@dataclass
class SchematicSheet:
    """A hierarchical sheet reference."""
    name: str
    filename: str
    at: Point
    size: Tuple[float, float]
    uuid: str


@dataclass
class SchematicData:
    """Structured data extracted from a schematic file."""
    version: int
    generator: str
    uuid: str
    paper_size: str
    title_block: Dict[str, str]
    symbols: List[SchematicSymbol]
    wires: List[Wire]
    junctions: List[Junction]
    labels: List[Label]
    power_symbols: List[PowerSymbol]
    no_connects: List[NoConnect]
    sheets: List[SchematicSheet]
    
    def get_symbol_by_ref(self, reference: str) -> Optional[SchematicSymbol]:
        """Find a symbol by its reference designator."""
        for sym in self.symbols:
            if sym.reference == reference:
                return sym
        return None
    
    def get_symbols_by_value(self, value: str) -> List[SchematicSymbol]:
        """Find all symbols with a specific value."""
        return [sym for sym in self.symbols if sym.value == value]
    
    @property
    def component_count(self) -> int:
        """Count of components (excluding power symbols)."""
        return len(self.symbols)
    
    @property
    def net_label_count(self) -> int:
        """Count of net labels."""
        return len(self.labels)


class SchematicParser:
    """Parser for KiCad schematic files."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.root: Optional[SExprNode] = None
    
    def parse(self) -> SchematicData:
        """Parse the schematic file and return structured data."""
        content = self.filepath.read_text(encoding='utf-8')
        self.root = parse_sexpr(content)
        
        if self.root.name != 'kicad_sch':
            raise SchematicParseError(f"Expected 'kicad_sch' root node, got '{self.root.name}'")
        
        return SchematicData(
            version=self._parse_version(),
            generator=self._parse_generator(),
            uuid=self._parse_uuid(),
            paper_size=self._parse_paper_size(),
            title_block=self._parse_title_block(),
            symbols=self._parse_symbols(),
            wires=self._parse_wires(),
            junctions=self._parse_junctions(),
            labels=self._parse_labels(),
            power_symbols=self._parse_power_symbols(),
            no_connects=self._parse_no_connects(),
            sheets=self._parse_sheets(),
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
    
    def _parse_uuid(self) -> str:
        """Parse the schematic UUID."""
        uuid_node = self.root.get_child('uuid')
        if uuid_node:
            return uuid_node.get_value(0, "")
        return ""
    
    def _parse_paper_size(self) -> str:
        """Parse the paper size."""
        paper_node = self.root.get_child('paper')
        if paper_node:
            return paper_node.get_value(0, "A4")
        return "A4"
    
    def _parse_title_block(self) -> Dict[str, str]:
        """Parse the title block information."""
        title_block = {}
        tb_node = self.root.get_child('title_block')
        if tb_node:
            for field_name in ['title', 'date', 'rev', 'company']:
                field_node = tb_node.get_child(field_name)
                if field_node:
                    title_block[field_name] = field_node.get_value(0, "")
            
            # Parse comment fields
            for comment_node in tb_node.get_children('comment'):
                num = comment_node.get_value(0)
                text = comment_node.get_value(1, "")
                if num is not None:
                    title_block[f'comment{num}'] = text
        
        return title_block
    
    def _parse_point(self, node: SExprNode) -> Point:
        """Parse a point from an at node."""
        return Point(
            x=float(node.get_value(0, 0)),
            y=float(node.get_value(1, 0))
        )
    
    def _parse_symbols(self) -> List[SchematicSymbol]:
        """Parse symbol instances."""
        symbols = []
        for sym_node in self.root.get_children('symbol'):
            lib_id_node = sym_node.get_child('lib_id')
            if not lib_id_node:
                continue
            
            lib_id = lib_id_node.get_value(0, "")
            if ':' in lib_id:
                library, symbol_name = lib_id.split(':', 1)
            else:
                library = ""
                symbol_name = lib_id
            
            # Skip power symbols (handled separately)
            if library.lower() == 'power':
                continue
            
            at_node = sym_node.get_child('at')
            at = self._parse_point(at_node) if at_node else Point(0, 0)
            rotation = float(at_node.get_value(2, 0)) if at_node else 0
            
            unit_node = sym_node.get_child('unit')
            unit = int(unit_node.get_value(0, 1)) if unit_node else 1
            
            in_bom_node = sym_node.get_child('in_bom')
            in_bom = in_bom_node.get_value(0, "yes") == "yes" if in_bom_node else True
            
            on_board_node = sym_node.get_child('on_board')
            on_board = on_board_node.get_value(0, "yes") == "yes" if on_board_node else True
            
            uuid_node = sym_node.get_child('uuid')
            uuid = uuid_node.get_value(0, "") if uuid_node else ""
            
            # Parse properties
            properties = {}
            reference = ""
            value = ""
            for prop_node in sym_node.get_children('property'):
                prop_name = prop_node.get_value(0, "")
                prop_value = prop_node.get_value(1, "")
                properties[prop_name] = prop_value
                
                if prop_name == "Reference":
                    reference = prop_value
                elif prop_name == "Value":
                    value = prop_value
            
            # Parse pins
            pins = []
            for pin_node in sym_node.get_children('pin'):
                pin_name = pin_node.get_value(0, "")
                pin_uuid_node = pin_node.get_child('uuid')
                # Pin details would be in the library, not instance
                pins.append(SchematicPin(
                    name=pin_name,
                    number="",
                    pin_type="",
                    at=Point(0, 0)
                ))
            
            symbols.append(SchematicSymbol(
                reference=reference,
                value=value,
                library=library,
                symbol_name=symbol_name,
                at=at,
                rotation=rotation,
                unit=unit,
                in_bom=in_bom,
                on_board=on_board,
                uuid=uuid,
                pins=pins,
                properties=properties
            ))
        
        return symbols
    
    def _parse_power_symbols(self) -> List[PowerSymbol]:
        """Parse power symbols (VCC, GND, etc.)."""
        power_symbols = []
        for sym_node in self.root.get_children('symbol'):
            lib_id_node = sym_node.get_child('lib_id')
            if not lib_id_node:
                continue
            
            lib_id = lib_id_node.get_value(0, "")
            if ':' in lib_id:
                library, _ = lib_id.split(':', 1)
            else:
                library = ""
            
            # Only process power symbols
            if library.lower() != 'power':
                continue
            
            at_node = sym_node.get_child('at')
            at = self._parse_point(at_node) if at_node else Point(0, 0)
            
            uuid_node = sym_node.get_child('uuid')
            uuid = uuid_node.get_value(0, "") if uuid_node else ""
            
            # Get reference and value from properties
            reference = ""
            value = ""
            for prop_node in sym_node.get_children('property'):
                prop_name = prop_node.get_value(0, "")
                prop_value = prop_node.get_value(1, "")
                if prop_name == "Reference":
                    reference = prop_value
                elif prop_name == "Value":
                    value = prop_value
            
            power_symbols.append(PowerSymbol(
                reference=reference,
                value=value,
                at=at,
                uuid=uuid
            ))
        
        return power_symbols
    
    def _parse_wires(self) -> List[Wire]:
        """Parse wire segments."""
        wires = []
        for wire_node in self.root.get_children('wire'):
            pts_node = wire_node.get_child('pts')
            if pts_node:
                xy_nodes = pts_node.get_children('xy')
                if len(xy_nodes) >= 2:
                    start = Point(
                        x=float(xy_nodes[0].get_value(0, 0)),
                        y=float(xy_nodes[0].get_value(1, 0))
                    )
                    end = Point(
                        x=float(xy_nodes[1].get_value(0, 0)),
                        y=float(xy_nodes[1].get_value(1, 0))
                    )
                    
                    uuid_node = wire_node.get_child('uuid')
                    uuid = uuid_node.get_value(0, "") if uuid_node else ""
                    
                    wires.append(Wire(start=start, end=end, uuid=uuid))
        
        return wires
    
    def _parse_junctions(self) -> List[Junction]:
        """Parse junction points."""
        junctions = []
        for junc_node in self.root.get_children('junction'):
            at_node = junc_node.get_child('at')
            uuid_node = junc_node.get_child('uuid')
            
            if at_node:
                junctions.append(Junction(
                    at=self._parse_point(at_node),
                    uuid=uuid_node.get_value(0, "") if uuid_node else ""
                ))
        
        return junctions
    
    def _parse_labels(self) -> List[Label]:
        """Parse net labels (local, global, hierarchical)."""
        labels = []
        
        # Local labels
        for label_node in self.root.get_children('label'):
            text = label_node.get_value(0, "")
            at_node = label_node.get_child('at')
            uuid_node = label_node.get_child('uuid')
            
            if at_node:
                labels.append(Label(
                    text=text,
                    at=self._parse_point(at_node),
                    label_type="local",
                    uuid=uuid_node.get_value(0, "") if uuid_node else ""
                ))
        
        # Global labels
        for label_node in self.root.get_children('global_label'):
            text = label_node.get_value(0, "")
            at_node = label_node.get_child('at')
            uuid_node = label_node.get_child('uuid')
            
            if at_node:
                labels.append(Label(
                    text=text,
                    at=self._parse_point(at_node),
                    label_type="global",
                    uuid=uuid_node.get_value(0, "") if uuid_node else ""
                ))
        
        # Hierarchical labels
        for label_node in self.root.get_children('hierarchical_label'):
            text = label_node.get_value(0, "")
            at_node = label_node.get_child('at')
            uuid_node = label_node.get_child('uuid')
            
            if at_node:
                labels.append(Label(
                    text=text,
                    at=self._parse_point(at_node),
                    label_type="hierarchical",
                    uuid=uuid_node.get_value(0, "") if uuid_node else ""
                ))
        
        return labels
    
    def _parse_no_connects(self) -> List[NoConnect]:
        """Parse no-connect markers."""
        no_connects = []
        for nc_node in self.root.get_children('no_connect'):
            at_node = nc_node.get_child('at')
            uuid_node = nc_node.get_child('uuid')
            
            if at_node:
                no_connects.append(NoConnect(
                    at=self._parse_point(at_node),
                    uuid=uuid_node.get_value(0, "") if uuid_node else ""
                ))
        
        return no_connects
    
    def _parse_sheets(self) -> List[SchematicSheet]:
        """Parse hierarchical sheet references."""
        sheets = []
        for sheet_node in self.root.get_children('sheet'):
            at_node = sheet_node.get_child('at')
            size_node = sheet_node.get_child('size')
            uuid_node = sheet_node.get_child('uuid')
            
            # Get sheet name and filename from properties
            name = ""
            filename = ""
            for prop_node in sheet_node.get_children('property'):
                prop_name = prop_node.get_value(0, "")
                prop_value = prop_node.get_value(1, "")
                if prop_name == "Sheetname":
                    name = prop_value
                elif prop_name == "Sheetfile":
                    filename = prop_value
            
            if at_node:
                sheets.append(SchematicSheet(
                    name=name,
                    filename=filename,
                    at=self._parse_point(at_node),
                    size=(
                        float(size_node.get_value(0, 0)),
                        float(size_node.get_value(1, 0))
                    ) if size_node else (0, 0),
                    uuid=uuid_node.get_value(0, "") if uuid_node else ""
                ))
        
        return sheets


class SchematicParseError(Exception):
    """Exception raised for schematic parsing errors."""
    pass
