"""
Board outline suggestion generator.

Generates a rectangular board outline that encloses all components
with a configurable margin. This is a deterministic suggestion based
purely on component bounding boxes.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging

from .base import (
    SuggestionGenerator,
    Suggestion,
    SuggestionStatus,
    GeometryChange,
)
from ..checks.base import Finding
from ..parsers.pcb_parser import PCBData, Footprint, Point

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2
        )
    
    def expand(self, margin: float) -> 'BoundingBox':
        """Return a new bounding box expanded by margin on all sides."""
        return BoundingBox(
            min_x=self.min_x - margin,
            min_y=self.min_y - margin,
            max_x=self.max_x + margin,
            max_y=self.max_y + margin
        )
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the bounding box."""
        return (
            self.min_x <= x <= self.max_x and
            self.min_y <= y <= self.max_y
        )


class BoardOutlineGenerator(SuggestionGenerator):
    """Generator for board outline suggestions.
    
    Creates a rectangular board outline that encloses all components
    with a configurable margin. The outline is deterministically
    calculated from component positions.
    
    Assumptions documented:
    - Uses axis-aligned rectangle (no rotation)
    - Margin is applied uniformly to all sides
    - Does not account for component physical dimensions (only positions)
    """
    
    # Default margin around components in mm
    DEFAULT_MARGIN_MM = 2.5
    
    # Minimum board size in mm
    MIN_BOARD_SIZE_MM = 10.0
    
    # Line width for board outline in mm
    OUTLINE_WIDTH_MM = 0.15

    # KiCad page center target for A4 as requested by user.
    # (KiCad A4 is 297x210; the geometric center is 148.5,105,
    # but users commonly expect "around (150,100)" as the practical center.)
    TARGET_CENTER_MM = (150.0, 100.0)
    
    @property
    def generator_id(self) -> str:
        return "BOARD_OUTLINE_GEN"
    
    @property
    def handles_rules(self) -> List[str]:
        return ["BOARD_OUTLINE_001"]  # Missing board outline check
    
    @property
    def description(self) -> str:
        return (
            "Creates a rectangular board outline that encloses all components "
            f"with a {self.DEFAULT_MARGIN_MM}mm margin."
        )
    
    def __init__(self, margin_mm: float = DEFAULT_MARGIN_MM):
        """Initialize the generator.
        
        Args:
            margin_mm: Margin around components in mm
        """
        self.margin_mm = margin_mm
    
    def can_generate(self, finding_rule_id: str, pcb_data: PCBData) -> bool:
        """Check if we can generate a board outline suggestion.
        
        We can generate if:
        - The finding is about a missing board outline
        - There are components on the board to bound
        """
        if finding_rule_id not in self.handles_rules:
            return False
        
        if pcb_data is None:
            return False
        
        # Need at least one component to create a meaningful outline
        return len(pcb_data.footprints) > 0
    
    def generate(self, finding: Finding, pcb_data: PCBData) -> Optional[Suggestion]:
        """Generate a board outline suggestion.
        
        Args:
            finding: The BOARD_OUTLINE_001 finding
            pcb_data: Current PCB data
        
        Returns:
            A Suggestion with the proposed rectangular outline
        """
        if not self.can_generate(finding.rule_id, pcb_data):
            return None
        
        # Calculate component bounding box
        bbox = self._calculate_component_bounds(pcb_data.footprints)
        
        if bbox is None:
            logger.warning("Could not calculate component bounds")
            return None
        
        # Expand by margin
        outline_bbox = bbox.expand(self.margin_mm)
        
        # Ensure minimum size
        if outline_bbox.width < self.MIN_BOARD_SIZE_MM:
            expand_x = (self.MIN_BOARD_SIZE_MM - outline_bbox.width) / 2
            outline_bbox.min_x -= expand_x
            outline_bbox.max_x += expand_x
        
        if outline_bbox.height < self.MIN_BOARD_SIZE_MM:
            expand_y = (self.MIN_BOARD_SIZE_MM - outline_bbox.height) / 2
            outline_bbox.min_y -= expand_y
            outline_bbox.max_y += expand_y

        # IMPORTANT: Center the proposed outline on the page.
        # Without this, if footprints are piled at (0,0) (very common right after
        # Update PCB from Schematic), the bounding box is also near (0,0) and the
        # suggestion draws the outline at the origin.
        cx, cy = self.TARGET_CENTER_MM
        half_w = outline_bbox.width / 2.0
        half_h = outline_bbox.height / 2.0
        outline_bbox = BoundingBox(
            min_x=cx - half_w,
            min_y=cy - half_h,
            max_x=cx + half_w,
            max_y=cy + half_h,
        )
        
        # Create the suggestion
        suggestion_id = self._generate_suggestion_id(finding.rule_id)
        
        # Create geometry change for rectangular outline
        rect_change = GeometryChange(
            change_type='add_rect',
            layer='Edge.Cuts',
            params={
                'x1': outline_bbox.min_x,
                'y1': outline_bbox.min_y,
                'x2': outline_bbox.max_x,
                'y2': outline_bbox.max_y,
                'width': self.OUTLINE_WIDTH_MM,
                'centered_at_mm': {'x': cx, 'y': cy},
            },
            description=(
                f"Add rectangular board outline: "
                f"({outline_bbox.min_x:.2f}, {outline_bbox.min_y:.2f}) to "
                f"({outline_bbox.max_x:.2f}, {outline_bbox.max_y:.2f}) mm"
            )
        )
        
        # Document assumptions
        assumptions = [
            f"Rectangular outline with {self.margin_mm}mm margin around all components",
            "Outline is axis-aligned (no rotation)",
            "Component positions used (not physical dimensions)",
            f"Minimum board size enforced: {self.MIN_BOARD_SIZE_MM}mm",
            f"Outline centered at approximately ({cx:.0f}, {cy:.0f}) mm on the page",
            f"Enclosing {len(pcb_data.footprints)} component(s)",
        ]
        
        # Build preview data for overlay rendering
        preview_data = {
            'type': 'rectangle',
            'layer': 'Edge.Cuts',
            'x1': outline_bbox.min_x,
            'y1': outline_bbox.min_y,
            'x2': outline_bbox.max_x,
            'y2': outline_bbox.max_y,
            'width': self.OUTLINE_WIDTH_MM,
            'style': 'dashed',  # Preview shows dashed
            'color': 'yellow',  # Suggestion color
            'components_enclosed': [fp.reference for fp in pcb_data.footprints],
        }
        
        # Finding context for LLM explanation
        finding_context = {
            'original_finding': finding.to_dict(),
            'component_count': len(pcb_data.footprints),
            'component_refs': [fp.reference for fp in pcb_data.footprints[:10]],
            'component_bounds': {
                'min_x': bbox.min_x,
                'min_y': bbox.min_y,
                'max_x': bbox.max_x,
                'max_y': bbox.max_y,
            },
            'proposed_outline': {
                'min_x': outline_bbox.min_x,
                'min_y': outline_bbox.min_y,
                'max_x': outline_bbox.max_x,
                'max_y': outline_bbox.max_y,
                'width_mm': outline_bbox.width,
                'height_mm': outline_bbox.height,
            },
            'margin_used_mm': self.margin_mm,
        }
        
        return Suggestion(
            suggestion_id=suggestion_id,
            rule_id=finding.rule_id,
            title="Create Rectangular Board Outline",
            description=(
                f"Create a {outline_bbox.width:.1f} × {outline_bbox.height:.1f} mm "
                f"rectangular board outline on Edge.Cuts layer, enclosing all "
                f"{len(pcb_data.footprints)} components with {self.margin_mm}mm margin."
            ),
            status=SuggestionStatus.PENDING,
            geometry_changes=[rect_change],
            assumptions=assumptions,
            preview_data=preview_data,
            finding_context=finding_context,
        )
    
    def _calculate_component_bounds(self, footprints: List[Footprint]) -> Optional[BoundingBox]:
        """Calculate the bounding box of all component positions.
        
        Args:
            footprints: List of footprints
        
        Returns:
            BoundingBox, or None if no footprints
        """
        if not footprints:
            return None
        
        # Initialize with first footprint
        first = footprints[0]
        min_x = max_x = first.at.x
        min_y = max_y = first.at.y
        
        # Expand to include all footprints
        for fp in footprints[1:]:
            min_x = min(min_x, fp.at.x)
            max_x = max(max_x, fp.at.x)
            min_y = min(min_y, fp.at.y)
            max_y = max(max_y, fp.at.y)
        
        return BoundingBox(min_x, min_y, max_x, max_y)
    
    def get_outline_bbox(self, pcb_data: PCBData) -> Optional[BoundingBox]:
        """Get the proposed outline bounding box (for external use).
        
        Args:
            pcb_data: Current PCB data
        
        Returns:
            BoundingBox of the proposed outline
        """
        if not pcb_data or not pcb_data.footprints:
            return None
        
        bbox = self._calculate_component_bounds(pcb_data.footprints)
        if bbox:
            return bbox.expand(self.margin_mm)
        return None
