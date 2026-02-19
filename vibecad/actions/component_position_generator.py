"""
Component position suggestion generator.

Suggests moving components that are fully outside the board outline
to a safe position within the board.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import logging
import math

from .base import (
    SuggestionGenerator,
    Suggestion,
    SuggestionStatus,
    GeometryChange,
)
from .board_outline_generator import BoundingBox
from ..checks.base import Finding
from ..parsers.pcb_parser import PCBData, Footprint, Point

logger = logging.getLogger(__name__)


class ComponentPositionGenerator(SuggestionGenerator):
    """Generator for component position suggestions.
    
    Identifies components that are outside the board outline and
    suggests moving them to a safe position inside the board.
    
    Assumptions:
    - Only moves components fully outside the outline
    - Target position is a configurable margin inside the board edge
    - Does not rotate components
    - Does not check for collisions with other components
    """
    
    # Margin from board edge for moved components (mm)
    DEFAULT_MARGIN_MM = 2.0
    
    # Minimum distance outside board to trigger suggestion (mm)
    MIN_OUTSIDE_DISTANCE_MM = 0.5
    
    @property
    def generator_id(self) -> str:
        return "COMPONENT_POSITION_GEN"
    
    @property
    def handles_rules(self) -> List[str]:
        return ["COMPONENT_OUTSIDE_001"]  # Component outside board outline
    
    @property
    def description(self) -> str:
        return (
            "Suggests moving components that are outside the board outline "
            "to a safe position within the board."
        )
    
    def __init__(self, margin_mm: float = DEFAULT_MARGIN_MM):
        """Initialize the generator.
        
        Args:
            margin_mm: Margin from board edge for placed components
        """
        self.margin_mm = margin_mm
    
    def can_generate(self, finding_rule_id: str, pcb_data: PCBData) -> bool:
        """Check if we can generate a component move suggestion."""
        if finding_rule_id not in self.handles_rules:
            return False
        
        if pcb_data is None:
            return False
        
        # Need a board outline to move components into
        return pcb_data.has_board_outline
    
    def generate(self, finding: Finding, pcb_data: PCBData) -> Optional[Suggestion]:
        """Generate a component move suggestion.
        
        Args:
            finding: The COMPONENT_OUTSIDE_001 finding
            pcb_data: Current PCB data
        
        Returns:
            A Suggestion to move the component
        """
        if not self.can_generate(finding.rule_id, pcb_data):
            return None
        
        # Get component reference from finding
        component_ref = finding.component_ref
        if not component_ref:
            # Try to get from details
            component_ref = finding.details.get('component_ref')
        
        if not component_ref:
            logger.warning("No component reference in finding")
            return None
        
        # Find the footprint
        footprint = pcb_data.get_footprint_by_ref(component_ref)
        if footprint is None:
            logger.warning(f"Footprint {component_ref} not found")
            return None
        
        # Calculate board outline bounds
        outline_bbox = self._calculate_outline_bounds(pcb_data)
        if outline_bbox is None:
            logger.warning("Could not calculate board outline bounds")
            return None
        
        # Check if component is actually outside
        if outline_bbox.contains_point(footprint.at.x, footprint.at.y):
            logger.info(f"{component_ref} is inside outline, no move needed")
            return None
        
        # Calculate safe position
        new_x, new_y = self._calculate_safe_position(
            footprint, outline_bbox, pcb_data.footprints
        )
        
        # Create the suggestion
        suggestion_id = self._generate_suggestion_id(finding.rule_id)
        
        # Distance moved
        distance = math.sqrt(
            (new_x - footprint.at.x) ** 2 + 
            (new_y - footprint.at.y) ** 2
        )
        
        move_change = GeometryChange(
            change_type='move_component',
            layer=footprint.layer,
            params={
                'reference': component_ref,
                'old_x': footprint.at.x,
                'old_y': footprint.at.y,
                'new_x': new_x,
                'new_y': new_y,
            },
            description=(
                f"Move {component_ref} from ({footprint.at.x:.2f}, {footprint.at.y:.2f}) "
                f"to ({new_x:.2f}, {new_y:.2f}) mm "
                f"(distance: {distance:.1f}mm)"
            )
        )
        
        assumptions = [
            f"Component {component_ref} is currently outside the board outline",
            f"Target position is {self.margin_mm}mm inside the board edge",
            "Component rotation is preserved",
            "No collision checking with other components",
            "Position is chosen to minimize movement",
        ]
        
        # Preview data
        preview_data = {
            'type': 'highlight_component',
            'reference': component_ref,
            'old_x': footprint.at.x,
            'old_y': footprint.at.y,
            'new_x': new_x,
            'new_y': new_y,
            'color': 'blue',
            'label': f"Move {component_ref} here",
        }
        
        finding_context = {
            'original_finding': finding.to_dict(),
            'component': {
                'reference': component_ref,
                'value': footprint.value,
                'current_x': footprint.at.x,
                'current_y': footprint.at.y,
                'layer': footprint.layer,
            },
            'proposed_position': {
                'x': new_x,
                'y': new_y,
                'distance_from_current': distance,
            },
            'board_outline': {
                'min_x': outline_bbox.min_x,
                'min_y': outline_bbox.min_y,
                'max_x': outline_bbox.max_x,
                'max_y': outline_bbox.max_y,
            },
        }
        
        return Suggestion(
            suggestion_id=suggestion_id,
            rule_id=finding.rule_id,
            title=f"Move {component_ref} Inside Board",
            description=(
                f"Move component {component_ref} ({footprint.value}) from outside "
                f"the board to ({new_x:.1f}, {new_y:.1f}) mm, "
                f"which is {self.margin_mm}mm inside the board edge."
            ),
            status=SuggestionStatus.PENDING,
            geometry_changes=[move_change],
            assumptions=assumptions,
            preview_data=preview_data,
            finding_context=finding_context,
        )
    
    def _calculate_outline_bounds(self, pcb_data: PCBData) -> Optional[BoundingBox]:
        """Calculate the bounding box of the board outline.
        
        This is a simplified calculation that takes the bounding box
        of all Edge.Cuts geometry.
        """
        all_points = []
        
        # Collect points from lines
        for line in pcb_data.board_outline_lines:
            all_points.append((line.start.x, line.start.y))
            all_points.append((line.end.x, line.end.y))
        
        # Collect points from arcs
        for arc in pcb_data.board_outline_arcs:
            all_points.append((arc.start.x, arc.start.y))
            all_points.append((arc.end.x, arc.end.y))
            all_points.append((arc.mid.x, arc.mid.y))
        
        # Collect from rectangles
        for rect in pcb_data.board_outline_rects:
            all_points.append((rect.start.x, rect.start.y))
            all_points.append((rect.end.x, rect.end.y))
        
        # Collect from circles (use diameter as bounds)
        for circle in pcb_data.board_outline_circles:
            all_points.append((circle.center.x - circle.radius, circle.center.y - circle.radius))
            all_points.append((circle.center.x + circle.radius, circle.center.y + circle.radius))
        
        # Collect from polygons
        for poly in pcb_data.board_outline_polygons:
            for point in poly.points:
                all_points.append((point.x, point.y))
        
        if not all_points:
            return None
        
        min_x = min(p[0] for p in all_points)
        max_x = max(p[0] for p in all_points)
        min_y = min(p[1] for p in all_points)
        max_y = max(p[1] for p in all_points)
        
        return BoundingBox(min_x, min_y, max_x, max_y)
    
    def _calculate_safe_position(self, footprint: Footprint,
                                  outline_bbox: BoundingBox,
                                  all_footprints: List[Footprint]) -> Tuple[float, float]:
        """Calculate a safe position inside the board for the component.
        
        Strategy: Move the component to the nearest point inside the board,
        plus a margin from the edge.
        """
        current_x = footprint.at.x
        current_y = footprint.at.y
        
        # Calculate the safe zone (board minus margin)
        safe_min_x = outline_bbox.min_x + self.margin_mm
        safe_max_x = outline_bbox.max_x - self.margin_mm
        safe_min_y = outline_bbox.min_y + self.margin_mm
        safe_max_y = outline_bbox.max_y - self.margin_mm
        
        # Clamp to safe zone
        new_x = max(safe_min_x, min(safe_max_x, current_x))
        new_y = max(safe_min_y, min(safe_max_y, current_y))
        
        # If the safe zone is too small, use center of board
        if safe_max_x < safe_min_x or safe_max_y < safe_min_y:
            new_x = (outline_bbox.min_x + outline_bbox.max_x) / 2
            new_y = (outline_bbox.min_y + outline_bbox.max_y) / 2
        
        return (new_x, new_y)
    
    def find_components_outside_outline(self, pcb_data: PCBData) -> List[Footprint]:
        """Find all components that are outside the board outline.
        
        Args:
            pcb_data: Current PCB data
        
        Returns:
            List of footprints outside the outline
        """
        if not pcb_data.has_board_outline:
            return []
        
        outline_bbox = self._calculate_outline_bounds(pcb_data)
        if outline_bbox is None:
            return []
        
        outside = []
        for fp in pcb_data.footprints:
            if not outline_bbox.contains_point(fp.at.x, fp.at.y):
                outside.append(fp)
        
        return outside
