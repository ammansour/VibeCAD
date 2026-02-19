"""
Component position checks for KiCad PCB files.

These checks verify that components are properly placed within
the board outline.
"""

from typing import List, Optional
import logging

from .base import Check, CheckResult, Finding, Severity
from ..parsers.pcb_parser import PCBData, Footprint

logger = logging.getLogger(__name__)


class ComponentOutsideBoardCheck(Check):
    """Check for components placed outside the board outline.
    
    A PCB should have all components placed within the board outline.
    Components outside the outline cannot be manufactured and typically
    indicate a placement error.
    """
    
    # Tolerance for determining if a component is "outside" (mm)
    TOLERANCE_MM = 0.1
    
    @property
    def check_id(self) -> str:
        return "COMPONENT_OUTSIDE_001"
    
    @property
    def check_name(self) -> str:
        return "Component Outside Board"
    
    @property
    def description(self) -> str:
        return (
            "Verifies that all components are placed within the board outline. "
            "Components outside the outline cannot be manufactured."
        )
    
    def run(self, pcb_data: Optional[PCBData] = None,
            schematic_data=None) -> CheckResult:
        """Check if any components are outside the board outline."""
        
        if pcb_data is None:
            return self._create_result(
                passed=False,
                findings=[Finding(
                    rule_id=self.check_id,
                    severity=Severity.ERROR,
                    message="No PCB data provided for analysis",
                    details={"reason": "pcb_data_missing"}
                )]
            )
        
        findings: List[Finding] = []
        
        # If there's no outline, we can't check component positions
        if not pcb_data.has_board_outline:
            return self._create_result(
                passed=True,
                findings=[],
                context={
                    "skipped": True,
                    "reason": "no_board_outline_to_check_against"
                }
            )
        
        # Calculate board outline bounds
        outline_bounds = self._calculate_outline_bounds(pcb_data)
        
        if outline_bounds is None:
            return self._create_result(
                passed=True,
                findings=[],
                context={
                    "skipped": True,
                    "reason": "could_not_calculate_outline_bounds"
                }
            )
        
        min_x, min_y, max_x, max_y = outline_bounds
        
        # Check each component
        components_outside = []
        
        for footprint in pcb_data.footprints:
            x, y = footprint.at.x, footprint.at.y
            
            # Check if component center is outside bounds
            outside_x = x < (min_x - self.TOLERANCE_MM) or x > (max_x + self.TOLERANCE_MM)
            outside_y = y < (min_y - self.TOLERANCE_MM) or y > (max_y + self.TOLERANCE_MM)
            
            if outside_x or outside_y:
                components_outside.append(footprint)
                
                # Determine which direction(s) it's outside
                directions = []
                if x < min_x:
                    directions.append("left")
                if x > max_x:
                    directions.append("right")
                if y < min_y:
                    directions.append("above")
                if y > max_y:
                    directions.append("below")
                
                direction_str = " and ".join(directions)
                
                findings.append(Finding(
                    rule_id=self.check_id,
                    severity=Severity.ERROR,
                    message=(
                        f"Component {footprint.reference} ({footprint.value}) "
                        f"is outside the board outline ({direction_str})"
                    ),
                    component_ref=footprint.reference,
                    layer=footprint.layer,
                    location_x=x,
                    location_y=y,
                    details={
                        "component_ref": footprint.reference,
                        "component_value": footprint.value,
                        "position_x_mm": x,
                        "position_y_mm": y,
                        "board_bounds": {
                            "min_x": min_x,
                            "max_x": max_x,
                            "min_y": min_y,
                            "max_y": max_y,
                        },
                        "outside_directions": directions,
                        "recommendation": "Move component inside the board outline"
                    }
                ))
        
        context = {
            "total_components": len(pcb_data.footprints),
            "components_outside_count": len(components_outside),
            "components_outside_refs": [fp.reference for fp in components_outside],
            "board_bounds": {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            },
            "tolerance_mm": self.TOLERANCE_MM,
        }
        
        return self._create_result(
            passed=len(findings) == 0,
            findings=findings,
            context=context
        )
    
    def _calculate_outline_bounds(self, pcb_data: PCBData):
        """Calculate the bounding box of the board outline.
        
        Returns (min_x, min_y, max_x, max_y) or None if no outline.
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
        
        return (min_x, min_y, max_x, max_y)
