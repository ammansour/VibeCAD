"""
Board outline checks for KiCad PCB files.

These checks verify that the PCB has a properly defined board outline
on the Edge.Cuts layer.
"""

from typing import List, Optional, Set, Tuple
from dataclasses import dataclass
import math

from .base import Check, CheckResult, Finding, Severity
from ..parsers.pcb_parser import PCBData, Point


class MissingBoardOutlineCheck(Check):
    """Check for missing board outline on Edge.Cuts layer.
    
    A PCB must have a board outline defined on the Edge.Cuts layer
    for manufacturing. This check verifies that at least some geometry
    exists on that layer.
    """
    
    @property
    def check_id(self) -> str:
        return "BOARD_OUTLINE_001"
    
    @property
    def check_name(self) -> str:
        return "Missing Board Outline"
    
    @property
    def description(self) -> str:
        return (
            "Verifies that the PCB has geometry defined on the Edge.Cuts layer. "
            "The board outline defines the physical boundary of the PCB and is "
            "required for manufacturing."
        )
    
    def run(self, pcb_data: Optional[PCBData] = None, 
            schematic_data=None) -> CheckResult:
        """Check if the PCB has any board outline geometry."""
        
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
        
        if not pcb_data.has_board_outline:
            findings.append(Finding(
                rule_id=self.check_id,
                severity=Severity.ERROR,
                message="No board outline found on Edge.Cuts layer",
                layer="Edge.Cuts",
                details={
                    "layer_checked": "Edge.Cuts",
                    "geometry_types_checked": [
                        "gr_line", "gr_arc", "gr_circle", "gr_rect", "gr_poly"
                    ],
                    "element_count": 0,
                    "recommendation": "Add board outline geometry to Edge.Cuts layer"
                }
            ))
        
        # Build context with factual information about what was found
        context = {
            "edge_cuts_elements": {
                "lines": len(pcb_data.board_outline_lines),
                "arcs": len(pcb_data.board_outline_arcs),
                "circles": len(pcb_data.board_outline_circles),
                "rectangles": len(pcb_data.board_outline_rects),
                "polygons": len(pcb_data.board_outline_polygons),
                "total": pcb_data.board_outline_element_count
            },
            "footprint_count": len(pcb_data.footprints),
            "has_components": len(pcb_data.footprints) > 0
        }
        
        return self._create_result(
            passed=len(findings) == 0,
            findings=findings,
            context=context
        )


class BoardOutlineOpenCheck(Check):
    """Check if the board outline forms a closed shape.
    
    The board outline must form a closed contour for proper manufacturing.
    This check analyzes the Edge.Cuts geometry to detect gaps.
    """
    
    @property
    def check_id(self) -> str:
        return "BOARD_OUTLINE_002"
    
    @property
    def check_name(self) -> str:
        return "Board Outline Not Closed"
    
    @property
    def description(self) -> str:
        return (
            "Verifies that the board outline on Edge.Cuts forms a closed contour. "
            "An open board outline will cause manufacturing issues as the board "
            "shape cannot be properly determined."
        )
    
    # Tolerance for point matching in mm
    POINT_TOLERANCE = 0.001
    
    def run(self, pcb_data: Optional[PCBData] = None,
            schematic_data=None) -> CheckResult:
        """Check if the board outline forms a closed shape."""
        
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
        
        # If there's no outline at all, skip this check (covered by BOARD_OUTLINE_001)
        if not pcb_data.has_board_outline:
            return self._create_result(
                passed=True,
                findings=[],
                context={"skipped": True, "reason": "no_board_outline_present"}
            )
        
        findings: List[Finding] = []
        
        # Collect all endpoints from line segments and arcs
        endpoints = self._collect_endpoints(pcb_data)
        
        # Check if all endpoints are paired (each point should appear exactly twice
        # in a closed contour, or the shape is a circle/rect which is inherently closed)
        
        # Circles and filled rectangles are inherently closed
        inherently_closed = (
            len(pcb_data.board_outline_circles) > 0 or
            len(pcb_data.board_outline_rects) > 0 or
            len(pcb_data.board_outline_polygons) > 0
        )
        
        if inherently_closed and len(pcb_data.board_outline_lines) == 0 and len(pcb_data.board_outline_arcs) == 0:
            # Only closed shapes present
            return self._create_result(
                passed=True,
                findings=[],
                context={
                    "outline_type": "inherently_closed",
                    "circles": len(pcb_data.board_outline_circles),
                    "rectangles": len(pcb_data.board_outline_rects),
                    "polygons": len(pcb_data.board_outline_polygons)
                }
            )
        
        # For lines and arcs, check endpoint connectivity
        open_endpoints = self._find_open_endpoints(endpoints)
        
        if open_endpoints:
            for point in open_endpoints:
                findings.append(Finding(
                    rule_id=self.check_id,
                    severity=Severity.ERROR,
                    message=f"Board outline has open endpoint at ({point[0]:.3f}, {point[1]:.3f}) mm",
                    layer="Edge.Cuts",
                    location_x=point[0],
                    location_y=point[1],
                    details={
                        "endpoint_x_mm": point[0],
                        "endpoint_y_mm": point[1],
                        "tolerance_mm": self.POINT_TOLERANCE,
                        "recommendation": "Connect this endpoint to close the board outline"
                    }
                ))
        
        context = {
            "total_endpoints_analyzed": len(endpoints),
            "open_endpoints_found": len(open_endpoints),
            "line_segments": len(pcb_data.board_outline_lines),
            "arc_segments": len(pcb_data.board_outline_arcs),
            "tolerance_mm": self.POINT_TOLERANCE
        }
        
        return self._create_result(
            passed=len(findings) == 0,
            findings=findings,
            context=context
        )
    
    def _collect_endpoints(self, pcb_data: PCBData) -> List[Tuple[float, float]]:
        """Collect all endpoints from outline geometry."""
        endpoints = []
        
        # Lines have two endpoints
        for line in pcb_data.board_outline_lines:
            endpoints.append((line.start.x, line.start.y))
            endpoints.append((line.end.x, line.end.y))
        
        # Arcs have start and end points
        for arc in pcb_data.board_outline_arcs:
            endpoints.append((arc.start.x, arc.start.y))
            endpoints.append((arc.end.x, arc.end.y))
        
        return endpoints
    
    def _find_open_endpoints(self, endpoints: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Find endpoints that don't have a matching pair.
        
        In a closed contour, each endpoint should appear exactly twice
        (once as the end of one segment, once as the start of another).
        """
        # Count occurrences of each point (within tolerance)
        point_counts: dict = {}
        
        for point in endpoints:
            matched = False
            for existing in point_counts:
                if self._points_match(point, existing):
                    point_counts[existing] += 1
                    matched = True
                    break
            if not matched:
                point_counts[point] = 1
        
        # Find points that appear only once (open endpoints)
        open_points = [p for p, count in point_counts.items() if count == 1]
        
        return open_points
    
    def _points_match(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """Check if two points are the same within tolerance."""
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance <= self.POINT_TOLERANCE
