"""
Base classes for deterministic design rule checks.

This module defines the interface for all checks in the VibeCAD system.
Checks are purely deterministic and produce structured output that can
be sent to the LLM for explanation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional
import json


class Severity(Enum):
    """Severity levels for check findings."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class Finding:
    """A single finding from a check.
    
    All findings must reference specific, verifiable facts from the design:
    - Component references (e.g., "R1", "U3")
    - Net names (e.g., "VCC", "GND", "NET_001")
    - Layer names (e.g., "F.Cu", "Edge.Cuts")
    - Coordinates (x, y in mm)
    - Rule IDs (e.g., "BOARD_OUTLINE_001")
    """
    rule_id: str
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Optional location information
    component_ref: Optional[str] = None
    net_name: Optional[str] = None
    layer: Optional[str] = None
    location_x: Optional[float] = None
    location_y: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to a dictionary for JSON serialization."""
        result = {
            'rule_id': self.rule_id,
            'severity': str(self.severity),
            'message': self.message,
            'details': self.details,
        }
        
        # Only include optional fields if they have values
        if self.component_ref:
            result['component_ref'] = self.component_ref
        if self.net_name:
            result['net_name'] = self.net_name
        if self.layer:
            result['layer'] = self.layer
        if self.location_x is not None:
            result['location_x_mm'] = self.location_x
        if self.location_y is not None:
            result['location_y_mm'] = self.location_y
        
        return result
    
    def to_json(self) -> str:
        """Convert finding to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class CheckResult:
    """Result of running a check.
    
    Contains all findings from the check along with metadata about
    the check itself.
    """
    check_id: str
    check_name: str
    description: str
    passed: bool
    findings: List[Finding] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for JSON serialization."""
        return {
            'check_id': self.check_id,
            'check_name': self.check_name,
            'description': self.description,
            'passed': self.passed,
            'finding_count': len(self.findings),
            'findings': [f.to_dict() for f in self.findings],
            'context': self.context,
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def error_count(self) -> int:
        """Count of error-severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)
    
    @property
    def info_count(self) -> int:
        """Count of info-severity findings."""
        return sum(1 for f in self.findings if f.severity == Severity.INFO)


class Check(ABC):
    """Abstract base class for all deterministic checks.
    
    Subclasses must implement:
    - check_id: Unique identifier for the check
    - check_name: Human-readable name
    - description: What the check looks for
    - run(): Execute the check and return results
    
    Checks must be:
    - Deterministic: Same input always produces same output
    - Factual: Only report verifiable facts from the design
    - Traceable: All findings reference specific design elements
    """
    
    @property
    @abstractmethod
    def check_id(self) -> str:
        """Unique identifier for this check."""
        pass
    
    @property
    @abstractmethod
    def check_name(self) -> str:
        """Human-readable name for this check."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this check looks for."""
        pass
    
    @abstractmethod
    def run(self, pcb_data=None, schematic_data=None) -> CheckResult:
        """Run the check and return results.
        
        Args:
            pcb_data: Parsed PCB data (PCBData instance)
            schematic_data: Parsed schematic data (SchematicData instance)
        
        Returns:
            CheckResult containing all findings
        """
        pass
    
    def _create_result(self, passed: bool, findings: List[Finding], 
                       context: Optional[Dict[str, Any]] = None) -> CheckResult:
        """Helper to create a CheckResult with this check's metadata."""
        return CheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            description=self.description,
            passed=passed,
            findings=findings,
            context=context or {}
        )
