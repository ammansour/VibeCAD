"""
Base classes for assisted design actions.

All suggestions in this module are:
- Deterministically generated (no LLM involvement in geometry)
- Preview-only by default
- Require explicit user approval
- Reversible via KiCad undo
- Logged for review
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import logging

logger = logging.getLogger(__name__)


class SuggestionStatus(Enum):
    """Status of a suggestion."""
    PENDING = "pending"          # Generated, awaiting user decision
    PREVIEWING = "previewing"    # Currently showing preview overlay
    APPROVED = "approved"        # User clicked Apply
    APPLIED = "applied"          # Successfully applied to board
    DISMISSED = "dismissed"      # User clicked Dismiss
    FAILED = "failed"            # Application failed
    
    def __str__(self) -> str:
        return self.value


@dataclass
class GeometryChange:
    """A single geometry change to be applied.
    
    This represents one atomic change (add line, move component, etc.)
    that will be applied when the user approves the suggestion.
    """
    change_type: str  # 'add_line', 'add_rect', 'move_component', 'highlight'
    layer: str
    params: Dict[str, Any]  # Type-specific parameters
    
    # For display purposes
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'change_type': self.change_type,
            'layer': self.layer,
            'params': self.params,
            'description': self.description,
        }


@dataclass
class Suggestion:
    """A suggested fix for a finding.
    
    Suggestions are:
    - Generated deterministically from design data
    - Displayed as preview before application
    - Applied only with explicit user approval
    - Logged for audit purposes
    
    Attributes:
        suggestion_id: Unique identifier for this suggestion
        rule_id: The check rule ID this suggestion addresses
        title: Short title for display
        description: What will change if applied
        status: Current status of the suggestion
        geometry_changes: List of changes to apply
        assumptions: What assumptions were made when generating this
        preview_data: Data for rendering the preview overlay
        finding_context: Context from the original finding
    """
    suggestion_id: str
    rule_id: str
    title: str
    description: str
    status: SuggestionStatus = SuggestionStatus.PENDING
    geometry_changes: List[GeometryChange] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    preview_data: Dict[str, Any] = field(default_factory=dict)
    finding_context: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps for audit
    created_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    
    # User notes
    user_notes: str = ""
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'suggestion_id': self.suggestion_id,
            'rule_id': self.rule_id,
            'title': self.title,
            'description': self.description,
            'status': str(self.status),
            'geometry_changes': [c.to_dict() for c in self.geometry_changes],
            'assumptions': self.assumptions,
            'finding_context': self.finding_context,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @property
    def is_highlight_only(self) -> bool:
        """Check if this suggestion only highlights issues (no actual changes)."""
        return all(c.change_type == 'highlight' for c in self.geometry_changes)
    
    @property
    def change_count(self) -> int:
        """Number of changes this suggestion will make."""
        return len([c for c in self.geometry_changes if c.change_type != 'highlight'])
    
    def get_llm_context(self) -> Dict[str, Any]:
        """Get context for LLM explanation.
        
        This provides structured data that the LLM can use to explain
        what the suggestion will do.
        """
        return {
            'suggestion_id': self.suggestion_id,
            'rule_id': self.rule_id,
            'title': self.title,
            'description': self.description,
            'assumptions': self.assumptions,
            'changes': [
                {
                    'type': c.change_type,
                    'layer': c.layer,
                    'description': c.description,
                    'params': c.params,
                }
                for c in self.geometry_changes
            ],
            'finding_context': self.finding_context,
        }


@dataclass
class ActionResult:
    """Result of applying a suggestion.
    
    Attributes:
        success: Whether the action was successful
        suggestion_id: ID of the applied suggestion
        message: Human-readable result message
        changes_made: List of actual changes that were made
        error: Error message if failed
        undo_available: Whether KiCad undo can revert this
    """
    success: bool
    suggestion_id: str
    message: str
    changes_made: List[str] = field(default_factory=list)
    error: Optional[str] = None
    undo_available: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'suggestion_id': self.suggestion_id,
            'message': self.message,
            'changes_made': self.changes_made,
            'error': self.error,
            'undo_available': self.undo_available,
        }


@dataclass
class ActionLog:
    """Log entry for an applied action.
    
    Used for audit trail and review of all changes made by VibeCAD.
    """
    timestamp: datetime
    suggestion_id: str
    rule_id: str
    action_type: str  # 'applied', 'dismissed', 'failed'
    description: str
    changes_made: List[str]
    user_approved: bool
    pcb_filename: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'suggestion_id': self.suggestion_id,
            'rule_id': self.rule_id,
            'action_type': self.action_type,
            'description': self.description,
            'changes_made': self.changes_made,
            'user_approved': self.user_approved,
            'pcb_filename': self.pcb_filename,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ActionLogger:
    """Logger for all actions taken by VibeCAD.
    
    Maintains an audit trail of all suggestions that were applied,
    dismissed, or failed.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the logger.
        
        Args:
            log_file: Optional path to write JSON logs. If None, only in-memory.
        """
        self.log_file = log_file
        self.entries: List[ActionLog] = []
    
    def log_action(self, suggestion: Suggestion, action_type: str,
                   changes_made: List[str], pcb_filename: str) -> ActionLog:
        """Log an action.
        
        Args:
            suggestion: The suggestion that was acted upon
            action_type: 'applied', 'dismissed', or 'failed'
            changes_made: List of changes that were made
            pcb_filename: Name of the PCB file
        
        Returns:
            The created log entry
        """
        entry = ActionLog(
            timestamp=datetime.now(),
            suggestion_id=suggestion.suggestion_id,
            rule_id=suggestion.rule_id,
            action_type=action_type,
            description=suggestion.description,
            changes_made=changes_made,
            user_approved=(action_type == 'applied'),
            pcb_filename=pcb_filename,
        )
        
        self.entries.append(entry)
        
        # Persist to file if configured
        if self.log_file:
            self._write_entry(entry)
        
        logger.info(f"Action logged: {action_type} suggestion {suggestion.suggestion_id}")
        
        return entry
    
    def _write_entry(self, entry: ActionLog):
        """Write a log entry to the log file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(entry.to_json() + '\n')
        except Exception as e:
            logger.error(f"Failed to write action log: {e}")
    
    def get_recent_entries(self, count: int = 10) -> List[ActionLog]:
        """Get the most recent log entries."""
        return self.entries[-count:]
    
    def get_entries_for_pcb(self, pcb_filename: str) -> List[ActionLog]:
        """Get all log entries for a specific PCB file."""
        return [e for e in self.entries if e.pcb_filename == pcb_filename]


class SuggestionGenerator(ABC):
    """Abstract base class for suggestion generators.
    
    Subclasses implement deterministic logic to generate suggestions
    for specific types of findings.
    
    Key principles:
    - All geometry is generated deterministically
    - No LLM involvement in generation
    - Clear assumptions documented
    - Preview data included for visualization
    """
    
    @property
    @abstractmethod
    def generator_id(self) -> str:
        """Unique identifier for this generator."""
        pass
    
    @property
    @abstractmethod
    def handles_rules(self) -> List[str]:
        """List of rule IDs this generator can create suggestions for."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this generator does."""
        pass
    
    @abstractmethod
    def can_generate(self, finding_rule_id: str, pcb_data: Any) -> bool:
        """Check if this generator can create a suggestion for the given finding.
        
        Args:
            finding_rule_id: The rule ID from the finding
            pcb_data: Current PCB data
        
        Returns:
            True if a suggestion can be generated
        """
        pass
    
    @abstractmethod
    def generate(self, finding: Any, pcb_data: Any) -> Optional[Suggestion]:
        """Generate a suggestion for the given finding.
        
        Args:
            finding: The Finding object to address
            pcb_data: Current PCB data
        
        Returns:
            A Suggestion object, or None if generation is not possible
        """
        pass
    
    def _generate_suggestion_id(self, rule_id: str) -> str:
        """Generate a unique suggestion ID."""
        import uuid
        short_uuid = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"SUG_{rule_id}_{timestamp}_{short_uuid}"


class SuggestionApplier:
    """Applies approved suggestions to the PCB board.
    
    This class handles the actual modification of the PCB through
    KiCad's pcbnew API. All changes are:
    - Applied only after user approval
    - Compatible with KiCad's undo system
    - Logged for audit
    """
    
    def __init__(self, action_logger: Optional[ActionLogger] = None):
        """Initialize the applier.
        
        Args:
            action_logger: Optional logger for audit trail
        """
        self.action_logger = action_logger or ActionLogger()
        self._pcbnew_available = False
        
        try:
            import pcbnew
            self._pcbnew_available = True
        except ImportError:
            logger.warning("pcbnew not available - suggestions cannot be applied")
    
    @property
    def can_apply(self) -> bool:
        """Check if suggestions can be applied (pcbnew available)."""
        return self._pcbnew_available
    
    def apply_suggestion(self, suggestion: Suggestion, 
                         board: Any = None) -> ActionResult:
        """Apply a suggestion to the PCB board.
        
        This method:
        1. Validates the suggestion can be applied
        2. Applies each geometry change
        3. Updates the board display
        4. Logs the action
        
        Args:
            suggestion: The approved Suggestion to apply
            board: KiCad board object (if None, gets current board)
        
        Returns:
            ActionResult with success status and details
        """
        if not self._pcbnew_available:
            return ActionResult(
                success=False,
                suggestion_id=suggestion.suggestion_id,
                message="Cannot apply: pcbnew not available",
                error="KiCad's pcbnew module is not available"
            )
        
        import pcbnew
        
        if board is None:
            board = pcbnew.GetBoard()
        
        if board is None:
            return ActionResult(
                success=False,
                suggestion_id=suggestion.suggestion_id,
                message="Cannot apply: No board loaded",
                error="No PCB board is currently loaded in KiCad"
            )
        
        changes_made = []
        commit = None
        
        try:
            # Use KiCad's BOARD_COMMIT when available so actions are undoable.
            try:
                commit_cls = getattr(pcbnew, "BOARD_COMMIT", None)
                if callable(commit_cls):
                    commit = commit_cls(board)
            except Exception:
                commit = None

            for change in suggestion.geometry_changes:
                if change.change_type == 'highlight':
                    # Highlights don't modify the board
                    continue
                
                result = self._apply_change(board, change, commit=commit)
                if result:
                    changes_made.append(result)

            # Push undo commit (single entry) if we used BOARD_COMMIT.
            if commit is not None and changes_made:
                try:
                    commit.Push(f"VibeCAD: {suggestion.title}")
                except Exception:
                    # If push fails, we still refresh; worst case undo isn't available.
                    commit = None
            
            # Refresh the board display
            pcbnew.Refresh()
            
            # Update suggestion status
            suggestion.status = SuggestionStatus.APPLIED
            suggestion.applied_at = datetime.now()
            
            # Log the action
            pcb_filename = board.GetFileName() or "unsaved.kicad_pcb"
            self.action_logger.log_action(
                suggestion, 'applied', changes_made, pcb_filename
            )
            
            return ActionResult(
                success=True,
                suggestion_id=suggestion.suggestion_id,
                message=f"Applied {len(changes_made)} change(s). Use Edit > Undo to revert.",
                changes_made=changes_made,
                undo_available=(commit is not None)
            )
            
        except Exception as e:
            logger.exception(f"Failed to apply suggestion {suggestion.suggestion_id}")
            suggestion.status = SuggestionStatus.FAILED
            
            # Log the failure
            pcb_filename = board.GetFileName() or "unsaved.kicad_pcb"
            self.action_logger.log_action(
                suggestion, 'failed', [], pcb_filename
            )
            
            return ActionResult(
                success=False,
                suggestion_id=suggestion.suggestion_id,
                message=f"Failed to apply: {e}",
                error=str(e)
            )
    
    def _apply_change(self, board: Any, change: GeometryChange, commit: Any = None) -> Optional[str]:
        """Apply a single geometry change to the board.
        
        Args:
            board: KiCad board object
            change: The change to apply
        
        Returns:
            Description of what was changed, or None if skipped
        """
        import pcbnew
        
        if change.change_type == 'add_line':
            return self._add_line(board, change, commit=commit)
        elif change.change_type == 'add_rect':
            return self._add_rect(board, change, commit=commit)
        elif change.change_type == 'move_component':
            return self._move_component(board, change, commit=commit)
        else:
            logger.warning(f"Unknown change type: {change.change_type}")
            return None

    def _add_line(self, board: Any, change: GeometryChange, commit: Any = None) -> str:
        """Add a line segment to the board."""
        import pcbnew
        
        params = change.params
        start_x = params['start_x']
        start_y = params['start_y']
        end_x = params['end_x']
        end_y = params['end_y']
        width = params.get('width', 0.15)
        
        # Create line segment
        line = pcbnew.PCB_SHAPE(board)
        line.SetShape(pcbnew.SHAPE_T_SEGMENT)
        
        # KiCad uses nanometers internally, convert from mm
        line.SetStart(pcbnew.VECTOR2I(
            pcbnew.FromMM(start_x),
            pcbnew.FromMM(start_y)
        ))
        line.SetEnd(pcbnew.VECTOR2I(
            pcbnew.FromMM(end_x),
            pcbnew.FromMM(end_y)
        ))
        line.SetWidth(pcbnew.FromMM(width))
        
        # Set layer
        layer_id = board.GetLayerID(change.layer)
        line.SetLayer(layer_id)
        
        if commit is not None:
            try:
                commit.Add(line)
            except Exception:
                board.Add(line)
        else:
            board.Add(line)
        
        return f"Added line on {change.layer}: ({start_x:.2f}, {start_y:.2f}) to ({end_x:.2f}, {end_y:.2f})"
    
    def _add_rect(self, board: Any, change: GeometryChange, commit: Any = None) -> str:
        """Add a rectangle to the board (as four line segments)."""
        import pcbnew
        
        params = change.params
        x1 = params['x1']
        y1 = params['y1']
        x2 = params['x2']
        y2 = params['y2']
        width = params.get('width', 0.15)

        try:
            centered = params.get('centered_at_mm')
            if isinstance(centered, dict) and 'x' in centered and 'y' in centered:
                logger.info(
                    "apply add_rect %s: (%.2f,%.2f)->(%.2f,%.2f)mm width=%.2fmm centered_at=(%.2f,%.2f)mm",
                    change.layer,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(width),
                    float(centered['x']),
                    float(centered['y']),
                )
            else:
                logger.info(
                    "apply add_rect %s: (%.2f,%.2f)->(%.2f,%.2f)mm width=%.2fmm",
                    change.layer,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(width),
                )
        except Exception:
            pass
        
        # Create four line segments for the rectangle
        corners = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ]
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            
            line = pcbnew.PCB_SHAPE(board)
            line.SetShape(pcbnew.SHAPE_T_SEGMENT)
            try:
                sd = getattr(line, 'SetDescription', None)
                if callable(sd):
                    sd('VibeCAD board outline')
            except Exception:
                pass
            line.SetStart(pcbnew.VECTOR2I(
                pcbnew.FromMM(start[0]),
                pcbnew.FromMM(start[1])
            ))
            line.SetEnd(pcbnew.VECTOR2I(
                pcbnew.FromMM(end[0]),
                pcbnew.FromMM(end[1])
            ))
            line.SetWidth(pcbnew.FromMM(width))
            
            layer_id = board.GetLayerID(change.layer)
            line.SetLayer(layer_id)
            
            if commit is not None:
                try:
                    commit.Add(line)
                except Exception:
                    board.Add(line)
            else:
                board.Add(line)
        
        return f"Added rectangle on {change.layer}: ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"
    
    def _move_component(self, board: Any, change: GeometryChange, commit: Any = None) -> str:
        """Move a component to a new position."""
        import pcbnew
        
        params = change.params
        reference = params['reference']
        new_x = params['new_x']
        new_y = params['new_y']
        
        # Find the footprint by reference
        footprints = board.GetFootprints()
        target_fp = None
        
        for fp in footprints:
            if fp.GetReference() == reference:
                target_fp = fp
                break
        
        if target_fp is None:
            raise ValueError(f"Component {reference} not found on board")
        
        # Get old position for logging
        old_pos = target_fp.GetPosition()
        old_x = pcbnew.ToMM(old_pos.x)
        old_y = pcbnew.ToMM(old_pos.y)
        
        # Move with undo support when possible.
        if commit is not None:
            try:
                commit.Modify(target_fp)
            except Exception:
                pass

        # Set requested position first
        target_fp.SetPosition(pcbnew.VECTOR2I(
            pcbnew.FromMM(new_x),
            pcbnew.FromMM(new_y)
        ))

        # Ensure the footprint ends up fully inside the board edges.
        # This corrects cases where the suggestion was based on footprint center only.
        try:
            adjusted = self._clamp_footprint_inside_board(board, target_fp, clearance_mm=1.0)
            if adjusted:
                new_pos = target_fp.GetPosition()
                new_x = pcbnew.ToMM(new_pos.x)
                new_y = pcbnew.ToMM(new_pos.y)
        except Exception:
            pass
        
        return f"Moved {reference} from ({old_x:.2f}, {old_y:.2f}) to ({new_x:.2f}, {new_y:.2f})"

    def _clamp_footprint_inside_board(self, board: Any, footprint: Any, clearance_mm: float = 1.0) -> bool:
        """Clamp a footprint so its bounding box stays inside board edges.

        Uses the board's Edge.Cuts bounding box as a conservative approximation.
        clearance_mm inflates the footprint bounding box to keep a safety margin.

        Returns True if an adjustment was applied.
        """
        import pcbnew

        get_edges_bb = getattr(board, "GetBoardEdgesBoundingBox", None)
        if not callable(get_edges_bb):
            return False

        board_bb = get_edges_bb()
        if board_bb is None:
            return False

        # Prefer courtyard bbox if KiCad provides it; otherwise fall back to overall bbox.
        fp_bb = None
        for attr in ("GetCourtyardBoundingBox", "GetBoundingBox"):
            fn = getattr(footprint, attr, None)
            if callable(fn):
                try:
                    fp_bb = fn()
                    if fp_bb is not None:
                        break
                except Exception:
                    fp_bb = None

        if fp_bb is None:
            return False

        margin = pcbnew.FromMM(max(0.0, float(clearance_mm)))

        # EDA_RECT accessors are consistent across KiCad 6/7.
        def left(r):
            return int(r.GetLeft())

        def right(r):
            return int(r.GetRight())

        def top(r):
            return int(r.GetTop())

        def bottom(r):
            return int(r.GetBottom())

        board_left = left(board_bb) + margin
        board_right = right(board_bb) - margin
        board_top = top(board_bb) + margin
        board_bottom = bottom(board_bb) - margin

        bb_left = left(fp_bb)
        bb_right = right(fp_bb)
        bb_top = top(fp_bb)
        bb_bottom = bottom(fp_bb)

        dx = 0
        dy = 0

        if bb_left < board_left:
            dx = board_left - bb_left
        elif bb_right > board_right:
            dx = board_right - bb_right

        if bb_top < board_top:
            dy = board_top - bb_top
        elif bb_bottom > board_bottom:
            dy = board_bottom - bb_bottom

        if dx == 0 and dy == 0:
            return False

        pos = footprint.GetPosition()
        footprint.SetPosition(pcbnew.VECTOR2I(pos.x + dx, pos.y + dy))
        return True
    
    def dismiss_suggestion(self, suggestion: Suggestion, 
                          pcb_filename: str = "") -> ActionResult:
        """Dismiss a suggestion without applying it.
        
        Args:
            suggestion: The Suggestion to dismiss
            pcb_filename: Name of the PCB file
        
        Returns:
            ActionResult confirming dismissal
        """
        suggestion.status = SuggestionStatus.DISMISSED
        
        self.action_logger.log_action(
            suggestion, 'dismissed', [], pcb_filename or "unknown.kicad_pcb"
        )
        
        return ActionResult(
            success=True,
            suggestion_id=suggestion.suggestion_id,
            message="Suggestion dismissed",
            undo_available=False
        )
