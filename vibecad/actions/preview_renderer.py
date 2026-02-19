"""
Preview overlay rendering for KiCad pcbnew.

This module renders ghosted/highlighted preview overlays for suggested
changes before they are applied. The overlays are:
- Clearly labeled as "Suggestion – not applied"
- Visually distinct from actual board geometry
- Removed when suggestion is dismissed or applied
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check for pcbnew availability
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False
    logger.warning("pcbnew not available - preview rendering disabled")

try:
    import wx
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False


@dataclass
class PreviewOverlay:
    """A preview overlay for a suggested change.
    
    Attributes:
        overlay_id: Unique identifier
        suggestion_id: ID of the suggestion this previews
        overlay_type: Type of overlay (rectangle, line, highlight, etc.)
        layer: Layer to render on (or 'overlay' for on-top rendering)
        params: Type-specific rendering parameters
        is_visible: Whether the overlay is currently shown
    """
    overlay_id: str
    suggestion_id: str
    overlay_type: str
    layer: str
    params: Dict[str, Any]
    is_visible: bool = True
    
    # Visual style for preview
    style: str = 'dashed'  # 'dashed', 'dotted', 'ghost'
    color: Tuple[int, int, int, int] = (255, 200, 0, 180)  # RGBA yellow with alpha
    label: str = "Suggestion"


class PreviewRenderer:
    """Renders preview overlays in KiCad's pcbnew.
    
    This class manages the creation, display, and removal of preview
    overlays that show suggested changes before they are applied.
    
    KiCad 7 approach:
    - Uses PCB_SHAPE objects on a user layer or the target layer
    - Objects are marked with a special property for identification
    - Objects are removed when preview is cleared
    
    Note: For full overlay support, KiCad 8's drawing API would be
    preferred. This implementation uses a pragmatic approach that
    works in KiCad 7.
    """
    
    # Property name used to identify preview objects
    PREVIEW_PROPERTY = "VIBECAD_PREVIEW"
    
    # User layer for overlays (can be changed in settings)
    OVERLAY_LAYER = "User.1"
    
    # Preview colors
    COLOR_SUGGESTION = (255, 200, 0)      # Yellow for suggestions
    COLOR_HIGHLIGHT = (255, 100, 100)     # Red for highlights/warnings
    COLOR_MOVE = (100, 200, 255)          # Blue for move previews
    
    def __init__(self):
        """Initialize the preview renderer."""
        self.overlays: Dict[str, PreviewOverlay] = {}
        self._preview_shapes: List[Any] = []  # KiCad shape objects
        self._board: Optional[Any] = None
    
    @property
    def is_available(self) -> bool:
        """Check if preview rendering is available."""
        return PCBNEW_AVAILABLE
    
    def set_board(self, board: Any):
        """Set the board to render previews on.
        
        Args:
            board: KiCad board object
        """
        self._board = board
    
    def create_overlay(self, suggestion_id: str, preview_data: Dict[str, Any]) -> Optional[PreviewOverlay]:
        """Create a preview overlay from suggestion preview data.
        
        Args:
            suggestion_id: ID of the suggestion
            preview_data: Preview data from the suggestion
        
        Returns:
            PreviewOverlay object, or None if creation failed
        """
        overlay_type = preview_data.get('type', 'unknown')
        layer = preview_data.get('layer', self.OVERLAY_LAYER)
        
        overlay_id = f"PREV_{suggestion_id}_{overlay_type}"
        
        overlay = PreviewOverlay(
            overlay_id=overlay_id,
            suggestion_id=suggestion_id,
            overlay_type=overlay_type,
            layer=layer,
            params=preview_data,
            style=preview_data.get('style', 'dashed'),
            label=preview_data.get('label', 'Suggestion – not applied'),
        )
        
        # Set color based on type
        color_name = preview_data.get('color', 'yellow')
        if color_name == 'yellow':
            overlay.color = (*self.COLOR_SUGGESTION, 180)
        elif color_name == 'red':
            overlay.color = (*self.COLOR_HIGHLIGHT, 180)
        elif color_name == 'blue':
            overlay.color = (*self.COLOR_MOVE, 180)
        
        self.overlays[overlay_id] = overlay
        return overlay
    
    def show_preview(self, overlay: PreviewOverlay) -> bool:
        """Render a preview overlay on the board.
        
        Args:
            overlay: The overlay to show
        
        Returns:
            True if successfully rendered
        """
        if not PCBNEW_AVAILABLE:
            logger.warning("Cannot show preview: pcbnew not available")
            return False
        
        board = self._board or pcbnew.GetBoard()
        if board is None:
            logger.warning("Cannot show preview: no board loaded")
            return False
        
        try:
            if overlay.overlay_type == 'rectangle':
                self._render_rectangle_preview(board, overlay)
            elif overlay.overlay_type == 'line':
                self._render_line_preview(board, overlay)
            elif overlay.overlay_type == 'highlight_component':
                self._render_component_highlight(board, overlay)
            elif overlay.overlay_type == 'highlight_net':
                self._render_net_highlight(board, overlay)
            else:
                logger.warning(f"Unknown overlay type: {overlay.overlay_type}")
                return False
            
            overlay.is_visible = True
            
            # Refresh the display
            pcbnew.Refresh()
            
            return True
            
        except Exception as e:
            logger.exception(f"Failed to show preview: {e}")
            return False
    
    def hide_preview(self, overlay_id: str):
        """Hide and remove a specific preview overlay.
        
        Args:
            overlay_id: ID of the overlay to hide
        """
        if overlay_id in self.overlays:
            self.overlays[overlay_id].is_visible = False
        
        self._remove_preview_shapes(overlay_id)
    
    def hide_all_previews(self):
        """Hide and remove all preview overlays."""
        for overlay_id in list(self.overlays.keys()):
            self.hide_preview(overlay_id)
        
        self.overlays.clear()
        self._preview_shapes.clear()
        
        if PCBNEW_AVAILABLE:
            try:
                pcbnew.Refresh()
            except:
                pass
    
    def clear_suggestion_previews(self, suggestion_id: str):
        """Clear all previews for a specific suggestion.
        
        Args:
            suggestion_id: ID of the suggestion
        """
        to_remove = [
            oid for oid, overlay in self.overlays.items()
            if overlay.suggestion_id == suggestion_id
        ]
        
        for overlay_id in to_remove:
            self.hide_preview(overlay_id)
            del self.overlays[overlay_id]
    
    def _render_rectangle_preview(self, board: Any, overlay: PreviewOverlay):
        """Render a rectangle preview overlay.
        
        Uses dashed/colored lines to show the proposed rectangle.
        """
        params = overlay.params
        x1 = params.get('x1', 0)
        y1 = params.get('y1', 0)
        x2 = params.get('x2', 100)
        y2 = params.get('y2', 100)
        width = params.get('width', 0.2)
        
        # Create four lines for the rectangle
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            
            line = pcbnew.PCB_SHAPE(board)
            line.SetShape(pcbnew.SHAPE_T_SEGMENT)
            line.SetStart(pcbnew.VECTOR2I(
                pcbnew.FromMM(start[0]),
                pcbnew.FromMM(start[1])
            ))
            line.SetEnd(pcbnew.VECTOR2I(
                pcbnew.FromMM(end[0]),
                pcbnew.FromMM(end[1])
            ))
            line.SetWidth(pcbnew.FromMM(width * 1.5))  # Slightly thicker for visibility
            
            # Use User.1 layer for preview to avoid conflicts
            try:
                layer_id = board.GetLayerID(self.OVERLAY_LAYER)
            except:
                layer_id = board.GetLayerID("User.Drawings")
            line.SetLayer(layer_id)
            
            # Mark as preview object (KiCad 7 doesn't have custom properties,
            # so we use a workaround with the shape's locked state)
            # For production, consider using PCB groups or a dedicated layer
            
            board.Add(line)
            self._preview_shapes.append(line)
        
        # Add a text label
        self._add_preview_label(board, overlay, (x1 + x2) / 2, y1 - 2)
    
    def _render_line_preview(self, board: Any, overlay: PreviewOverlay):
        """Render a line preview overlay."""
        params = overlay.params
        start_x = params.get('start_x', 0)
        start_y = params.get('start_y', 0)
        end_x = params.get('end_x', 100)
        end_y = params.get('end_y', 100)
        width = params.get('width', 0.15)
        
        line = pcbnew.PCB_SHAPE(board)
        line.SetShape(pcbnew.SHAPE_T_SEGMENT)
        line.SetStart(pcbnew.VECTOR2I(
            pcbnew.FromMM(start_x),
            pcbnew.FromMM(start_y)
        ))
        line.SetEnd(pcbnew.VECTOR2I(
            pcbnew.FromMM(end_x),
            pcbnew.FromMM(end_y)
        ))
        line.SetWidth(pcbnew.FromMM(width * 1.5))
        
        try:
            layer_id = board.GetLayerID(self.OVERLAY_LAYER)
        except:
            layer_id = board.GetLayerID("User.Drawings")
        line.SetLayer(layer_id)
        
        board.Add(line)
        self._preview_shapes.append(line)
    
    def _render_component_highlight(self, board: Any, overlay: PreviewOverlay):
        """Render a component highlight overlay.
        
        Draws a box around the component to highlight it.
        """
        params = overlay.params
        reference = params.get('reference', '')
        
        # Find the footprint
        footprints = board.GetFootprints()
        target_fp = None
        
        for fp in footprints:
            if fp.GetReference() == reference:
                target_fp = fp
                break
        
        if target_fp is None:
            logger.warning(f"Component {reference} not found for highlight")
            return
        
        # Get bounding box
        bbox = target_fp.GetBoundingBox()
        x1 = pcbnew.ToMM(bbox.GetX())
        y1 = pcbnew.ToMM(bbox.GetY())
        x2 = pcbnew.ToMM(bbox.GetX() + bbox.GetWidth())
        y2 = pcbnew.ToMM(bbox.GetY() + bbox.GetHeight())
        
        # Create highlight box (slightly expanded)
        margin = 0.5
        corners = [
            (x1 - margin, y1 - margin),
            (x2 + margin, y1 - margin),
            (x2 + margin, y2 + margin),
            (x1 - margin, y2 + margin)
        ]
        
        for i in range(4):
            start = corners[i]
            end = corners[(i + 1) % 4]
            
            line = pcbnew.PCB_SHAPE(board)
            line.SetShape(pcbnew.SHAPE_T_SEGMENT)
            line.SetStart(pcbnew.VECTOR2I(
                pcbnew.FromMM(start[0]),
                pcbnew.FromMM(start[1])
            ))
            line.SetEnd(pcbnew.VECTOR2I(
                pcbnew.FromMM(end[0]),
                pcbnew.FromMM(end[1])
            ))
            line.SetWidth(pcbnew.FromMM(0.3))
            
            try:
                layer_id = board.GetLayerID(self.OVERLAY_LAYER)
            except:
                layer_id = board.GetLayerID("User.Drawings")
            line.SetLayer(layer_id)
            
            board.Add(line)
            self._preview_shapes.append(line)
    
    def _render_net_highlight(self, board: Any, overlay: PreviewOverlay):
        """Render a net highlight overlay.
        
        Highlights all pads connected to the specified net.
        """
        params = overlay.params
        net_name = params.get('net_name', '')
        
        # Find net by name
        netinfo = board.FindNet(net_name)
        if netinfo is None:
            logger.warning(f"Net {net_name} not found for highlight")
            return
        
        net_code = netinfo.GetNetCode()
        
        # Highlight all pads on this net
        for fp in board.GetFootprints():
            for pad in fp.Pads():
                if pad.GetNetCode() == net_code:
                    pos = pad.GetPosition()
                    size = max(pcbnew.ToMM(pad.GetSizeX()), pcbnew.ToMM(pad.GetSizeY()))
                    
                    # Draw a circle around the pad
                    self._draw_highlight_circle(
                        board,
                        pcbnew.ToMM(pos.x),
                        pcbnew.ToMM(pos.y),
                        size / 2 + 0.5
                    )
    
    def _draw_highlight_circle(self, board: Any, cx: float, cy: float, radius: float):
        """Draw a highlight circle at the specified position."""
        circle = pcbnew.PCB_SHAPE(board)
        circle.SetShape(pcbnew.SHAPE_T_CIRCLE)
        circle.SetCenter(pcbnew.VECTOR2I(
            pcbnew.FromMM(cx),
            pcbnew.FromMM(cy)
        ))
        circle.SetEnd(pcbnew.VECTOR2I(
            pcbnew.FromMM(cx + radius),
            pcbnew.FromMM(cy)
        ))
        circle.SetWidth(pcbnew.FromMM(0.2))
        
        try:
            layer_id = board.GetLayerID(self.OVERLAY_LAYER)
        except:
            layer_id = board.GetLayerID("User.Drawings")
        circle.SetLayer(layer_id)
        
        board.Add(circle)
        self._preview_shapes.append(circle)
    
    def _add_preview_label(self, board: Any, overlay: PreviewOverlay, 
                           x: float, y: float):
        """Add a text label for the preview."""
        try:
            text = pcbnew.PCB_TEXT(board)
            text.SetText(overlay.label)
            text.SetPosition(pcbnew.VECTOR2I(
                pcbnew.FromMM(x),
                pcbnew.FromMM(y)
            ))
            text.SetTextSize(pcbnew.VECTOR2I(
                pcbnew.FromMM(1.5),
                pcbnew.FromMM(1.5)
            ))
            text.SetTextThickness(pcbnew.FromMM(0.15))
            
            try:
                layer_id = board.GetLayerID(self.OVERLAY_LAYER)
            except:
                layer_id = board.GetLayerID("User.Drawings")
            text.SetLayer(layer_id)
            
            # Center alignment
            text.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_CENTER)
            
            board.Add(text)
            self._preview_shapes.append(text)
        except Exception as e:
            logger.warning(f"Could not add preview label: {e}")
    
    def _remove_preview_shapes(self, overlay_id: Optional[str] = None):
        """Remove preview shapes from the board.
        
        Args:
            overlay_id: If specified, only remove shapes for this overlay
        """
        if not PCBNEW_AVAILABLE:
            return
        
        board = self._board or pcbnew.GetBoard()
        if board is None:
            return
        
        # Remove all tracked preview shapes
        shapes_to_remove = list(self._preview_shapes)
        self._preview_shapes.clear()
        
        for shape in shapes_to_remove:
            try:
                board.Remove(shape)
            except Exception as e:
                logger.debug(f"Could not remove preview shape: {e}")


class PreviewManager:
    """High-level manager for preview overlays.
    
    Coordinates between suggestions and the preview renderer to
    show/hide previews as suggestions are selected.
    """
    
    def __init__(self):
        self.renderer = PreviewRenderer()
        self.active_suggestion_id: Optional[str] = None
    
    def show_suggestion_preview(self, suggestion: 'Suggestion') -> bool:
        """Show preview for a suggestion.
        
        Args:
            suggestion: The suggestion to preview
        
        Returns:
            True if preview was shown successfully
        """
        # Hide any existing preview
        if self.active_suggestion_id:
            self.hide_active_preview()
        
        preview_data = suggestion.preview_data
        if not preview_data:
            logger.warning(f"No preview data for suggestion {suggestion.suggestion_id}")
            return False
        
        overlay = self.renderer.create_overlay(
            suggestion.suggestion_id,
            preview_data
        )
        
        if overlay is None:
            return False
        
        success = self.renderer.show_preview(overlay)
        
        if success:
            self.active_suggestion_id = suggestion.suggestion_id
            logger.info(f"Showing preview for suggestion {suggestion.suggestion_id}")
        
        return success
    
    def hide_active_preview(self):
        """Hide the currently active preview."""
        if self.active_suggestion_id:
            self.renderer.clear_suggestion_previews(self.active_suggestion_id)
            self.active_suggestion_id = None
    
    def hide_all(self):
        """Hide all previews."""
        self.renderer.hide_all_previews()
        self.active_suggestion_id = None
