"""
Suggestion manager for coordinating all suggestion operations.

This manager:
- Generates suggestions from findings
- Manages preview overlays
- Handles apply/dismiss actions
- Logs all actions for audit
- Coordinates with LLM for explanations
"""

import logging
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path

from .base import (
    Suggestion,
    SuggestionStatus,
    SuggestionGenerator,
    SuggestionApplier,
    ActionLogger,
    ActionResult,
)
from .board_outline_generator import BoardOutlineGenerator
from .component_position_generator import ComponentPositionGenerator
from .unrouted_nets_highlighter import UnroutedNetsHighlighter
from .preview_renderer import PreviewManager
from ..checks.base import Finding, CheckResult
from ..parsers.pcb_parser import PCBData

logger = logging.getLogger(__name__)


class SuggestionManager:
    """Manages all suggestion operations for VibeCAD.
    
    This is the main coordinator for Phase 3 assisted design actions:
    - Generates suggestions from check findings
    - Shows/hides preview overlays
    - Applies or dismisses suggestions
    - Logs all actions for audit
    - Gets LLM explanations for suggestions
    
    Core principles:
    - All geometry is generated deterministically
    - LLM only explains, never generates
    - User must approve before any modification
    - All changes are logged and reversible
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize the suggestion manager.
        
        Args:
            log_file: Optional path for action log file
        """
        self.action_logger = ActionLogger(log_file)
        self.applier = SuggestionApplier(self.action_logger)
        self.preview_manager = PreviewManager()
        
        # Initialize generators
        self.generators: List[SuggestionGenerator] = [
            BoardOutlineGenerator(),
            ComponentPositionGenerator(),
            UnroutedNetsHighlighter(),
        ]
        
        # Current suggestions
        self.suggestions: List[Suggestion] = []
        
        # LLM explainer (set externally)
        self._suggestion_explainer = None
    
    def set_suggestion_explainer(self, explainer):
        """Set the suggestion explainer.
        
        Args:
            explainer: SuggestionExplainer instance
        """
        self._suggestion_explainer = explainer
    
    def generate_suggestions(self, check_results: List[CheckResult],
                            pcb_data: PCBData) -> List[Suggestion]:
        """Generate suggestions for all findings.
        
        Args:
            check_results: List of check results with findings
            pcb_data: Current PCB data
        
        Returns:
            List of generated suggestions
        """
        self.suggestions.clear()
        
        for result in check_results:
            if result.passed:
                continue  # No suggestions for passed checks
            
            for finding in result.findings:
                suggestion = self._generate_for_finding(finding, pcb_data)
                if suggestion:
                    self.suggestions.append(suggestion)
                    logger.info(f"Generated suggestion: {suggestion.suggestion_id}")
        
        logger.info(f"Generated {len(self.suggestions)} suggestions from {len(check_results)} checks")
        
        return self.suggestions
    
    def _generate_for_finding(self, finding: Finding,
                              pcb_data: PCBData) -> Optional[Suggestion]:
        """Generate a suggestion for a specific finding.
        
        Args:
            finding: The Finding to address
            pcb_data: Current PCB data
        
        Returns:
            Suggestion, or None if no generator can handle it
        """
        for generator in self.generators:
            if generator.can_generate(finding.rule_id, pcb_data):
                try:
                    suggestion = generator.generate(finding, pcb_data)
                    if suggestion:
                        return suggestion
                except Exception as e:
                    logger.exception(f"Generator {generator.generator_id} failed: {e}")
        
        return None
    
    def show_preview(self, suggestion: Suggestion) -> bool:
        """Show preview overlay for a suggestion.
        
        Args:
            suggestion: The suggestion to preview
        
        Returns:
            True if preview was shown successfully
        """
        success = self.preview_manager.show_suggestion_preview(suggestion)
        
        if success:
            suggestion.status = SuggestionStatus.PREVIEWING
            logger.info(f"Showing preview for {suggestion.suggestion_id}")
        
        return success
    
    def hide_preview(self, suggestion: Suggestion):
        """Hide preview for a suggestion.
        
        Args:
            suggestion: The suggestion to hide preview for
        """
        self.preview_manager.hide_active_preview()
        
        if suggestion.status == SuggestionStatus.PREVIEWING:
            suggestion.status = SuggestionStatus.PENDING
    
    def hide_all_previews(self):
        """Hide all preview overlays."""
        self.preview_manager.hide_all()
        
        for suggestion in self.suggestions:
            if suggestion.status == SuggestionStatus.PREVIEWING:
                suggestion.status = SuggestionStatus.PENDING
    
    def apply_suggestion(self, suggestion: Suggestion,
                         board: Any = None) -> ActionResult:
        """Apply a suggestion to the PCB board.
        
        This requires explicit user approval before calling.
        
        Args:
            suggestion: The approved Suggestion to apply
            board: Optional KiCad board object
        
        Returns:
            ActionResult with success status
        """
        # Hide any active preview
        self.hide_preview(suggestion)
        
        result = self.applier.apply_suggestion(suggestion, board)
        
        if result.success:
            logger.info(f"Applied suggestion {suggestion.suggestion_id}")
        else:
            logger.warning(f"Failed to apply {suggestion.suggestion_id}: {result.error}")
        
        return result
    
    def dismiss_suggestion(self, suggestion: Suggestion,
                           pcb_filename: str = "") -> ActionResult:
        """Dismiss a suggestion without applying it.
        
        Args:
            suggestion: The Suggestion to dismiss
            pcb_filename: Name of the PCB file
        
        Returns:
            ActionResult confirming dismissal
        """
        # Hide any active preview
        self.hide_preview(suggestion)
        
        result = self.applier.dismiss_suggestion(suggestion, pcb_filename)
        
        logger.info(f"Dismissed suggestion {suggestion.suggestion_id}")
        
        return result
    
    def get_explanation(self, suggestion: Suggestion) -> str:
        """Get LLM explanation for a suggestion.
        
        Args:
            suggestion: The suggestion to explain
        
        Returns:
            Formatted explanation string
        """
        if self._suggestion_explainer is None:
            # Return basic explanation without LLM
            return self._format_basic_explanation(suggestion)
        
        try:
            explanation = self._suggestion_explainer.explain_suggestion(suggestion)
            return self._suggestion_explainer.format_explanation(explanation)
        except Exception as e:
            logger.exception(f"Failed to get LLM explanation: {e}")
            return self._format_basic_explanation(suggestion)
    
    def _format_basic_explanation(self, suggestion: Suggestion) -> str:
        """Format a basic explanation without LLM."""
        parts = [
            f"📝 Suggestion: {suggestion.title}",
            "",
            f"Rule: {suggestion.rule_id}",
            "",
            "What will change:",
        ]
        
        for change in suggestion.geometry_changes:
            if change.description:
                parts.append(f"  • {change.description}")
        
        if suggestion.assumptions:
            parts.append("")
            parts.append("Assumptions:")
            for assumption in suggestion.assumptions:
                parts.append(f"  • {assumption}")
        
        parts.append("")
        parts.append("Note: This suggestion was generated deterministically.")
        parts.append("Use Edit > Undo to revert if needed.")
        
        return '\n'.join(parts)
    
    def get_pending_suggestions(self) -> List[Suggestion]:
        """Get all pending suggestions."""
        return [s for s in self.suggestions if s.status == SuggestionStatus.PENDING]
    
    def get_applied_suggestions(self) -> List[Suggestion]:
        """Get all applied suggestions."""
        return [s for s in self.suggestions if s.status == SuggestionStatus.APPLIED]
    
    def get_dismissed_suggestions(self) -> List[Suggestion]:
        """Get all dismissed suggestions."""
        return [s for s in self.suggestions if s.status == SuggestionStatus.DISMISSED]
    
    def get_recent_actions(self, count: int = 10) -> list:
        """Get recent action log entries.
        
        Args:
            count: Number of entries to retrieve
        
        Returns:
            List of ActionLog entries
        """
        return self.action_logger.get_recent_entries(count)
    
    def clear_suggestions(self):
        """Clear all suggestions and previews."""
        self.hide_all_previews()
        self.suggestions.clear()
