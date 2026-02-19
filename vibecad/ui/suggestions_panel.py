"""
Suggestions panel for VibeCAD UI.

This panel displays suggested fixes with:
- Clear "Suggestion – not applied" labeling
- Preview button to show overlay
- Apply button for user approval
- Dismiss button to reject
- LLM explanation of the suggestion
"""

import logging
import threading
from typing import Optional, List, Callable, Any
from dataclasses import dataclass

try:
    import wx
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False
    class wx:
        class Panel:
            pass
        class BoxSizer:
            pass
        VERTICAL = 0
        HORIZONTAL = 0
        EXPAND = 0
        ALL = 0
        ID_ANY = -1

from ..actions.base import Suggestion, SuggestionStatus, ActionResult

logger = logging.getLogger(__name__)


@dataclass
class SuggestionCallbacks:
    """Callbacks for suggestion actions."""
    on_preview: Optional[Callable[[Suggestion], None]] = None
    on_apply: Optional[Callable[[Suggestion], ActionResult]] = None
    on_dismiss: Optional[Callable[[Suggestion], None]] = None
    on_explain: Optional[Callable[[Suggestion], str]] = None


class SuggestionCard(wx.Panel if WX_AVAILABLE else object):
    """A card displaying a single suggestion.
    
    Shows:
    - Title and description
    - Status badge
    - Preview/Apply/Dismiss buttons
    - Assumptions made
    - LLM explanation (when requested)
    """
    
    def __init__(self, parent, suggestion: Suggestion,
                 callbacks: Optional[SuggestionCallbacks] = None):
        if not WX_AVAILABLE:
            return
        
        super().__init__(parent, wx.ID_ANY, style=wx.BORDER_SIMPLE)
        
        self.suggestion = suggestion
        self.callbacks = callbacks or SuggestionCallbacks()
        self._explanation_text = ""
        
        self._create_ui()
        self._update_state()
    
    def _create_ui(self):
        """Create the card UI."""
        # Card background color based on status
        self.SetBackgroundColour(wx.Colour(255, 250, 230))  # Light yellow
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # === Header with title and status ===
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Suggestion icon
        icon = wx.StaticText(self, label="💡")
        header_sizer.Add(icon, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        # Title
        self.title_label = wx.StaticText(self, label=self.suggestion.title)
        font = self.title_label.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.title_label.SetFont(font)
        header_sizer.Add(self.title_label, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        # Status badge
        self.status_badge = wx.StaticText(self, label="PENDING")
        self.status_badge.SetForegroundColour(wx.Colour(180, 140, 0))
        header_sizer.Add(self.status_badge, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        main_sizer.Add(header_sizer, 0, wx.EXPAND)
        
        # === Not Applied Warning ===
        warning_text = wx.StaticText(
            self, 
            label="⚠️ Suggestion – not applied. Review before applying."
        )
        warning_text.SetForegroundColour(wx.Colour(180, 100, 0))
        font = warning_text.GetFont()
        font.SetStyle(wx.FONTSTYLE_ITALIC)
        warning_text.SetFont(font)
        main_sizer.Add(warning_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # === Description ===
        desc_label = wx.StaticText(self, label=self.suggestion.description)
        desc_label.Wrap(400)
        main_sizer.Add(desc_label, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # === Rule Reference ===
        rule_text = wx.StaticText(
            self, 
            label=f"Addresses: {self.suggestion.rule_id}"
        )
        rule_text.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(rule_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # === Assumptions ===
        if self.suggestion.assumptions:
            assumptions_box = wx.StaticBox(self, label="Assumptions Made")
            assumptions_sizer = wx.StaticBoxSizer(assumptions_box, wx.VERTICAL)
            
            for assumption in self.suggestion.assumptions:
                assumption_text = wx.StaticText(
                    assumptions_box, 
                    label=f"• {assumption}"
                )
                assumption_text.SetForegroundColour(wx.Colour(80, 80, 80))
                assumptions_sizer.Add(assumption_text, 0, wx.LEFT | wx.TOP, 5)
            
            main_sizer.Add(assumptions_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # === Changes Summary ===
        if self.suggestion.geometry_changes:
            changes_text = wx.StaticText(
                self,
                label=f"Will make {self.suggestion.change_count} change(s):"
            )
            main_sizer.Add(changes_text, 0, wx.LEFT | wx.TOP, 10)
            
            for change in self.suggestion.geometry_changes:
                if change.change_type != 'highlight':
                    change_item = wx.StaticText(
                        self,
                        label=f"  → {change.description}"
                    )
                    change_item.SetForegroundColour(wx.Colour(60, 60, 60))
                    main_sizer.Add(change_item, 0, wx.LEFT, 15)
        
        # === Explanation Area ===
        self.explanation_box = wx.StaticBox(self, label="LLM Explanation")
        self.explanation_sizer = wx.StaticBoxSizer(self.explanation_box, wx.VERTICAL)
        
        self.explanation_label = wx.StaticText(
            self.explanation_box,
            label="Click 'Get Explanation' to have the LLM explain this suggestion."
        )
        self.explanation_label.SetForegroundColour(wx.Colour(100, 100, 100))
        self.explanation_label.Wrap(380)
        self.explanation_sizer.Add(self.explanation_label, 0, wx.ALL, 5)
        
        main_sizer.Add(self.explanation_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # === Action Buttons ===
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Preview button
        self.preview_btn = wx.Button(self, label="👁 Preview")
        self.preview_btn.SetToolTip("Show preview overlay on the board")
        self.preview_btn.Bind(wx.EVT_BUTTON, self._on_preview)
        button_sizer.Add(self.preview_btn, 0, wx.ALL, 5)
        
        # Explain button
        self.explain_btn = wx.Button(self, label="💬 Get Explanation")
        self.explain_btn.SetToolTip("Get LLM explanation of this suggestion")
        self.explain_btn.Bind(wx.EVT_BUTTON, self._on_explain)
        button_sizer.Add(self.explain_btn, 0, wx.ALL, 5)
        
        button_sizer.AddStretchSpacer()
        
        # Dismiss button
        self.dismiss_btn = wx.Button(self, label="✕ Dismiss")
        self.dismiss_btn.SetToolTip("Dismiss this suggestion without applying")
        self.dismiss_btn.Bind(wx.EVT_BUTTON, self._on_dismiss)
        button_sizer.Add(self.dismiss_btn, 0, wx.ALL, 5)
        
        # Apply button (prominent)
        self.apply_btn = wx.Button(self, label="✓ Apply This Fix")
        self.apply_btn.SetToolTip("Apply this suggestion to the board (can be undone)")
        self.apply_btn.SetBackgroundColour(wx.Colour(200, 255, 200))
        self.apply_btn.Bind(wx.EVT_BUTTON, self._on_apply)
        button_sizer.Add(self.apply_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # === Undo Note ===
        undo_note = wx.StaticText(
            self,
            label="ℹ️ Applied changes can be undone via Edit > Undo (Ctrl+Z)"
        )
        undo_note.SetForegroundColour(wx.Colour(100, 100, 100))
        font = undo_note.GetFont()
        font.SetPointSize(9)
        undo_note.SetFont(font)
        main_sizer.Add(undo_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        self.SetSizer(main_sizer)
    
    def _update_state(self):
        """Update UI based on suggestion status."""
        status = self.suggestion.status
        
        # Update status badge
        status_text = str(status).upper()
        self.status_badge.SetLabel(status_text)
        
        # Update colors based on status
        if status == SuggestionStatus.PENDING:
            self.SetBackgroundColour(wx.Colour(255, 250, 230))  # Light yellow
            self.status_badge.SetForegroundColour(wx.Colour(180, 140, 0))
            self.apply_btn.Enable(True)
            self.dismiss_btn.Enable(True)
            self.preview_btn.Enable(True)
        elif status == SuggestionStatus.PREVIEWING:
            self.SetBackgroundColour(wx.Colour(255, 255, 200))  # Bright yellow
            self.status_badge.SetForegroundColour(wx.Colour(180, 140, 0))
        elif status == SuggestionStatus.APPLIED:
            self.SetBackgroundColour(wx.Colour(220, 255, 220))  # Light green
            self.status_badge.SetForegroundColour(wx.Colour(0, 150, 0))
            self.apply_btn.Enable(False)
            self.dismiss_btn.Enable(False)
            self.preview_btn.Enable(False)
        elif status == SuggestionStatus.DISMISSED:
            self.SetBackgroundColour(wx.Colour(240, 240, 240))  # Gray
            self.status_badge.SetForegroundColour(wx.Colour(128, 128, 128))
            self.apply_btn.Enable(False)
            self.dismiss_btn.Enable(False)
            self.preview_btn.Enable(False)
        elif status == SuggestionStatus.FAILED:
            self.SetBackgroundColour(wx.Colour(255, 230, 230))  # Light red
            self.status_badge.SetForegroundColour(wx.Colour(200, 0, 0))
        
        self.Refresh()
    
    def set_explanation(self, explanation: str):
        """Set the LLM explanation text."""
        self._explanation_text = explanation
        self.explanation_label.SetLabel(explanation)
        self.explanation_label.Wrap(380)
        self.Layout()
    
    def _on_preview(self, event):
        """Handle Preview button click."""
        if self.callbacks.on_preview:
            self.callbacks.on_preview(self.suggestion)
    
    def _on_apply(self, event):
        """Handle Apply button click."""
        # Confirmation dialog
        dlg = wx.MessageDialog(
            self,
            f"Apply this suggestion?\n\n{self.suggestion.description}\n\n"
            "This action can be undone via Edit > Undo.",
            "Confirm Apply",
            wx.YES_NO | wx.ICON_QUESTION
        )
        
        if dlg.ShowModal() == wx.ID_YES:
            if self.callbacks.on_apply:
                result = self.callbacks.on_apply(self.suggestion)
                if result and result.success:
                    self.suggestion.status = SuggestionStatus.APPLIED
                    self._update_state()
                    wx.MessageBox(
                        result.message,
                        "Success",
                        wx.OK | wx.ICON_INFORMATION
                    )
                elif result:
                    wx.MessageBox(
                        f"Failed to apply: {result.error}",
                        "Error",
                        wx.OK | wx.ICON_ERROR
                    )
        
        dlg.Destroy()
    
    def _on_dismiss(self, event):
        """Handle Dismiss button click."""
        if self.callbacks.on_dismiss:
            self.callbacks.on_dismiss(self.suggestion)
        self.suggestion.status = SuggestionStatus.DISMISSED
        self._update_state()
    
    def _on_explain(self, event):
        """Handle Get Explanation button click (runs LLM off-thread)."""
        if not self.callbacks.on_explain:
            return
        self.explain_btn.Enable(False)
        self.explanation_label.SetLabel("Getting explanation...")

        def worker():
            try:
                explanation = self.callbacks.on_explain(self.suggestion)
                wx.CallAfter(self.set_explanation, explanation or "No explanation available.")
            except Exception as e:
                wx.CallAfter(self.set_explanation, f"Failed to get explanation: {e}")
            finally:
                wx.CallAfter(self.explain_btn.Enable, True)

        threading.Thread(target=worker, daemon=True).start()


class SuggestionsPanel(wx.Panel if WX_AVAILABLE else object):
    """Panel displaying all suggestions for the current findings.
    
    Shows:
    - List of suggestion cards
    - Summary of pending/applied/dismissed
    - Controls for preview management
    """
    
    def __init__(self, parent, callbacks: Optional[SuggestionCallbacks] = None):
        if not WX_AVAILABLE:
            return
        
        super().__init__(parent, wx.ID_ANY)
        
        self.callbacks = callbacks or SuggestionCallbacks()
        self.suggestion_cards: List[SuggestionCard] = []
        self.suggestions: List[Suggestion] = []
        
        self._create_ui()
    
    def _create_ui(self):
        """Create the panel UI."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # === Header ===
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        title = wx.StaticText(self, label="💡 Suggested Fixes")
        font = title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        header_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        header_sizer.AddStretchSpacer()
        
        # Hide all previews button
        self.hide_previews_btn = wx.Button(self, label="Hide All Previews")
        self.hide_previews_btn.Bind(wx.EVT_BUTTON, self._on_hide_all_previews)
        header_sizer.Add(self.hide_previews_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(header_sizer, 0, wx.EXPAND)
        
        # === Summary ===
        self.summary_label = wx.StaticText(self, label="No suggestions yet")
        self.summary_label.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(self.summary_label, 0, wx.LEFT | wx.BOTTOM, 10)
        
        # === Scrolled area for suggestion cards ===
        self.scroll_panel = wx.ScrolledWindow(self, style=wx.VSCROLL)
        self.scroll_panel.SetScrollRate(0, 20)
        
        self.cards_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll_panel.SetSizer(self.cards_sizer)
        
        main_sizer.Add(self.scroll_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # === Info Text ===
        info_text = wx.StaticText(
            self,
            label="Suggestions are generated deterministically. "
                  "The LLM only explains, never modifies geometry."
        )
        info_text.SetForegroundColour(wx.Colour(120, 120, 120))
        font = info_text.GetFont()
        font.SetPointSize(9)
        info_text.SetFont(font)
        main_sizer.Add(info_text, 0, wx.ALL, 5)
        
        self.SetSizer(main_sizer)
    
    def set_suggestions(self, suggestions: List[Suggestion]):
        """Set the list of suggestions to display.
        
        Args:
            suggestions: List of Suggestion objects
        """
        self.suggestions = suggestions
        self._rebuild_cards()
        self._update_summary()
    
    def add_suggestion(self, suggestion: Suggestion):
        """Add a single suggestion.
        
        Args:
            suggestion: Suggestion to add
        """
        self.suggestions.append(suggestion)
        self._add_card(suggestion)
        self._update_summary()
    
    def clear_suggestions(self):
        """Clear all suggestions."""
        self.suggestions.clear()
        self._rebuild_cards()
        self._update_summary()
    
    def _rebuild_cards(self):
        """Rebuild all suggestion cards."""
        # Remove existing cards
        for card in self.suggestion_cards:
            card.Destroy()
        self.suggestion_cards.clear()
        
        # Add new cards
        for suggestion in self.suggestions:
            self._add_card(suggestion)
        
        self.scroll_panel.Layout()
        self.scroll_panel.FitInside()
    
    def _add_card(self, suggestion: Suggestion):
        """Add a card for a suggestion."""
        card = SuggestionCard(
            self.scroll_panel,
            suggestion,
            self.callbacks
        )
        self.cards_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 5)
        self.suggestion_cards.append(card)
        
        self.scroll_panel.Layout()
        self.scroll_panel.FitInside()
    
    def _update_summary(self):
        """Update the summary label."""
        if not self.suggestions:
            self.summary_label.SetLabel("No suggestions available")
            return
        
        pending = sum(1 for s in self.suggestions if s.status == SuggestionStatus.PENDING)
        applied = sum(1 for s in self.suggestions if s.status == SuggestionStatus.APPLIED)
        dismissed = sum(1 for s in self.suggestions if s.status == SuggestionStatus.DISMISSED)
        
        parts = [f"{len(self.suggestions)} suggestion(s)"]
        if pending:
            parts.append(f"{pending} pending")
        if applied:
            parts.append(f"{applied} applied")
        if dismissed:
            parts.append(f"{dismissed} dismissed")
        
        self.summary_label.SetLabel(" | ".join(parts))
    
    def _on_hide_all_previews(self, event):
        """Handle Hide All Previews button."""
        # This would call back to the plugin to clear all previews
        # For now, just update all cards
        for card in self.suggestion_cards:
            if card.suggestion.status == SuggestionStatus.PREVIEWING:
                card.suggestion.status = SuggestionStatus.PENDING
                card._update_state()
