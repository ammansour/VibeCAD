"""
KiCad dockable panel for VibeCAD.

This module provides the main UI panel that integrates with KiCad's
window system using wxPython.
"""

import logging
from typing import Optional, List, Callable
from pathlib import Path

# wx is provided by KiCad's Python environment
try:
    import wx
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False
    # Create dummy classes for type hints when wx is not available
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

from .results_view import ResultsView, ResultsViewModel
from ..checks.base import CheckResult

logger = logging.getLogger(__name__)


class VibeCADPanel(wx.Panel if WX_AVAILABLE else object):
    """Main panel for VibeCAD plugin.
    
    This panel provides:
    - Button to run design checks
    - Results display area
    - LLM explanation section
    - User question input (future)
    """
    
    def __init__(self, parent, on_run_checks: Optional[Callable] = None,
                 on_explain: Optional[Callable] = None,
                 on_set_verbose: Optional[Callable[[bool], None]] = None):
        """Initialize the panel.
        
        Args:
            parent: Parent wx window
            on_run_checks: Callback when user clicks Run Checks
            on_explain: Callback when user requests LLM explanation
        """
        if not WX_AVAILABLE:
            logger.error("wxPython not available. Panel cannot be created.")
            return
        
        super().__init__(parent, wx.ID_ANY)
        
        self.on_run_checks = on_run_checks
        self.on_explain = on_explain
        self.on_set_verbose = on_set_verbose
        self.results_view = ResultsView()
        self.results_view.add_update_callback(self._on_model_update)
        
        self._create_ui()

        # React to system theme/light-dark changes.
        try:
            self.Bind(wx.EVT_SYS_COLOUR_CHANGED, self._on_sys_colour_changed)
        except Exception:
            pass
    
    def _create_ui(self):
        """Create the UI elements."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="VibeCAD Design Review")
        title_font = title.GetFont()
        title_font.SetPointSize(14)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        main_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 10)
        
        # Description
        desc = wx.StaticText(
            self, 
            label="LLM-assisted design review for KiCad.\nRun deterministic checks and get AI explanations."
        )
        desc.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.ALIGN_CENTER, 10)
        
        # Separator
        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 5)
        
        # Button panel
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.run_btn = wx.Button(self, label="▶ Run Checks")
        self.run_btn.Bind(wx.EVT_BUTTON, self._on_run_clicked)
        self.run_btn.SetToolTip("Run deterministic design rule checks")
        button_sizer.Add(self.run_btn, 0, wx.ALL, 5)
        
        self.explain_btn = wx.Button(self, label="💡 Get Explanation")
        self.explain_btn.Bind(wx.EVT_BUTTON, self._on_explain_clicked)
        self.explain_btn.SetToolTip("Get LLM explanation of results")
        self.explain_btn.Enable(False)
        button_sizer.Add(self.explain_btn, 0, wx.ALL, 5)
        
        self.clear_btn = wx.Button(self, label="Clear")
        self.clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_clicked)
        button_sizer.Add(self.clear_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)

        # Verbosity toggle
        self.verbose_chk = wx.CheckBox(self, label="Verbose (debug)")
        self.verbose_chk.SetToolTip("Show debug details and enable verbose logging")
        self.verbose_chk.Bind(wx.EVT_CHECKBOX, self._on_verbose_changed)
        main_sizer.Add(self.verbose_chk, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.ALIGN_CENTER, 5)
        
        # Separator
        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 5)
        
        # Results display
        results_label = wx.StaticText(self, label="Results:")
        main_sizer.Add(results_label, 0, wx.LEFT | wx.TOP, 10)
        
        self.results_text = wx.TextCtrl(
            self, 
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
            size=(-1, 300)
        )
        self.results_text.SetFont(wx.Font(
            12, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL
        ))
        main_sizer.Add(self.results_text, 1, wx.EXPAND | wx.ALL, 10)
        
        # Status bar
        self.status_text = wx.StaticText(self, label="Ready")
        self.status_text.SetForegroundColour(wx.Colour(100, 100, 100))
        main_sizer.Add(self.status_text, 0, wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        
        # Initial display
        self._update_display()

    def _apply_system_theme_to_textboxes(self) -> None:
        if not WX_AVAILABLE:
            return
        try:
            bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        except Exception:
            return

        for ctrl_name in ("results_text", "input_text"):
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is None:
                continue
            try:
                ctrl.SetBackgroundColour(bg)
                ctrl.SetForegroundColour(fg)
            except Exception:
                pass

    def _on_sys_colour_changed(self, event):
        try:
            self._apply_system_theme_to_textboxes()
            self.Refresh()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass
    
    def _on_run_clicked(self, event):
        """Handle Run Checks button click."""
        if self.on_run_checks:
            self.set_status("Running checks...")
            self.results_view.set_loading(True)
            self.on_run_checks()

    def _on_verbose_changed(self, event):
        """Handle verbose toggle changes."""
        verbose = bool(self.verbose_chk.GetValue())
        self.results_view.set_verbose(verbose)
        if self.on_set_verbose:
            try:
                self.on_set_verbose(verbose)
            except Exception:
                logger.exception("Failed to set verbose mode")
    
    def _on_explain_clicked(self, event):
        """Handle Get Explanation button click."""
        if self.on_explain:
            self.set_status("Getting LLM explanation...")
            self.on_explain()
    
    def _on_clear_clicked(self, event):
        """Handle Clear button click."""
        self.results_view.clear()
        self.set_status("Cleared")
    
    def _on_model_update(self, model: ResultsViewModel):
        """Handle model updates from ResultsView."""
        self._update_display()
    
    def _update_display(self):
        """Update the display with current results."""
        text = self.results_view.format_summary_text()
        self.results_text.SetValue(text)
        
        # Enable/disable explain button based on results
        has_results = bool(self.results_view.model.check_results)
        self.explain_btn.Enable(has_results and not self.results_view.model.is_loading)

        # Disable run button while busy
        self.run_btn.Enable(not self.results_view.model.is_loading)
        
        # Update status
        if self.results_view.model.is_loading:
            self.set_status("Running...")
        elif self.results_view.model.error_message:
            self.set_status(f"Error: {self.results_view.model.error_message[:50]}")
        elif has_results:
            model = self.results_view.model
            self.set_status(
                f"Completed: {model.total_errors} errors, {model.total_warnings} warnings"
            )
        else:
            self.set_status("Ready")
    
    def set_status(self, message: str):
        """Set the status bar message."""
        if hasattr(self, 'status_text'):
            self.status_text.SetLabel(message)
    
    def set_results(self, results: List[CheckResult]):
        """Set the check results to display."""
        self.results_view.set_results(results)
    
    def set_explanation(self, explanation):
        """Set the LLM explanation."""
        self.results_view.set_explanation(explanation)
    
    def set_error(self, message: str):
        """Set an error message."""
        self.results_view.set_error(message)


class VibeCADDialog(wx.Dialog if WX_AVAILABLE else object):
    """Standalone dialog for VibeCAD when not docked."""
    
    def __init__(self, parent, title="VibeCAD Design Review",
                 on_run_checks=None, on_explain=None, on_set_verbose=None):
        if not WX_AVAILABLE:
            return
        
        super().__init__(
            parent, 
            title=title,
            size=(600, 500),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )
        
        self.panel = VibeCADPanel(
            self, 
            on_run_checks=on_run_checks,
            on_explain=on_explain,
            on_set_verbose=on_set_verbose
        )
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.panel, 1, wx.EXPAND)
        self.SetSizer(sizer)
        
        self.Centre()
