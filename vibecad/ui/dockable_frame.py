"""
Dockable/Separable frame for VibeCAD.

This module provides a frame that can be docked into KiCad's main window
using AUI (Advanced User Interface) manager, or floated as a separate window.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, List, Any

try:
    import wx
    import wx.aui as aui
    try:
        import wx.html2
        HTML2_AVAILABLE = True
    except Exception:
        HTML2_AVAILABLE = False
    # Some environments may provide a stub `wx` module (or partial bindings)
    # that lacks core widgets; treat that as unavailable.
    WX_AVAILABLE = bool(hasattr(wx, "Frame") and hasattr(wx, "Panel"))
except ImportError:
    WX_AVAILABLE = False
    HTML2_AVAILABLE = False
    class wx:
        class Frame:
            pass
    class aui:
        pass

from .results_view import ResultsView, ResultsViewModel
from .design_panel import DesignPanel
from .debug_panel import DebugPanel
from ..checks.base import CheckResult

logger = logging.getLogger(__name__)


class VibeCADFrame(wx.Frame if WX_AVAILABLE else object):
    """
    Dockable/separable frame for VibeCAD.
    
    Features:
    - Can float as independent window
    - Uses AUI for potential docking into KiCad
    - Run Checks button
    - Results panel with per-finding display
    - Question input for LLM queries
    - Get Explanation button
    - Phase 3: Suggestions panel with Apply/Dismiss buttons
    """
    
    def __init__(self, parent=None, title="VibeCAD Design Review",
                 on_run_checks: Optional[Callable] = None,
                 on_explain: Optional[Callable[[Optional[str]], None]] = None,
                 on_ask: Optional[Callable[[str], None]] = None,
                 on_set_verbose: Optional[Callable[[bool], None]] = None,
                 initial_verbose: bool = False,
                 on_toggle_dock: Optional[Callable[[], bool]] = None,
                 on_open_settings: Optional[Callable[[], None]] = None,
                 # Phase 3: Suggestion callbacks
                 on_preview_suggestion: Optional[Callable] = None,
                 on_apply_suggestion: Optional[Callable] = None,
                 on_dismiss_suggestion: Optional[Callable] = None,
                 on_explain_suggestion: Optional[Callable] = None,
                 on_hide_all_previews: Optional[Callable] = None,
                 # Phase 4: Design agent callbacks
                 on_design_message: Optional[Callable[[str], None]] = None,
                 on_approve_action: Optional[Callable] = None,
                 on_reject_action: Optional[Callable] = None,
                 on_new_chat: Optional[Callable[[], None]] = None,
                 # Debug tab callbacks
                 on_get_debug_text: Optional[Callable[[], str]] = None,
                 on_clear_debug: Optional[Callable[[], None]] = None):
        """
        Initialize the VibeCAD frame.
        
        Args:
            parent: Parent window (can be None for floating)
            title: Window title
            on_run_checks: Callback when Run Checks is clicked
            on_explain: Callback when Get Explanation is clicked (receives optional question)
            on_ask: Callback for user questions (receives question string)
            on_set_verbose: Callback for verbose mode toggle
            on_preview_suggestion: Callback to show suggestion preview
            on_apply_suggestion: Callback to apply a suggestion
            on_dismiss_suggestion: Callback to dismiss a suggestion
            on_explain_suggestion: Callback to get LLM explanation for suggestion
            on_hide_all_previews: Callback to hide all previews
            on_design_message: Phase 4 - callback for design assistant messages
            on_approve_action: Phase 4 - callback to approve a design action
            on_reject_action: Phase 4 - callback to reject a design action
        """
        if not WX_AVAILABLE:
            logger.error("wxPython not available")
            return
        
        style = (wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT)
        super().__init__(parent, wx.ID_ANY, title, size=(700, 700), style=style)

        self._base_title = title
        self._is_pinned = False
        self._docking_warned = False
        self._ask_in_progress = False
        self._qa_log: List[str] = []
        
        self.on_run_checks = on_run_checks
        self.on_explain = on_explain
        self.on_ask = on_ask
        self.on_set_verbose = on_set_verbose
        self._initial_verbose = bool(initial_verbose)
        self.on_toggle_dock = on_toggle_dock
        self.on_open_settings = on_open_settings
        
        # Phase 3: Suggestion callbacks
        self.on_preview_suggestion = on_preview_suggestion
        self.on_apply_suggestion = on_apply_suggestion
        self.on_dismiss_suggestion = on_dismiss_suggestion
        self.on_explain_suggestion = on_explain_suggestion
        self.on_hide_all_previews = on_hide_all_previews
        
        # Phase 4: Design agent callbacks
        self.on_design_message = on_design_message
        self.on_approve_action = on_approve_action
        self.on_reject_action = on_reject_action
        self.on_new_chat = on_new_chat

        self.on_get_debug_text = on_get_debug_text
        self.on_clear_debug = on_clear_debug
        
        # Current suggestions
        self._suggestions: List[Any] = []
        
        self.results_view = ResultsView()
        self.results_view.add_update_callback(self._on_model_update)
        
        # AUI manager for internal docking
        self._mgr = aui.AuiManager(self)
        
        self._create_ui()
        self._setup_aui()

        # React to system theme/light-dark changes (macOS, Windows).
        try:
            self.Bind(wx.EVT_SYS_COLOUR_CHANGED, self._on_sys_colour_changed)
        except Exception:
            pass

        # Apply current system theme to text controls.
        try:
            self._apply_system_theme_to_textboxes()
        except Exception:
            pass

        # Restore persisted verbose checkbox state.
        try:
            self.set_verbose_ui(self._initial_verbose)
        except Exception:
            pass
        
        # Bind close event
        self.Bind(wx.EVT_CLOSE, self._on_close)
        
        self.Centre()
    
    def _create_ui(self):
        """Create all UI components."""
        # Main panel that will contain everything
        self.main_panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # === Header Section ===
        header_panel = self._create_header_panel(self.main_panel)
        main_sizer.Add(header_panel, 0, wx.EXPAND | wx.ALL, 5)
        
        # === Toolbar Section ===
        toolbar_panel = self._create_toolbar_panel(self.main_panel)
        main_sizer.Add(toolbar_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        
        # === Notebook for Results, Suggestions, and Design ===
        self.notebook = wx.Notebook(self.main_panel)
        
        # Design tab (Phase 4) - now the primary tab
        design_panel = self._create_design_tab(self.notebook)
        self.notebook.AddPage(design_panel, "🎨 Design")
        
        # Results page
        results_panel = self._create_results_panel(self.notebook)
        self.notebook.AddPage(results_panel, "📋 Check Results")
        
        # Suggestions page (Phase 3)
        suggestions_panel = self._create_suggestions_panel(self.notebook)
        self.notebook.AddPage(suggestions_panel, "💡 Suggestions")

        # Debug page (user-visible command output)
        debug_panel = self._create_debug_panel(self.notebook)
        self.notebook.AddPage(debug_panel, "🐞 Debug")
        
        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)
        
        # === Status Bar ===
        self.status_bar = self.CreateStatusBar(2)
        self.status_bar.SetStatusWidths([-3, -1])
        self.set_status("Ready", "LLM: Unknown")
        
        self.main_panel.SetSizer(main_sizer)

    def _create_debug_panel(self, parent) -> wx.Panel:
        panel = DebugPanel(
            parent,
            on_get_text=self.on_get_debug_text,
            on_clear=self.on_clear_debug,
        )
        return panel
    
    def _create_header_panel(self, parent) -> wx.Panel:
        """Create the header panel with title and description."""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(panel, label="⚡ VibeCAD")
        font = title.GetFont()
        font.SetPointSize(14)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        
        # Separator
        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.TOP, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_toolbar_panel(self, parent) -> wx.Panel:
        """Create the toolbar with action buttons."""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Run Checks button
        self.run_btn = wx.Button(panel, label="▶ Run Checks")
        self.run_btn.SetToolTip("Run all deterministic design rule checks")
        self.run_btn.Bind(wx.EVT_BUTTON, self._on_run_clicked)
        sizer.Add(self.run_btn, 0, wx.ALL, 3)
        
        # Get Explanation button
        self.explain_btn = wx.Button(panel, label="💡 Explain Results")
        self.explain_btn.SetToolTip("Get LLM explanation of all findings")
        self.explain_btn.Bind(wx.EVT_BUTTON, self._on_explain_clicked)
        self.explain_btn.Enable(False)
        sizer.Add(self.explain_btn, 0, wx.ALL, 3)
        
        # Clear button
        self.clear_btn = wx.Button(panel, label="🗑 Clear")
        self.clear_btn.SetToolTip("Clear all results")
        self.clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_clicked)
        sizer.Add(self.clear_btn, 0, wx.ALL, 3)

        # Settings
        self.settings_btn = wx.Button(panel, label="⚙ Settings")
        self.settings_btn.SetToolTip("Configure LLM API key, endpoint, model, etc")
        self.settings_btn.Bind(wx.EVT_BUTTON, self._on_settings_clicked)
        sizer.Add(self.settings_btn, 0, wx.ALL, 3)
        
        sizer.AddStretchSpacer()
        
        # Verbose checkbox
        self.verbose_chk = wx.CheckBox(panel, label="Verbose")
        self.verbose_chk.SetToolTip("Show debug details in output")
        self.verbose_chk.Bind(wx.EVT_CHECKBOX, self._on_verbose_changed)
        sizer.Add(self.verbose_chk, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        
        # Dock/Pin toggle button
        self.dock_btn = wx.Button(panel, label="📌 Dock / Pin")
        self.dock_btn.SetToolTip(
            "Dock into KiCad if possible; otherwise pin window on top"
        )
        self.dock_btn.Bind(wx.EVT_BUTTON, self._on_dock_toggle)
        sizer.Add(self.dock_btn, 0, wx.ALL, 3)
        
        panel.SetSizer(sizer)
        return panel

    def set_verbose_ui(self, verbose: bool) -> None:
        """Set verbose state in the UI without requiring a user click."""
        if not WX_AVAILABLE:
            return
        v = bool(verbose)
        try:
            if hasattr(self, 'verbose_chk') and self.verbose_chk is not None:
                self.verbose_chk.SetValue(v)
        except Exception:
            pass

    def _apply_system_theme_to_textboxes(self) -> None:
        """Apply system window colors to key text inputs/outputs."""
        if not WX_AVAILABLE:
            return
        try:
            bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        except Exception:
            return

        for ctrl_name in ("results_text", "question_input"):
            ctrl = getattr(self, ctrl_name, None)
            if ctrl is None:
                continue
            try:
                ctrl.SetBackgroundColour(bg)
                ctrl.SetForegroundColour(fg)
            except Exception:
                pass

        # Also update the Design tab's message input if present.
        try:
            if hasattr(self, 'design_panel') and self.design_panel is not None:
                if hasattr(self.design_panel, 'apply_system_theme'):
                    self.design_panel.apply_system_theme()
        except Exception:
            pass

    def _on_sys_colour_changed(self, event):
        """Handle OS theme/system color changes."""
        try:
            self._apply_system_theme_to_textboxes()
            self.Refresh()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass
        try:
            if hasattr(self, 'results_view') and self.results_view is not None:
                self.results_view.set_verbose(v)
        except Exception:
            pass
    
    def _create_results_panel(self, parent) -> wx.Panel:
        """Create the results display panel."""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # Results output: prefer HTML rendering so LLM markdown displays correctly.
        self.results_html = None
        if HTML2_AVAILABLE:
            try:
                self.results_html = wx.html2.WebView.New(panel)
                try:
                    self._install_copy_shortcuts(self.results_html)
                except Exception:
                    pass
                try:
                    # Slightly increase legibility without changing layout too much.
                    if hasattr(self.results_html, 'SetZoomFactor'):
                        self.results_html.SetZoomFactor(1.08)
                except Exception:
                    pass
                sizer.Add(self.results_html, 1, wx.EXPAND | wx.ALL, 5)
            except Exception:
                self.results_html = None

        if self.results_html is None:
            # Fallback: plain text control.
            self.results_text = wx.TextCtrl(
                panel,
                style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.TE_AUTO_URL,
                size=(-1, 250)
            )
            self._apply_results_font()
            try:
                self._install_copy_shortcuts(self.results_text)
            except Exception:
                pass
            sizer.Add(self.results_text, 1, wx.EXPAND | wx.ALL, 5)

        # Keep the legacy Q&A input, but scope it to the Results tab so the
        # Design tab doesn't show a duplicate input area.
        question_panel = self._create_question_panel(panel)
        sizer.Add(question_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        panel.SetSizer(sizer)
        return panel

    def _install_copy_shortcuts(self, widget) -> None:
        """Ensure Cmd/Ctrl+C copies selected text from read-only output widgets."""
        try:
            import wx
        except Exception:
            return

        def _on_char_hook(evt):
            try:
                key = evt.GetKeyCode()
                is_copy = (evt.CmdDown() or evt.ControlDown()) and key in (ord('C'), ord('c'))
                if is_copy:
                    try:
                        # wx.TextCtrl and wx.html2.WebView both provide Copy() on most builds.
                        if hasattr(widget, 'Copy'):
                            widget.Copy()
                            return
                    except Exception:
                        pass
            except Exception:
                pass
            evt.Skip()

        try:
            widget.Bind(wx.EVT_CHAR_HOOK, _on_char_hook)
        except Exception:
            # Some platforms/widgets don't support EVT_CHAR_HOOK; ignore.
            pass
    
    def _create_design_tab(self, parent) -> wx.Panel:
        """Create the Design tab with Copilot-style interface (Phase 4)."""
        self.design_panel = DesignPanel(
            parent,
            on_send_message=self._on_design_message,
            on_approve_action=self._on_approve_design_action,
            on_reject_action=self._on_reject_design_action,
            on_suggestion_click=self._on_design_suggestion_click,
        )
        return self.design_panel

    def set_thinking_output_enabled(self, enabled: bool) -> None:
        """Enable/disable thinking/status output in the Design tab."""
        try:
            if hasattr(self, 'design_panel') and self.design_panel is not None:
                self.design_panel.set_thinking_output_enabled(bool(enabled))
        except Exception:
            pass
    
    def _on_design_message(self, message: str):
        """Handle message from the design panel."""
        if self.on_design_message:
            try:
                return self.on_design_message(message)
            except Exception as e:
                logger.exception(f"Design message failed: {e}")
                if hasattr(self, 'design_panel'):
                    self.design_panel.add_response(f"❌ Error: {e}")
        return False
    
    def _on_approve_design_action(self, action_data):
        """Handle action approval from design panel."""
        if self.on_approve_action:
            try:
                self.on_approve_action(action_data)
            except Exception as e:
                logger.exception(f"Action approval failed: {e}")
    
    def _on_reject_design_action(self, action_data):
        """Handle action rejection from design panel."""
        if self.on_reject_action:
            try:
                self.on_reject_action(action_data)
            except Exception as e:
                logger.exception(f"Action rejection failed: {e}")
    
    def _on_design_suggestion_click(self, suggestion: str):
        """Handle suggestion chip click from design panel."""
        # Just puts the text in the input - send not triggered automatically
        pass
    
    def _create_suggestions_panel(self, parent) -> wx.Panel:
        """Create the suggestions panel (Phase 3)."""
        panel = wx.Panel(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header with info
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        title = wx.StaticText(panel, label="💡 Suggested Fixes")
        font = title.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        header_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        header_sizer.AddStretchSpacer()
        
        # Hide all previews button
        self.hide_previews_btn = wx.Button(panel, label="Hide All Previews")
        self.hide_previews_btn.Bind(wx.EVT_BUTTON, self._on_hide_all_previews_clicked)
        header_sizer.Add(self.hide_previews_btn, 0, wx.ALL, 5)
        
        sizer.Add(header_sizer, 0, wx.EXPAND)
        
        # Info text
        info = wx.StaticText(
            panel,
            label="Suggestions are generated deterministically. "
                  "The LLM only explains, never generates geometry.\n"
                  "Review each suggestion carefully before applying."
        )
        # Use theme-aware muted text so it's readable in KiCad dark mode.
        try:
            info.SetForegroundColour(self._muted_text_colour())
        except Exception:
            info.SetForegroundColour(wx.Colour(100, 100, 100))
        sizer.Add(info, 0, wx.LEFT | wx.BOTTOM, 10)
        
        # Suggestions summary
        self.suggestions_summary = wx.StaticText(panel, label="No suggestions yet. Run checks to generate suggestions.")
        try:
            self.suggestions_summary.SetForegroundColour(self._muted_text_colour())
        except Exception:
            pass
        sizer.Add(self.suggestions_summary, 0, wx.LEFT | wx.BOTTOM, 10)
        
        # Scrolled window for suggestion cards
        self.suggestions_scroll = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        self.suggestions_scroll.SetScrollRate(0, 20)
        self.suggestions_sizer = wx.BoxSizer(wx.VERTICAL)
        self.suggestions_scroll.SetSizer(self.suggestions_sizer)
        
        sizer.Add(self.suggestions_scroll, 1, wx.EXPAND | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel
    
    def _on_hide_all_previews_clicked(self, event):
        """Handle Hide All Previews button click."""
        if self.on_hide_all_previews:
            try:
                self.on_hide_all_previews()
            except Exception as e:
                logger.exception(f"Failed to hide previews: {e}")
    
    def _create_question_panel(self, parent) -> wx.Panel:
        """Create the question input panel for user queries."""
        panel = wx.Panel(parent)
        # Use system colors so the input remains readable in KiCad dark/light themes.
        try:
            bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
            panel.SetBackgroundColour(bg)
        except Exception:
            fg = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Label
        label = wx.StaticText(panel, label="💬 Ask a Question (LLM will answer based on current design data):")
        sizer.Add(label, 0, wx.LEFT | wx.TOP, 8)
        
        # Input row
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.question_input = wx.TextCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER,
            size=(-1, -1)
        )
        if fg is not None:
            try:
                self.question_input.SetBackgroundColour(
                    wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
                )
                self.question_input.SetForegroundColour(
                    wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
                )
            except Exception:
                pass
        self.question_input.SetHint("e.g., 'Why is the board outline missing?' or 'Are all power nets connected?'")
        self.question_input.Bind(wx.EVT_TEXT_ENTER, self._on_ask_clicked)
        input_sizer.Add(self.question_input, 1, wx.ALL | wx.EXPAND, 5)
        
        self.ask_btn = wx.Button(panel, label="Ask")
        self.ask_btn.SetToolTip("Send question to LLM (uses current check results as context)")
        self.ask_btn.Bind(wx.EVT_BUTTON, self._on_ask_clicked)
        input_sizer.Add(self.ask_btn, 0, wx.ALL, 5)
        
        sizer.Add(input_sizer, 0, wx.EXPAND)
        
        # Help text
        help_text = wx.StaticText(
            panel,
            label="Note: The LLM can only reference data from the current design. "
                  "It will not infer specs or fetch external datasheets."
        )
        help_text.SetForegroundColour(wx.Colour(120, 120, 120))
        font = help_text.GetFont()
        font.SetPointSize(9)
        help_text.SetFont(font)
        sizer.Add(help_text, 0, wx.LEFT | wx.BOTTOM, 8)
        
        panel.SetSizer(sizer)
        return panel
    
    def _setup_aui(self):
        """Set up AUI manager for docking support."""
        self._mgr.AddPane(
            self.main_panel,
            aui.AuiPaneInfo()
            .Name("main")
            .CenterPane()
            .PaneBorder(False)
        )
        self._mgr.Update()
    
    def _on_close(self, event):
        """Handle window close."""
        # Single-instance behavior: on user close, hide instead of destroying.
        # This prevents multiple windows when the toolbar icon is clicked repeatedly.
        try:
            if event.CanVeto():
                try:
                    self.Hide()
                except Exception:
                    pass
                try:
                    event.Veto()
                except Exception:
                    return
                return
        except Exception:
            pass

        # If the close cannot be vetoed (e.g., app shutdown), clean up.
        try:
            if getattr(self, "_mgr", None) is not None:
                self._mgr.UnInit()
        except Exception:
            pass
        event.Skip()
    
    # === Event Handlers ===
    
    def _on_run_clicked(self, event):
        """Handle Run Checks button."""
        if self.on_run_checks:
            self.set_status("Running checks...", "")
            self.results_view.set_loading(True)
            self._update_display()
            self.on_run_checks()
    
    def _on_explain_clicked(self, event):
        """Handle Explain Results button."""
        if self.on_explain:
            question = self.question_input.GetValue().strip() or None
            self.set_status("Getting LLM explanation...", "")
            self.results_view.set_loading(True)
            self._update_display()
            self.on_explain(question)
    
    def _on_ask_clicked(self, event):
        """Handle Ask button or Enter in question input."""
        question = self.question_input.GetValue().strip()
        if not question:
            wx.MessageBox(
                "Please enter a question.",
                "No Question",
                wx.OK | wx.ICON_INFORMATION
            )
            return
        
        if self.on_ask:
            self.set_status(f"Asking: {question[:40]}...", "")
            # Do NOT set results_view loading here; that would disable all buttons.
            # We only disable the Ask button while a question is in flight.
            self._ask_in_progress = True
            self._update_display()
            try:
                self.on_ask(question)
            except Exception as e:
                # If the callback fails synchronously, ensure Ask is not left disabled.
                self._ask_in_progress = False
                self.set_error(f"Failed to start Ask: {e}")

    def _on_settings_clicked(self, event):
        """Open settings dialog."""
        if self.on_open_settings:
            try:
                self.on_open_settings()
            except Exception:
                logger.exception("Failed to open settings")
    
    def _on_clear_clicked(self, event):
        """Handle Clear button — starts a new chat session."""
        self.results_view.clear()
        self.question_input.SetValue("")
        # Clear the Design tab chat history
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.clear_history()
        # Tell the plugin to reset the agent's conversation context
        if self.on_new_chat:
            try:
                self.on_new_chat()
            except Exception:
                logger.exception("on_new_chat callback failed")
        self.set_status("New chat started", "")
    
    def _on_verbose_changed(self, event):
        """Handle verbose checkbox toggle."""
        verbose = self.verbose_chk.GetValue()
        self.results_view.set_verbose(verbose)
        if self.on_set_verbose:
            self.on_set_verbose(verbose)
    
    def _on_dock_toggle(self, event):
        """Attempt to dock into KiCad; otherwise fall back to pin/unpin."""
        docked = False
        if self.on_toggle_dock:
            try:
                docked = bool(self.on_toggle_dock())
            except Exception:
                logger.exception("Dock toggle callback failed")
                docked = False

        if docked:
            # If we successfully docked, the frame will be hidden by the plugin.
            self.set_status("Docked into KiCad", "")
            return

        # If docking was attempted but not possible, tell the user once.
        if self.on_toggle_dock and not self._docking_warned:
            try:
                wx.MessageBox(
                    "Could not dock into KiCad. Falling back to pin/unpin (always-on-top).\n\n"
                    "Tip: Enable Verbose and check KiCad's scripting console for details.",
                    "Docking not available",
                    wx.OK | wx.ICON_INFORMATION,
                )
                self._docking_warned = True
            except Exception:
                pass

        # Fallback: Toggle always-on-top style (pin behavior).
        self._set_pinned(not self._is_pinned)

    def _set_pinned(self, pinned: bool):
        """Pin/unpin the window and update visible UI state."""
        self._is_pinned = bool(pinned)

        try:
            style = self.GetWindowStyleFlag()
            if self._is_pinned:
                style |= wx.STAY_ON_TOP
            else:
                style &= ~wx.STAY_ON_TOP
            self.SetWindowStyleFlag(style)
        except Exception:
            # Fallback to old API if needed
            try:
                style = self.GetWindowStyle()
                if self._is_pinned:
                    self.SetWindowStyle(style | wx.STAY_ON_TOP)
                else:
                    self.SetWindowStyle(style & ~wx.STAY_ON_TOP)
            except Exception:
                pass

        if self._is_pinned:
            self.dock_btn.SetLabel("📌 Unpin")
            self.SetTitle(f"{self._base_title}  [PINNED]")
            self.set_status("Window pinned on top", "")
            try:
                self.Raise()
            except Exception:
                pass
        else:
            self.dock_btn.SetLabel("📌 Dock / Pin")
            self.SetTitle(self._base_title)
            self.set_status("Window unpinned", "")

        try:
            self.Refresh()
            self.Update()
        except Exception:
            pass
    
    def _on_model_update(self, model: ResultsViewModel):
        """Handle model updates."""
        self._update_display()
    
    def _update_display(self):
        """Update the display with current state."""
        text = self.results_view.format_summary_text()

        # If the user clicked Ask before running checks, don't show the
        # "No checks have been run yet" hint while the request is in-flight.
        try:
            if (
                self._ask_in_progress
                and (not self.results_view.model.check_results)
                and (not self.results_view.model.is_loading)
                and (not self.results_view.model.error_message)
            ):
                text = "Thinking..."
        except Exception:
            pass
        if self._qa_log:
            text = text + "\n\n" + "".join(self._qa_log)

        if getattr(self, 'results_html', None) is not None:
            try:
                html = self.results_view.format_html()
                self.results_html.SetPage(html, "")
            except Exception:
                # If HTML rendering fails, fall back to text (if available).
                if hasattr(self, 'results_text') and self.results_text is not None:
                    self.results_text.SetValue(text)
        else:
            self.results_text.SetValue(text)
        
        # Update button states
        has_results = bool(self.results_view.model.check_results)
        is_loading = self.results_view.model.is_loading
        
        self.run_btn.Enable(not is_loading)
        self.explain_btn.Enable(has_results and not is_loading)
        # Ask should be usable even before running checks (it can still answer about config or
        # explain that no design context is available). Only disable it while asking.
        self.ask_btn.Enable((not self._ask_in_progress) and (not is_loading))
        self.clear_btn.Enable(not is_loading)
        self.settings_btn.Enable(True)
        
        # Update status
        if is_loading:
            pass  # Keep current loading message
        elif self.results_view.model.error_message:
            self.set_status(f"Error: {self.results_view.model.error_message[:50]}", "")
        elif has_results:
            m = self.results_view.model
            self.set_status(f"Found {m.total_errors} error(s), {m.total_warnings} warning(s)", "")
    
    # === Public Methods ===
    
    def set_status(self, message: str, llm_status: str = ""):
        """Set status bar message."""
        self.status_bar.SetStatusText(message, 0)
        if llm_status:
            self.status_bar.SetStatusText(llm_status, 1)
    
    def set_llm_status(self, configured: bool):
        """Update LLM status indicator."""
        if configured:
            self.status_bar.SetStatusText("LLM: Ready", 1)
        else:
            self.status_bar.SetStatusText("LLM: Not configured", 1)
    
    def set_results(self, results: List[CheckResult]):
        """Set check results."""
        self.results_view.set_results(results)
    
    def set_explanation(self, explanation: Any):
        """Set LLM explanation."""
        self.results_view.set_explanation(explanation)
    
    def set_error(self, message: str):
        """Set error message."""
        self._ask_in_progress = False
        self.results_view.set_error(message)
    
    def set_suggestions(self, suggestions: List[Any]):
        """Set the list of suggestions to display (Phase 3).
        
        Args:
            suggestions: List of Suggestion objects
        """
        self._suggestions = suggestions
        self._rebuild_suggestion_cards()

        # Do not auto-switch tabs; the user controls the active view.
    
    # === Phase 4: Design Panel Methods ===
    
    def add_design_response(self, content: str):
        """Add a response message to the design panel."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.add_response(content)
    
    def add_design_action_preview(self, action_type: str, description: str, 
                                   preview_text: str, action_data: Any):
        """Add an action preview to the design panel that requires user approval."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.add_action_preview(action_type, description, preview_text, action_data)
    
    def set_design_suggestions(self, suggestions: List[str]):
        """Set the suggestion chips in the design panel."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.set_suggestions(suggestions)
    
    def set_design_thinking(self, thinking: bool = True):
        """Show/hide thinking indicator in design panel."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.set_thinking(thinking)

    def set_agent_running(self, running: bool):
        """Toggle send/pause button in design panel."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.set_agent_running(running)

    def set_agent_awaiting_input(self, awaiting: bool):
        """Re-enable input when agent asks a question."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.set_agent_awaiting_input(awaiting)

    def add_thinking_message(self, text: str):
        """Add a thinking/status message to design panel."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.add_thinking_message(text)

    def set_pause_callback(self, cb):
        """Set the pause callback on the design panel."""
        if hasattr(self, 'design_panel') and self.design_panel:
            self.design_panel.set_pause_callback(cb)

    def _rebuild_suggestion_cards(self):
        """Rebuild the suggestion cards in the scroll panel."""
        # Clear existing cards
        self.suggestions_sizer.Clear(delete_windows=True)
        
        if not self._suggestions:
            self.suggestions_summary.SetLabel(
                "No suggested fixes were generated for this design. "
                "Some findings may not have automated fixes yet."
            )
            self.suggestions_scroll.Layout()
            self.suggestions_scroll.FitInside()
            return
        
        # Update summary
        pending = sum(1 for s in self._suggestions if str(s.status) == 'pending')
        applied = sum(1 for s in self._suggestions if str(s.status) == 'applied')
        self.suggestions_summary.SetLabel(
            f"{len(self._suggestions)} suggestion(s): {pending} pending, {applied} applied"
        )
        
        # Create cards for each suggestion
        for suggestion in self._suggestions:
            card = self._create_suggestion_card(self.suggestions_scroll, suggestion)
            self.suggestions_sizer.Add(card, 0, wx.EXPAND | wx.ALL, 5)
        
        self.suggestions_scroll.Layout()
        self.suggestions_scroll.FitInside()
    
    def _create_suggestion_card(self, parent, suggestion) -> wx.Panel:
        """Create a card UI for a single suggestion."""
        status = str(suggestion.status)

        # Theme-aware coloring: in dark mode we tint darker backgrounds and
        # use explicit foreground colors for readability.
        is_dark = self._is_dark_mode()
        base_bg = self._window_bg_colour()
        base_fg = self._window_text_colour()
        muted_fg = self._muted_text_colour()

        if status == 'applied':
            accent = wx.Colour(60, 180, 90)
        elif status == 'dismissed':
            accent = wx.Colour(140, 140, 140)
        else:
            accent = wx.Colour(220, 170, 60)

        if is_dark:
            bg_color = self._blend_colours(base_bg, accent, 0.22)
        else:
            # Keep the previous light tints for light mode.
            if status == 'applied':
                bg_color = wx.Colour(220, 255, 220)
            elif status == 'dismissed':
                bg_color = wx.Colour(240, 240, 240)
            else:
                bg_color = wx.Colour(255, 250, 230)
        
        card = wx.Panel(parent, style=wx.BORDER_SIMPLE)
        card.SetBackgroundColour(bg_color)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title row
        title_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        icon = wx.StaticText(card, label="💡")
        title_sizer.Add(icon, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        title = wx.StaticText(card, label=suggestion.title)
        title.SetForegroundColour(base_fg)
        font = title.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        title_sizer.Add(title, 1, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        status_label = wx.StaticText(card, label=status.upper())
        status_label.SetForegroundColour(muted_fg)
        title_sizer.Add(status_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        sizer.Add(title_sizer, 0, wx.EXPAND)
        
        # Warning text
        if status == 'pending':
            warning = wx.StaticText(card, label="⚠️ Suggestion – not applied. Review before applying.")
            # Brighter warning in dark mode.
            warning.SetForegroundColour(wx.Colour(240, 170, 80) if is_dark else wx.Colour(180, 100, 0))
            sizer.Add(warning, 0, wx.LEFT | wx.BOTTOM, 10)
        
        # Description
        desc = wx.StaticText(card, label=suggestion.description)
        desc.SetForegroundColour(base_fg)
        desc.Wrap(500)
        sizer.Add(desc, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Rule reference
        rule_text = wx.StaticText(card, label=f"Addresses: {suggestion.rule_id}")
        rule_text.SetForegroundColour(muted_fg)
        sizer.Add(rule_text, 0, wx.LEFT | wx.BOTTOM, 10)
        
        # Buttons (only for pending suggestions)
        if status == 'pending':
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            # Preview button
            preview_btn = wx.Button(card, label="👁 Preview")
            preview_btn.SetToolTip("Show preview overlay on the board")
            preview_btn.Bind(wx.EVT_BUTTON, lambda e, s=suggestion: self._on_preview_clicked(s))
            btn_sizer.Add(preview_btn, 0, wx.ALL, 3)
            
            # Explain button
            explain_btn = wx.Button(card, label="💬 Explain")
            explain_btn.SetToolTip("Get LLM explanation of this suggestion")
            explain_btn.Bind(wx.EVT_BUTTON, lambda e, s=suggestion: self._on_explain_suggestion_clicked(s))
            btn_sizer.Add(explain_btn, 0, wx.ALL, 3)
            
            btn_sizer.AddStretchSpacer()
            
            # Dismiss button
            dismiss_btn = wx.Button(card, label="✕ Dismiss")
            dismiss_btn.Bind(wx.EVT_BUTTON, lambda e, s=suggestion: self._on_dismiss_clicked(s))
            btn_sizer.Add(dismiss_btn, 0, wx.ALL, 3)
            
            # Apply button
            apply_btn = wx.Button(card, label="✓ Apply")
            # Theme-aware emphasis color.
            try:
                if self._is_dark_mode():
                    base_bg = self._window_bg_colour()
                    apply_bg = self._blend_colours(base_bg, wx.Colour(60, 180, 90), 0.35)
                    apply_btn.SetBackgroundColour(apply_bg)
                    apply_btn.SetForegroundColour(self._window_text_colour())
                else:
                    apply_btn.SetBackgroundColour(wx.Colour(200, 255, 200))
            except Exception:
                pass
            apply_btn.Bind(wx.EVT_BUTTON, lambda e, s=suggestion: self._on_apply_clicked(s))
            btn_sizer.Add(apply_btn, 0, wx.ALL, 3)
            
            sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Undo note
        undo = wx.StaticText(card, label="ℹ️ Changes can be undone via Edit > Undo")
        undo.SetForegroundColour(muted_fg)
        font = undo.GetFont()
        font.SetPointSize(9)
        undo.SetFont(font)
        sizer.Add(undo, 0, wx.LEFT | wx.BOTTOM, 10)
        
        card.SetSizer(sizer)
        return card

    def _window_bg_colour(self) -> "wx.Colour":
        try:
            return wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        except Exception:
            return wx.Colour(30, 30, 30)

    def _window_text_colour(self) -> "wx.Colour":
        try:
            return wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        except Exception:
            return wx.Colour(230, 230, 230)

    def _muted_text_colour(self) -> "wx.Colour":
        # Create a muted text color derived from theme fg/bg.
        fg = self._window_text_colour()
        bg = self._window_bg_colour()
        return self._blend_colours(fg, bg, 0.45 if self._is_dark_mode() else 0.35)

    def _is_dark_mode(self) -> bool:
        # Prefer wx Appearance API when available.
        try:
            app = wx.SystemSettings.GetAppearance()
            is_dark = getattr(app, "IsDark", None)
            if callable(is_dark):
                return bool(is_dark())
        except Exception:
            pass

        # Fallback: estimate via window background luminance.
        try:
            c = self._window_bg_colour()
            r, g, b = int(c.Red()), int(c.Green()), int(c.Blue())
            luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return luma < 128
        except Exception:
            return False

    def _blend_colours(self, a: "wx.Colour", b: "wx.Colour", t: float) -> "wx.Colour":
        # Linear interpolate a->b with t in [0,1].
        try:
            t = max(0.0, min(1.0, float(t)))
        except Exception:
            t = 0.5
        ar, ag, ab = int(a.Red()), int(a.Green()), int(a.Blue())
        br, bg, bb = int(b.Red()), int(b.Green()), int(b.Blue())
        r = int(round(ar + (br - ar) * t))
        g = int(round(ag + (bg - ag) * t))
        b2 = int(round(ab + (bb - ab) * t))
        return wx.Colour(r, g, b2)
    
    def _on_preview_clicked(self, suggestion):
        """Handle Preview button click for a suggestion."""
        if self.on_preview_suggestion:
            try:
                self.on_preview_suggestion(suggestion)
            except Exception as e:
                logger.exception(f"Preview failed: {e}")
    
    def _on_explain_suggestion_clicked(self, suggestion):
        """Handle Explain button click for a suggestion (runs LLM in background)."""
        if not self.on_explain_suggestion:
            return

        import threading

        def _worker():
            try:
                explanation = self.on_explain_suggestion(suggestion)
            except Exception as e:
                logger.exception(f"Explain failed: {e}")
                explanation = f"Error: {e}"

            def _show():
                try:
                    dlg = wx.MessageDialog(
                        self._modal_parent(),
                        explanation,
                        f"Explanation: {suggestion.title}",
                        wx.OK | wx.ICON_INFORMATION,
                    )
                    dlg.ShowModal()
                    dlg.Destroy()
                except Exception:
                    pass

            wx.CallAfter(_show)

        threading.Thread(target=_worker, daemon=True).start()
    
    def _on_dismiss_clicked(self, suggestion):
        """Handle Dismiss button click for a suggestion."""
        if self.on_dismiss_suggestion:
            try:
                self.on_dismiss_suggestion(suggestion)
                self._rebuild_suggestion_cards()
            except Exception as e:
                logger.exception(f"Dismiss failed: {e}")
    
    def _on_apply_clicked(self, suggestion):
        """Handle Apply button click for a suggestion."""
        # Confirmation dialog
        dlg = wx.MessageDialog(
            self._modal_parent(),
            f"Apply this suggestion?\n\n{suggestion.description}\n\n"
            "This action can be undone via Edit > Undo.",
            "Confirm Apply",
            wx.YES_NO | wx.ICON_QUESTION
        )
        
        if dlg.ShowModal() == wx.ID_YES:
            if self.on_apply_suggestion:
                try:
                    result = self.on_apply_suggestion(suggestion)
                    if result and result.success:
                        wx.MessageBox(result.message, "Success", wx.OK | wx.ICON_INFORMATION, parent=self._modal_parent())
                    elif result:
                        wx.MessageBox(f"Failed: {result.error}", "Error", wx.OK | wx.ICON_ERROR, parent=self._modal_parent())
                    self._rebuild_suggestion_cards()
                except Exception as e:
                    logger.exception(f"Apply failed: {e}")
                    wx.MessageBox(f"Apply failed: {e}", "Error", wx.OK | wx.ICON_ERROR, parent=self._modal_parent())
        
        dlg.Destroy()

    def _modal_parent(self):
        """Best-effort parent for modal dialogs.

        When docked, the VibeCADFrame is intentionally hidden and the visible
        UI is reparented into KiCad's top-level frame. Parenting dialogs to the
        hidden frame can create ghost/blank windows on macOS.
        """
        try:
            if hasattr(self, "main_panel") and self.main_panel is not None:
                p = wx.GetTopLevelParent(self.main_panel)
                if p is not None:
                    return p
        except Exception:
            pass

        try:
            wins = wx.GetTopLevelWindows()
            if wins:
                return wins[0]
        except Exception:
            pass

        return self
    
    def append_answer(self, question: str, answer: str):
        """Append a Q&A pair to the results."""
        self._ask_in_progress = False
        qa_text = f"\n\n{'─' * 50}\n💬 Q: {question}\n\n🤖 A: {answer}\n"
        self._qa_log.append(qa_text)
        self._update_display()
        try:
            self.results_text.ShowPosition(self.results_text.GetLastPosition())
        except Exception:
            pass

    def _apply_results_font(self):
        """Apply a larger, more modern UI font to the output box."""
        try:
            base = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
            point_size = max(int(getattr(base, "GetPointSize", lambda: 10)() or 10) + 2, 12)

            # Prefer a modern UI font; fall back to system default.
            # On macOS, SF Pro is usually the system UI font.
            preferred_faces = [
                "SF Pro Text",
                ".AppleSystemUIFont",
                "San Francisco",
                "Helvetica Neue",
                "Segoe UI",
                "Inter",
            ]

            font = wx.Font(wx.FontInfo(point_size).Family(wx.FONTFAMILY_SWISS))

            try:
                enum = wx.FontEnumerator()
                enum.EnumerateFacenames()
                available = set(enum.GetFacenames())
                for face in preferred_faces:
                    if face in available:
                        font.SetFaceName(face)
                        break
            except Exception:
                pass

            self.results_text.SetFont(font)
        except Exception:
            # Last-resort fallback
            try:
                self.results_text.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
            except Exception:
                pass


def create_vibecad_frame(parent=None, **callbacks) -> Optional[VibeCADFrame]:
    """
    Factory function to create a VibeCAD frame.
    
    Args:
        parent: Optional parent window
        **callbacks: Callback functions (on_run_checks, on_explain, on_ask, on_set_verbose)
    
    Returns:
        VibeCADFrame instance or None if wx not available
    """
    if not WX_AVAILABLE:
        logger.error("Cannot create frame: wxPython not available")
        return None
    
    return VibeCADFrame(parent, **callbacks)
