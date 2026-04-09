"""
Dockable/Separable frame for VibeCAD.

This frame hosts the Design assistant chat and a Debug tab. The legacy
"Check Results" and "Suggestions" tabs have been removed.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, Any, List

from . import theme

try:
    import wx
    import wx.aui as aui

    # Some environments may provide a stub `wx` module (or partial bindings)
    # that lacks core widgets; treat that as unavailable.
    WX_AVAILABLE = bool(hasattr(wx, "Frame") and hasattr(wx, "Panel"))
except ImportError:
    WX_AVAILABLE = False

    class wx:  # type: ignore
        class Frame:
            pass
            pass

    class aui:  # type: ignore
        class AuiManager:
            def __init__(self, *_args, **_kwargs):
                pass
                pass

logger = logging.getLogger(__name__)

from .design_panel import DesignPanel
from .debug_panel import DebugPanel


class VibeCADFrame(wx.Frame if WX_AVAILABLE else object):
    """Dockable/separable frame for VibeCAD."""

    def __init__(
        self,
        parent=None,
        title: str = "VibeCAD Design Review",
        *,
        on_set_verbose: Optional[Callable[[bool], None]] = None,
        initial_verbose: bool = False,
        on_toggle_dock: Optional[Callable[[], bool]] = None,
        on_open_settings: Optional[Callable[[], None]] = None,
        on_design_message: Optional[Callable[[str], object]] = None,
        on_run_benchmark: Optional[Callable[[], None]] = None,
        on_approve_action: Optional[Callable[[Any], None]] = None,
        on_reject_action: Optional[Callable[[Any], None]] = None,
        on_new_chat: Optional[Callable[[], None]] = None,
        on_get_debug_text: Optional[Callable[[], str]] = None,
        on_clear_debug: Optional[Callable[[], None]] = None,
        on_llm_controls_changed: Optional[Callable[[str, bool], None]] = None,
        on_before_destroy: Optional[Callable[[], None]] = None,
    ):
        if not WX_AVAILABLE:
            logger.error("wxPython not available")
            return

        style = wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT
        super().__init__(parent, wx.ID_ANY, title, size=(700, 700), style=style)

        self._base_title = str(title or "VibeCAD")
        self._is_pinned = False
        self._docking_warned = False

        self.on_toggle_dock = on_toggle_dock
        self.on_open_settings = on_open_settings
        self.on_design_message = on_design_message
        self.on_run_benchmark = on_run_benchmark
        self.on_approve_action = on_approve_action
        self.on_reject_action = on_reject_action
        self.on_new_chat = on_new_chat
        self.on_get_debug_text = on_get_debug_text
        self.on_clear_debug = on_clear_debug
        self.on_llm_controls_changed = on_llm_controls_changed
        self.on_before_destroy = on_before_destroy
        self._force_destroy = False

        self._mgr = aui.AuiManager(self)

        self._create_ui()
        self._setup_aui()

        try:
            self.Bind(wx.EVT_SYS_COLOUR_CHANGED, self._on_sys_colour_changed)
        except Exception:
            pass

        try:
            self.set_verbose_ui(bool(initial_verbose))
        except Exception:
            pass

        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Centre()

    # ── UI ───────────────────────────────────────────────────────

    def _create_ui(self) -> None:
        self.main_panel = wx.Panel(self)
        try:
            self.main_panel.SetBackgroundColour(self._panel_bg_colour())
            self.main_panel.SetForegroundColour(theme.window_text_colour())
            if hasattr(self.main_panel, "SetBackgroundStyle") and hasattr(wx, "BG_STYLE_PAINT"):
                self.main_panel.SetBackgroundStyle(wx.BG_STYLE_PAINT)
            self.main_panel.Bind(wx.EVT_PAINT, self._on_main_panel_paint)
        except Exception:
            pass
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.header_panel = self._create_header_panel(self.main_panel)
        main_sizer.Add(self.header_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.toolbar_panel = self._create_toolbar_panel(self.main_panel)
        main_sizer.Add(self.toolbar_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        self.notebook = wx.Notebook(self.main_panel)
        try:
            self.notebook.SetBackgroundColour(theme.panel_bg_colour())
            self.notebook.SetForegroundColour(theme.window_text_colour())
        except Exception:
            pass

        self.design_panel = self._create_design_tab(self.notebook)
        self.notebook.AddPage(self.design_panel, "🎨 Design")

        self.debug_panel = self._create_debug_tab(self.notebook)
        self.notebook.AddPage(self.debug_panel, "🐞 Debug")

        main_sizer.Add(self.notebook, 1, wx.EXPAND | wx.ALL, 5)

        self.status_bar = self.CreateStatusBar(2)
        self.status_bar.SetStatusWidths([-3, -1])
        self.set_status("Ready", "LLM: Unknown")
        self._apply_frame_theme()

        self.main_panel.SetSizer(main_sizer)
        try:
            self.main_panel.Layout()
            self.main_panel.Refresh()
        except Exception:
            pass

    def _panel_bg_colour(self):
        return theme.panel_bg_colour()

    def _apply_frame_theme(self) -> None:
        if not WX_AVAILABLE:
            return

        panel_bg = theme.panel_bg_colour()
        fg = theme.window_text_colour()

        for panel in (
            getattr(self, "main_panel", None),
            getattr(self, "header_panel", None),
            getattr(self, "toolbar_panel", None),
            getattr(self, "notebook", None),
        ):
            if panel is None:
                continue
            try:
                panel.SetBackgroundColour(panel_bg)
                panel.SetForegroundColour(fg)
            except Exception:
                pass

        try:
            if hasattr(self, "status_bar") and self.status_bar is not None:
                self.status_bar.SetBackgroundColour(panel_bg)
                self.status_bar.SetForegroundColour(fg)
        except Exception:
            pass

    def _on_main_panel_paint(self, event) -> None:
        if not WX_AVAILABLE:
            return
        try:
            dc_class = getattr(wx, "AutoBufferedPaintDC", None) or getattr(wx, "PaintDC", None)
            if dc_class is None:
                return
            dc = dc_class(self.main_panel)
            dc.SetBackground(wx.Brush(self._panel_bg_colour()))
            dc.Clear()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _create_header_panel(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        try:
            panel.SetBackgroundColour(self._panel_bg_colour())
            panel.SetForegroundColour(theme.window_text_colour())
        except Exception:
            pass
        sizer = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="VibeCAD")
        font = title.GetFont()
        font.SetPointSize(14)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        try:
            title.SetForegroundColour(theme.window_text_colour())
        except Exception:
            pass
        sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER, 5)

        sizer.Add(wx.StaticLine(panel), 0, wx.EXPAND | wx.TOP, 5)

        panel.SetSizer(sizer)
        return panel

    def _create_toolbar_panel(self, parent) -> wx.Panel:
        panel = wx.Panel(parent)
        try:
            panel.SetBackgroundColour(self._panel_bg_colour())
            panel.SetForegroundColour(theme.window_text_colour())
        except Exception:
            pass
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.clear_btn = wx.Button(panel, label="🗑 Clear")
        self.clear_btn.SetToolTip("Start a new chat session")
        self.clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_clicked)
        sizer.Add(self.clear_btn, 0, wx.ALL, 3)

        self.settings_btn = wx.Button(panel, label="⚙ Settings")
        self.settings_btn.SetToolTip("Configure LLM API key, endpoint, model, etc")
        self.settings_btn.Bind(wx.EVT_BUTTON, self._on_settings_clicked)
        sizer.Add(self.settings_btn, 0, wx.ALL, 3)

        self.benchmark_btn = wx.Button(panel, label="Run Arduino Uno Benchmark")
        self.benchmark_btn.SetToolTip("Run the Arduino Uno R3 v4 workflow benchmark and report where it fails")
        self.benchmark_btn.Bind(wx.EVT_BUTTON, self._on_benchmark_clicked)
        sizer.Add(self.benchmark_btn, 0, wx.ALL, 3)

        sizer.AddStretchSpacer()

        self.dock_btn = wx.Button(panel, label="📌 Dock")
        self.dock_btn.SetToolTip("Dock into KiCad if possible; otherwise toggle always-on-top")
        self.dock_btn.Bind(wx.EVT_BUTTON, self._on_dock_toggle)
        sizer.Add(self.dock_btn, 0, wx.ALL, 3)

        panel.SetSizer(sizer)
        return panel

    def _create_design_tab(self, parent) -> DesignPanel:
        return DesignPanel(
            parent,
            on_send_message=self._on_design_message,
            on_run_benchmark=self._on_run_benchmark,
            on_approve_action=self._on_approve_design_action,
            on_reject_action=self._on_reject_design_action,
            on_suggestion_click=self._on_design_suggestion_click,
            on_llm_controls_changed=self.on_llm_controls_changed,
        )

    def _create_debug_tab(self, parent) -> wx.Panel:
        return DebugPanel(parent, on_get_text=self.on_get_debug_text, on_clear=self.on_clear_debug)

    def _setup_aui(self) -> None:
        try:
            self._mgr.AddPane(
                self.main_panel,
                aui.AuiPaneInfo().Name("main").CenterPane().PaneBorder(False),
            )
            self._mgr.Update()
        except Exception:
            pass

    # ── Theme / lifecycle ────────────────────────────────────────

    def _on_sys_colour_changed(self, event):
        try:
            logger.debug("VibeCADFrame system colour change -> reapplying theme palette")
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.apply_system_theme(rebuild_chat=True)
            self._apply_frame_theme()
        except Exception:
            pass
        try:
            if hasattr(self, "main_panel") and self.main_panel is not None:
                self.main_panel.Refresh()
                self.main_panel.Update()
        except Exception:
            pass
        try:
            self.Refresh()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _on_close(self, event):
        # Single-instance behavior: on user close, hide instead of destroying.
        force_destroy = bool(getattr(self, "_force_destroy", False))
        try:
            if (not force_destroy) and event.CanVeto():
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

        try:
            if callable(getattr(self, "on_before_destroy", None)):
                self.on_before_destroy()
        except Exception:
            logger.exception("Frame pre-destroy callback failed")

        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                shutdown = getattr(self.design_panel, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        except Exception:
            pass

        try:
            if hasattr(self, "debug_panel") and self.debug_panel is not None:
                shutdown = getattr(self.debug_panel, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        except Exception:
            pass

        try:
            if getattr(self, "_mgr", None) is not None:
                self._mgr.UnInit()
        except Exception:
            pass
        event.Skip()

    # ── Toolbar handlers ─────────────────────────────────────────

    def _on_settings_clicked(self, _event):
        if self.on_open_settings:
            try:
                self.on_open_settings()
            except Exception:
                logger.exception("Failed to open settings")

    def _on_clear_clicked(self, _event):
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.clear_history()
        except Exception:
            pass

        if self.on_new_chat:
            try:
                self.on_new_chat()
            except Exception:
                logger.exception("on_new_chat callback failed")

        self.set_status("New chat started", "")

    def _on_benchmark_clicked(self, _event):
        try:
            self._on_run_benchmark()
        except Exception:
            logger.exception("Benchmark launch failed")

    def _on_dock_toggle(self, _event):
        docked = False
        if self.on_toggle_dock:
            try:
                docked = bool(self.on_toggle_dock())
            except Exception:
                logger.exception("Dock toggle callback failed")
                docked = False

        try:
            logger.debug("VibeCADFrame dock toggle requested docked=%s", docked)
        except Exception:
            pass

        if docked:
            try:
                if hasattr(self, "design_panel") and self.design_panel is not None:
                    self.design_panel.set_docked_mode(True)
            except Exception:
                pass
            self.set_status("Docked into KiCad", "")
            return

        # If docking isn't available, toggle always-on-top (pin behavior).
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_docked_mode(False)
        except Exception:
            pass
        if self.on_toggle_dock and not self._docking_warned:
            try:
                wx.MessageBox(
                    "Could not dock into KiCad. Falling back to always-on-top toggle.",
                    "Docking not available",
                    wx.OK | wx.ICON_INFORMATION,
                )
                self._docking_warned = True
            except Exception:
                pass

        self._set_pinned(not self._is_pinned)

    def _set_pinned(self, pinned: bool) -> None:
        self._is_pinned = bool(pinned)
        try:
            style = self.GetWindowStyle()
            if self._is_pinned:
                style |= wx.STAY_ON_TOP
                self.dock_btn.SetLabel("Undock")
                self.SetTitle(self._base_title + " (Pinned)")
                self.set_status("Window pinned (always-on-top)", "")
            else:
                style &= ~wx.STAY_ON_TOP
                self.dock_btn.SetLabel("Dock")
                self.SetTitle(self._base_title)
                self.set_status("Window unpinned", "")
            self.SetWindowStyle(style)
        except Exception:
            pass
        try:
            if hasattr(self, "main_panel") and self.main_panel is not None:
                self.main_panel.Refresh()
                self.main_panel.Update()
        except Exception:
            pass
        try:
            self.Refresh()
            self.Update()
        except Exception:
            pass

    # ── Design tab wiring ────────────────────────────────────────

    def _on_design_message(self, message: str):
        if self.on_design_message:
            try:
                return self.on_design_message(message)
            except Exception as e:
                logger.exception("Design message failed")
                try:
                    self.design_panel.add_response(f"❌ Error: {e}")
                except Exception:
                    pass
        return False

    def _on_approve_design_action(self, action_data):
        if self.on_approve_action:
            try:
                self.on_approve_action(action_data)
            except Exception:
                logger.exception("Action approval failed")

    def _on_reject_design_action(self, action_data):
        if self.on_reject_action:
            try:
                self.on_reject_action(action_data)
            except Exception:
                logger.exception("Action rejection failed")

    def _on_design_suggestion_click(self, _suggestion: str):
        # Intentionally no-op; suggestion chips are handled within DesignPanel.
        return

    def _on_run_benchmark(self):
        if self.on_run_benchmark:
            try:
                self.on_run_benchmark()
            except Exception:
                logger.exception("Benchmark callback failed")

    # ── Public API used by plugin ────────────────────────────────

    def set_status(self, message: str, llm_status: str = "") -> None:
        try:
            self.status_bar.SetStatusText(message or "", 0)
        except Exception:
            pass
        if llm_status:
            try:
                self.status_bar.SetStatusText(llm_status, 1)
            except Exception:
                pass

    def set_llm_status(self, configured: bool) -> None:
        try:
            self.status_bar.SetStatusText("LLM: Ready" if configured else "LLM: Not configured", 1)
        except Exception:
            pass

    def set_verbose_ui(self, verbose: bool) -> None:
        # Verbose toggle was removed from the toolbar; keep API compatibility.
        _ = bool(verbose)

    def set_thinking_output_enabled(self, enabled: bool) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_thinking_output_enabled(bool(enabled))
        except Exception:
            pass

    def set_llm_controls(self, model: str, extended_reasoning: bool) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_llm_controls(model, bool(extended_reasoning))
        except Exception:
            pass

    def add_design_response(self, content: str) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.add_response(content)
        except Exception:
            pass

    def add_design_action_preview(self, action_type: str, description: str, preview_text: str, action_data: Any) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.add_action_preview(action_type, description, preview_text, action_data)
        except Exception:
            pass

    def set_design_suggestions(self, suggestions: List[str]) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_suggestions(list(suggestions or []))
        except Exception:
            pass

    def set_design_thinking(self, thinking: bool = True) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_thinking(bool(thinking))
        except Exception:
            pass

    def set_agent_running(self, running: bool) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_agent_running(bool(running))
        except Exception:
            pass
        try:
            if hasattr(self, "benchmark_btn") and self.benchmark_btn is not None:
                self.benchmark_btn.Enable(not bool(running))
        except Exception:
            pass

    def set_agent_awaiting_input(self, awaiting: bool) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_agent_awaiting_input(bool(awaiting))
        except Exception:
            pass

    def add_thinking_message(self, text: str) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.add_thinking_message(text)
        except Exception:
            pass

    def set_pause_callback(self, cb) -> None:
        try:
            if hasattr(self, "design_panel") and self.design_panel is not None:
                self.design_panel.set_pause_callback(cb)
        except Exception:
            pass


def create_vibecad_frame(parent=None, **callbacks) -> Optional[VibeCADFrame]:
    """Factory function to create a VibeCAD frame."""
    if not WX_AVAILABLE:
        logger.error("Cannot create frame: wxPython not available")
        return None
    return VibeCADFrame(parent, **callbacks)
