"""
Design Panel - Copilot-style interface for design assistance.

This panel provides a conversational interface for design requests,
similar to GitHub Copilot's chat interface.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, Callable, List, Any, Dict
from datetime import datetime

from ..config.settings import (
    DEFAULT_LLM_MODEL,
    LLM_MODEL_CHOICES,
    normalize_llm_model_choice,
)
from . import theme
from .markdown_utils import html_document, markdown_to_html_fragment
from .markdown_utils import render_basic_latex

try:
    import wx
    import wx.html2
    try:
        from wx.lib.wordwrap import wordwrap
    except Exception:  # pragma: no cover
        wordwrap = None
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False
    class wx:
        class Panel:
            pass
    wordwrap = None

logger = logging.getLogger(__name__)


class ChatMessage:
    """A single message in the chat history."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # "user", "assistant", "system"
        self.content = content
        self.timestamp = timestamp or datetime.now()
        
        # For action messages
        self.action_type: Optional[str] = None
        self.action_status: Optional[str] = None  # "pending", "approved", "executed", "failed"
        self.action_data: Optional[Dict] = None


class DesignPanel(wx.Panel if WX_AVAILABLE else object):
    """
    Copilot-style design assistance panel.
    
    Features:
    - Chat-like interface for design requests
    - Action previews with approve/reject buttons
    - Context-aware suggestions
    - History of interactions
    """
    
    def __init__(self, parent,
                 on_send_message: Optional[Callable[[str], object]] = None,
                 on_run_benchmark: Optional[Callable[[], None]] = None,
                 on_approve_action: Optional[Callable[[Any], None]] = None,
                 on_reject_action: Optional[Callable[[Any], None]] = None,
                 on_suggestion_click: Optional[Callable[[str], None]] = None,
                 on_llm_controls_changed: Optional[Callable[[str, bool], None]] = None):
        """
        Initialize the design panel.
        
        Args:
            parent: Parent window
            on_send_message: Callback when user sends a message
            on_approve_action: Callback when user approves an action
            on_reject_action: Callback when user rejects an action
            on_suggestion_click: Callback when user clicks a suggestion chip
        """
        if not WX_AVAILABLE:
            return
        
        super().__init__(parent)
        
        self.on_send_message = on_send_message
        self.on_run_benchmark = on_run_benchmark
        self.on_approve_action = on_approve_action
        self.on_reject_action = on_reject_action
        self.on_suggestion_click = on_suggestion_click
        self.on_llm_controls_changed = on_llm_controls_changed
        
        self._messages: List[ChatMessage] = []
        self._pending_action: Optional[Any] = None
        self._suggestions: List[str] = []
        self._agent_running: bool = False
        self._on_pause_agent: Optional[Callable] = None
        self._on_resume_agent: Optional[Callable] = None

        # Performance/stability guards.
        self._max_rendered_bubbles: int = 160
        self._max_webview_bubbles: int = 1
        self._docked_mode: bool = False
        self._rendered_bubbles: List[ChatMessage] = []

        # User-configurable output toggles
        self._thinking_output_enabled: bool = True

        # Keep only one live thinking/status message to avoid flooding the UI.
        self._live_thinking_msg: Optional[ChatMessage] = None

        # Debounced chat refresh (Layout/FitInside/scroll can be expensive with many messages)
        self._chat_refresh_scheduled: bool = False
        self._chat_refresh_needs_scroll: bool = False
        self._chat_rewrap_scheduled: bool = False
        self._last_chat_client_width: int = -1
        self._last_rewrapped_chat_width: int = -1

        # Use platform-native/default fonts.
        self._chat_font = None
        self._suppress_llm_control_events: bool = False
        self._model_choice = None
        self._extended_reasoning = None
        
        self._create_ui()

        # React to system theme/light-dark changes.
        try:
            self.Bind(wx.EVT_SYS_COLOUR_CHANGED, self._on_sys_colour_changed)
        except Exception:
            pass
            
        # 3. Add Working Animation
        try:
            self._working_timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self._on_working_timer, self._working_timer)
            self._working_dots = 0
        except Exception:
            self._working_timer = None
        try:
            # Reflow existing bubbles when the panel width changes.
            self.Bind(wx.EVT_SIZE, self._on_size_changed)
        except Exception:
            pass
        try:
            self.Bind(wx.EVT_SHOW, self._on_panel_shown)
        except Exception:
            pass
        
        # Add welcome message
        self._add_system_message(
            "👋 Welcome to VibeCAD Design Assistant!\n\n"
            "I can help you with:\n"
            "• Finding and downloading components (symbols & footprints)\n"
            "• Drawing connections between components\n"
            "• Generating BOM exports\n"
            "• Modifying your layout\n\n"
            "Just describe what you want to do, and I'll create a preview for your approval."
        )

        # Ensure initial colors match current system appearance.
        try:
            self.apply_system_theme(rebuild_chat=True)
        except Exception:
            pass
        try:
            wx.CallAfter(self._force_chat_layout, "init")
            wx.CallLater(120, self._force_chat_layout, "init-delayed")
        except Exception:
            pass

    def _on_panel_shown(self, event):
        try:
            if event.IsShown():
                wx.CallAfter(self._force_chat_layout, "panel-shown")
                wx.CallLater(80, self._force_chat_layout, "panel-shown-delayed")
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _force_chat_layout(self, reason: str = "") -> None:
        """Force the same geometry updates a manual resize would trigger."""
        if not WX_AVAILABLE:
            return
        try:
            self.Layout()
            self.chat_scroll.Layout()
            self.chat_scroll.SetVirtualSize(self.chat_sizer.GetMinSize())
            self.chat_scroll.FitInside()
        except Exception:
            pass

        try:
            self._schedule_chat_rewrap(force=True)
        except Exception:
            pass

        try:
            self.chat_scroll.Refresh()
            self.Refresh()
        except Exception:
            pass

    def _on_sys_colour_changed(self, event):
        """Handle OS theme/system color changes."""
        try:
            self.apply_system_theme(rebuild_chat=True)
            self.Refresh()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def apply_system_theme(self, rebuild_chat: bool = False) -> None:
        """Apply the current palette to the panel chrome and visible bubbles.

        Args:
            rebuild_chat: If True, restyle rendered bubbles in place.
        """
        if not WX_AVAILABLE:
            return

        try:
            panel_bg = self._window_bg_colour()
            chat_bg = self._chat_surface_colour()
            fg = self._window_text_colour()
        except Exception:
            return

        try:
            if hasattr(self, 'SetBackgroundColour'):
                self.SetBackgroundColour(panel_bg)
            if hasattr(self, 'SetForegroundColour'):
                self.SetForegroundColour(fg)
            if hasattr(self, 'chat_scroll') and self.chat_scroll is not None:
                self.chat_scroll.SetBackgroundColour(chat_bg)
                self.chat_scroll.SetForegroundColour(fg)
            if hasattr(self, 'input_panel') and self.input_panel is not None:
                self.input_panel.SetBackgroundColour(panel_bg)
                if hasattr(self.input_panel, 'SetForegroundColour'):
                    self.input_panel.SetForegroundColour(fg)
            if hasattr(self, 'input_controls_panel') and self.input_controls_panel is not None:
                self.input_controls_panel.SetBackgroundColour(panel_bg)
                if hasattr(self.input_controls_panel, 'SetForegroundColour'):
                    self.input_controls_panel.SetForegroundColour(fg)
            if hasattr(self, '_model_choice') and self._model_choice is not None:
                self._model_choice.SetBackgroundColour(panel_bg)
                self._model_choice.SetForegroundColour(fg)
            if hasattr(self, '_extended_reasoning') and self._extended_reasoning is not None:
                self._extended_reasoning.SetBackgroundColour(panel_bg)
                if hasattr(self._extended_reasoning, 'SetOwnBackgroundColour'):
                    self._extended_reasoning.SetOwnBackgroundColour(panel_bg)
            if hasattr(self, '_extended_reasoning_label') and self._extended_reasoning_label is not None:
                self._extended_reasoning_label.SetBackgroundColour(panel_bg)
                self._extended_reasoning_label.SetForegroundColour(fg)
        except Exception:
            pass

        try:
            if hasattr(self, 'input_text') and self.input_text is not None:
                self.input_text.SetBackgroundColour(chat_bg)
                self.input_text.SetForegroundColour(fg)
                if hasattr(self.input_text, 'SetHintTextColour'):
                    self.input_text.SetHintTextColour(self._muted_text_colour())
        except Exception:
            pass

        if rebuild_chat:
            try:
                self._rebuild_chat_bubbles()
            except Exception:
                pass

    def set_docked_mode(self, docked: bool) -> None:
        """Switch between normal and docked rendering profiles."""
        if not WX_AVAILABLE:
            return

        docked = bool(docked)
        if docked == getattr(self, "_docked_mode", False):
            return

        self._docked_mode = docked
        self._max_webview_bubbles = 1
        self._max_rendered_bubbles = 60 if docked else 160

        try:
            self._rebuild_chat_bubbles()
        except Exception:
            pass

    def _rebuild_chat_bubbles(self) -> None:
        """Re-render the chat bubbles (used after theme change)."""
        if not WX_AVAILABLE:
            return

        try:
            # Clear existing bubble widgets
            self.chat_sizer.Clear(delete_windows=True)
        except Exception:
            return

        try:
            self._rendered_bubbles = []
        except Exception:
            pass

        messages = list(self._messages)
        if getattr(self, "_docked_mode", False):
            messages = messages[-int(getattr(self, "_max_rendered_bubbles", 100)):]

        # Recreate visible messages.
        for msg in messages:
            try:
                bubble = self._create_message_bubble(msg)
                self.chat_sizer.Add(bubble, 0, wx.EXPAND | wx.ALL, 5)
                try:
                    self._rendered_bubbles.append(msg)
                    msg._bubble_panel = bubble
                except Exception:
                    pass
            except Exception:
                continue

        self._schedule_chat_refresh(scroll_to_bottom=True)

    def _resize_chat_bubbles(self) -> None:
        """Update wrap sizes of existing chat bubbles instead of destroying them to eliminate glitches."""
        if not WX_AVAILABLE:
            return

        changed = False
        try:
            self.chat_scroll.Freeze()
        except Exception:
            pass

        for msg in list(self._messages):
            panel = getattr(msg, '_bubble_panel', None)
            widget = getattr(msg, '_content_widget', None)
            if not panel or not widget:
                continue

            max_wrap_width = self._max_wrap_width_for_role(msg.role)
            raw_text = self._truncate_for_display(msg.content or "")
            wrap_width = max_wrap_width

            if msg.role == "user" and raw_text:
                try:
                    dc = wx.ClientDC(panel)
                    dc.SetFont(panel.GetFont())
                    text_w = dc.GetTextExtent(raw_text)[0] + 24
                    wrap_width = max(80, min(text_w, max_wrap_width))
                except Exception:
                    pass

            try:
                display_text = raw_text
                if callable(wordwrap):
                    dc = wx.ClientDC(panel)
                    dc.SetFont(panel.GetFont())
                    display_text = wordwrap(raw_text, wrap_width, dc)

                try:
                    dc = wx.ClientDC(panel)
                    dc.SetFont(panel.GetFont())
                    _w, line_h = dc.GetTextExtent("Ag")
                except Exception:
                    line_h = 14
                line_count = max(1, display_text.count("\n") + 1)
                needed_h = max(28, int(line_h * line_count + 4))

                if isinstance(widget, wx.html2.WebView):
                    needed_h += (raw_text.count("\n\n")) * 8 + 4
                    try:
                        current_size = widget.GetSize()
                    except Exception:
                        current_size = None
                    if current_size is None or current_size.GetWidth() != wrap_width or current_size.GetHeight() != needed_h:
                        widget.SetMinSize((wrap_width, needed_h))
                        widget.SetSize((wrap_width, needed_h))
                        changed = True
                elif isinstance(widget, wx.TextCtrl):
                    display_text_plain = render_basic_latex(display_text)
                    if widget.GetValue() != display_text_plain:
                        if hasattr(widget, "ChangeValue"):
                            widget.ChangeValue(display_text_plain)
                        else:
                            widget.SetValue(display_text_plain)
                        changed = True

                    try:
                        current_min = widget.GetMinSize()
                    except Exception:
                        current_min = None
                    if current_min is None or current_min.GetWidth() != wrap_width or current_min.GetHeight() != needed_h:
                        widget.SetMinSize((wrap_width, needed_h))
                        widget.SetSize((wrap_width, needed_h))
                        changed = True
                    self._hide_message_scrollbars(widget)
                elif isinstance(widget, wx.StaticText):
                    display_text_plain = render_basic_latex(display_text)
                    if widget.GetLabel() != display_text_plain:
                        widget.SetLabel(display_text_plain)
                        changed = True

                    if not callable(wordwrap):
                        try:
                            widget.Wrap(int(wrap_width))
                        except Exception:
                            pass

                    try:
                        needed_h = self._estimate_message_height(panel, display_text_plain, min_height=28)
                    except Exception:
                        needed_h = max(28, int(widget.GetBestSize().GetHeight()) + 2)

                    try:
                        current_min = widget.GetMinSize()
                    except Exception:
                        current_min = None
                    if current_min is None or current_min.GetWidth() != wrap_width or current_min.GetHeight() != needed_h:
                        widget.SetMinSize((wrap_width, needed_h))
                        widget.SetSize((wrap_width, needed_h))
                        changed = True
            except Exception:
                continue

        if changed:
            try:
                self.chat_scroll.Layout()
                self.chat_scroll.SetVirtualSize(self.chat_sizer.GetMinSize())
                self.chat_scroll.FitInside()
            except Exception:
                pass

        try:
            self.chat_scroll.Thaw()
        except Exception:
            pass

    def _create_ui(self):
        """Create the panel UI."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self._apply_chat_font(self)
        
        # === Chat History Area ===
        self.chat_scroll = wx.ScrolledWindow(self, style=wx.VSCROLL)
        self._apply_chat_font(self.chat_scroll)
        
        # Prevent background vanishing during popups/saves
        try:
            self.SetBackgroundColour(self._window_bg_colour())
        except Exception:
            pass
        self.chat_scroll.SetScrollRate(0, 20)
        self.chat_sizer = wx.BoxSizer(wx.VERTICAL)
        self.chat_scroll.SetSizer(self.chat_sizer)
        try:
            self.chat_scroll.SetDoubleBuffered(True)
            self.SetDoubleBuffered(True)
        except Exception:
            pass
        
        # Set background
        try:
            bg = self._chat_surface_colour()
            self.chat_scroll.SetBackgroundColour(bg)
        except:
            pass
        
        main_sizer.Add(self.chat_scroll, 1, wx.EXPAND | wx.ALL, 5)
        
        # === Input Area ===
        input_panel = self._create_input_area()
        main_sizer.Add(input_panel, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)

    def _build_chat_transcript(self) -> str:
        """Return a plain-text transcript of all chat messages."""
        rows: List[str] = []
        for msg in list(self._messages):
            try:
                ts = msg.timestamp
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.astimezone()
                time_str = ts.strftime("%I:%M:%S %p").lstrip("0")
            except Exception:
                time_str = msg.timestamp.strftime("%I:%M:%S %p").lstrip("0")
            content = getattr(msg, "content", "") or ""
            rows.append(f"[{str(msg.role or '').upper()}] {time_str}\n{content}\n")
        return "\n".join(rows)

    def _show_transcript_selection_dialog(self) -> None:
        """Open a selectable full transcript for cross-message selection."""
        if not WX_AVAILABLE:
            return

        try:
            transcript = self._build_chat_transcript()
        except Exception:
            transcript = ""

        dlg = wx.Dialog(
            self,
            title="VibeCAD Transcript",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        sizer = wx.BoxSizer(wx.VERTICAL)

        text = wx.TextCtrl(
            dlg,
            value=transcript,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
        )
        text.SetMinSize((760, 420))
        sizer.Add(text, 1, wx.EXPAND | wx.ALL, 8)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        btn_row.AddStretchSpacer()
        close_btn = wx.Button(dlg, wx.ID_CLOSE, "Close")
        close_btn.Bind(wx.EVT_BUTTON, lambda _e: dlg.Destroy())
        btn_row.Add(close_btn, 0, wx.ALL, 6)
        sizer.Add(btn_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)

        dlg.SetSizer(sizer)
        dlg.Layout()
        dlg.CentreOnParent()
        dlg.Show()

        try:
            text.SetFocus()
            text.SetSelection(-1, -1)
        except Exception:
            pass

    def _install_copy_shortcuts(self, widget, scope: str = "message") -> None:
        """Install Cmd/Ctrl+A/C/V shortcuts on chat widgets.

        scope:
            - input: standard text-entry behavior
            - message/chat: supports cross-message selection via transcript dialog
        """
        if not WX_AVAILABLE or widget is None:
            return

        def _on_char_hook(evt):
            try:
                if not (evt.CmdDown() or evt.ControlDown()):
                    evt.Skip()
                    return

                key = evt.GetKeyCode()

                if key in (ord('A'), ord('a')):
                    if scope in ("chat", "message"):
                        self._show_transcript_selection_dialog()
                        return
                    if hasattr(widget, 'SetSelection'):
                        try:
                            widget.SetSelection(-1, -1)
                            return
                        except Exception:
                            pass

                if key in (ord('C'), ord('c')):
                    if hasattr(widget, 'Copy'):
                        try:
                            widget.Copy()
                            return
                        except Exception:
                            pass
                    if scope in ("chat", "message"):
                        self._copy_text_to_clipboard(self._build_chat_transcript())
                        return

                if key in (ord('V'), ord('v')):
                    if scope == "input" and hasattr(widget, 'Paste'):
                        try:
                            widget.Paste()
                            return
                        except Exception:
                            pass
                    try:
                        self.input_text.SetFocus()
                        self.input_text.Paste()
                        return
                    except Exception:
                        pass
            except Exception:
                pass
            evt.Skip()

        try:
            widget.Bind(wx.EVT_CHAR_HOOK, _on_char_hook)
        except Exception:
            pass
    
    def _create_suggestions_bar(self) -> wx.Panel:
        """Create the suggestions chip bar."""
        panel = wx.Panel(self)
        try:
            panel.SetBackgroundColour(self._window_bg_colour())
        except Exception:
            pass
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Label
        label = wx.StaticText(panel, label="Try:")
        try:
            label.SetForegroundColour(self._muted_text_colour())
        except:
            pass
        sizer.Add(label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        # Chips container (will be populated dynamically)
        self.chips_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.chips_sizer, 1, wx.EXPAND)
        
        panel.SetSizer(sizer)
        return panel
    
    def _create_input_area(self) -> wx.Panel:
        """Create the message input area."""
        panel = wx.Panel(self)
        self.input_panel = panel
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel_bg = self._window_bg_colour()
        input_bg = self._chat_surface_colour()
        fg = self._window_text_colour()

        try:
            panel.SetBackgroundColour(panel_bg)
            if hasattr(panel, 'SetForegroundColour'):
                panel.SetForegroundColour(fg)
        except Exception:
            pass

        # Keep Enter-to-send behavior but expand composer height now that
        # bottom action buttons are removed.
        self.input_text = wx.TextCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER,
            size=(-1, 96)
        )
        self._apply_chat_font(self.input_text)
        self.input_text.SetMinSize((-1, 96))
        try:
            # Slight rightward visual alignment for typed text.
            self.input_text.SetMargins(18, 10)
        except Exception:
            pass
        self.input_text.SetHint("Describe what you want to do... (e.g., 'design an Arduino UNO')")
        self.input_text.Bind(wx.EVT_TEXT_ENTER, self._on_send_clicked)
        
        # Apply theme colors
        try:
            self.input_text.SetBackgroundColour(input_bg)
            self.input_text.SetForegroundColour(fg)
            if hasattr(self.input_text, 'SetHintTextColour'):
                self.input_text.SetHintTextColour(self._muted_text_colour())
        except:
            pass

        sizer.Add(self.input_text, 0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, 5)

        controls_panel = wx.Panel(panel)
        self.input_controls_panel = controls_panel
        controls_sizer = wx.BoxSizer(wx.VERTICAL)
        try:
            controls_panel.SetBackgroundColour(panel_bg)
            if hasattr(controls_panel, 'SetForegroundColour'):
                controls_panel.SetForegroundColour(fg)
        except Exception:
            pass

        model_row = wx.BoxSizer(wx.HORIZONTAL)
        model_label = wx.StaticText(controls_panel, label="Model")
        try:
            model_label.SetForegroundColour(fg)
        except Exception:
            pass
        self._model_choice = wx.Choice(controls_panel, choices=[label for label, _ in LLM_MODEL_CHOICES])
        self._model_choice.SetToolTip("Switch between Gemini 3 Flash Preview and Gemini 3.1 Pro Preview")
        self._model_choice.SetSelection(0)
        try:
            self._model_choice.SetBackgroundColour(panel_bg)
            self._model_choice.SetForegroundColour(fg)
        except Exception:
            pass
        model_row.Add(model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        model_row.Add(self._model_choice, 1, wx.EXPAND)
        controls_sizer.Add(model_row, 0, wx.EXPAND)

        reasoning_row = wx.BoxSizer(wx.HORIZONTAL)
        reasoning_row.AddSpacer(model_label.GetBestSize().GetWidth() + 8)
        self._extended_reasoning = wx.CheckBox(controls_panel, label="")
        self._extended_reasoning.SetToolTip("Enable the model's extended reasoning budget")
        self._extended_reasoning.SetValue(False)
        try:
            self._extended_reasoning.SetBackgroundColour(panel_bg)
            if hasattr(self._extended_reasoning, 'SetOwnBackgroundColour'):
                self._extended_reasoning.SetOwnBackgroundColour(panel_bg)
        except Exception:
            pass
        reasoning_row.Add(self._extended_reasoning, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP, 6)

        self._extended_reasoning_label = wx.StaticText(controls_panel, label="Extended reasoning (may increase response time)")
        self._extended_reasoning_label.SetToolTip("Enable the model's extended reasoning budget")
        try:
            self._extended_reasoning_label.SetBackgroundColour(panel_bg)
            self._extended_reasoning_label.SetForegroundColour(fg)
        except Exception:
            pass
        reasoning_row.Add(self._extended_reasoning_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.TOP, 8)
        controls_sizer.Add(reasoning_row, 0, wx.EXPAND)

        controls_panel.SetSizer(controls_sizer)
        sizer.Add(controls_panel, 0, wx.EXPAND | wx.ALL, 5)

        try:
            self._model_choice.Bind(wx.EVT_CHOICE, self._on_model_choice_changed)
            self._extended_reasoning.Bind(wx.EVT_CHECKBOX, self._on_extended_reasoning_changed)
        except Exception:
            pass
        
        panel.SetSizer(sizer)
        return panel

    def _model_choice_index_for_value(self, model: str) -> int:
        normalized = normalize_llm_model_choice(model)
        for idx, (_label, value) in enumerate(LLM_MODEL_CHOICES):
            if value == normalized:
                return idx
        return 0

    def _selected_model_value(self) -> str:
        choice = getattr(self, "_model_choice", None)
        if choice is None:
            return DEFAULT_LLM_MODEL
        try:
            idx = choice.GetSelection()
        except Exception:
            idx = -1
        if 0 <= idx < len(LLM_MODEL_CHOICES):
            return LLM_MODEL_CHOICES[idx][1]
        return DEFAULT_LLM_MODEL

    def _notify_llm_controls_changed(self) -> None:
        if self._suppress_llm_control_events:
            return
        callback = getattr(self, "on_llm_controls_changed", None)
        if not callable(callback):
            return
        try:
            callback(self._selected_model_value(), bool(self._extended_reasoning.GetValue() if self._extended_reasoning is not None else False))
        except Exception:
            logger.exception("LLM control callback failed")

    def set_llm_controls(self, model: str, extended_reasoning: bool) -> None:
        """Update the model dropdown and reasoning checkbox without firing callbacks."""
        if not WX_AVAILABLE:
            return
        self._suppress_llm_control_events = True
        try:
            if self._model_choice is not None:
                self._model_choice.SetSelection(self._model_choice_index_for_value(model))
            if self._extended_reasoning is not None:
                self._extended_reasoning.SetValue(bool(extended_reasoning))
        finally:
            self._suppress_llm_control_events = False

    def _on_model_choice_changed(self, event) -> None:
        try:
            self._notify_llm_controls_changed()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _on_extended_reasoning_changed(self, event) -> None:
        try:
            self._notify_llm_controls_changed()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _on_copy_chat_clicked(self, event):
        """Copy the entire chat history to clipboard."""
        if not WX_AVAILABLE:
            return
            
        try:
            full_log = []
            for msg in self._messages:
                # Format: [Role] Time: Content
                try:
                    ts = msg.timestamp
                    if getattr(ts, "tzinfo", None) is not None:
                        ts = ts.astimezone()
                    time_str = ts.strftime("%I:%M:%S %p").lstrip("0")
                except Exception:
                    time_str = msg.timestamp.strftime("%I:%M:%S %p").lstrip("0")
                content = getattr(msg, 'content', '') or ''
                full_log.append(f"[{msg.role.upper()}] {time_str}\n{content}\n")
            
            text_data = wx.TextDataObject("\n".join(full_log))
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(text_data)
                wx.TheClipboard.Close()
                
                # Feedback
                self.input_text.SetHint("Chat log copied to clipboard!")
                def restore_hint():
                    try:
                        self.input_text.SetHint("Describe what you want to do... (e.g., 'design an Arduino UNO')")
                    except: pass
                wx.CallLater(2000, restore_hint)
        except Exception as e:
            logger.error(f"Failed to copy chat log: {e}")

    def _on_send_or_pause_clicked(self, event):
        """Handle send/pause button click."""
        if self._agent_running:
            # Currently running → pause
            self._agent_running = False
            self.set_agent_running(False)
            if self._on_pause_agent:
                try:
                    self._on_pause_agent()
                except Exception as e:
                    logger.exception(f"Pause agent failed: {e}")
            return
        # Not running → treat as send
        self._on_send_clicked(event)

    def _on_benchmark_clicked(self, event):
        if self._agent_running:
            return
        if self.on_run_benchmark:
            try:
                self.on_run_benchmark()
            except Exception as e:
                logger.exception(f"Benchmark launch failed: {e}")
                self._add_error_message(f"Failed to run benchmark: {e}")

    def _on_send_clicked(self, event):
        """Handle send button click or Enter key."""
        message = self.input_text.GetValue().strip()
        if not message:
            return
        
        # Clear input
        self.input_text.SetValue("")
        
        # Add user message to chat
        self._add_user_message(message)
        
        # Show immediate thinking indicator so the user sees something right away.
        # Backend will also toggle thinking as it progresses.
        self.set_thinking(True)

        # Call the callback. It may return a bool indicating whether the
        # autonomous agent loop was started.
        started_agent = False
        if self.on_send_message:
            try:
                result = self.on_send_message(message)
                started_agent = bool(result)
            except Exception as e:
                logger.exception(f"Send message failed: {e}")
                self._add_error_message(f"Failed to process request: {e}")
                self.set_thinking(False)

        # Only enter agent-run mode if the backend actually started/resumed the agent.
        # For simple Q&A, keep the UI in normal chat mode.
        try:
            self.set_agent_running(started_agent)
        except Exception:
            pass
        
        # Do not re-enable input here; the agent loop controls this via
        # set_agent_running / set_agent_awaiting_input.
    
    def _add_message(self, msg: ChatMessage):
        """Add a message to the chat display."""
        self._messages.append(msg)

        # Assign a stable index so we can decide rich vs plain rendering.
        try:
            msg._index = len(self._messages) - 1
        except Exception:
            pass
        
        # Create message bubble
        bubble = self._create_message_bubble(msg)
        self.chat_sizer.Add(bubble, 0, wx.EXPAND | wx.ALL, 5)
        self._animate_message_bubble(bubble)

        # Track rendered widgets so we can prune old ones.
        try:
            self._rendered_bubbles.append(msg)
            msg._bubble_panel = bubble
        except Exception:
            pass

        self._prune_rendered_bubbles()

        self._schedule_chat_refresh(scroll_to_bottom=True)

    def _animate_message_bubble(self, bubble) -> None:
        """Keep bubble chrome stable; message motion is handled in HTML."""
        if not WX_AVAILABLE or bubble is None:
            return
        return

    def _prune_rendered_bubbles(self) -> None:
        """Destroy oldest rendered bubble widgets once we exceed the cap."""
        if not WX_AVAILABLE:
            return

        try:
            while len(self._rendered_bubbles) > self._max_rendered_bubbles:
                oldest = self._rendered_bubbles.pop(0)
                panel = getattr(oldest, '_bubble_panel', None)
                if panel is None:
                    continue
                try:
                    self.chat_sizer.Detach(panel)
                except Exception:
                    pass
                try:
                    panel.Destroy()
                except Exception:
                    pass
        except Exception:
            pass

    def _schedule_chat_refresh(self, scroll_to_bottom: bool = False) -> None:
        """Debounce expensive chat layout/scroll operations."""
        if not WX_AVAILABLE:
            return

        if scroll_to_bottom:
            self._chat_refresh_needs_scroll = True

        if self._chat_refresh_scheduled:
            return

        self._chat_refresh_scheduled = True

        def _do():
            self._chat_refresh_scheduled = False
            try:
                self.chat_scroll.Freeze()
            except Exception:
                pass
            try:
                self.chat_scroll.Layout()
                self.chat_scroll.SetVirtualSize(self.chat_sizer.GetMinSize())
                self.chat_scroll.FitInside()
                if self._chat_refresh_needs_scroll:
                    self._scroll_to_bottom()
            except Exception:
                pass
            finally:
                self._chat_refresh_needs_scroll = False
                try:
                    self.chat_scroll.Thaw()
                except Exception:
                    pass

        try:
            wx.CallLater(50, _do)
        except Exception:
            _do()

    def _on_size_changed(self, event):
        """Re-wrap message bubbles when the chat viewport width changes."""
        try:
            self._schedule_chat_rewrap()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _current_chat_client_width(self) -> int:
        """Return current chat viewport width in pixels."""
        if not WX_AVAILABLE:
            return 0
        try:
            sz = self.chat_scroll.GetClientSize()
            return max(0, int(sz.GetWidth()))
        except Exception:
            return 0

    def _schedule_chat_rewrap(self, force: bool = False) -> None:
        """Continuously re-wrap while resizing, coalesced to ~60fps."""
        if not WX_AVAILABLE:
            return
        width = self._current_chat_client_width()
        if width <= 0:
            return

        self._last_chat_client_width = width

        if self._chat_rewrap_scheduled:
            return
        self._chat_rewrap_scheduled = True

        def _do():
            self._chat_rewrap_scheduled = False
            target_width = self._last_chat_client_width
            if target_width <= 0:
                return

            if not force and target_width == self._last_rewrapped_chat_width:
                return

            try:
                self._resize_chat_bubbles()
                self._last_rewrapped_chat_width = target_width
            except Exception:
                pass

            # If another width arrived while rebuilding, run another pass quickly.
            try:
                if self._last_chat_client_width != self._last_rewrapped_chat_width:
                    self._schedule_chat_rewrap(force=False)
            except Exception:
                pass

        try:
            if force:
                _do()
            else:
                wx.CallLater(32, _do)
        except Exception:
            _do()

    def _max_wrap_width_for_role(self, role: str) -> int:
        """Calculate a responsive max wrap width for a given message role."""
        base = 520 if role != "user" else 420
        client_w = self._current_chat_client_width()
        if client_w <= 0:
            return base

        # Account for chat bubble margins/padding.
        usable = max(140, client_w - 70)
        if role == "user":
            return max(120, min(base, int(usable * 0.78)))
        return max(180, min(680, int(usable * 0.92)))
    
    def _add_user_message(self, content: str):
        """Add a user message."""
        self._add_message(ChatMessage("user", content))
    
    def _add_assistant_message(self, content: str):
        """Add an assistant message."""
        self._add_message(ChatMessage("assistant", content))
    
    def _add_system_message(self, content: str):
        """Add a system message."""
        self._add_message(ChatMessage("system", content))
    
    def _add_error_message(self, content: str):
        """Add an error message."""
        msg = ChatMessage("system", f"❌ {content}")
        self._add_message(msg)

    @staticmethod
    def _truncate_for_display(text: str, *, max_chars: int = 25000, head_chars: int = 18000, tail_chars: int = 4000) -> str:
        """Truncate very large tool/search outputs so the UI stays responsive."""
        s = str(text or "")
        if len(s) <= max_chars:
            return s

        head = max(0, int(head_chars))
        tail = max(0, int(tail_chars))

        # Ensure head+tail leaves room for the marker (avoid edge cases where we
        # accidentally return something larger than max_chars).
        marker = "\n\n...[VibeCAD truncated output in UI: {omitted} chars omitted]...\n\n"
        marker_budget = len(marker.format(omitted=0)) + 16
        if head + tail + marker_budget > max_chars:
            # Prefer keeping the head.
            head = max(0, max_chars - marker_budget - min(tail, 2000))
            tail = min(tail, max(0, max_chars - marker_budget - head))

        omitted = max(0, len(s) - (head + tail))
        if tail <= 0:
            return s[:head] + marker.format(omitted=omitted)
        return s[:head] + marker.format(omitted=omitted) + s[-tail:]
    
    def _create_message_bubble(self, msg: ChatMessage) -> wx.Panel:
        """Create a chat bubble for a message."""
        panel = wx.Panel(self.chat_scroll)
        self._apply_chat_font(panel)
        
        # Message colors track the active appearance via the shared theme.
        base_fg = self._window_text_colour()
        bg_color = self._bubble_colour_for_role(msg.role)
        alignment = wx.ALIGN_RIGHT if msg.role == "user" else wx.ALIGN_LEFT
        
        panel.SetBackgroundColour(bg_color)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header with timestamp (local AM/PM)
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)

        try:
            ts = msg.timestamp
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.astimezone()
            time_str = ts.strftime("%I:%M %p").lstrip("0")
        except Exception:
            time_str = msg.timestamp.strftime("%I:%M %p").lstrip("0")

        time_text = wx.StaticText(panel, label=time_str)
        self._apply_chat_font(time_text)
        time_text.SetForegroundColour(self._muted_text_colour())
        try:
            time_text.SetBackgroundColour(bg_color)
        except Exception:
            pass
        header_sizer.Add(time_text, 0, wx.ALL, 3)
        
        sizer.Add(header_sizer, 0, alignment)

        # Message content:
        # - WebView-based Markdown rendering looks nice but is very heavy.
        # - Limit it to only the newest messages to avoid crashes/lag.
        max_wrap_width = self._max_wrap_width_for_role(msg.role)
        raw_text = self._truncate_for_display(msg.content or "")
        is_working_indicator = bool(str(raw_text).startswith("⏳ Working"))
        is_welcome_message = bool(msg.role == "system" and "Welcome to VibeCAD" in raw_text)

        wrap_width = max_wrap_width

        if msg.role == "user" and raw_text:
            try:
                dc = wx.ClientDC(panel)
                dc.SetFont(panel.GetFont())
                text_w = dc.GetTextExtent(raw_text)[0] + 24
                wrap_width = max(80, min(text_w, max_wrap_width))
            except Exception:
                pass

        try:
            msg_index = getattr(msg, '_index', None)
            if msg_index is None:
                msg_index = len(self._messages) - 1
            newest_rank = (len(self._messages) - 1) - int(msg_index)
        except Exception:
            newest_rank = 0

        # WebView backgrounds are still unreliable on macOS, so use the native
        # render path there. Other platforms can keep the richer HTML bubble.
        use_webview = bool(
            sys.platform != "darwin"
            and newest_rank < getattr(self, "_max_webview_bubbles", 1)
            and not is_welcome_message
        )

        # Estimate height from wrapped plain-text line count (WebView does not
        # reliably provide content height across platforms).
        display_text = raw_text
        try:
            if callable(wordwrap):
                dc = wx.ClientDC(panel)
                dc.SetFont(panel.GetFont())
                display_text = wordwrap(raw_text, wrap_width, dc)
        except Exception:
            display_text = raw_text

        min_height = 28
        try:
            dc = wx.ClientDC(panel)
            dc.SetFont(panel.GetFont())
            _w, line_h = dc.GetTextExtent("Ag")
            line_count = max(1, (display_text.count("\n") + 1))
            # Reduce base padding significantly now that bottom margin is handled by removing
            # wx.ALL and HTML p:last-child margin-bottom: 0
            needed_h = max(min_height, int(line_h * line_count + 4))
            if use_webview:
                # Add space for HTML paragraph margins (8px each matching the CSS)
                needed_h += (raw_text.count("\n\n")) * 8
                # Minimal extra buffer since body margin is 0
                needed_h += 4
        except Exception:
            needed_h = min_height

        content_widget = None
        if use_webview:
            try:
                web = wx.html2.WebView.New(panel, size=(wrap_width, needed_h))
                try:
                    web.SetBackgroundColour(bg_color)
                except Exception:
                    pass
                frag = markdown_to_html_fragment(raw_text)
                doc = html_document(
                    frag,
                    bg_hex=self._colour_to_hex(bg_color),
                    fg_hex=self._colour_to_hex(base_fg),
                    border_hex=self._colour_to_hex(self._blend_colours(base_fg, bg_color, 0.65)),
                    text_align="right" if msg.role == "user" else "left",
                    animate=not is_working_indicator,
                    color_scheme=theme.html_color_scheme(),
                )
                web.SetPage(doc, "")
                content_widget = web
            except Exception:
                content_widget = None

        if content_widget is None:
            # Fallback path for older messages or if WebView creation fails.
            display_text_plain = render_basic_latex(display_text if callable(wordwrap) or is_working_indicator else raw_text)

            content_widget = wx.StaticText(panel, label=display_text_plain)
            try:
                content_widget.SetForegroundColour(base_fg)
                content_widget.SetBackgroundColour(bg_color)
                if hasattr(content_widget, "SetOwnBackgroundColour"):
                    content_widget.SetOwnBackgroundColour(bg_color)
            except Exception:
                pass
            if not callable(wordwrap):
                try:
                    content_widget.Wrap(int(wrap_width))
                except Exception:
                    pass
            try:
                needed_h = self._estimate_message_height(panel, display_text_plain, min_height=min_height)
            except Exception:
                needed_h = max(min_height, int(content_widget.GetBestSize().GetHeight()) + 2)

        self._apply_chat_font(content_widget)

        if isinstance(content_widget, wx.html2.WebView) or isinstance(content_widget, wx.TextCtrl) or isinstance(content_widget, wx.StaticText):
            try:
                content_widget.SetMinSize((wrap_width, needed_h))
            except Exception:
                pass
        self._hide_message_scrollbars(content_widget)
        try:
            msg._content_widget = content_widget
        except Exception: pass

        # Keep the message body sized to its content so resize reflow stays
        # cheap and the bubble does not stretch across the whole pane.
        content_flags = wx.TOP | wx.LEFT | wx.RIGHT | alignment
        if msg.role == "user":
            content_flags = wx.TOP | wx.LEFT | wx.RIGHT | wx.ALIGN_RIGHT
        sizer.Add(content_widget, 0, content_flags, 8)

        copy_row = wx.BoxSizer(wx.HORIZONTAL)
        copy_row.AddStretchSpacer()
        copy_link = wx.StaticText(panel, label="Copy")
        self._apply_chat_font(copy_link)
        copy_link.SetToolTip("Copy this message")
        try:
            copy_link.SetForegroundColour(self._muted_text_colour())
            copy_link.SetBackgroundColour(bg_color)
        except Exception:
            pass
        try:
            copy_link.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        except Exception:
            pass
        copy_link.Bind(wx.EVT_LEFT_UP, lambda e, m=msg: self._on_copy_message_clicked(e, m))
        # Remove top margin to pull copy up tightly
        copy_row.Add(copy_link, 0, wx.RIGHT, 4)
        sizer.Add(copy_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        try:
            panel._bubble_target_bg = bg_color
            panel._bubble_content_widget = content_widget
            panel._bubble_chrome_widgets = [time_text, copy_link]
        except Exception:
            pass

        try:
            self._set_bubble_bg(panel, bg_color)
        except Exception:
            pass
        
        # Action buttons if this is an action message
        if msg.action_type and msg.action_status == "pending":
            btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            reject_btn = wx.Button(panel, label="✕ Reject")
            reject_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_reject_action(msg))
            btn_sizer.Add(reject_btn, 0, wx.ALL, 3)
            
            approve_btn = wx.Button(panel, label="✓ Apply")
            # Use native button styling (no custom background) to avoid
            # rendering artefacts (green rectangle) on macOS / GTK.
            approve_btn.Bind(wx.EVT_BUTTON, lambda e: self._on_approve_action(msg))
            btn_sizer.Add(approve_btn, 0, wx.ALL, 3)

            # Keep references so we can disable/update labels without rebuilding.
            try:
                msg._approve_btn = approve_btn
                msg._reject_btn = reject_btn
            except Exception:
                pass
            
            sizer.Add(btn_sizer, 0, alignment | wx.ALL, 5)
        
        panel.SetSizer(sizer)
        return panel

    def _copy_text_to_clipboard(self, text: str) -> bool:
        """Copy plain text to the system clipboard."""
        if not WX_AVAILABLE:
            return False
        try:
            text_data = wx.TextDataObject(str(text or ""))
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(text_data)
                wx.TheClipboard.Close()
                return True
        except Exception:
            pass
        return False

    def _on_copy_message_clicked(self, _event, msg: ChatMessage):
        """Copy a single message body to clipboard."""
        try:
            self._copy_text_to_clipboard(getattr(msg, "content", "") or "")
        except Exception:
            pass
    
    def _on_approve_action(self, msg: ChatMessage):
        """Handle approve action button."""
        if self.on_approve_action and msg.action_data:
            try:
                self.on_approve_action(msg.action_data)
                # In batched approval rounds, Apply selects the action; the
                # agent proceeds only after all actions in the round are decided.
                msg.action_status = "approved"
                try:
                    if hasattr(msg, '_approve_btn'):
                        msg._approve_btn.SetLabel("✓ Selected")
                        msg._approve_btn.Enable(False)
                    if hasattr(msg, '_reject_btn'):
                        msg._reject_btn.Enable(False)
                except Exception:
                    pass
            except Exception as e:
                msg.action_status = "failed"
                self._add_error_message(f"Action failed: {e}")
    
    def _on_reject_action(self, msg: ChatMessage):
        """Handle reject action button."""
        if self.on_reject_action and msg.action_data:
            try:
                self.on_reject_action(msg.action_data)
            except:
                pass
        msg.action_status = "rejected"
        try:
            if hasattr(msg, '_reject_btn'):
                msg._reject_btn.SetLabel("✕ Skipped")
                msg._reject_btn.Enable(False)
            if hasattr(msg, '_approve_btn'):
                msg._approve_btn.Enable(False)
        except Exception:
            pass
    
    def _scroll_to_bottom(self):
        """Smooth-scroll chat to the bottom."""
        try:
            _vx, vy = self.chat_scroll.GetVirtualSize()
            _sx, cur_units = self.chat_scroll.GetViewStart()
            _ux, py_per_unit = self.chat_scroll.GetScrollPixelsPerUnit()
            if int(py_per_unit or 0) <= 0:
                self.chat_scroll.Scroll(0, vy)
                return

            target_units = int(vy / py_per_unit)
            steps = 5
            delta = max(0, target_units - int(cur_units))
            if delta <= 0:
                self.chat_scroll.Scroll(0, target_units)
                return

            for i in range(1, steps + 1):
                y_step = int(cur_units + (delta * i / steps))
                wx.CallLater(i * 16, self.chat_scroll.Scroll, 0, y_step)
        except:
            pass
    
    def set_suggestions(self, suggestions: List[str]):
        """Update the suggestion chips."""
        self._suggestions = list(suggestions or [])

        # Suggestions bar is hidden from the current UI layout.
        if not hasattr(self, "chips_sizer") or self.chips_sizer is None:
            return

        # Clear existing chips
        self.chips_sizer.Clear(delete_windows=True)

        # Add new chips
        for suggestion in self._suggestions[:4]:  # Max 4 chips
            chip = self._create_chip(suggestion)
            self.chips_sizer.Add(chip, 0, wx.ALL, 2)

        try:
            self.suggestions_panel.Layout()
        except Exception:
            pass
    
    def _create_chip(self, text: str) -> wx.Button:
        """Create a suggestion chip button."""
        btn = wx.Button(self.suggestions_panel, label=text, size=(-1, 24))
        btn.SetFont(btn.GetFont().Smaller())
        btn.Bind(wx.EVT_BUTTON, lambda e, t=text: self._on_chip_clicked(t))
        
        # Style
        try:
            btn.SetBackgroundColour(theme.chip_colour())
            btn.SetForegroundColour(self._window_text_colour())
        except:
            pass
        
        return btn
    
    def _on_chip_clicked(self, text: str):
        """Handle suggestion chip click."""
        # Put the text in the input
        self.input_text.SetValue(text)
        self.input_text.SetFocus()
        
        if self.on_suggestion_click:
            try:
                self.on_suggestion_click(text)
            except:
                pass
    
    def set_agent_running(self, running: bool):
        """Enable or disable composer editing while the agent is running."""
        self._agent_running = running
        try:
            self._set_input_editable(not running)
            if not running:
                self.input_text.SetFocus()
        except Exception:
            pass

    def set_agent_awaiting_input(self, awaiting: bool):
        """Toggle composer editability when the agent asks a clarifying question."""
        try:
            self._set_input_editable(awaiting)
            if awaiting:
                self.input_text.SetFocus()
                self.input_text.SetHint("Answer the agent's question...")
            else:
                self.input_text.SetHint("Describe what you want to do... (e.g., 'design an Arduino UNO')")
        except Exception:
            pass

    def _set_input_editable(self, editable: bool) -> None:
        """Keep the composer enabled while toggling editability."""
        if not WX_AVAILABLE:
            return
        try:
            if hasattr(self.input_text, "Enable"):
                self.input_text.Enable(True)
            if hasattr(self.input_text, "SetEditable"):
                self.input_text.SetEditable(bool(editable))
            elif hasattr(self.input_text, "Enable"):
                self.input_text.Enable(bool(editable))
        except Exception:
            pass

    def set_pause_callback(self, cb):
        """Set callback for pause button."""
        self._on_pause_agent = cb

    def set_resume_callback(self, cb):
        """Set callback for resume."""
        self._on_resume_agent = cb

    def add_thinking_message(self, text: str):
        """Add an ephemeral thinking/status message."""
        if not getattr(self, '_thinking_output_enabled', True):
            return
        # Keep only one live thinking message to avoid UI flooding.
        try:
            if self._live_thinking_msg is not None:
                self._remove_message_widget(self._live_thinking_msg)
                try:
                    self._messages.remove(self._live_thinking_msg)
                except Exception:
                    pass
        except Exception:
            pass

        msg = ChatMessage("system", f"💭 {text}")
        self._live_thinking_msg = msg
        self._add_message(msg)

    def set_thinking_output_enabled(self, enabled: bool) -> None:
        """Enable/disable thinking/status message output."""
        self._thinking_output_enabled = bool(enabled)

    def _remove_message_widget(self, msg: ChatMessage) -> None:
        """Remove and destroy a bubble widget for a given message, if rendered."""
        if not WX_AVAILABLE:
            return
        panel = getattr(msg, '_bubble_panel', None)
        if panel is None:
            return
        try:
            self.chat_sizer.Detach(panel)
        except Exception:
            pass
        try:
            panel.Destroy()
        except Exception:
            pass
        try:
            if msg in self._rendered_bubbles:
                self._rendered_bubbles.remove(msg)
        except Exception:
            pass

    def add_action_preview(self, action_type: str, description: str, preview_text: str, action_data: Any):
        """Add an action preview message that requires user approval."""
        content = f"**Proposed Action: {action_type}**\n\n{description}\n\n{preview_text}"
        
        msg = ChatMessage("assistant", content)
        msg.action_type = action_type
        msg.action_status = "pending"
        msg.action_data = action_data
        
        self._add_message(msg)
    
    def add_response(self, content: str):
        """Add a simple response message from the assistant."""
        self._add_assistant_message(content)
    
    def _on_working_timer(self, event):
        """Animate the 'Working' message."""
        if not WX_AVAILABLE: return
        if not getattr(self, '_thinking_msg', None): return
        
        widget = getattr(self._thinking_msg, '_content_widget', None)
        if not widget: return
        try:
            self._working_dots = (getattr(self, '_working_dots', 0) + 1) % 4
            dots = "." * self._working_dots
            spaces = " " * (3 - self._working_dots)
            new_text = f"⏳ Working{dots}{spaces}"
            if isinstance(widget, wx.html2.WebView):
                try:
                    panel = getattr(self._thinking_msg, '_bubble_panel', None)
                    bg_color = getattr(panel, '_bubble_target_bg', self._window_bg_colour())
                    doc = html_document(
                        markdown_to_html_fragment(new_text),
                        bg_hex=self._colour_to_hex(bg_color),
                        fg_hex=self._colour_to_hex(self._window_text_colour()),
                        border_hex=self._colour_to_hex(self._blend_colours(self._window_text_colour(), bg_color, 0.65)),
                        text_align="left",
                        animate=False,
                        color_scheme=theme.html_color_scheme(),
                    )
                    widget.SetPage(doc, "")
                except Exception:
                    pass
            elif isinstance(widget, wx.TextCtrl):
                if widget.GetValue() != new_text:
                    if hasattr(widget, "ChangeValue"):
                        widget.ChangeValue(new_text)
                    else:
                        widget.SetValue(new_text)
                    try:
                        widget.SetSelection(0, 0)
                    except Exception:
                        pass
            elif isinstance(widget, wx.StaticText):
                if widget.GetLabel() != new_text:
                    widget.SetLabel(new_text)
                    try:
                        widget.Wrap(int(self._max_wrap_width_for_role("system")))
                    except Exception:
                        pass
        except Exception: pass

    def set_thinking(self, thinking: bool = True):
        """Show/hide thinking indicator and start/stop animations."""
        if thinking:
            if getattr(self, '_thinking_msg', None) is None:
                self._thinking_msg = ChatMessage("system", "⏳ Working...")
                self._add_message(self._thinking_msg)
            try:
                if getattr(self, '_working_timer', None) and not self._working_timer.IsRunning():
                    self._working_timer.Start(400) # update every 400ms
            except Exception: pass
        else:
            try:
                if getattr(self, '_working_timer', None) and self._working_timer.IsRunning():
                    self._working_timer.Stop()
            except Exception: pass
            
            if hasattr(self, '_thinking_msg') and self._thinking_msg in self._messages:
                try:
                    self._remove_message_widget(self._thinking_msg)
                except Exception:
                    pass
                try:
                    self._messages.remove(self._thinking_msg)
                except Exception:
                    pass
            self._thinking_msg = None

            try:
                if self._live_thinking_msg is not None:
                    self._remove_message_widget(self._live_thinking_msg)
                    try:
                        self._messages.remove(self._live_thinking_msg)
                    except Exception:
                        pass
                    self._live_thinking_msg = None
            except Exception:
                pass

    def clear_history(self):
        """Clear the chat history."""
        self._messages.clear()
        self.chat_sizer.Clear(delete_windows=True)
        self.chat_scroll.Layout()

    def _resolve_chat_font(self):
        """Return None so controls use native/default fonts."""
        return None

    def _apply_chat_font(self, widget) -> None:
        """No-op: keep platform default widget fonts."""
        return

    def _estimate_message_height(self, panel, text: str, min_height: int = 28) -> int:
        """Estimate required bubble content height from line count and font metrics."""
        try:
            dc = wx.ClientDC(panel)
            dc.SetFont(panel.GetFont())
            _w, line_h = dc.GetTextExtent("Ag")
            line_count = max(1, str(text or "").count("\n") + 1)
            return max(min_height, int(line_h * line_count + 4))
        except Exception:
            return min_height

    def _hide_message_scrollbars(self, widget) -> None:
        """Best-effort hide of inner message scrollbars."""
        if widget is None:
            return
        try:
            if hasattr(widget, "ShowScrollbars") and hasattr(wx, "SHOW_SB_NEVER"):
                widget.ShowScrollbars(wx.SHOW_SB_NEVER, wx.SHOW_SB_NEVER)
        except Exception:
            pass

    def _set_bubble_bg(self, bubble, colour) -> None:
        """Apply a background color to a bubble and its child controls."""
        if bubble is None:
            return

        try:
            if hasattr(bubble, "SetBackgroundColour"):
                bubble.SetBackgroundColour(colour)
            if hasattr(bubble, "SetOwnBackgroundColour"):
                bubble.SetOwnBackgroundColour(colour)
        except Exception:
            pass

        try:
            if hasattr(bubble, "Refresh"):
                bubble.Refresh()
        except Exception:
            pass

        try:
            content_widget = getattr(bubble, "_bubble_content_widget", None)
            if content_widget is not None:
                if hasattr(content_widget, "SetBackgroundColour"):
                    content_widget.SetBackgroundColour(colour)
                if hasattr(content_widget, "SetOwnBackgroundColour"):
                    content_widget.SetOwnBackgroundColour(colour)
                if hasattr(content_widget, "Refresh"):
                    content_widget.Refresh()
        except Exception:
            pass

        try:
            for widget in list(getattr(bubble, "_bubble_chrome_widgets", [])):
                if widget is None:
                    continue
                if hasattr(widget, "SetBackgroundColour"):
                    widget.SetBackgroundColour(colour)
                if hasattr(widget, "SetOwnBackgroundColour"):
                    widget.SetOwnBackgroundColour(colour)
                if hasattr(widget, "Refresh"):
                    widget.Refresh()
        except Exception:
            pass

    def _refresh_existing_bubble_backgrounds(self) -> None:
        """Re-apply the current bubble palette without destroying widgets."""
        if not WX_AVAILABLE:
            return
        try:
            for msg in list(self._messages):
                panel = getattr(msg, "_bubble_panel", None)
                if panel is None:
                    continue
                try:
                    self._set_bubble_bg(panel, self._bubble_colour_for_role(msg.role))
                except Exception:
                    pass
        except Exception:
            pass
    
    # === Theme helpers ===
    
    def _window_bg_colour(self):
        return theme.panel_bg_colour()
    
    def _window_text_colour(self):
        return theme.window_text_colour()
    
    def _muted_text_colour(self):
        return theme.muted_text_colour()
    
    def _is_dark_mode(self) -> bool:
        return theme.is_dark_mode()

    def _chat_surface_colour(self):
        return theme.chat_surface_colour()

    def _bubble_colour_for_role(self, role: str):
        return theme.bubble_colour_for_role(role)
    
    def _blend_colours(self, a, b, t: float):
        try:
            t = max(0.0, min(1.0, float(t)))
            ar, ag, ab = int(a.Red()), int(a.Green()), int(a.Blue())
            br, bg, bb = int(b.Red()), int(b.Green()), int(b.Blue())
            r = int(round(ar + (br - ar) * t))
            g = int(round(ag + (bg - ag) * t))
            b2 = int(round(ab + (bb - ab) * t))
            return wx.Colour(r, g, b2)
        except:
            return a

    @staticmethod
    def _colour_to_hex(c) -> str:
        try:
            return f"#{int(c.Red()):02x}{int(c.Green()):02x}{int(c.Blue()):02x}"
        except Exception:
            return "#000000"
