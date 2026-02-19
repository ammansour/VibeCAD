"""
Design Panel - Copilot-style interface for design assistance.

This panel provides a conversational interface for design requests,
similar to GitHub Copilot's chat interface.
"""

from __future__ import annotations

import logging
from typing import Optional, Callable, List, Any, Dict
from datetime import datetime

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
                 on_approve_action: Optional[Callable[[Any], None]] = None,
                 on_reject_action: Optional[Callable[[Any], None]] = None,
                 on_suggestion_click: Optional[Callable[[str], None]] = None):
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
        self.on_approve_action = on_approve_action
        self.on_reject_action = on_reject_action
        self.on_suggestion_click = on_suggestion_click
        
        self._messages: List[ChatMessage] = []
        self._pending_action: Optional[Any] = None
        self._suggestions: List[str] = []
        self._agent_running: bool = False
        self._on_pause_agent: Optional[Callable] = None
        self._on_resume_agent: Optional[Callable] = None

        # Performance/stability guards. Lots of bubbles (especially WebViews)
        # can make KiCad sluggish or crash on some platforms.
        self._max_rendered_bubbles: int = 160
        self._max_webview_bubbles: int = 18  # only for the newest N messages
        self._rendered_bubbles: List[ChatMessage] = []

        # User-configurable output toggles
        self._thinking_output_enabled: bool = True

        # Keep only one live thinking/status message to avoid flooding the UI.
        self._live_thinking_msg: Optional[ChatMessage] = None

        # Debounced chat refresh (Layout/FitInside/scroll can be expensive with many messages)
        self._chat_refresh_scheduled: bool = False
        self._chat_refresh_needs_scroll: bool = False
        
        self._create_ui()

        # React to system theme/light-dark changes.
        try:
            self.Bind(wx.EVT_SYS_COLOUR_CHANGED, self._on_sys_colour_changed)
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
            self.apply_system_theme()
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
        """Apply system window colors to the input and chat background.

        Args:
            rebuild_chat: If True, rebuild existing chat bubbles so their colors
                match the new theme.
        """
        if not WX_AVAILABLE:
            return

        try:
            bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            fg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        except Exception:
            return

        try:
            if hasattr(self, 'chat_scroll') and self.chat_scroll is not None:
                self.chat_scroll.SetBackgroundColour(bg)
        except Exception:
            pass

        try:
            if hasattr(self, 'input_text') and self.input_text is not None:
                self.input_text.SetBackgroundColour(bg)
                self.input_text.SetForegroundColour(fg)
        except Exception:
            pass

        if rebuild_chat:
            self._rebuild_chat_bubbles()

    def _rebuild_chat_bubbles(self) -> None:
        """Re-render the chat bubbles (used after theme change)."""
        if not WX_AVAILABLE:
            return

        try:
            # Clear existing bubble widgets
            self.chat_sizer.Clear(delete_windows=True)
        except Exception:
            return

        # Recreate all messages
        for msg in list(self._messages):
            try:
                bubble = self._create_message_bubble(msg)
                self.chat_sizer.Add(bubble, 0, wx.EXPAND | wx.ALL, 5)
            except Exception:
                continue

        self._schedule_chat_refresh(scroll_to_bottom=True)
    
    def _create_ui(self):
        """Create the panel UI."""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # === Chat History Area ===
        self.chat_scroll = wx.ScrolledWindow(self, style=wx.VSCROLL)
        self.chat_scroll.SetScrollRate(0, 20)
        self.chat_sizer = wx.BoxSizer(wx.VERTICAL)
        self.chat_scroll.SetSizer(self.chat_sizer)
        
        # Set background
        try:
            bg = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            self.chat_scroll.SetBackgroundColour(bg)
        except:
            pass
        
        main_sizer.Add(self.chat_scroll, 1, wx.EXPAND | wx.ALL, 5)
        
        # === Suggestions Bar ===
        self.suggestions_panel = self._create_suggestions_bar()
        main_sizer.Add(self.suggestions_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        
        # === Input Area ===
        input_panel = self._create_input_area()
        main_sizer.Add(input_panel, 0, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)

    def _install_copy_shortcuts(self, widget) -> None:
        """Ensure Cmd/Ctrl+C copies selected text from chat bubble widgets."""
        if not WX_AVAILABLE or widget is None:
            return

        def _on_char_hook(evt):
            try:
                key = evt.GetKeyCode()
                is_copy = (evt.CmdDown() or evt.ControlDown()) and key in (ord('C'), ord('c'))
                if is_copy and hasattr(widget, 'Copy'):
                    try:
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
            pass
    
    def _create_suggestions_bar(self) -> wx.Panel:
        """Create the suggestions chip bar."""
        panel = wx.Panel(self)
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
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Text input - make this single-line and behave like the Results question box
        # so typing does not accidentally trigger actions. Use TE_PROCESS_ENTER
        # and bind EVT_TEXT_ENTER so Enter sends the message explicitly.
        self.input_text = wx.TextCtrl(
            panel,
            style=wx.TE_PROCESS_ENTER,
            size=(-1, -1)
        )
        self.input_text.SetHint("Describe what you want to do... (e.g., 'design an Arduino UNO')")
        self.input_text.Bind(wx.EVT_TEXT_ENTER, self._on_send_clicked)
        
        # Apply theme colors
        try:
            self.input_text.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            self.input_text.SetForegroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT))
        except:
            pass
        
        sizer.Add(self.input_text, 1, wx.EXPAND | wx.ALL, 5)
        
        # Send / Pause button (toggles during agent execution)
        self.send_btn = wx.Button(panel, label="Send")
        self.send_btn.Bind(wx.EVT_BUTTON, self._on_send_or_pause_clicked)
        sizer.Add(self.send_btn, 0, wx.ALL | wx.ALIGN_BOTTOM, 5)
        
        # Copy log button
        self.copy_btn = wx.Button(panel, label="📋")
        self.copy_btn.SetToolTip("Copy entire chat log to clipboard")
        self.copy_btn.Bind(wx.EVT_BUTTON, self._on_copy_chat_clicked)
        sizer.Add(self.copy_btn, 0, wx.ALL | wx.ALIGN_BOTTOM, 5)
        
        panel.SetSizer(sizer)
        return panel

    def _on_copy_chat_clicked(self, event):
        """Copy the entire chat history to clipboard."""
        if not WX_AVAILABLE:
            return
            
        try:
            full_log = []
            for msg in self._messages:
                # Format: [Role] Time: Content
                role_icon = "👤" if msg.role == "user" else "🤖" if msg.role == "assistant" else "ℹ️"
                time_str = msg.timestamp.strftime("%H:%M:%S")
                content = getattr(msg, 'content', '') or ''
                full_log.append(f"{role_icon} [{msg.role.upper()}] {time_str}\n{content}\n")
            
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
            self.send_btn.Enable(True)
            self.input_text.Enable(not started_agent)
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

        # Track rendered widgets so we can prune old ones.
        try:
            self._rendered_bubbles.append(msg)
            msg._bubble_panel = bubble
        except Exception:
            pass

        self._prune_rendered_bubbles()

        self._schedule_chat_refresh(scroll_to_bottom=True)

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
    
    def _create_message_bubble(self, msg: ChatMessage) -> wx.Panel:
        """Create a chat bubble for a message."""
        panel = wx.Panel(self.chat_scroll)
        
        # Determine styling based on role
        is_dark = self._is_dark_mode()
        base_bg = self._window_bg_colour()
        base_fg = self._window_text_colour()
        
        if msg.role == "user":
            # User messages: right-aligned, accent color
            accent = wx.Colour(50, 130, 220)
            if is_dark:
                bg_color = self._blend_colours(base_bg, accent, 0.3)
            else:
                bg_color = wx.Colour(220, 240, 255)
            alignment = wx.ALIGN_RIGHT
            icon = "👤"
        elif msg.role == "assistant":
            # Assistant messages: left-aligned, subtle bg
            if is_dark:
                bg_color = self._blend_colours(base_bg, wx.Colour(100, 100, 100), 0.15)
            else:
                bg_color = wx.Colour(245, 245, 245)
            alignment = wx.ALIGN_LEFT
            icon = "🤖"
        else:
            # System messages: centered, info styling
            if is_dark:
                bg_color = self._blend_colours(base_bg, wx.Colour(80, 80, 120), 0.2)
            else:
                bg_color = wx.Colour(240, 245, 255)
            alignment = wx.ALIGN_LEFT
            icon = "ℹ️"
        
        panel.SetBackgroundColour(bg_color)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Header with icon and timestamp
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        icon_text = wx.StaticText(panel, label=icon)
        header_sizer.Add(icon_text, 0, wx.ALL, 3)
        
        time_str = msg.timestamp.strftime("%H:%M")
        time_text = wx.StaticText(panel, label=time_str)
        time_text.SetForegroundColour(self._muted_text_colour())
        header_sizer.Add(time_text, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)
        
        sizer.Add(header_sizer, 0, alignment)

        # Message content:
        # - WebView-based Markdown rendering looks nice but is very heavy.
        # - Limit it to only the newest messages to avoid crashes/lag.
        max_wrap_width = 520 if msg.role != "user" else 420
        raw_text = msg.content or ""

        # For user messages, shrink the bubble to fit the text instead of
        # using the full max width.  This keeps short one-liners compact.
        wrap_width = max_wrap_width
        if msg.role == "user" and raw_text:
            try:
                dc = wx.ClientDC(panel)
                dc.SetFont(panel.GetFont())
                _tw, _th = dc.GetTextExtent(raw_text)
                # Add horizontal padding (icon + timestamp header + inner margin)
                text_w = _tw + 24
                # Clamp between a reasonable minimum and the max bubble width
                wrap_width = max(80, min(text_w, max_wrap_width))
            except Exception:
                pass

        # Determine whether this message should use a WebView.
        try:
            msg_index = getattr(msg, '_index', None)
            if msg_index is None:
                msg_index = len(self._messages) - 1
            newest_rank = (len(self._messages) - 1) - int(msg_index)
        except Exception:
            newest_rank = 0

        use_webview = bool(newest_rank <= self._max_webview_bubbles)

        # Estimate height from wrapped plain-text line count (WebView does not
        # reliably provide content height across platforms).
        display_text = raw_text
        try:
            if wordwrap is not None:
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
            needed_h = max(min_height, int(line_h * line_count + 18))
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
                )
                web.SetPage(doc, "")
                content_widget = web
            except Exception:
                content_widget = None

        if content_widget is None:
            # Plain rendering path (much lighter than WebView).
            try:
                display_text_plain = render_basic_latex(display_text)
                # Removed wx.StaticText usage to ensure all messages are selectable/copyable.
                # Always use wx.TextCtrl for text content.
                content_widget = wx.TextCtrl(
                    panel,
                    value=display_text_plain,
                    style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.BORDER_NONE,
                )
                try:
                    content_widget.SetBackgroundColour(bg_color)
                    content_widget.SetForegroundColour(base_fg)
                except Exception:
                    pass
            except Exception:
                display_text_plain = render_basic_latex(display_text)
                content_widget = wx.TextCtrl(
                    panel,
                    value=display_text_plain,
                    style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2 | wx.BORDER_NONE,
                )
                try:
                    content_widget.SetBackgroundColour(bg_color)
                    content_widget.SetForegroundColour(base_fg)
                except Exception:
                    pass

        content_widget.SetMinSize((wrap_width, needed_h))
        content_widget.SetSize((wrap_width, needed_h))
        try:
            self._install_copy_shortcuts(content_widget)
        except Exception:
            pass

        # For compact user one-liners, right-align content inside the bubble
        # so text sits flush against the right edge.
        content_flags = wx.ALL | alignment
        if msg.role == "user":
            content_flags = wx.ALL | wx.ALIGN_RIGHT
        sizer.Add(content_widget, 0, content_flags, 8)
        
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
        """Scroll chat to the bottom."""
        try:
            x, y = self.chat_scroll.GetVirtualSize()
            self.chat_scroll.Scroll(0, y)
        except:
            pass
    
    def set_suggestions(self, suggestions: List[str]):
        """Update the suggestion chips."""
        self._suggestions = suggestions
        
        # Clear existing chips
        self.chips_sizer.Clear(delete_windows=True)
        
        # Add new chips
        for suggestion in suggestions[:4]:  # Max 4 chips
            chip = self._create_chip(suggestion)
            self.chips_sizer.Add(chip, 0, wx.ALL, 2)
        
        self.suggestions_panel.Layout()
    
    def _create_chip(self, text: str) -> wx.Button:
        """Create a suggestion chip button."""
        btn = wx.Button(self.suggestions_panel, label=text, size=(-1, 24))
        btn.SetFont(btn.GetFont().Smaller())
        btn.Bind(wx.EVT_BUTTON, lambda e, t=text: self._on_chip_clicked(t))
        
        # Style
        try:
            if self._is_dark_mode():
                bg = self._blend_colours(self._window_bg_colour(), wx.Colour(100, 150, 200), 0.2)
            else:
                bg = wx.Colour(230, 240, 255)
            btn.SetBackgroundColour(bg)
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
        """Toggle the send button between Send and Pause modes."""
        self._agent_running = running
        try:
            if running:
                self.send_btn.SetLabel("⏸ Pause")
                self.input_text.Enable(False)
            else:
                self.send_btn.SetLabel("Send")
                try:
                    self.send_btn.SetBackgroundColour(wx.NullColour)
                    self.send_btn.SetForegroundColour(wx.NullColour)
                except Exception:
                    pass
                self.input_text.Enable(True)
                self.input_text.SetFocus()
            self.send_btn.Refresh()
        except Exception:
            pass

    def set_agent_awaiting_input(self, awaiting: bool):
        """Re-enable input when the agent asks a clarifying question."""
        try:
            self.input_text.Enable(awaiting)
            if awaiting:
                self.input_text.SetFocus()
                self.input_text.SetHint("Answer the agent's question...")
            else:
                self.input_text.SetHint("Describe what you want to do... (e.g., 'design an Arduino UNO')")
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
    
    def set_thinking(self, thinking: bool = True):
        """Show/hide thinking indicator."""
        if thinking:
            # Show a visible "Working..." bubble so the user knows the plugin isn't stuck.
            if getattr(self, '_thinking_msg', None) is None:
                self._thinking_msg = ChatMessage("system", "⏳ Working...")
                self._add_message(self._thinking_msg)
        else:
            # Remove the thinking bubble when we're done.
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

            # Also clear live thinking/status message if present.
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
    
    # === Theme helpers ===
    
    def _window_bg_colour(self):
        try:
            return wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
        except:
            return wx.Colour(30, 30, 30)
    
    def _window_text_colour(self):
        try:
            return wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOWTEXT)
        except:
            return wx.Colour(230, 230, 230)
    
    def _muted_text_colour(self):
        fg = self._window_text_colour()
        bg = self._window_bg_colour()
        return self._blend_colours(fg, bg, 0.45 if self._is_dark_mode() else 0.35)
    
    def _is_dark_mode(self) -> bool:
        try:
            app = wx.SystemSettings.GetAppearance()
            is_dark = getattr(app, "IsDark", None)
            if callable(is_dark):
                return bool(is_dark())
        except:
            pass
        
        try:
            c = self._window_bg_colour()
            luma = 0.2126 * c.Red() + 0.7152 * c.Green() + 0.0722 * c.Blue()
            return luma < 128
        except:
            return False
    
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
