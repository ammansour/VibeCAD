"""Debug tab UI.

Shows captured plugin log output so users can verify what commands/actions
were executed (e.g., exact board-outline coordinates).
"""

from __future__ import annotations

from typing import Callable, Optional
import logging

try:
    import wx
    WX_AVAILABLE = True
except Exception:
    WX_AVAILABLE = False
    class wx:  # type: ignore
        class Panel:
            pass


logger = logging.getLogger(__name__)


class DebugPanel(wx.Panel if WX_AVAILABLE else object):
    def __init__(
        self,
        parent,
        on_get_text: Optional[Callable[[], str]] = None,
        on_clear: Optional[Callable[[], None]] = None,
    ):
        if not WX_AVAILABLE:
            return

        super().__init__(parent)
        self._on_get_text = on_get_text
        self._on_clear = on_clear

        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self._timer)

        self._create_ui()

        # Auto-refresh so the user sees live output.
        try:
            self._timer.Start(750)
        except Exception:
            pass

        self.refresh_now()

    def _create_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)

        self.refresh_btn = wx.Button(self, label="Refresh")
        self.refresh_btn.Bind(wx.EVT_BUTTON, lambda evt: self.refresh_now())
        btn_row.Add(self.refresh_btn, 0, wx.ALL, 3)

        self.clear_btn = wx.Button(self, label="Clear")
        self.clear_btn.Bind(wx.EVT_BUTTON, self._on_clear_clicked)
        btn_row.Add(self.clear_btn, 0, wx.ALL, 3)

        self.copy_btn = wx.Button(self, label="📋")
        self.copy_btn.SetToolTip("Copy debug log to clipboard")
        self.copy_btn.Bind(wx.EVT_BUTTON, self._on_copy_clicked)
        btn_row.Add(self.copy_btn, 0, wx.ALL, 3)

        btn_row.AddStretchSpacer()

        self._hint = wx.StaticText(
            self,
            label="Shows recent VibeCAD command/log output (copyable).",
        )
        btn_row.Add(self._hint, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        sizer.Add(btn_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        self.text = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
        )
        sizer.Add(self.text, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)

        self._last_content: str = ""
        self._pinned_to_bottom: bool = True

        # If the user scrolls up, do not auto-scroll on refresh. Only scroll
        # to bottom when already at bottom.
        for evt in (getattr(wx, "EVT_MOUSEWHEEL", None), getattr(wx, "EVT_SCROLLWIN", None)):
            if evt is None:
                continue
            try:
                self.text.Bind(evt, self._on_user_scroll)
            except Exception:
                pass

    def _on_user_scroll(self, evt):
        try:
            # Update after wx applies the scroll.
            wx.CallAfter(self._update_pinned_flag)
        except Exception:
            pass
        try:
            evt.Skip()
        except Exception:
            pass

    def _update_pinned_flag(self) -> None:
        self._pinned_to_bottom = self._is_at_bottom()

    def _is_at_bottom(self) -> bool:
        if not WX_AVAILABLE:
            return True
        try:
            if not hasattr(self.text, "GetScrollPos"):
                return self._pinned_to_bottom
            if not hasattr(wx, "VERTICAL"):
                return self._pinned_to_bottom
            pos = int(self.text.GetScrollPos(wx.VERTICAL))
            rng = int(self.text.GetScrollRange(wx.VERTICAL))
            thumb = int(self.text.GetScrollThumb(wx.VERTICAL))
            # Consider within 1 scroll unit of bottom as pinned.
            return pos >= max(0, rng - thumb - 1)
        except Exception:
            return self._pinned_to_bottom

    def _restore_scroll(self, pos: int) -> None:
        if not WX_AVAILABLE:
            return
        try:
            if hasattr(self.text, "SetScrollPos") and hasattr(wx, "VERTICAL"):
                self.text.SetScrollPos(wx.VERTICAL, int(pos), True)
        except Exception:
            pass

    def _on_copy_clicked(self, _evt):
        if not WX_AVAILABLE:
            return
        try:
            content = self.text.GetValue() or ""
            text_data = wx.TextDataObject(content)
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(text_data)
                wx.TheClipboard.Close()
                try:
                    old = self._hint.GetLabel()
                    self._hint.SetLabel("Copied debug log to clipboard.")
                    wx.CallLater(2000, lambda: self._hint.SetLabel(old))
                except Exception:
                    pass
        except Exception:
            logger.debug("Debug copy failed", exc_info=True)

    def _on_timer(self, _evt):
        self.refresh_now()

    def _on_clear_clicked(self, _evt):
        try:
            if callable(self._on_clear):
                self._on_clear()
        except Exception:
            logger.debug("Debug clear failed", exc_info=True)
        self.refresh_now()

    def refresh_now(self) -> None:
        try:
            if not callable(self._on_get_text):
                return
            content = self._on_get_text() or ""
            # Avoid flicker if unchanged.
            if content == self._last_content:
                return

            pinned = self._is_at_bottom()
            prev_scroll = None
            try:
                if hasattr(self.text, "GetScrollPos") and hasattr(wx, "VERTICAL"):
                    prev_scroll = int(self.text.GetScrollPos(wx.VERTICAL))
            except Exception:
                prev_scroll = None

            try:
                self.text.Freeze()
            except Exception:
                pass

            try:
                # Preserve selection/insertion if possible.
                ip = None
                sel = None
                try:
                    ip = int(self.text.GetInsertionPoint())
                    sel = self.text.GetSelection()
                except Exception:
                    ip = None
                    sel = None

                # Prefer incremental append when content grows.
                if self._last_content and content.startswith(self._last_content):
                    delta = content[len(self._last_content):]
                    if delta:
                        try:
                            self.text.AppendText(delta)
                        except Exception:
                            self.text.SetValue(content)
                else:
                    self.text.SetValue(content)

                if sel is not None:
                    try:
                        self.text.SetSelection(sel[0], sel[1])
                    except Exception:
                        pass
                if ip is not None:
                    try:
                        self.text.SetInsertionPoint(ip)
                    except Exception:
                        pass
            finally:
                try:
                    self.text.Thaw()
                except Exception:
                    pass

            self._last_content = content

            if pinned:
                try:
                    self.text.ShowPosition(self.text.GetLastPosition())
                except Exception:
                    pass
                self._pinned_to_bottom = True
            else:
                if prev_scroll is not None:
                    self._restore_scroll(prev_scroll)
                self._pinned_to_bottom = False
        except Exception:
            logger.debug("Debug refresh failed", exc_info=True)
