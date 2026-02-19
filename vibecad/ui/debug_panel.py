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

        btn_row.AddStretchSpacer()

        hint = wx.StaticText(
            self,
            label="Shows recent VibeCAD command/log output (copyable).",
        )
        btn_row.Add(hint, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 3)

        sizer.Add(btn_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        self.text = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2,
        )
        sizer.Add(self.text, 1, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(sizer)

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
            if self.text.GetValue() != content:
                self.text.SetValue(content)
                try:
                    self.text.ShowPosition(self.text.GetLastPosition())
                except Exception:
                    pass
        except Exception:
            logger.debug("Debug refresh failed", exc_info=True)
