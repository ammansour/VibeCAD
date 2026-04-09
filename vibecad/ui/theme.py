"""Shared theme helpers for VibeCAD UI.

The palette is intentionally centralized so dark-mode handling stays
consistent across the design panel, debug panel, settings dialog, and
WebView-rendered chat bubbles.
"""

from __future__ import annotations

try:
    import wx
    WX_AVAILABLE = True
except Exception:  # pragma: no cover
    WX_AVAILABLE = False

    class _Colour:
        def __init__(self, red: int = 0, green: int = 0, blue: int = 0):
            self._red = int(red)
            self._green = int(green)
            self._blue = int(blue)

        def Red(self) -> int:
            return self._red

        def Green(self) -> int:
            return self._green

        def Blue(self) -> int:
            return self._blue

    class _WxStub:
        Colour = _Colour

    wx = _WxStub()  # type: ignore


_LIGHT_PANEL = (240, 240, 240)
_LIGHT_SURFACE = (255, 255, 255)
_LIGHT_TEXT = (28, 28, 28)
_LIGHT_MUTED = (110, 110, 110)
_LIGHT_USER = (220, 240, 255)
_LIGHT_ASSISTANT = (255, 255, 255)
_LIGHT_SYSTEM = (228, 241, 255)
_LIGHT_WARNING = (160, 90, 90)
_LIGHT_CHIP = (230, 240, 255)

_DARK_PANEL = (31, 31, 33)
_DARK_SURFACE = (40, 40, 44)
_DARK_TEXT = (242, 242, 242)
_DARK_MUTED = (168, 168, 168)
_DARK_USER = (51, 94, 140)
_DARK_ASSISTANT = (46, 46, 50)
_DARK_SYSTEM = (52, 70, 104)
_DARK_WARNING = (219, 128, 128)
_DARK_CHIP = (60, 76, 96)


def _colour(rgb):
    return wx.Colour(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _colour_luma(colour) -> float:
    try:
        return 0.2126 * float(colour.Red()) + 0.7152 * float(colour.Green()) + 0.0722 * float(colour.Blue())
    except Exception:
        return 0.0


def is_dark_mode() -> bool:
    """Best-effort detection of the active system appearance."""
    if WX_AVAILABLE:
        try:
            appearance = wx.SystemSettings.GetAppearance()
            is_dark = getattr(appearance, "IsDark", None)
            if callable(is_dark):
                return bool(is_dark())
        except Exception:
            pass

        try:
            colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW)
            return _colour_luma(colour) < 128.0
        except Exception:
            pass

    return False


def html_color_scheme() -> str:
    return "dark" if is_dark_mode() else "light"


def window_bg_colour():
    return _colour(_DARK_PANEL if is_dark_mode() else _LIGHT_PANEL)


def panel_bg_colour():
    return window_bg_colour()


def chat_surface_colour():
    return _colour(_DARK_SURFACE if is_dark_mode() else _LIGHT_SURFACE)


def window_text_colour():
    return _colour(_DARK_TEXT if is_dark_mode() else _LIGHT_TEXT)


def muted_text_colour():
    return _colour(_DARK_MUTED if is_dark_mode() else _LIGHT_MUTED)


def warning_text_colour():
    return _colour(_DARK_WARNING if is_dark_mode() else _LIGHT_WARNING)


def chip_colour():
    return _colour(_DARK_CHIP if is_dark_mode() else _LIGHT_CHIP)


def bubble_colour_for_role(role: str):
    role = str(role or "").strip().lower()
    if is_dark_mode():
        if role == "user":
            return _colour(_DARK_USER)
        if role == "assistant":
            return _colour(_DARK_ASSISTANT)
        return _colour(_DARK_SYSTEM)

    if role == "user":
        return _colour(_LIGHT_USER)
    if role == "assistant":
        return _colour(_LIGHT_ASSISTANT)
    return _colour(_LIGHT_SYSTEM)
