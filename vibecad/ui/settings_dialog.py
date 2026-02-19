"""Settings dialog for configuring LLM provider parameters."""

import importlib
import logging
from typing import Any, Optional

try:
    wx: Any = importlib.import_module("wx")
    WX_AVAILABLE = True
except Exception:  # pragma: no cover
    wx = None
    WX_AVAILABLE = False

from ..config import VibeCADSettings

logger = logging.getLogger(__name__)


class _SettingsDialogStub(object):
    def __init__(self, parent, settings: VibeCADSettings):
        self._settings = settings

    @property
    def settings(self) -> VibeCADSettings:
        return self._settings

    def ShowModal(self) -> int:
        return 0

    def Destroy(self) -> None:
        return None


# Default to stub; override below if wx is available
_SettingsDialogImpl = _SettingsDialogStub


if WX_AVAILABLE:

    class _SettingsDialogWx(wx.Dialog):
        def __init__(self, parent, settings: VibeCADSettings):
            super().__init__(
                parent,
                title="VibeCAD Settings",
                style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                size=(620, 460),
            )

            self._settings = settings

            panel = wx.Panel(self)
            sizer = wx.BoxSizer(wx.VERTICAL)

            header = wx.StaticText(panel, label="LLM Provider Settings")
            font = header.GetFont()
            font.SetPointSize(12)
            font.SetWeight(wx.FONTWEIGHT_BOLD)
            header.SetFont(font)
            sizer.Add(header, 0, wx.ALL, 10)

            help_text = wx.StaticText(
                panel,
                label=(
                    "These settings are stored in ~/.vibecad/settings.json and are used by the plugin.\n"
                    "You can also use environment variables (VIBECAD_* or GITHUB_TOKEN) to override."
                ),
            )
            help_text.SetForegroundColour(wx.Colour(120, 120, 120))
            sizer.Add(help_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            grid = wx.FlexGridSizer(cols=2, hgap=10, vgap=10)
            grid.AddGrowableCol(1, 1)

            def add_row(label: str, ctrl):
                st = wx.StaticText(panel, label=label)
                grid.Add(st, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
                grid.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 10)

            self.api_key = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
            self.api_key.SetValue(settings.api_key or "")
            add_row("API key", self.api_key)

            self.api_base = wx.TextCtrl(panel)
            self.api_base.SetValue(settings.api_base or "")
            self.api_base.SetHint("e.g. https://models.github.ai/inference or https://api.openai.com/v1")
            add_row("API base URL", self.api_base)

            self.model = wx.TextCtrl(panel)
            self.model.SetValue(settings.model or "")
            self.model.SetHint("e.g. openai/gpt-5")
            add_row("Model", self.model)

            self.temperature = wx.TextCtrl(panel)
            self.temperature.SetValue("" if settings.temperature is None else str(settings.temperature))
            self.temperature.SetHint("0.0 - 2.0 (some models only support default)")
            add_row("Temperature (optional)", self.temperature)

            self.max_tokens = wx.TextCtrl(panel)
            self.max_tokens.SetValue("" if settings.max_tokens is None else str(settings.max_tokens))
            self.max_tokens.SetHint("e.g. 1024")
            add_row("Max tokens (or completion tokens)", self.max_tokens)

            self.timeout = wx.TextCtrl(panel)
            self.timeout.SetValue("" if settings.timeout is None else str(settings.timeout))
            self.timeout.SetHint("seconds")
            add_row("Timeout", self.timeout)

            self.verify_ssl = wx.CheckBox(panel, label="Verify TLS certificates (recommended)")
            self.verify_ssl.SetValue(bool(getattr(settings, "verify_ssl", True)))
            add_row("TLS", self.verify_ssl)

            self.ca_bundle_path = wx.TextCtrl(panel)
            self.ca_bundle_path.SetValue(getattr(settings, "ca_bundle_path", "") or "")
            self.ca_bundle_path.SetHint("Optional path to a CA bundle (PEM). Leave blank to use defaults.")
            add_row("CA bundle path", self.ca_bundle_path)

            # --- Agent UX ---
            agent_header = wx.StaticText(panel, label="Design Agent")
            f2 = agent_header.GetFont()
            f2.SetWeight(wx.FONTWEIGHT_BOLD)
            agent_header.SetFont(f2)
            sizer.Add(agent_header, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

            agent_help = wx.StaticText(
                panel,
                label=(
                    "These settings affect the Design tab agent behavior."
                ),
            )
            agent_help.SetForegroundColour(wx.Colour(120, 120, 120))
            sizer.Add(agent_help, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            self.thinking_output_enabled = wx.CheckBox(panel, label="Show thinking output (step-by-step status messages)")
            self.thinking_output_enabled.SetValue(bool(getattr(settings, "thinking_output_enabled", True)))
            sizer.Add(self.thinking_output_enabled, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            self.yolo_auto_apply = wx.CheckBox(panel, label="YOLO mode: auto-apply actions without asking")
            self.yolo_auto_apply.SetValue(bool(getattr(settings, "yolo_auto_apply", False)))
            sizer.Add(self.yolo_auto_apply, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            yolo_disclaimer = wx.StaticText(
                panel,
                label=(
                    "Disclaimer: YOLO mode can modify your board without approvals.\n"
                    "Use at your own risk; keep backups / use version control."
                ),
            )
            yolo_disclaimer.SetForegroundColour(wx.Colour(160, 90, 90))
            sizer.Add(yolo_disclaimer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            sizer.Add(grid, 1, wx.EXPAND | wx.TOP, 5)

            btn_sizer = wx.StdDialogButtonSizer()
            self.btn_save = wx.Button(panel, wx.ID_OK, label="Save")
            self.btn_cancel = wx.Button(panel, wx.ID_CANCEL, label="Cancel")
            btn_sizer.AddButton(self.btn_save)
            btn_sizer.AddButton(self.btn_cancel)
            btn_sizer.Realize()
            sizer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

            panel.SetSizer(sizer)

            self.Bind(wx.EVT_BUTTON, self._on_save, self.btn_save)

        @property
        def settings(self) -> VibeCADSettings:
            return self._settings

        def _on_save(self, event):
            try:
                s = VibeCADSettings(
                    api_key=self.api_key.GetValue().strip(),
                    api_base=self.api_base.GetValue().strip(),
                    model=self.model.GetValue().strip(),
                    temperature=self._parse_float(self.temperature.GetValue().strip()),
                    max_tokens=self._parse_int(self.max_tokens.GetValue().strip()),
                    timeout=self._parse_int(self.timeout.GetValue().strip()),
                    verify_ssl=bool(self.verify_ssl.GetValue()),
                    ca_bundle_path=self.ca_bundle_path.GetValue().strip(),
                    thinking_output_enabled=bool(self.thinking_output_enabled.GetValue()),
                    yolo_auto_apply=bool(self.yolo_auto_apply.GetValue()),
                )
                self._settings = s
            except ValueError as e:
                wx.MessageBox(str(e), "Invalid settings", wx.OK | wx.ICON_ERROR)
                return

            event.Skip()

        def _parse_float(self, value: str) -> Optional[float]:
            if value == "":
                return None
            try:
                f = float(value)
            except Exception:
                raise ValueError("Temperature must be a number (or blank).")
            if f < 0.0 or f > 2.0:
                raise ValueError("Temperature must be between 0.0 and 2.0.")
            return f

        def _parse_int(self, value: str) -> Optional[int]:
            if value == "":
                return None
            try:
                return int(value)
            except Exception:
                raise ValueError("Max tokens / timeout must be an integer (or blank).")

    _SettingsDialogImpl = _SettingsDialogWx

SettingsDialog = _SettingsDialogImpl
