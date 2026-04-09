"""Settings dialog for configuring LLM provider parameters."""

import importlib
import logging
from pathlib import Path
from typing import Any, Optional

try:
    wx: Any = importlib.import_module("wx")
    WX_AVAILABLE = True
except Exception:  # pragma: no cover
    wx = None
    WX_AVAILABLE = False

from ..config import VibeCADSettings
from ..config.settings import (
    DEFAULT_LLM_MODEL,
    LLM_PROVIDER_OPENROUTER,
    LLM_PROVIDER_VERTEX,
    LLM_PROVIDER_GITHUB,
    LLM_PROVIDER_OPENAI,
    LLM_PROVIDER_CUSTOM,
    _PROVIDER_API_BASES,
    default_api_key_override_path,
    default_settings_path,
)
from . import theme

logger = logging.getLogger(__name__)

# Human-readable labels for the provider dropdown (order matters).
_PROVIDER_LABELS = [
    ("OpenRouter",         LLM_PROVIDER_OPENROUTER),
    ("Vertex AI (Google)", LLM_PROVIDER_VERTEX),
    ("GitHub Models",      LLM_PROVIDER_GITHUB),
    ("OpenAI",             LLM_PROVIDER_OPENAI),
    ("Custom / Other",     LLM_PROVIDER_CUSTOM),
]
_PROVIDER_IDS   = [p for _, p in _PROVIDER_LABELS]
_PROVIDER_NAMES = [n for n, _ in _PROVIDER_LABELS]


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
                size=(760, 860),
            )

            self._settings = settings

            self._root_sizer = wx.BoxSizer(wx.VERTICAL)
            self._scroll = wx.ScrolledWindow(self, style=wx.VSCROLL)
            self._scroll.SetScrollRate(0, 20)
            try:
                # Show vertical scrollbar only when content overflows.
                self._scroll.ShowScrollbars(wx.SHOW_SB_NEVER, wx.SHOW_SB_AUTO)
            except Exception:
                pass
            self._root_sizer.Add(self._scroll, 1, wx.EXPAND)
            self.SetSizer(self._root_sizer)
            try:
                self.SetBackgroundColour(theme.panel_bg_colour())
                self.SetForegroundColour(theme.window_text_colour())
                self._scroll.SetBackgroundColour(theme.panel_bg_colour())
            except Exception:
                pass

            panel = wx.Panel(self._scroll)
            try:
                panel.SetBackgroundColour(theme.panel_bg_colour())
                panel.SetForegroundColour(theme.window_text_colour())
            except Exception:
                pass
            outer = wx.BoxSizer(wx.VERTICAL)

            # ── Header ────────────────────────────────────────────────────────
            header = wx.StaticText(panel, label="LLM Provider Settings")
            font = header.GetFont()
            font.SetPointSize(12)
            font.SetWeight(wx.FONTWEIGHT_BOLD)
            header.SetFont(font)
            try:
                header.SetForegroundColour(theme.window_text_colour())
            except Exception:
                pass
            outer.Add(header, 0, wx.ALL, 10)

            help_text = wx.StaticText(
                panel,
                label=(
                    f"Stored in {default_settings_path()}.\n"
                    f"Optional API key file: {default_api_key_override_path()} (takes precedence when set).\n"
                    "Environment variables (VIBECAD_* / GITHUB_TOKEN) override these at runtime."
                ),
            )
            help_text.SetForegroundColour(theme.muted_text_colour())
            outer.Add(help_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            # ── Provider preset selector ──────────────────────────────────────
            provider_row = wx.BoxSizer(wx.HORIZONTAL)
            provider_row.Add(
                wx.StaticText(panel, label="Provider preset:"),
                0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 10,
            )
            self._provider_choice = wx.Choice(panel, choices=_PROVIDER_NAMES)
            cur_provider = str(getattr(settings, "llm_provider", LLM_PROVIDER_OPENROUTER) or LLM_PROVIDER_OPENROUTER)
            idx = _PROVIDER_IDS.index(cur_provider) if cur_provider in _PROVIDER_IDS else 0
            self._provider_choice.SetSelection(idx)
            provider_row.Add(self._provider_choice, 1, wx.EXPAND | wx.RIGHT, 10)
            outer.Add(provider_row, 0, wx.EXPAND | wx.BOTTOM, 6)

            provider_note = wx.StaticText(
                panel,
                label="Selecting a preset fills the API base URL below. You can always edit it manually.",
            )
            provider_note.SetForegroundColour(theme.muted_text_colour())
            outer.Add(provider_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            chat_note = wx.StaticText(
                panel,
                label="Model selection and extended reasoning are now controlled from the Design tab below the chat input.",
            )
            chat_note.SetForegroundColour(theme.muted_text_colour())
            outer.Add(chat_note, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            # ── Common fields grid ────────────────────────────────────────────
            grid = wx.FlexGridSizer(cols=2, hgap=10, vgap=8)
            grid.AddGrowableCol(1, 1)

            def add_row(label: str, ctrl):
                st = wx.StaticText(panel, label=label)
                grid.Add(st, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 10)
                grid.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 10)

            self.api_key = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
            self.api_key.SetValue(settings.api_key or "")
            add_row("API key", self.api_key)

            self.github_token = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
            self.github_token.SetValue(getattr(settings, "github_token", "") or "")
            self.github_token.SetHint("Optional GitHub token for GitHub Models / part search")
            add_row("GitHub token (optional)", self.github_token)

            self.api_base = wx.TextCtrl(panel)
            self.api_base.SetValue(settings.api_base or "")
            self.api_base.SetHint("e.g. https://openrouter.ai/api/v1")
            add_row("API base URL", self.api_base)

            self.temperature = wx.TextCtrl(panel)
            self.temperature.SetValue("" if settings.temperature is None else str(settings.temperature))
            self.temperature.SetHint("0.0 – 2.0 (blank = use provider default)")
            add_row("Temperature (optional)", self.temperature)

            self.top_p = wx.TextCtrl(panel)
            self.top_p.SetValue("" if settings.top_p is None else str(settings.top_p))
            self.top_p.SetHint("0.0 – 1.0 (blank = provider default; try 0.95 for reasoning models)")
            add_row("Top-P (optional)", self.top_p)

            self.max_tokens = wx.TextCtrl(panel)
            self.max_tokens.SetValue("" if settings.max_tokens is None else str(settings.max_tokens))
            self.max_tokens.SetHint("e.g. 16384")
            add_row("Max tokens (optional)", self.max_tokens)

            self.timeout = wx.TextCtrl(panel)
            self.timeout.SetValue("" if settings.timeout is None else str(settings.timeout))
            self.timeout.SetHint("seconds (blank = 120)")
            add_row("Timeout (optional)", self.timeout)

            self.verify_ssl = wx.CheckBox(panel, label="Verify TLS certificates (recommended)")
            self.verify_ssl.SetValue(bool(getattr(settings, "verify_ssl", True)))
            add_row("TLS", self.verify_ssl)

            self.ca_bundle_path = wx.TextCtrl(panel)
            self.ca_bundle_path.SetValue(getattr(settings, "ca_bundle_path", "") or "")
            self.ca_bundle_path.SetHint("Optional path to CA bundle (.pem). Leave blank for defaults.")
            add_row("CA bundle (optional)", self.ca_bundle_path)

            outer.Add(grid, 0, wx.EXPAND | wx.TOP, 2)

            self._vertex_box = wx.StaticBoxSizer(
                wx.StaticBox(panel, label="Vertex AI configuration"), wx.VERTICAL
            )
            vgrid = wx.FlexGridSizer(cols=2, hgap=10, vgap=8)
            vgrid.AddGrowableCol(1, 1)

            def add_vrow(label: str, ctrl):
                st = wx.StaticText(panel, label=label)
                vgrid.Add(st, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
                vgrid.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 6)

            self.vertex_project = wx.TextCtrl(panel)
            self.vertex_project.SetValue(getattr(settings, "vertex_project", "") or "")
            self.vertex_project.SetHint("your-gcp-project-id")
            add_vrow("GCP project ID", self.vertex_project)

            self.vertex_location = wx.TextCtrl(panel)
            self.vertex_location.SetValue(getattr(settings, "vertex_location", "us-central1") or "us-central1")
            self.vertex_location.SetHint("us-central1 or global (for Gemini 3 preview)")
            add_vrow("Location / region", self.vertex_location)

            self.vertex_credentials_path = wx.TextCtrl(panel)
            self.vertex_credentials_path.SetValue(getattr(settings, "vertex_credentials_path", "") or "")
            self.vertex_credentials_path.SetHint("blank = Application Default Credentials (ADC)")

            # Browse button + text field in one horizontal sizer
            creds_row = wx.BoxSizer(wx.HORIZONTAL)
            creds_row.Add(self.vertex_credentials_path, 1, wx.EXPAND | wx.RIGHT, 4)
            self._btn_browse_creds = wx.Button(panel, label="Browse…")
            creds_row.Add(self._btn_browse_creds, 0, wx.ALIGN_CENTER_VERTICAL)
            add_vrow("Service account JSON", creds_row)

            # Auto-detect ADC path if field is blank
            if not self.vertex_credentials_path.GetValue().strip():
                import os as _os
                _adc_candidates = [
                    _os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
                    _os.path.expanduser("~/.config/gcloud/legacy_credentials/default/adc.json"),
                ]
                for _adc in _adc_candidates:
                    if _os.path.isfile(_adc):
                        # Don't pre-fill — ADC is used automatically without a path.
                        # Just update the hint so the user knows where it is.
                        self.vertex_credentials_path.SetHint(
                            f"ADC found: {_adc} (leave blank to use it automatically)"
                        )
                        break

            vertex_auth_note = wx.StaticText(
                panel,
                label=(
                    "Leave service account blank to use ADC (Application Default Credentials).\n"
                    "Run: gcloud auth application-default login"
                ),
            )
            vertex_auth_note.SetForegroundColour(theme.muted_text_colour())

            self._vertex_box.Add(vgrid, 0, wx.EXPAND | wx.ALL, 6)
            self._vertex_box.Add(vertex_auth_note, 0, wx.LEFT | wx.BOTTOM, 6)
            outer.Add(self._vertex_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

            # ── DigiKey API section ───────────────────────────────────────────
            self._digikey_box = wx.StaticBoxSizer(
                wx.StaticBox(panel, label="DigiKey API (real datasheet lookups)"), wx.VERTICAL
            )
            dkgrid = wx.FlexGridSizer(cols=2, hgap=10, vgap=8)
            dkgrid.AddGrowableCol(1, 1)

            def add_dkrow(label: str, ctrl):
                dkgrid.Add(wx.StaticText(panel, label=label), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 6)
                dkgrid.Add(ctrl, 1, wx.EXPAND | wx.RIGHT, 6)

            self.digikey_client_id = wx.TextCtrl(panel)
            self.digikey_client_id.SetValue(getattr(settings, "digikey_client_id", "") or "")
            self.digikey_client_id.SetHint("DigiKey OAuth2 Client ID")
            add_dkrow("Client ID", self.digikey_client_id)

            self.digikey_client_secret = wx.TextCtrl(panel, style=wx.TE_PASSWORD)
            self.digikey_client_secret.SetValue(getattr(settings, "digikey_client_secret", "") or "")
            self.digikey_client_secret.SetHint("DigiKey OAuth2 Client Secret")
            add_dkrow("Client Secret", self.digikey_client_secret)

            dk_note = wx.StaticText(
                panel,
                label=(
                    "Register a free app at developer.digikey.com → My Apps → Create App.\n"
                    "Subscribe to \"Product Information\". When set, the SPEC agent downloads\n"
                    "real datasheets instead of relying on training-memory recall."
                ),
            )
            dk_note.SetForegroundColour(theme.muted_text_colour())
            self._digikey_box.Add(dkgrid, 0, wx.EXPAND | wx.ALL, 6)
            self._digikey_box.Add(dk_note, 0, wx.LEFT | wx.BOTTOM, 6)
            outer.Add(self._digikey_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 10)

            # ── Design Agent section ──────────────────────────────────────────
            agent_header = wx.StaticText(panel, label="Design Agent")
            f2 = agent_header.GetFont()
            f2.SetWeight(wx.FONTWEIGHT_BOLD)
            agent_header.SetFont(f2)
            outer.Add(agent_header, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

            agent_help = wx.StaticText(panel, label="These settings affect the Design tab agent behaviour.")
            agent_help.SetForegroundColour(theme.muted_text_colour())
            outer.Add(agent_help, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

            self.thinking_output_enabled = wx.CheckBox(panel, label="Show thinking output (step-by-step status messages)")
            self.thinking_output_enabled.SetValue(bool(getattr(settings, "thinking_output_enabled", True)))
            outer.Add(self.thinking_output_enabled, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            self.yolo_auto_apply = wx.CheckBox(panel, label="YOLO mode: auto-apply actions without asking")
            self.yolo_auto_apply.SetValue(bool(getattr(settings, "yolo_auto_apply", False)))
            outer.Add(self.yolo_auto_apply, 0, wx.LEFT | wx.RIGHT, 10)

            yolo_disclaimer = wx.StaticText(
                panel,
                label="⚠ YOLO mode can modify your board without approvals. Keep backups / use version control.",
            )
            yolo_disclaimer.SetForegroundColour(theme.warning_text_colour())
            outer.Add(yolo_disclaimer, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

            control_bg = theme.chat_surface_colour()
            control_fg = theme.window_text_colour()
            for ctrl in (
                self._provider_choice,
                self.api_key,
                self.github_token,
                self.api_base,
                self.temperature,
                self.top_p,
                self.max_tokens,
                self.timeout,
                self.verify_ssl,
                self.ca_bundle_path,
                self.vertex_project,
                self.vertex_location,
                self.vertex_credentials_path,
                self.digikey_client_id,
                self.digikey_client_secret,
                self.thinking_output_enabled,
                self.yolo_auto_apply,
            ):
                try:
                    ctrl.SetBackgroundColour(control_bg)
                    ctrl.SetForegroundColour(control_fg)
                except Exception:
                    pass

            # ── Buttons ───────────────────────────────────────────────────────
            btn_sizer = wx.StdDialogButtonSizer()
            self.btn_save = wx.Button(panel, wx.ID_OK, label="Save")
            self.btn_cancel = wx.Button(panel, wx.ID_CANCEL, label="Cancel")
            btn_sizer.AddButton(self.btn_save)
            btn_sizer.AddButton(self.btn_cancel)
            btn_sizer.Realize()
            outer.Add(btn_sizer, 0, wx.EXPAND | wx.ALL, 10)

            panel.SetSizer(outer)
            panel.Layout()
            try:
                outer.Fit(panel)
            except Exception:
                pass

            self._content_panel = panel
            self._scroll_content_sizer = wx.BoxSizer(wx.VERTICAL)
            self._scroll_content_sizer.Add(panel, 0, wx.EXPAND)
            self._scroll.SetSizer(self._scroll_content_sizer)
            self._sync_scroll_area()

            # Open large enough to show the full settings UI by default.
            best_size = panel.GetBestSize()
            try:
                best_w = int(best_size.GetWidth())
                best_h = int(best_size.GetHeight())
            except Exception:
                best_w, best_h = 760, 860
            target_w = max(760, int(best_w + 80))
            target_h = max(860, int(best_h + 120))
            try:
                display_idx = wx.Display.GetFromWindow(self)
                if display_idx != wx.NOT_FOUND:
                    area = wx.Display(display_idx).GetClientArea()
                    aw = int(getattr(area, "width", area.GetWidth()))
                    ah = int(getattr(area, "height", area.GetHeight()))
                    target_w = min(target_w, max(680, int(aw * 0.95)))
                    target_h = min(target_h, max(700, int(ah * 0.95)))
            except Exception:
                pass
            self.SetMinSize((640, 680))
            self.SetSize((target_w, target_h))
            try:
                self.CentreOnParent()
            except Exception:
                pass

            # Wire events
            self.Bind(wx.EVT_BUTTON, self._on_save, self.btn_save)
            self.Bind(wx.EVT_CHOICE, self._on_provider_changed, self._provider_choice)
            self.Bind(wx.EVT_BUTTON, self._on_browse_credentials, self._btn_browse_creds)
            self.Bind(wx.EVT_SIZE, self._on_scroll_resized)
            self._scroll.Bind(wx.EVT_SIZE, self._on_scroll_resized)

            # Initialise visibility
            self._update_provider_ui(cur_provider, fill_url=False)
            try:
                wx.CallAfter(self._sync_scroll_area)
            except Exception:
                pass

        # ── helpers ───────────────────────────────────────────────────────────

        def _current_provider_id(self) -> str:
            sel = self._provider_choice.GetSelection()
            if 0 <= sel < len(_PROVIDER_IDS):
                return _PROVIDER_IDS[sel]
            return LLM_PROVIDER_CUSTOM

        def _update_provider_ui(self, provider_id: str, fill_url: bool = True) -> None:
            """Show/hide the Vertex section and optionally fill preset URL."""
            show_vertex = (provider_id == LLM_PROVIDER_VERTEX)
            self._vertex_box.ShowItems(show_vertex)
            self._vertex_box.GetStaticBox().Show(show_vertex)

            if fill_url:
                preset_url = _PROVIDER_API_BASES.get(provider_id, "")
                if preset_url:
                    self.api_base.SetValue(preset_url)
                elif provider_id == LLM_PROVIDER_VERTEX:
                    self.api_base.SetValue("")  # built at runtime; leave blank

            try:
                self.Layout()
                self.Refresh()
                self._sync_scroll_area()
            except Exception:
                pass

        def _on_scroll_resized(self, event) -> None:
            try:
                self._sync_scroll_area()
            except Exception:
                pass
            try:
                event.Skip()
            except Exception:
                pass

        def _sync_scroll_area(self) -> None:
            """Keep scrolled content width in sync and refresh scrollbars."""
            try:
                if not hasattr(self, "_scroll") or self._scroll is None:
                    return
                if not hasattr(self, "_content_panel") or self._content_panel is None:
                    return
                client_w = int(self._scroll.GetClientSize().GetWidth())
                if client_w > 0:
                    self._content_panel.SetMinSize((max(1, client_w), -1))
                self._scroll.Layout()
                self._scroll.FitInside()
            except Exception:
                pass

        def _on_provider_changed(self, event) -> None:
            self._update_provider_ui(self._current_provider_id(), fill_url=True)

        def _on_browse_credentials(self, event) -> None:
            """Open a file-picker to locate the service-account JSON."""
            import os as _os
            # Start in the directory of the current value, or ~/.config/gcloud
            cur = self.vertex_credentials_path.GetValue().strip()
            start_dir = ""
            if cur:
                cur_path = Path(cur).expanduser()
                if not cur_path.is_absolute():
                    cur_path = default_settings_path().parent / cur_path
                if cur_path.parent.exists():
                    start_dir = str(cur_path.parent)
            if not start_dir:
                start_dir = _os.path.expanduser("~/.config/gcloud")
                if not _os.path.isdir(start_dir):
                    start_dir = _os.path.expanduser("~")
            dlg = wx.FileDialog(
                self,
                message="Select service-account credentials JSON",
                defaultDir=start_dir,
                defaultFile="",
                wildcard="JSON files (*.json)|*.json|All files (*.*)|*.*",
                style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
            )
            if dlg.ShowModal() == wx.ID_OK:
                self.vertex_credentials_path.SetValue(dlg.GetPath())
            dlg.Destroy()

        @property
        def settings(self) -> VibeCADSettings:
            return self._settings

        def _on_save(self, event):
            try:
                s = VibeCADSettings(
                    api_key=self.api_key.GetValue().strip(),
                    github_token=self.github_token.GetValue().strip(),
                    api_base=self.api_base.GetValue().strip(),
                    model=str(getattr(self._settings, "model", "") or "").strip() or DEFAULT_LLM_MODEL,
                    temperature=self._parse_float(self.temperature.GetValue().strip()),
                    max_tokens=self._parse_int(self.max_tokens.GetValue().strip()),
                    timeout=self._parse_int(self.timeout.GetValue().strip()),
                    verify_ssl=bool(self.verify_ssl.GetValue()),
                    ca_bundle_path=self.ca_bundle_path.GetValue().strip(),
                    llm_provider=self._current_provider_id(),
                    vertex_project=self.vertex_project.GetValue().strip(),
                    vertex_location=self.vertex_location.GetValue().strip() or "us-central1",
                    vertex_credentials_path=self.vertex_credentials_path.GetValue().strip(),
                    thinking_output_enabled=bool(self.thinking_output_enabled.GetValue()),
                    yolo_auto_apply=bool(self.yolo_auto_apply.GetValue()),
                    enable_thinking=bool(getattr(self._settings, "enable_thinking", False)),
                    thinking_budget=getattr(self._settings, "thinking_budget", None),
                    top_p=self._parse_top_p(self.top_p.GetValue().strip()),
                    digikey_client_id=self.digikey_client_id.GetValue().strip(),
                    digikey_client_secret=self.digikey_client_secret.GetValue().strip(),
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

        def _parse_top_p(self, value: str) -> Optional[float]:
            if value == "":
                return None
            try:
                f = float(value)
            except Exception:
                raise ValueError("Top-P must be a number between 0.0 and 1.0 (or blank).")
            if f < 0.0 or f > 1.0:
                raise ValueError("Top-P must be between 0.0 and 1.0.")
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
