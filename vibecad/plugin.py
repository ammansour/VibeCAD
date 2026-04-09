"""
VibeCAD Plugin for KiCad 7.

This is the main plugin entry point that integrates with KiCad's
plugin system.
"""

import logging
import os
import re
import tempfile
import threading
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict, Set

from .debug_log import InMemoryLogBuffer, install_debug_log_capture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vibecad')

# Enable basic crash logging via faulthandler
try:
    import faulthandler
    crash_log_path = os.path.join(
        os.path.expanduser("~"), ".kicad", "9.0", "vibecad_crash.log"
    )
    os.makedirs(os.path.dirname(crash_log_path), exist_ok=True)
    crash_file = open(crash_log_path, "a", encoding="utf-8")
    crash_file.write(f"\n--- VibeCAD Loaded: {datetime.now().isoformat()} ---\n")
    crash_file.flush()
    faulthandler.enable(file=crash_file)
except Exception as e:
    logger.warning(f"Could not enable faulthandler: {e}")

# KiCad imports - these are available in KiCad's Python environment
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False
    logger.warning("pcbnew not available - running outside KiCad")

# Check for eeschema (schematic editor) API
try:
    import eeschema
    EESCHEMA_AVAILABLE = True
except ImportError:
    EESCHEMA_AVAILABLE = False
    # Not logged as warning since it's commonly unavailable

try:
    import wx
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False
    logger.warning("wxPython not available")

from .parsers import PCBParser, PCBData, SchematicParser, SchematicData
from .llm import LLMClient, LLMConfig, IssueExplainer
from .llm.explainer import Explanation, AnswerResponse
from .llm.vertex_client import VertexAIClient
from .config import VibeCADSettings
from .config.settings import LLM_PROVIDER_VERTEX
from .design.intent_router import decide_route
from .plugin_benchmark_entry_mixin import BenchmarkEntryMixin
from .plugin_sleep_guard_mixin import SleepGuardMixin
from .plugin_benchmark_mixin import BenchmarkMixin

# Phase 4: Design assistance imports
try:
    from .design import (
        DesignAgent, DesignAction, DesignActionType,
        LibraryManager, ConnectionManager, BOMExporter,
        AgentLoop, AgentState, AgentLoopConfig,
        CircuitContextBuilder, CircuitSnapshot,
        ComponentWebSearch, ComponentInfo,
    )
    DESIGN_AVAILABLE = True
    DESIGN_IMPORT_ERROR = ""
except ImportError as e:
    DESIGN_AVAILABLE = False
    DESIGN_IMPORT_ERROR = str(e)
    logger.warning("Design module not available")


class VibeCADPlugin(BenchmarkEntryMixin, BenchmarkMixin, SleepGuardMixin):
    """Main VibeCAD plugin class for KiCad.
    
    This plugin provides LLM-assisted design review:
    - Deterministic checks analyze the design
    - LLM explains findings (never modifies)
    - User can ask questions about the design
    - Results displayed in a dockable/separable panel
    
    Phase 3 features:
    - Generates deterministic suggestions for fixes
    - Shows preview overlays before applying
    - Requires explicit user approval
    - LLM explains suggestions (never generates geometry)
    """
    
    def __init__(self):
        self.name = "VibeCAD"
        self.description = "LLM-assisted design review for KiCad"
        self.version = "0.4.0"  # Phase 4
        
        # Components
        self.pcb_data: Optional[PCBData] = None
        self.schematic_data: Optional[SchematicData] = None
        self.active_editor: str = "pcb"  # "pcb" or "schematic"
        self.check_results: List[Any] = []
        self.llm_client: Optional[LLMClient] = None
        # Always set; explainer methods will raise if LLM is unavailable.
        self.explainer: IssueExplainer = IssueExplainer(None)
        # Persisted user settings are needed before design component init.
        self.settings = VibeCADSettings.load()
        self._design_init_error: str = ""
        
        # Phase 4: Design assistance components
        self.design_agent: Optional['DesignAgent'] = None
        self.library_manager: Optional['LibraryManager'] = None
        self.connection_manager: Optional['ConnectionManager'] = None
        self.bom_exporter: Optional['BOMExporter'] = None
        self._agent_loop: Optional['AgentLoop'] = None
        self._active_benchmark: Optional[Dict[str, Any]] = None
        self._sleep_guard_proc: Optional[Any] = None
        self._init_design_components()
        
        # Legacy deterministic checks/suggestions removed in v4-only mode.
        self.checks: List[Any] = []
        
        # UI - now uses the new dockable frame
        self.frame = None

        # Host KiCad window captured at activation time (used for docking).
        self._host_frame = None
        self._host_close_bound = False
        self._host_close_parent = None
        self._shutting_down = False

        # Persisted UI handles for cross-reload single-instance behavior.
        # (KiCad may reload the plugin module; pcbnew usually stays loaded.)
        try:
            if PCBNEW_AVAILABLE:
                existing = getattr(pcbnew, "_VIBECAD_FRAME", None)
                if existing is not None:
                    self.frame = existing
        except Exception:
            pass

        # Verbose state (used for debug logging + richer error messages).
        self._verbose_enabled = False

        # Debug log buffer (shown in UI Debug tab)
        self._debug_log = InMemoryLogBuffer(max_lines=1200)
        # Install capture on root logger so vibecad.* logs are visible in UI.
        try:
            install_debug_log_capture(self._debug_log, level=logging.DEBUG, logger_name='')
        except Exception:
            pass

        # Ensure the Debug tab isn't empty even before any actions run.
        try:
            self._debug_log.append("Debug capture active (VibeCAD).")
        except Exception:
            pass

        # Prevent re-entrancy issues with modal dialogs.
        self._settings_dialog_open = False

        # Docking state (when embedded into KiCad's AUI)
        self._docked_panel = None
        self._docked_parent = None
        self._docked_mgr = None
        
        # (settings already loaded above before design component init)

        # Fixed output mode defaults:
        # - Design output stays concise.
        # - Debug logging stays verbose.
        try:
            self._verbose_enabled = False
            self._apply_debug_verbosity()
        except Exception:
            # Never fail plugin init due to logging configuration issues.
            self._verbose_enabled = False

        # Initialize LLM if configured
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client from environment + persisted settings."""
        def _set_llm(client):
            """Apply LLM client (or None) to all subsystems."""
            self.llm_client = client
            self.explainer = IssueExplainer(client)
            if getattr(self, "library_manager", None) is not None:
                try:
                    self.library_manager.set_llm_client(client)
                except Exception:
                    logger.exception("Failed to update LibraryManager LLM client")
            if DESIGN_AVAILABLE and self.design_agent:
                try:
                    self.design_agent.set_llm_client(client)
                except Exception:
                    logger.exception("Failed to update DesignAgent LLM client")

        try:
            provider = str(getattr(self.settings, "llm_provider", "") or "")

            if provider == LLM_PROVIDER_VERTEX:
                # ── Vertex AI path ────────────────────────────────────────────
                project = str(getattr(self.settings, "vertex_project", "") or "")
                location = str(getattr(self.settings, "vertex_location", "us-central1") or "us-central1")
                creds = str(getattr(self.settings, "vertex_credentials_path", "") or "")
                model = str(getattr(self.settings, "model", "") or "")
                temperature = getattr(self.settings, "temperature", None)
                max_tokens = getattr(self.settings, "max_tokens", None)
                timeout = getattr(self.settings, "timeout", None)
                verify_ssl = bool(getattr(self.settings, "verify_ssl", True))
                ca_bundle = str(getattr(self.settings, "ca_bundle_path", "") or "")
                enable_thinking = bool(getattr(self.settings, "enable_thinking", False))
                thinking_budget = getattr(self.settings, "thinking_budget", None)
                if not project:
                    logger.info("Vertex AI not configured (no GCP project ID)")
                    _set_llm(None)
                    return False
                client = VertexAIClient(
                    project=project,
                    location=location,
                    model=model or "google/gemini-2.0-flash-001",
                    credentials_json_path=creds,
                    verify_ssl=verify_ssl,
                    ca_bundle=ca_bundle,
                    temperature=float(temperature) if temperature is not None else 0.3,
                    max_tokens=int(max_tokens) if max_tokens is not None else 16384,
                    timeout=int(timeout) if timeout is not None else 120,
                    enable_thinking=enable_thinking,
                    thinking_budget=int(thinking_budget) if thinking_budget is not None else 8000,
                )
                _set_llm(client)
                logger.info(f"LLM configured: Vertex AI project={project} location={location} model={client._model}")
                return True
            else:
                # ── OpenAI-compatible path (OpenRouter, GitHub, custom…) ──────
                config = LLMConfig.from_environment()

                # Apply persisted settings as overrides
                try:
                    overrides = self.settings.to_llm_overrides()
                    for key, value in overrides.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                except Exception:
                    logger.exception("Failed to apply persisted settings")

                _saved_timeout = getattr(self.settings, "timeout", None)
                config.timeout = int(_saved_timeout) if _saved_timeout is not None else 30

                if config.is_configured:
                    client = LLMClient(config)
                    _set_llm(client)
                    logger.info(f"LLM configured: {config.model} at {config.api_base}")
                    return True
                else:
                    logger.info("LLM not configured (no API key)")
                    _set_llm(None)
                    return False
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            _set_llm(None)
            return False
    
    def _init_design_components(self):
        """Initialize Phase 4 design assistance components."""
        if not DESIGN_AVAILABLE:
            self._design_init_error = f"Design import failed: {DESIGN_IMPORT_ERROR or 'unknown import error'}"
            logger.info("Design components not available")
            return
        
        try:
            # Initialize library manager for symbol/footprint downloads
            # Enable keyless GitHub search by default in the KiCad plugin runtime.
            # Curated repo downloads are left empty to avoid large/slow downloads.
            self.library_manager = LibraryManager(
                llm_client=self.llm_client,
                github_token=(getattr(self.settings, "github_token", "") or ""),
                enable_easyeda_sources=False,
                enable_snapeda_sources=False,
                enable_github_sources=True,
                enable_github_search=True,
                github_curated_repos=[],
            )
            
            # Initialize connection manager for routing
            self.connection_manager = ConnectionManager()
            
            # Initialize BOM exporter
            self.bom_exporter = BOMExporter()

            # Initialize circuit context builder (smart, compact board snapshots)
            self.circuit_context_builder = CircuitContextBuilder()
            self._circuit_snapshot: Optional['CircuitSnapshot'] = None

            # Initialize component web search (free LCSC/Mouser/Nexar APIs)
            self.component_search = ComponentWebSearch()
            
            # Initialize design agent (LLM-powered interpreter)
            self.design_agent = DesignAgent(self.llm_client)
            self.design_agent.set_library_manager(self.library_manager)
            self.design_agent.set_connection_manager(self.connection_manager)
            self.design_agent.set_bom_exporter(self.bom_exporter)
            self._design_init_error = ""
            
            logger.info("Phase 4 design components initialized")
        except Exception as e:
            logger.exception(f"Failed to initialize design components: {e}")
            self._design_init_error = str(e)
            self.design_agent = None
    
    @property
    def llm_configured(self) -> bool:
        """Check if LLM is configured and available."""
        return self.explainer is not None and self.explainer.is_available
    
    def Run(self):
        """KiCad plugin entry point - called when user activates the plugin."""
        logger.info("VibeCAD plugin activated")
        # Help debug "why didn't my code change apply?" by logging import paths.
        try:
            import vibecad as _v
            from vibecad.design import agent_loop as _al
            logger.info("VibeCAD module path: %s", getattr(_v, "__file__", "?"))
            logger.info("AgentLoop module path: %s", getattr(_al, "__file__", "?"))
        except Exception:
            pass
        
        if WX_AVAILABLE:
            self._show_frame()
        else:
            logger.error("wxPython not available - cannot show UI")

    def set_verbose(self, verbose: bool):
        """Deprecated UI toggle: design output remains concise, debug logs stay verbose."""
        _ = bool(verbose)
        self._verbose_enabled = False
        self._apply_debug_verbosity()

    def _apply_debug_verbosity(self) -> None:
        """Keep debug logging verbose regardless of design-output verbosity."""
        level = logging.DEBUG
        logger.setLevel(level)
        logging.getLogger('vibecad').setLevel(level)
        logging.getLogger('vibecad.llm').setLevel(level)
        logging.getLogger('vibecad.llm.client').setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)

    def _frame_alive(self) -> bool:
        """Return True if self.frame exists and hasn't been destroyed."""
        if self.frame is None or not WX_AVAILABLE:
            return False

        try:
            # Accessing any wx method will raise if the C++ object is gone.
            _ = self.frame.IsShown()
            return True
        except Exception:
            self.frame = None
            return False

    def _wx_window_alive(self, win) -> bool:
        if win is None or not WX_AVAILABLE:
            return False
        try:
            _ = win.IsShown()
            return True
        except Exception:
            return False

    def _clear_persisted_ui_handles(self) -> None:
        if not PCBNEW_AVAILABLE:
            return
        for attr in ("_VIBECAD_FRAME", "_VIBECAD_DOCKED_MGR", "_VIBECAD_DOCKED_PANEL"):
            try:
                setattr(pcbnew, attr, None)
            except Exception:
                pass

    def _on_frame_before_destroy(self) -> None:
        """Best-effort teardown before destroying the floating VibeCAD frame."""
        try:
            self._stop_sleep_guard()
        except Exception:
            pass

        try:
            if self._agent_loop and self._agent_loop.is_running:
                self._agent_loop.pause()
        except Exception:
            pass

        try:
            if self._docked_panel is not None and self._docked_mgr is not None:
                try:
                    pane = self._docked_mgr.GetPane("VibeCAD")
                    if pane.IsOk():
                        try:
                            self._docked_mgr.DetachPane(self._docked_panel)
                        except Exception:
                            pass
                        try:
                            self._docked_mgr.Update()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

        self._docked_panel = None
        self._docked_parent = None
        self._docked_mgr = None
        self.frame = None
        self._clear_persisted_ui_handles()

    def _on_host_frame_close(self, event) -> None:
        """Let KiCad own host shutdown so the docked pane is not torn down early."""
        try:
            if getattr(self, "frame", None) is not None:
                main_panel = getattr(self.frame, "main_panel", None)
                if main_panel is not None:
                    main_panel.Refresh()
                    main_panel.Update()
        except Exception:
            pass
        try:
            event.Skip()
        except Exception:
            pass

    def _raise_existing_ui(self) -> bool:
        """If the UI already exists, bring it to front and return True."""
        if not WX_AVAILABLE:
            return False

        # If docked into KiCad AUI, ensure the docked pane is visible.
        try:
            if self._docked_mgr is not None:
                pane = self._docked_mgr.GetPane("VibeCAD")
                if pane.IsOk():
                    try:
                        pane.Show(True)
                        self._docked_mgr.Update()
                    except Exception:
                        pass
                    try:
                        if self._docked_panel is not None:
                            self._docked_panel.SetFocus()
                    except Exception:
                        pass
                    return True
                self._docked_panel = None
                self._docked_parent = None
                self._docked_mgr = None
                self._clear_persisted_ui_handles()
        except Exception:
            pass

        if not self._frame_alive():
            return False

        try:
            # Make sure it is visible and focused.
            try:
                if self.frame.IsIconized():
                    self.frame.Restore()
            except Exception:
                pass
            try:
                if not self.frame.IsShown():
                    self.frame.Show(True)
            except Exception:
                pass
            try:
                self.frame.Raise()
            except Exception:
                pass
            try:
                self.frame.SetFocus()
            except Exception:
                pass
            return True
        except Exception:
            return False
    
    def _show_frame(self):
        """Show the plugin frame (dockable/separable window)."""
        # Reuse existing UI if it already exists.
        if self._raise_existing_ui():
            return

        from .ui.dockable_frame import VibeCADFrame
        
        # Get parent window from KiCad.
        # Important: when KiCad launcher + PCB Editor are both open, picking
        # `wx.GetTopLevelWindows()[0]` can incorrectly parent to the launcher.
        parent = None
        if WX_AVAILABLE:
            try:
                focused = wx.Window.FindFocus()
                if focused is not None:
                    parent = focused.GetTopLevelParent()
            except Exception:
                parent = None

        if parent is None and PCBNEW_AVAILABLE:
            try:
                get_frame = getattr(pcbnew, "GetFrame", None)
                if callable(get_frame):
                    parent = get_frame()
            except Exception:
                parent = None

        if parent is None and WX_AVAILABLE:
            try:
                parent = wx.GetActiveWindow()
            except Exception:
                parent = None

        if parent is None and WX_AVAILABLE:
            try:
                wins = wx.GetTopLevelWindows()
                # Prefer a visible window; last resort is the first.
                for w in wins:
                    try:
                        if w and w.IsShown():
                            parent = w
                            break
                    except Exception:
                        continue
                if parent is None:
                    parent = wins[0] if wins else None
            except Exception:
                parent = None

        # Cache the host frame we were launched from (used for docking).
        if self._wx_window_alive(parent):
            self._host_frame = parent
            try:
                if (not self._host_close_bound) or (self._host_close_parent is not parent):
                    parent.Bind(wx.EVT_CLOSE, self._on_host_frame_close)
                    self._host_close_bound = True
                    self._host_close_parent = parent
            except Exception:
                pass
        
        try:
            self.frame = VibeCADFrame(
                parent,
                title="VibeCAD Design Review",
                on_toggle_dock=self._on_toggle_dock,
                on_open_settings=self._on_open_settings,
                # Phase 4: Design agent callbacks
                on_design_message=self._on_design_message,
                on_run_benchmark=self._on_run_benchmark,
                on_approve_action=self._on_approve_design_action,
                on_reject_action=self._on_reject_design_action,
                on_new_chat=self._on_new_chat,
                on_llm_controls_changed=self._on_design_llm_controls_changed,
                # Debug tab callbacks
                on_get_debug_text=self.get_debug_text,
                on_clear_debug=self.clear_debug_text,
                on_before_destroy=self._on_frame_before_destroy,
            )
        except Exception:
            self.frame = None
            logger.exception("Failed to construct VibeCAD frame")
            try:
                self._clear_persisted_ui_handles()
            except Exception:
                pass
            try:
                if WX_AVAILABLE:
                    wx.MessageBox(
                        "VibeCAD failed to open. Check KiCad's Scripting Console for traceback.",
                        "VibeCAD Error",
                        wx.OK | wx.ICON_ERROR,
                    )
            except Exception:
                pass
            return

        if self.frame is None:
            logger.error("VibeCAD frame was not created")
            return
        
        # Update LLM status indicator
        self.frame.set_llm_status(self.llm_configured)
        try:
            self.frame.set_llm_controls(
                str(getattr(self.settings, "model", "") or ""),
                bool(getattr(self.settings, "enable_thinking", False)),
            )
        except Exception:
            pass

        try:
            logger.info("VibeCAD UI opened")
        except Exception:
            pass

        # Give the window a stable name so we can locate it across reloads.
        try:
            self.frame.SetName("VibeCADFrame")
        except Exception:
            pass

        # Persist handle on pcbnew for cross-reload reuse.
        try:
            if PCBNEW_AVAILABLE:
                setattr(pcbnew, "_VIBECAD_FRAME", self.frame)
        except Exception:
            pass

        # Wire pause callback for the agent loop
        try:
            self.frame.set_pause_callback(self._on_pause_agent)
        except Exception:
            logger.exception("Failed to wire pause callback")

        try:
            self.frame.Show()
        except Exception:
            logger.exception("Failed to show VibeCAD frame")
            try:
                self.frame.Destroy()
            except Exception:
                pass
            self.frame = None
            try:
                self._clear_persisted_ui_handles()
            except Exception:
                pass
            return

        # Emulate the geometry updates that happen on a manual window resize.
        try:
            self.frame.Layout()
            self.frame.SendSizeEvent()
            wx.CallAfter(self.frame.SendSizeEvent)
            wx.CallLater(90, self.frame.SendSizeEvent)
        except Exception:
            pass

        try:
            dp = getattr(self.frame, "design_panel", None)
            if dp is not None and hasattr(dp, "_force_chat_layout"):
                wx.CallAfter(dp._force_chat_layout, "plugin-frame-show")
                wx.CallLater(120, dp._force_chat_layout, "plugin-frame-show-delayed")
        except Exception:
            pass

    def get_debug_text(self) -> str:
        """Return captured debug output for the UI."""
        try:
            return self._debug_log.get_text()
        except Exception:
            return ""

    def clear_debug_text(self) -> None:
        """Clear captured debug output."""
        try:
            self._debug_log.clear()
        except Exception:
            pass

    def _on_open_settings(self):
        """Open settings dialog and apply changes."""
        if not WX_AVAILABLE or self.frame is None:
            return

        # Avoid opening multiple modal dialogs if the user double-clicks
        # or the UI generates duplicate events.
        if self._settings_dialog_open:
            return
        self._settings_dialog_open = True

        try:
            from .ui.settings_dialog import SettingsDialog
        except Exception:
            logger.exception("Failed to import SettingsDialog")
            return

        # When docked, self.frame is intentionally hidden and no longer owns
        # the visible UI panel. Parenting a modal dialog to a hidden/empty
        # frame can create focus/ghost-window issues on macOS.
        dlg_parent = None
        try:
            if self._docked_parent is not None:
                dlg_parent = self._docked_parent
        except Exception:
            dlg_parent = None

        if dlg_parent is None and PCBNEW_AVAILABLE:
            try:
                get_frame = getattr(pcbnew, "GetFrame", None)
                if callable(get_frame):
                    dlg_parent = get_frame()
            except Exception:
                dlg_parent = None

        if dlg_parent is None:
            try:
                # Prefer an on-screen toplevel window if available.
                wins = wx.GetTopLevelWindows()
                dlg_parent = wins[0] if wins else None
            except Exception:
                dlg_parent = None

        if dlg_parent is None:
            dlg_parent = self.frame

        dlg = SettingsDialog(dlg_parent, settings=self.settings)
        try:
            res = dlg.ShowModal()
            if res == wx.ID_OK:
                new_settings = dlg.settings
                new_settings.save()
                self.settings = new_settings
                self._init_llm()
                try:
                    if getattr(self, "library_manager", None) is not None:
                        self.library_manager.github_token = str(getattr(self.settings, "github_token", "") or "").strip()
                        self.library_manager._warned_no_github_token = False
                        self.library_manager.enable_easyeda_sources = False
                        self.library_manager.enable_snapeda_sources = False
                except Exception:
                    logger.exception("Failed to apply library-source settings")
                try:
                    self.frame.set_llm_status(self.llm_configured)
                except Exception:
                    pass

                try:
                    if self.frame and hasattr(self.frame, 'set_llm_controls'):
                        self.frame.set_llm_controls(
                            str(getattr(self.settings, 'model', '') or ''),
                            bool(getattr(self.settings, 'enable_thinking', False)),
                        )
                except Exception:
                    pass

                # Apply Design tab UX settings immediately.
                try:
                    if self.frame and hasattr(self.frame, 'set_thinking_output_enabled'):
                        self.frame.set_thinking_output_enabled(
                            bool(getattr(self.settings, 'thinking_output_enabled', True))
                        )
                except Exception:
                    pass
        finally:
            try:
                dlg.Destroy()
            except Exception:
                pass
            self._settings_dialog_open = False

    def _on_toggle_dock(self):
        """Best-effort docking into KiCad.

        KiCad's main window uses AUI internally; if we can locate its
        AuiManager, we can dock our panel there. If not available, we fall
        back to pin/unpin behavior in the frame.
        """
        if not WX_AVAILABLE or not PCBNEW_AVAILABLE:
            return False

        try:
            import wx.aui as aui
        except Exception:
            return False

        # Find the KiCad host frame to dock into.
        # Prefer the frame we were launched from; this avoids docking into
        # the KiCad launcher when both are open.
        parent = None
        try:
            if self._wx_window_alive(self._host_frame):
                parent = self._host_frame
        except Exception:
            parent = None

        if parent is None:
            try:
                get_frame = getattr(pcbnew, "GetFrame", None)
                if callable(get_frame):
                    parent = get_frame()
            except Exception:
                parent = None

        if parent is None:
            try:
                wins = wx.GetTopLevelWindows()
                # Prefer a shown window that's not our own frame.
                for w in wins:
                    try:
                        if w is None or w is self.frame:
                            continue
                        if hasattr(w, "IsShown") and w.IsShown():
                            parent = w
                            break
                    except Exception:
                        continue
                if parent is None:
                    parent = wins[0] if wins else None
            except Exception:
                parent = None

        if parent is None or self.frame is None:
            return False

        # Try to locate an existing AuiManager on the KiCad frame.
        mgr = None

        # Preferred: wx.aui provides a manager lookup in some builds
        get_mgr_for_window = getattr(aui.AuiManager, "GetManager", None)
        if callable(get_mgr_for_window):
            try:
                mgr = get_mgr_for_window(parent)
            except Exception:
                mgr = None

        if mgr is None:
            get_mgr_func = getattr(aui, "GetManager", None)
            if callable(get_mgr_func):
                try:
                    mgr = get_mgr_func(parent)
                except Exception:
                    mgr = None

        for attr in ("_mgr", "mgr", "m_mgr", "aui_mgr", "_auiManager"):
            candidate = getattr(parent, attr, None)
            if candidate is not None:
                mgr = candidate
                break
        if mgr is None:
            get_mgr = getattr(parent, "GetAuiManager", None)
            if callable(get_mgr):
                try:
                    mgr = get_mgr()
                except Exception:
                    mgr = None

        if mgr is None:
            return False

        pane_name = "VibeCAD"

        # Toggle: if already docked, undock back into the floating frame.
        if self._docked_panel is not None and self._docked_mgr is not None:
            try:
                try:
                    pane = self._docked_mgr.GetPane(pane_name)
                    if pane.IsOk():
                        self._docked_mgr.DetachPane(self._docked_panel)
                        self._docked_mgr.Update()
                except Exception:
                    # Best-effort detach; continue restoring UI
                    pass

                # Restore into the floating frame
                self._docked_panel.Reparent(self.frame)

                # Recreate the frame's internal AUI manager and re-add main_panel
                try:
                    import wx.aui as aui  # already imported above, but keep local safety
                    self.frame._mgr = aui.AuiManager(self.frame)
                    self.frame._mgr.AddPane(
                        self._docked_panel,
                        aui.AuiPaneInfo().Name("main").CenterPane().PaneBorder(False),
                    )
                    self.frame._mgr.Update()
                except Exception:
                    logger.exception("Failed to restore frame AUI manager")

                try:
                    self.frame.Layout()
                    self.frame.Refresh()
                except Exception:
                    pass

                self.frame.Show()
                self._docked_panel = None
                self._docked_parent = None
                self._docked_mgr = None
                try:
                    if PCBNEW_AVAILABLE:
                        setattr(pcbnew, "_VIBECAD_DOCKED_MGR", None)
                        setattr(pcbnew, "_VIBECAD_DOCKED_PANEL", None)
                except Exception:
                    pass
                return True
            except Exception:
                logger.exception("Failed to undock from KiCad")
                return False

        # Dock by reparenting the frame's main_panel into the KiCad frame.
        # This is a pragmatic approach that keeps UI code in one place.
        try:
            panel = getattr(self.frame, "main_panel", None)
            if panel is None:
                return False

            # If anything fails mid-dock, we MUST restore the panel back into
            # the floating frame; otherwise the user sees a blank/glitched frame.
            def _rollback_dock():
                try:
                    try:
                        pane = mgr.GetPane(pane_name)
                        if pane.IsOk():
                            try:
                                mgr.DetachPane(panel)
                            except Exception:
                                pass
                            try:
                                mgr.Update()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        panel.Reparent(self.frame)
                    except Exception:
                        pass

                    try:
                        import wx.aui as aui
                        self.frame._mgr = aui.AuiManager(self.frame)
                        self.frame._mgr.AddPane(
                            panel,
                            aui.AuiPaneInfo().Name("main").CenterPane().PaneBorder(False),
                        )
                        self.frame._mgr.Update()
                    except Exception:
                        logger.exception("Failed to rollback docking")

                    try:
                        self.frame.Layout()
                        self.frame.Refresh()
                    except Exception:
                        pass

                    try:
                        self.frame.Show()
                    except Exception:
                        pass
                except Exception:
                    pass

            # Uninitialize the frame's internal AUI manager before reparenting.
            try:
                if getattr(self.frame, "_mgr", None) is not None:
                    self.frame._mgr.UnInit()
            except Exception:
                pass

            panel.Reparent(parent)

            existing = mgr.GetPane(pane_name)
            if existing.IsOk():
                try:
                    mgr.DetachPane(panel)
                except Exception:
                    pass

            mgr.AddPane(
                panel,
                aui.AuiPaneInfo()
                .Name(pane_name)
                .Caption("VibeCAD")
                .Right()
                .BestSize(700, 600)
                .FloatingSize(700, 600)
                .Dockable(True)
                .Floatable(True)
                .CloseButton(True)
                .PinButton(True)
                .MinimizeButton(True)
                .MaximizeButton(True),
            )
            mgr.Update()

            try:
                panel.Refresh()
                panel.Update()
            except Exception:
                pass

            self._docked_panel = panel
            self._docked_parent = parent
            self._docked_mgr = mgr

            # Persist docked handles so a future module reload can re-show it.
            try:
                if PCBNEW_AVAILABLE:
                    setattr(pcbnew, "_VIBECAD_DOCKED_MGR", mgr)
                    setattr(pcbnew, "_VIBECAD_DOCKED_PANEL", panel)
            except Exception:
                pass

            # Hide the old frame chrome; the content now lives as a docked pane.
            self.frame.Hide()
            return True
        except Exception:
            logger.exception("Failed to dock into KiCad")
            try:
                _rollback_dock()
            except Exception:
                pass
            return False
    
    def _on_run_checks(self):
        """Handle run checks request from UI."""
        if self.frame:
            self.frame.add_design_response("ℹ️ Deterministic checks are removed in v4-only mode.")
    
    # === Phase 3: Suggestion Action Handlers ===
    
    def _on_preview_suggestion(self, suggestion: Any):
        """Show preview overlay for a suggestion."""
        if self.frame:
            self.frame.add_design_response("ℹ️ Suggestion previews are removed in v4-only mode.")
    
    def _on_apply_suggestion(self, suggestion: Any) -> Dict[str, Any]:
        """Apply a suggestion after user approval."""
        if self.frame:
            self.frame.add_design_response("ℹ️ Suggestions are removed in v4-only mode.")
        return {"success": False, "message": "suggestions removed"}
    
    def _on_dismiss_suggestion(self, suggestion: Any):
        """Dismiss a suggestion."""
        if self.frame:
            self.frame.add_design_response("ℹ️ Suggestions are removed in v4-only mode.")
    
    def _on_explain_suggestion(self, suggestion: Any) -> str:
        """Get LLM explanation for a suggestion."""
        return "Suggestion explanations are removed in v4-only mode."
    
    def _on_hide_all_previews(self):
        """Hide all preview overlays."""
        return
    
    # === Phase 4: Design Agent Handlers ===

    def _on_new_chat(self):
        """Reset the agent for a fresh conversation."""
        # Stop any running agent loop
        if self._agent_loop:
            try:
                self._agent_loop.pause()
            except Exception:
                pass
            self._agent_loop = None
        # Clear agent conversation history
        if self.design_agent:
            try:
                self.design_agent.clear_history()
            except Exception:
                pass
        # Reset UI state
        if self.frame:
            try:
                self.frame.set_agent_running(False)
            except Exception:
                pass
        self._stop_sleep_guard()
        self._active_benchmark = None
        logger.info("New chat session started")

    def _on_design_message(self, message: str):
        """Handle design message from the copilot-style UI.

        If an agent loop is already running and awaiting input (clarifying
        question), feed the answer back to the loop.  Otherwise, start a
        new agent loop for the user's goal.
        """
        if not DESIGN_AVAILABLE or not self.design_agent:
            if self.frame:
                detail = ""
                try:
                    detail = (getattr(self, "_design_init_error", "") or "").strip()
                except Exception:
                    detail = ""
                self.frame.add_design_response(
                    "\u274c Design assistance is not available. "
                    + (f"Reason: {detail}" if detail else "Please check the installation.")
                )
            return False

        # If the agent loop is awaiting user input, feed the answer
        if (self._agent_loop and
                self._agent_loop.state == AgentState.AWAITING_INPUT):
            self._start_sleep_guard()
            self._agent_loop.resume(message)
            return True

        # If paused, resume with the new message
        if (self._agent_loop and
                self._agent_loop.state == AgentState.PAUSED):
            self._start_sleep_guard()
            self._agent_loop.resume(message)
            if self.frame:
                wx.CallAfter(self.frame.set_agent_running, True)
            return True

        # New message: use LLM intent routing (in background) so the UI stays responsive.
        self._route_design_message_async(message)
        return False

    def _on_design_llm_controls_changed(self, model: str, extended_reasoning: bool) -> None:
        """Persist live model/reasoning changes coming from the Design tab."""
        try:
            self.settings.model = str(model or "").strip() or str(getattr(self.settings, "model", "") or "")
            self.settings.enable_thinking = bool(extended_reasoning)
            self.settings.thinking_budget = None
            self.settings.save()
        except Exception:
            logger.exception("Failed to persist LLM control changes")
            return

        try:
            self._init_llm()
        except Exception:
            logger.exception("Failed to reinitialize LLM after control change")

        try:
            if self.frame:
                self.frame.set_llm_status(self.llm_configured)
                self.frame.set_llm_controls(self.settings.model, self.settings.enable_thinking)
        except Exception:
            pass

    def _route_design_message_async(self, message: str) -> None:
        """Route message to Q&A, direct-tool mode, or AgentLoop (non-blocking)."""
        if not WX_AVAILABLE:
            try:
                decision = decide_route(self.llm_client, message)
                if decision.route == 'qa':
                    self._answer_design_question_async(message)
                elif decision.route == 'direct_tool':
                    # Non-wx fallback keeps behavior functional.
                    self._start_agent_loop(message)
                else:
                    self._start_agent_loop(message)
            except Exception:
                logger.exception("Failed to route design message (non-wx)")
            return

        def worker():
            try:
                decision = decide_route(self.llm_client, message)
                if decision.route == 'qa':
                    self._answer_design_question_async(message)
                    try:
                        if self.frame:
                            wx.CallAfter(self.frame.set_agent_running, False)
                    except Exception:
                        pass
                    return

                if decision.route == 'direct_tool':
                    self._handle_direct_tool_message_async(message)
                    try:
                        if self.frame:
                            wx.CallAfter(self.frame.set_agent_running, False)
                    except Exception:
                        pass
                    return

                # Agent route: start the loop on the GUI thread (pcbnew safety).
                try:
                    if self.frame:
                        wx.CallAfter(self.frame.set_agent_running, True)
                except Exception:
                    pass
                wx.CallAfter(self._start_agent_loop, message)
            except Exception:
                logger.exception("Failed to route design message")
                try:
                    if self.frame:
                        wx.CallAfter(self.frame.add_design_response, "❌ LLM routing failed. Check your API/model settings and retry.")
                        wx.CallAfter(self.frame.set_agent_running, False)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _handle_direct_tool_message_async(self, message: str) -> None:
        """Handle a bounded tool request without launching the full AgentLoop."""
        if not WX_AVAILABLE or not self.frame:
            return
        if not DESIGN_AVAILABLE or not self.design_agent:
            wx.CallAfter(self.frame.add_design_response, "❌ Direct tool mode is unavailable because the design module failed to load.")
            return
        if not self.llm_client or not self.llm_client.is_available:
            wx.CallAfter(self.frame.add_design_response, "❌ LLM is not available/configured. Configure it in ⚙ Settings and retry.")
            return

        # Load data before planning/execution so direct tool requests can reason
        # over up-to-date board/schematic state.
        try:
            self._load_pcb_data()
            self._load_schematic_data()
        except Exception:
            logger.debug("Pre-load board data failed for direct-tool mode", exc_info=True)

        active_editor = self._detect_active_editor()
        project_dir = ''
        board = None
        try:
            if PCBNEW_AVAILABLE:
                board = pcbnew.GetBoard()
                if board:
                    fn = board.GetFileName() or ''
                    if fn:
                        project_dir = str(Path(fn).expanduser().resolve().parent)
        except Exception:
            pass

        outline_context = self._extract_outline_context(board=board, pcb_data=self.pcb_data)
        context = {
            'active_editor': active_editor,
            'pcb_data': self.pcb_data,
            'schematic_data': self.schematic_data,
            'verbose': False,
            'project_dir': project_dir,
            **outline_context,
        }

        auto_execute_types = {
            DesignActionType.SEARCH_PART,
            DesignActionType.SEARCH_WEB,
            DesignActionType.LOOKUP_DATASHEET,
            DesignActionType.RUN_DRC,
            DesignActionType.RUN_ERC,
            DesignActionType.EXPORT_BOM,
        }
        yolo_enabled = bool(getattr(self.settings, 'yolo_auto_apply', False))

        def _execute_action_blocking(action: 'DesignAction') -> Optional['DesignAction']:
            try:
                action.approved = True
            except Exception:
                pass
            done = threading.Event()
            result_box: Dict[str, Any] = {"result": None}

            def _on_result(result):
                result_box["result"] = result
                done.set()

            self._execute_action_on_gui(action, context, _on_result)
            if not done.wait(timeout=120):
                return None
            res = result_box.get("result")
            return res if res is not None else action

        def worker():
            try:
                wx.CallAfter(self.frame.set_design_thinking, True)

                assistant_message, request = self.design_agent.chat(message, context)
                actions = list(getattr(request, 'interpreted_actions', []) or [])

                if assistant_message:
                    wx.CallAfter(self.frame.add_design_response, assistant_message)

                if not actions:
                    # Clarifying questions remain valid in YOLO mode.
                    # If planning returns no actions, we wait for user input.
                    return

                # Direct-tool mode is intentionally bounded.
                max_direct_actions = 3
                if len(actions) > max_direct_actions:
                    actions = actions[:max_direct_actions]
                    wx.CallAfter(
                        self.frame.add_design_response,
                        f"ℹ️ Direct tool mode is limited to {max_direct_actions} actions per request; queued the first {max_direct_actions}.",
                    )

                yolo_notice_emitted = False
                for action in actions:
                    atype = getattr(action, "action_type", DesignActionType.UNKNOWN)
                    try:
                        preview = self.design_agent.create_preview(action, context)
                    except Exception:
                        preview = str(getattr(action, "preview_text", "") or "")

                    if atype not in auto_execute_types and not yolo_enabled:
                        wx.CallAfter(
                            self.frame.add_design_action_preview,
                            atype.name,
                            str(getattr(action, "description", "") or atype.name),
                            preview,
                            action,
                        )
                        continue

                    if atype not in auto_execute_types and yolo_enabled and not yolo_notice_emitted:
                        yolo_notice_emitted = True
                        wx.CallAfter(
                            self.frame.add_design_response,
                            "⚠️ YOLO mode is enabled — auto-applying direct-tool actions without approval.",
                        )

                    result = _execute_action_blocking(action)
                    if result is None:
                        wx.CallAfter(self.frame.add_design_response, f"❌ {atype.name} timed out.")
                        continue

                    ok = bool(getattr(result, "success", False))
                    msg = str(getattr(result, "result_message", "") or "").strip()
                    if msg:
                        if msg.lstrip().startswith(("✅", "❌")):
                            wx.CallAfter(self.frame.add_design_response, msg)
                        else:
                            wx.CallAfter(self.frame.add_design_response, f"{'✅' if ok else '❌'} {msg}")
                    else:
                        wx.CallAfter(
                            self.frame.add_design_response,
                            f"{'✅' if ok else '❌'} {atype.name} {'completed' if ok else 'failed'}.",
                        )
            except Exception as e:
                logger.exception("Direct tool handling failed")
                wx.CallAfter(self.frame.add_design_response, f"❌ Direct tool execution failed: {e}")
            finally:
                try:
                    wx.CallAfter(self.frame.set_design_thinking, False)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _qa_recent_conversation_text(self, max_turns: int = 8) -> str:
        """Build compact recent chat context for follow-up Q&A continuity."""
        if not self.design_agent:
            return ""
        try:
            entries = self.design_agent.recent_history(max_turns=max_turns * 2)
        except Exception:
            return ""
        if not entries:
            return ""

        lines: List[str] = []
        for row in entries:
            role = str(row.get("role", "") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            text = str(row.get("content", "") or "").strip()
            if not text:
                continue
            if "PROPOSED_ACTIONS:" in text:
                text = text.split("PROPOSED_ACTIONS:", 1)[0].strip()
            if len(text) > 700:
                text = text[:700] + "\n...[truncated]"
            lines.append(f"[{role.upper()}] {text}")

        return "\n".join(lines[-max_turns:]).strip()

    def _answer_design_question_async(self, question: str) -> None:
        """Answer a user question in the Design tab without starting AgentLoop."""
        if not WX_AVAILABLE or not self.frame:
            return
        if not self.llm_client or not self.llm_client.is_available:
            wx.CallAfter(self.frame.add_design_response, "❌ LLM is not available/configured. Configure it in ⚙ Settings and retry.")
            return

        recent_conversation = self._qa_recent_conversation_text(max_turns=8)
        if self.design_agent:
            try:
                self.design_agent.record_history_turn("user", question)
            except Exception:
                pass

        # Load PCB/schematic data NOW on the GUI thread (pcbnew not thread-safe).
        try:
            self._load_pcb_data()
            self._load_schematic_data()
        except Exception:
            logger.debug("Pre-load board data failed for Q&A", exc_info=True)

        def worker():
            try:
                wx.CallAfter(self.frame.set_design_thinking, True)

                # ── Build circuit context from pre-loaded data (no pcbnew) ──
                circuit_context_str = ""
                if DESIGN_AVAILABLE and hasattr(self, 'circuit_context_builder'):
                    try:
                        snapshot = self.circuit_context_builder.build_snapshot(
                            self.pcb_data, self.schematic_data,
                        )
                        self._circuit_snapshot = snapshot
                        circuit_context_str = self.circuit_context_builder.build_context_for_query(
                            question, snapshot,
                        )
                    except Exception:
                        logger.debug("Failed to build circuit context for Q&A", exc_info=True)

                # ── Optional web search for component data ──
                web_context_str = ""
                if DESIGN_AVAILABLE and hasattr(self, 'component_search'):
                    try:
                        import re as _re
                        refs_in_q = _re.findall(r'\b([A-Z]{1,3}\d{1,4})\b', question.upper())
                        search_terms = []
                        for ref in refs_in_q:
                            if self._circuit_snapshot and ref in self._circuit_snapshot.components:
                                comp = self._circuit_snapshot.components[ref]
                                if comp.value and comp.value not in ('~', ref):
                                    search_terms.append(comp.value)
                        # Check if user mentions a part number directly
                        mpn_match = _re.search(
                            r'\b([A-Z]{2,}[\dA-Z]*[-/][\dA-Z]+)\b', question, _re.IGNORECASE,
                        )
                        if mpn_match:
                            search_terms.append(mpn_match.group(1))

                        if search_terms:
                            for term in search_terms[:2]:
                                try:
                                    web_data = self.component_search.search_for_llm_context(term, limit=2)
                                    if web_data:
                                        web_context_str += f"\n\n### Web Data for '{term}':\n{web_data}"
                                except Exception:
                                    pass
                    except Exception:
                        logger.debug("Web search for Q&A failed", exc_info=True)

                from .llm.client import LLMMessage

                system_prompt = (
                    "You are VibeCAD's Q&A assistant. The user is asking an informational question "
                    "about their PCB/schematic design. You have real-time access to their board state "
                    "(component values, net connections, footprints) provided below.\n\n"
                    "Answer directly and concisely. Reference specific component designators (R1, U3, etc.) "
                    "and net names from the board data when relevant. "
                    "Use the recent conversation context to resolve follow-up answers. "
                    "If the latest user message is short (for example a numbered reply like '1. TQFP-32'), "
                    "treat it as a continuation of the most recent assistant clarification question. "
                    "If critical context is missing, ask 1-3 clarifying questions "
                    "and provide typical ranges or rules of thumb. "
                    "Do NOT propose tool/actions or start multi-step planning."
                )

                prompt_parts = []
                if recent_conversation:
                    prompt_parts.append(f"--- RECENT CONVERSATION ---\n{recent_conversation}")
                prompt_parts.append(f"--- LATEST USER MESSAGE ---\n{question.strip()}")
                if circuit_context_str:
                    prompt_parts.append(f"\n\n--- CURRENT BOARD STATE ---\n{circuit_context_str}")
                if web_context_str:
                    prompt_parts.append(f"\n\n--- COMPONENT WEB DATA ---{web_context_str}")
                user_prompt = "\n".join(prompt_parts)

                resp = self.llm_client.chat([LLMMessage(role='user', content=user_prompt)], system_prompt=system_prompt)
                answer = (resp.content or "").strip()
                if not answer:
                    raise ValueError("LLM returned empty answer content.")

                if self.design_agent:
                    try:
                        self.design_agent.record_history_turn("assistant", answer)
                    except Exception:
                        pass
                wx.CallAfter(self.frame.add_design_response, answer)
            except Exception as e:
                logger.exception("Design Q&A failed")
                wx.CallAfter(self.frame.add_design_response, f"❌ Error answering question: {e}")
            finally:
                try:
                    wx.CallAfter(self.frame.set_design_thinking, False)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()





















    @staticmethod
    def _bench_norm_part_id(value: str) -> str:
        s = str(value or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    @classmethod
    def _bench_id_variants(cls, value: str) -> Set[str]:
        raw = str(value or "").strip()
        if not raw:
            return set()
        out: Set[str] = set()
        n0 = cls._bench_norm_part_id(raw)
        if n0:
            out.add(n0)
        if ":" in raw:
            tail = raw.split(":", 1)[1].strip()
            nt = cls._bench_norm_part_id(tail)
            if nt:
                out.add(nt)
        return out

    @classmethod
    def _bench_ids_match(cls, a: str, b: str) -> bool:
        av = cls._bench_id_variants(a)
        bv = cls._bench_id_variants(b)
        if not av or not bv:
            return False
        for x in av:
            for y in bv:
                if x == y:
                    return True
                # Allow fuzzy suffix/prefix match for library-qualified IDs.
                if len(x) >= 10 and len(y) >= 10 and (x in y or y in x):
                    return True
        return False








































    def _start_agent_loop(self, message: str, benchmark: Optional[Dict[str, Any]] = None):
        """Create and wire an AgentLoop, then start it."""
        self._load_pcb_data()
        self._load_schematic_data()
        active_editor = self._detect_active_editor()

        pcb_filename = ''
        project_dir = ''
        try:
            if PCBNEW_AVAILABLE:
                board = pcbnew.GetBoard()
                if board:
                    pcb_filename = board.GetFileName() or ''
            if pcb_filename:
                try:
                    project_dir = str(Path(pcb_filename).expanduser().resolve().parent)
                except Exception:
                    project_dir = str(Path(pcb_filename).expanduser().parent)
        except Exception:
            pass

        initial_context = {
            'active_editor': active_editor,
            'pcb_data': self.pcb_data,
            'schematic_data': self.schematic_data,
            'verbose': False,
            'project_dir': project_dir,
        }
        if benchmark:
            initial_context.update(
                {
                    "benchmark_mode": True,
                    "benchmark_bom_only": bool(benchmark.get("bom_only_mode", False)),
                    "benchmark_workflow_version": "v4",
                }
            )

        config = AgentLoopConfig(
            max_iterations=50,
            max_drc_retries=10,
            auto_approve_readonly=True,
            yolo_auto_apply=bool(getattr(self.settings, 'yolo_auto_apply', False)),
            # Component-by-component replanning is intentionally *not* forced by YOLO.
            # We instead default to batch placement to keep the UI/logs manageable.
            component_by_component_placement=bool(
                getattr(self.settings, 'component_by_component_placement', False)
            ),
            placement_batch_size=int(getattr(self.settings, 'placement_batch_size', 0) or 0),
            require_full_workflow=(False if benchmark else None),
        )
        self._agent_loop = AgentLoop(self.design_agent, config)

        # Wire callbacks (all use wx.CallAfter -- loop runs on background thread)
        self._agent_loop.set_ui_message_callback(
            (lambda text: None) if benchmark else (lambda text: wx.CallAfter(self.frame.add_design_response, text))
        )
        self._agent_loop.set_ui_thinking_callback(
            (
                (lambda text: wx.CallAfter(self.frame.add_thinking_message, text))
                if bool(getattr(self.settings, 'thinking_output_enabled', True))
                else (lambda text: None)
            )
        )
        self._agent_loop.set_ui_action_preview_callback(
            lambda atype, desc, prev, act: wx.CallAfter(
                self.frame.add_design_action_preview, atype, desc, prev, act
            )
        )
        self._agent_loop.set_ui_response_callback(
            lambda text: self._forward_agent_response_with_benchmark_capture(text, benchmark=benchmark)
        )
        self._agent_loop.set_phase_complete_callback(
            lambda phase_name, phase_result: self._benchmark_queue_phase_score(
                benchmark, self._agent_loop, phase_name, phase_result
            )
        )

        def on_state_change(new_state):
            self._benchmark_record_state_transition(benchmark, new_state)
            if benchmark and self._agent_loop is not None:
                self._benchmark_flush_pending_phase_scores(benchmark, self._agent_loop, new_state)
            def _handle():
                if new_state in (AgentState.DONE, AgentState.ERROR, AgentState.PAUSED, AgentState.AWAITING_INPUT):
                    self._stop_sleep_guard()
                if not self.frame:
                    return
                if new_state in (AgentState.DONE, AgentState.ERROR, AgentState.PAUSED):
                    self.frame.set_agent_running(False)
                    self.frame.set_design_thinking(False)
                elif new_state == AgentState.AWAITING_INPUT:
                    self.frame.set_agent_awaiting_input(True)
                    if benchmark and self._agent_loop is not None:
                        reply = self._benchmark_auto_clarify_reply(benchmark)
                        if reply:
                            try:
                                self.frame.set_agent_awaiting_input(False)
                            except Exception:
                                pass
                            try:
                                self._agent_loop.resume(reply)
                            except Exception:
                                logger.exception("Benchmark auto-clarify resume failed")
                            return
                elif new_state == AgentState.AWAITING_APPROVAL:
                    self.frame.set_design_thinking(False)
                elif new_state in (AgentState.PLANNING, AgentState.EXECUTING,
                                   AgentState.OBSERVING):
                    self._start_sleep_guard()
                    self.frame.set_design_thinking(True)
                    if benchmark and self._agent_loop is not None:
                        ff = self._benchmark_fail_fast_before_geom(benchmark, self._agent_loop)
                        if isinstance(ff, dict):
                            warn_only = bool(benchmark.get("fail_fast_warn_only", False))
                            try:
                                gate = str(ff.get("gate", "") or "benchmark_fail_fast")
                                msg = str(ff.get("message", "") or "Benchmark fail-fast triggered")
                                if warn_only:
                                    self.frame.add_design_response(
                                        f"⚠️ Benchmark fail-fast warning ({gate}): {msg} (continuing run)"
                                    )
                                else:
                                    self.frame.add_design_response(f"❌ Benchmark fail-fast ({gate}): {msg}")
                            except Exception:
                                pass
                            if not warn_only:
                                try:
                                    self._agent_loop.stop()
                                except Exception:
                                    logger.exception("Benchmark fail-fast stop failed")
                                return
                # Benchmark run: finalize only on terminal states that naturally end execution.
                # Keep PAUSED responsive (no heavy report build on pause click).
                if benchmark and new_state in (AgentState.DONE, AgentState.ERROR):
                    self._finalize_benchmark_report(benchmark, self._agent_loop)
            wx.CallAfter(_handle)

        self._agent_loop.set_state_change_callback(on_state_change)

        def refresh_context():
            return {
                'active_editor': self._detect_active_editor(),
                'pcb_data': self.pcb_data,
                'verbose': False,
                'project_dir': project_dir,
            }

        self._agent_loop.set_context_callback(refresh_context)
        self._agent_loop.set_execute_on_gui_callback(self._execute_action_on_gui)

        if self.frame:
            self.frame.set_agent_running(True)
            try:
                if hasattr(self.frame, 'set_thinking_output_enabled'):
                    self.frame.set_thinking_output_enabled(
                        bool(getattr(self.settings, 'thinking_output_enabled', True))
                    )
            except Exception:
                pass
        self._start_sleep_guard()
        self._agent_loop.run(message, initial_context)

    # -- Agent-loop helpers -----------------------------------------------

    def _on_pause_agent(self):
        """Pause the running agent loop (kill-switch)."""
        if self._agent_loop and self._agent_loop.is_running:
            self._agent_loop.pause()
            self._stop_sleep_guard()
            if self.frame:
                self.frame.add_design_response("Agent paused by user.")
                self.frame.set_agent_running(False)
                self.frame.set_design_thinking(False)

    def _execute_action_on_gui(self, action, context, result_callback):
        """Execute an action: I/O off-thread, pcbnew on the GUI thread.

        Footprint resolution (HTTP search/download) runs in a background
        thread so the UI stays responsive.  Only the final pcbnew calls
        (board.Add, SetPosition, SaveBoard, Refresh) bounce to the GUI thread.
        """
        def _io_worker():
            """Background thread: resolve footprint / do any HTTP I/O."""
            try:
                # Pre-resolve footprint path (involves HTTP — must stay off GUI)
                if action.action_type == DesignActionType.ADD_COMPONENT:
                    # Build a minimal context (no board — that's not thread-safe)
                    io_context = {
                        'project_dir': context.get('project_dir', ''),
                    }
                    self._resolve_footprint_for_action(action, io_context)
            except Exception as e:
                logger.warning("Footprint pre-resolution failed: %s", e)

            # Now bounce to the GUI thread for pcbnew work
            if WX_AVAILABLE:
                wx.CallAfter(_gui_run)
            else:
                result_callback(action)

        def _gui_run():
            """GUI thread: touch pcbnew, execute action, save, refresh."""
            try:
                board = None
                if PCBNEW_AVAILABLE:
                    board = pcbnew.GetBoard()
                    # Defensive: sometimes KiCad returns a low-level SwigPyObject
                    # rather than a proper pcbnew.BOARD wrapper (e.g. no board
                    # open / editor context mismatch). Avoid passing it through.
                    try:
                        is_valid = (
                            board is not None
                            and hasattr(board, "GetFootprints")
                            and hasattr(board, "Add")
                            and hasattr(board, "GetFileName")
                        )
                        if not is_valid:
                            sample = []
                            try:
                                sample = sorted(set(dir(board)))[:30] if board is not None else []
                            except Exception:
                                sample = []
                            logger.error(
                                "pcbnew.GetBoard() returned invalid board object; type=%s has_Add=%s has_GetFileName=%s attrs_sample=%s",
                                type(board).__name__ if board is not None else "None",
                                bool(hasattr(board, "Add")),
                                bool(hasattr(board, "GetFileName")),
                                sample,
                            )
                            board = None
                    except Exception:
                        # If inspection fails, treat board as unusable.
                            logger.exception("Failed to validate pcbnew board object")
                            board = None
                    if board is not None:
                        try:
                            self._last_valid_board = board
                        except Exception:
                            pass
                    elif hasattr(self, "_last_valid_board"):
                        fallback_board = getattr(self, "_last_valid_board", None)
                        if (
                            fallback_board is not None
                            and hasattr(fallback_board, "GetFootprints")
                            and hasattr(fallback_board, "Add")
                            and hasattr(fallback_board, "GetFileName")
                        ):
                            logger.warning("Reusing last valid pcbnew board after GetBoard() returned an invalid object")
                            board = fallback_board

                # If we don't have a valid board, fail fast with a user-visible error.
                # Otherwise actions may appear to "succeed" in logs but not affect
                # the actual PCB editor (wrong editor context / no board open).
                if board is None and action.action_type in {
                    DesignActionType.ADD_COMPONENT,
                    DesignActionType.DELETE_COMPONENT,
                    DesignActionType.MOVE_COMPONENT,
                    DesignActionType.ROTATE_COMPONENT,
                    DesignActionType.ALIGN_COMPONENTS,
                    DesignActionType.DEFINE_BOARD_OUTLINE,
                    DesignActionType.ADD_MOUNTING_HOLE,
                    DesignActionType.ASSIGN_NETS,
                    DesignActionType.DEFINE_NET,
                    DesignActionType.DRAW_TRACK,
                    DesignActionType.ADD_VIA,
                    DesignActionType.AUTOROUTE_BOARD,
                    DesignActionType.DELETE_TRACKS,
                    DesignActionType.ADD_POLYGON,
                    DesignActionType.ADD_TEXT,
                    DesignActionType.SET_LAYER_COUNT,
                }:
                    action.success = False
                    action.result_message = (
                        "No active PCB board found. Open the PCB Editor with a board file "
                        "(not the schematic editor) and try again."
                    )
                    action.executed = True
                    result_callback(action)
                    return

                project_dir = context.get('project_dir', '')
                try:
                    if board and not project_dir:
                        fn = board.GetFileName() or ''
                        if fn:
                            project_dir = str(Path(fn).expanduser().resolve().parent)
                except Exception:
                    pass

                gui_context = {
                    'active_editor': 'pcb',
                    'pcb_data': self.pcb_data,
                    'verbose': False,
                    'board': board,
                    'project_dir': project_dir,
                }

                import asyncio
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        self.design_agent.execute_action(action, gui_context)
                    )
                finally:
                    loop.close()

                # Save board (best-effort)
                try:
                    if PCBNEW_AVAILABLE and board and hasattr(board, "GetFileName"):
                        fn = board.GetFileName() or ''
                        if fn:
                            pcbnew.SaveBoard(fn, board)
                except Exception:
                    logger.debug("pcbnew.SaveBoard() failed (non-fatal)", exc_info=True)

                self._safe_pcbnew_refresh()

                result_callback(result)
            except Exception as e:
                logger.exception("GUI-thread action execution failed: %s", e)
                action.success = False
                action.result_message = f"GUI execution failed: {e}"
                action.executed = True
                result_callback(action)

        # Kick off the I/O worker (non-blocking)
        import threading
        threading.Thread(target=_io_worker, daemon=True).start()

    def _resolve_footprint_for_action(self, action, context):
        """Resolve a footprint path for an ADD_COMPONENT action (GUI thread)."""
        params = getattr(action, 'parameters', None) or {}
        if params.get('local_footprint_path'):
            return
        query = None
        for key in ('query', 'part_name', 'mpn', 'part', 'name'):
            v = params.get(key)
            if isinstance(v, str) and v.strip():
                query = v.strip()
                break
        if not query or not self.library_manager:
            return

        # Extract package hint from action parameters for better resolution
        package_hint = None
        pkg_val = params.get('package')
        if isinstance(pkg_val, str) and pkg_val.strip():
            package_hint = pkg_val.strip()

        project_dir = context.get('project_dir', '')
        try:
            resolved = None
            resolver = getattr(self.library_manager, 'resolve_best_footprint_path', None)
            if callable(resolver):
                resolved = resolver(query, package_hint=package_hint)
            if not resolved:
                # Also try with just the package hint if available
                if package_hint and callable(resolver):
                    resolved = resolver(package_hint, package_hint=package_hint)
            if not resolved:
                results = self.library_manager.search_parts_sync(query)
                if results:
                    best = None
                    for item in results:
                        if getattr(item, 'local_footprint_path', None):
                            if package_hint and not self.library_manager.footprint_path_matches_hint(str(getattr(item, 'local_footprint_path', '')), package_hint):
                                continue
                            best = item
                            break
                    if best is None:
                        if not package_hint:
                            best = results[0]
                    fp_path = getattr(best, 'local_footprint_path', None) if best is not None else None
                    if best is not None and not fp_path and getattr(best, 'footprint_url', None):
                        dl = self.library_manager.download_item(
                            best, install=True, project_dir=project_dir
                        )
                        if dl.success and dl.footprint_path:
                            fp_path = dl.footprint_path
                    if best is not None and fp_path:
                        resolved = (getattr(best, 'mpn', query), fp_path)
            if resolved:
                if not action.parameters:
                    action.parameters = {}
                action.parameters['local_footprint_path'] = resolved[1]
                action.parameters['resolved_mpn'] = resolved[0]
        except Exception as e:
            logger.warning(f"Footprint resolution failed for '{query}': {e}")

    def _on_approve_design_action(self, action: 'DesignAction'):
        """Handle approval of a design action.

        If an agent loop is running, signal approval and let the loop execute.
        Otherwise use _execute_action_on_gui for single-shot execution.
        """
        if not DESIGN_AVAILABLE or not self.design_agent:
            return

        # If the agent loop is awaiting approval, signal it
        if (self._agent_loop and
                self._agent_loop.state == AgentState.AWAITING_APPROVAL):
            self._agent_loop.approve_action(action, True)
            return

        # Legacy single-shot fallback — reuse the existing non-blocking executor
        action.approved = True
        if self.frame and hasattr(self.frame, 'set_design_thinking'):
            self.frame.set_design_thinking(True)

        # Get project_dir on the GUI thread (pcbnew not thread-safe)
        project_dir = ''
        try:
            if PCBNEW_AVAILABLE:
                board = pcbnew.GetBoard()
                if board:
                    fn = board.GetFileName() or ''
                    if fn:
                        project_dir = str(Path(fn).expanduser().resolve().parent)
        except Exception:
            pass

        context = {
            'active_editor': 'pcb',
            'pcb_data': self.pcb_data,
            'verbose': False,
            'project_dir': project_dir,
        }

        def _on_result(result):
            if not WX_AVAILABLE or not self.frame:
                return
            try:
                success = getattr(result, 'success', False)
                msg = getattr(result, 'result_message', '') or ''
                if success:
                    self.frame.add_design_response(f"✅ {msg or 'Action completed!'}")
                else:
                    self.frame.add_design_response(f"❌ {msg or 'Action failed'}")
            except Exception as e:
                logger.exception("Result callback failed: %s", e)
            finally:
                if self.frame and hasattr(self.frame, 'set_design_thinking'):
                    self.frame.set_design_thinking(False)

        self._execute_action_on_gui(action, context, _on_result)

    @staticmethod
    def _safe_pcbnew_refresh():
        """Refresh pcbnew on the main (GUI) thread."""
        try:
            if PCBNEW_AVAILABLE:
                pcbnew.Refresh()
        except Exception:
            logger.debug("pcbnew.Refresh() failed (non-fatal)", exc_info=True)
    
    def _on_reject_design_action(self, action: 'DesignAction'):
        """Handle rejection of a design action.
        
        Args:
            action: The DesignAction that was rejected
        """
        action.approved = False
        logger.info(f"Design action rejected: {action.description}")

        # If the agent loop is awaiting approval, signal rejection
        if (self._agent_loop and
                self._agent_loop.state == AgentState.AWAITING_APPROVAL):
            self._agent_loop.approve_action(action, False, "User rejected")
    
    def _on_explain(self, question: Optional[str] = None):
        """Handle explain request from UI.
        
        Args:
            question: Optional user question to include in explanation
        """
        if not self.check_results:
            self.frame.set_error("Run checks first before getting explanations")
            return
        
        def worker():
            try:
                explanation = self._get_explanation(question)
                wx.CallAfter(self.frame.set_explanation, explanation)
                wx.CallAfter(self.frame.set_status, "Explanation ready", "")
            except Exception as e:
                logger.exception("Error getting explanation")
                wx.CallAfter(self.frame.set_error, f"Explanation failed: {e}")

        if not WX_AVAILABLE:
            return
        threading.Thread(target=worker, daemon=True).start()
    
    def _on_ask(self, question: str):
        """Handle user question from UI.
        
        Args:
            question: User's question about the design
        """
        def worker():
            try:
                # Build design context from PCB data, focused on the question
                design_context = self._build_design_context(query=question)

                explainer = self.explainer or IssueExplainer(None)
                answer = explainer.answer_question(
                    question=question,
                    check_results=self.check_results,
                    design_context=design_context
                )
                
                wx.CallAfter(self.frame.append_answer, question, answer.answer)
                wx.CallAfter(self.frame.set_status, "Question answered", "")
                
                # Log referenced elements
                if answer.referenced_components:
                    logger.debug(f"Referenced components: {answer.referenced_components}")
                if answer.referenced_nets:
                    logger.debug(f"Referenced nets: {answer.referenced_nets}")
                if answer.referenced_rules:
                    logger.debug(f"Referenced rules: {answer.referenced_rules}")
                    
            except Exception as e:
                logger.exception("Error answering question")
                wx.CallAfter(self.frame.set_error, f"Failed to answer: {e}")

        if not WX_AVAILABLE:
            return
        threading.Thread(target=worker, daemon=True).start()
    
    def _build_design_context(self, query: str = "") -> dict:
        """Build design context dictionary from PCB data.

        When a query is provided, uses CircuitContextBuilder to produce
        a compact, query-relevant snapshot that includes full component
        details and net connections for mentioned parts without
        overwhelming the LLM context window.
        """
        if not self.pcb_data:
            return {}

        # Refresh the lightweight circuit snapshot
        if DESIGN_AVAILABLE and hasattr(self, 'circuit_context_builder'):
            try:
                self._circuit_snapshot = self.circuit_context_builder.build_snapshot(
                    self.pcb_data, self.schematic_data,
                )
            except Exception:
                logger.debug("Failed to build circuit snapshot", exc_info=True)
                self._circuit_snapshot = None

        context = {
            "board": {
                "has_outline": self.pcb_data.has_board_outline,
                "outline_elements": self.pcb_data.board_outline_element_count,
            },
            "components": {
                "count": len(self.pcb_data.footprints),
                "references": [fp.reference for fp in self.pcb_data.footprints[:50]],
                "details": [
                    {"ref": fp.reference, "value": fp.value, "footprint": fp.footprint_name}
                    for fp in self.pcb_data.footprints[:80]
                ],
            },
            "nets": {
                "count": len(self.pcb_data.nets),
                "names": [n.name for n in self.pcb_data.nets[:50] if n.name],
            },
            "tracks": {
                "count": len(self.pcb_data.tracks),
            },
            "vias": {
                "count": len(self.pcb_data.vias),
            },
        }

        # If we have a smart snapshot AND a query, add the focused context
        if self._circuit_snapshot and query:
            try:
                context["focused_context"] = (
                    self.circuit_context_builder.build_context_for_query(
                        query, self._circuit_snapshot,
                        include_full_table=False,  # Already have component details above
                    )
                )
            except Exception:
                logger.debug("Failed to build focused context", exc_info=True)

        return context

    def _extract_outline_context(self, board: Any = None, pcb_data: Optional[PCBData] = None) -> Dict[str, Any]:
        """Extract board outline geometry for prompt context.

        Returns keys expected by DesignAgent chat prompts:
        outline_defined / has_board_outline / board_width / board_height /
        board_origin_x / board_origin_y / board_center_x / board_center_y
        """
        out: Dict[str, Any] = {
            "outline_defined": False,
            "has_board_outline": False,
            "board_width": None,
            "board_height": None,
            "board_origin_x": None,
            "board_origin_y": None,
            "board_center_x": None,
            "board_center_y": None,
        }

        def _set_bbox(min_x: float, min_y: float, max_x: float, max_y: float) -> None:
            w = float(max_x) - float(min_x)
            h = float(max_y) - float(min_y)
            if w <= 0.0 or h <= 0.0:
                return
            out["outline_defined"] = True
            out["has_board_outline"] = True
            out["board_width"] = round(w, 3)
            out["board_height"] = round(h, 3)
            out["board_origin_x"] = round(float(min_x), 3)
            out["board_origin_y"] = round(float(min_y), 3)
            out["board_center_x"] = round((float(min_x) + float(max_x)) / 2.0, 3)
            out["board_center_y"] = round((float(min_y) + float(max_y)) / 2.0, 3)

        # Prefer live board bounds when available (captures unsaved edits).
        try:
            if board is None and PCBNEW_AVAILABLE:
                board = pcbnew.GetBoard()
            if board is not None and PCBNEW_AVAILABLE:
                to_mm = getattr(pcbnew, "ToMM", None)
                if callable(to_mm):
                    bb = board.GetBoardEdgesBoundingBox()
                    bw_iu = int(bb.GetWidth())
                    bh_iu = int(bb.GetHeight())
                    if bw_iu > 0 and bh_iu > 0:
                        x0 = float(to_mm(bb.GetX()))
                        y0 = float(to_mm(bb.GetY()))
                        w = float(to_mm(bw_iu))
                        h = float(to_mm(bh_iu))
                        _set_bbox(x0, y0, x0 + w, y0 + h)
                        return out
        except Exception:
            logger.debug("Failed to read live board outline bounds", exc_info=True)

        pdata = pcb_data if pcb_data is not None else self.pcb_data
        if pdata is None:
            return out
        try:
            has_outline = bool(getattr(pdata, "has_board_outline", False))
        except Exception:
            has_outline = False
        if has_outline:
            out["outline_defined"] = True
            out["has_board_outline"] = True

        xs: List[float] = []
        ys: List[float] = []

        try:
            for ln in getattr(pdata, "board_outline_lines", []) or []:
                xs.extend([float(ln.start.x), float(ln.end.x)])
                ys.extend([float(ln.start.y), float(ln.end.y)])
            for arc in getattr(pdata, "board_outline_arcs", []) or []:
                xs.extend([float(arc.start.x), float(arc.mid.x), float(arc.end.x)])
                ys.extend([float(arc.start.y), float(arc.mid.y), float(arc.end.y)])
            for rect in getattr(pdata, "board_outline_rects", []) or []:
                xs.extend([float(rect.start.x), float(rect.end.x)])
                ys.extend([float(rect.start.y), float(rect.end.y)])
            for poly in getattr(pdata, "board_outline_polygons", []) or []:
                for pt in getattr(poly, "points", []) or []:
                    xs.append(float(pt.x))
                    ys.append(float(pt.y))
            for circ in getattr(pdata, "board_outline_circles", []) or []:
                cx = float(circ.center.x)
                cy = float(circ.center.y)
                r = float(circ.radius)
                xs.extend([cx - r, cx + r])
                ys.extend([cy - r, cy + r])
        except Exception:
            logger.debug("Failed parsing PCB outline geometry from parser snapshot", exc_info=True)
            return out

        if xs and ys:
            _set_bbox(min(xs), min(ys), max(xs), max(ys))
        return out
    
    def _load_pcb_data(self):
        """Load PCB data from KiCad or file."""
        if PCBNEW_AVAILABLE:
            board = pcbnew.GetBoard()
            if board:
                # Prefer exporting the *current in-memory board* to a temp file.
                # This makes checks reflect unsaved modifications.
                tmp_path = None
                try:
                    fd, tmp_path = tempfile.mkstemp(suffix=".kicad_pcb", prefix="vibecad_")
                    os.close(fd)

                    save_board = getattr(pcbnew, "SaveBoard", None)
                    if callable(save_board):
                        save_board(tmp_path, board)
                        parser = PCBParser(tmp_path)
                        self.pcb_data = parser.parse()
                        logger.info("Loaded PCB from in-memory board export")
                        return

                    # Fallback: parse the saved file on disk
                    filename = board.GetFileName()
                    if filename and Path(filename).exists():
                        parser = PCBParser(filename)
                        self.pcb_data = parser.parse()
                        logger.info(f"Loaded PCB from file: {filename}")
                        return
                finally:
                    if tmp_path:
                        try:
                            Path(tmp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
        
        # Fallback: no PCB loaded
        self.pcb_data = None
        logger.warning("No PCB board loaded in KiCad")
    
    def _detect_active_editor(self) -> str:
        """Detect whether we're in PCB or schematic editor.
        
        Returns:
            "pcb" or "schematic"
        """
        # KiCad 7+ doesn't have a direct API for this in pcbnew
        # We infer from what's available
        
        if PCBNEW_AVAILABLE:
            try:
                board = pcbnew.GetBoard()
                if board is not None:
                    self.active_editor = "pcb"
                    return "pcb"
            except:
                pass
        
        if EESCHEMA_AVAILABLE:
            try:
                # eeschema module may have GetSchematic() or similar
                get_sch = getattr(eeschema, "GetSchematic", None)
                if callable(get_sch):
                    sch = get_sch()
                    if sch is not None:
                        self.active_editor = "schematic"
                        return "schematic"
            except:
                pass
        
        # Default to PCB
        self.active_editor = "pcb"
        return "pcb"
    
    def _load_schematic_data(self):
        """Load schematic data from KiCad or file."""
        self.schematic_data = None

        if EESCHEMA_AVAILABLE:
            try:
                # eeschema API (if available in KiCad 7+)
                get_sch = getattr(eeschema, "GetSchematic", None)
                if callable(get_sch):
                    sch = get_sch()
                    if sch:
                        filename = getattr(sch, "GetFileName", lambda: None)()
                        if filename and Path(filename).exists():
                            parser = SchematicParser(filename)
                            self.schematic_data = parser.parse()
                            logger.info(f"Loaded schematic from file: {filename}")
                            return
            except Exception as e:
                logger.debug(f"Could not load schematic via eeschema API: {e}")
        
        # Try to find schematic file in the same directory as PCB
        if PCBNEW_AVAILABLE:
            try:
                board = pcbnew.GetBoard()
                if board:
                    pcb_file = board.GetFileName()
                    if pcb_file:
                        sch_file = Path(pcb_file).with_suffix('.kicad_sch')
                        if sch_file.exists():
                            parser = SchematicParser(str(sch_file))
                            self.schematic_data = parser.parse()
                            logger.info(f"Loaded schematic from companion file: {sch_file}")
                            return
            except Exception as e:
                logger.debug(f"Could not load companion schematic: {e}")
    
    def _get_explanation(self, question: Optional[str] = None) -> Explanation:
        """Get explanation for current check results.
        
        Args:
            question: Optional user question to include in explanation
        """
        from .llm.explainer import ExplanationRequest
        
        request = ExplanationRequest(
            check_results=self.check_results,
            user_question=question
        )
        
        from .llm.client import LLMError
        if not self.explainer or not self.explainer.is_available:
            raise LLMError("LLM is required for explanations but is not available/configured.")
        return self.explainer.explain(request)
    
    # CLI interface for standalone usage
    def run_checks_on_file(self, pcb_path: str) -> List[Any]:
        """Run checks on a PCB file (for CLI/testing).
        
        Args:
            pcb_path: Path to .kicad_pcb file
        
        Returns:
            List of CheckResult objects
        """
        _ = pcb_path
        logger.info("run_checks_on_file is removed in v4-only mode.")
        self.check_results = []
        return []
    
    def explain_results(self, user_question: Optional[str] = None) -> Explanation:
        """Get explanation for current results.
        
        Args:
            user_question: Optional user question to address
        
        Returns:
            Explanation object
        """
        from .llm.explainer import ExplanationRequest
        
        request = ExplanationRequest(
            check_results=self.check_results,
            user_question=user_question
        )
        
        explainer = self.explainer or IssueExplainer(None)
        return explainer.explain(request)


# KiCad plugin registration
class VibeCADAction(pcbnew.ActionPlugin if PCBNEW_AVAILABLE else object):
    """KiCad ActionPlugin wrapper for VibeCAD."""
    
    def defaults(self):
        self.name = "VibeCAD"
        self.category = "Design Review"
        self.description = "LLM-assisted design review for KiCad"
        self.show_toolbar_button = True
        icon_path = Path(__file__).resolve().with_name("icon.png")
        icon_value = str(icon_path) if icon_path.exists() else ""
        self.icon_file_name = icon_value
        self.dark_icon_file_name = icon_value
    
    def Run(self):
        global _VIBECAD_SINGLETON

        def _clear_cached_state() -> None:
            global _VIBECAD_SINGLETON
            _VIBECAD_SINGLETON = None
            if not PCBNEW_AVAILABLE:
                return
            for attr in (
                "_VIBECAD_SINGLETON",
                "_VIBECAD_FRAME",
                "_VIBECAD_DOCKED_MGR",
                "_VIBECAD_DOCKED_PANEL",
            ):
                try:
                    setattr(pcbnew, attr, None)
                except Exception:
                    pass

        try:
            # First: if a prior instance exists (even across module reloads), raise it.
            if WX_AVAILABLE and PCBNEW_AVAILABLE:
                try:
                    mgr = getattr(pcbnew, "_VIBECAD_DOCKED_MGR", None)
                    panel = getattr(pcbnew, "_VIBECAD_DOCKED_PANEL", None)
                    if mgr is not None and panel is not None:
                        try:
                            pane = mgr.GetPane("VibeCAD")
                            if pane.IsOk():
                                pane.Show(True)
                                mgr.Update()
                                try:
                                    panel.SetFocus()
                                except Exception:
                                    pass
                                return
                        except Exception:
                            pass

                    frame = getattr(pcbnew, "_VIBECAD_FRAME", None)
                    if frame is not None:
                        try:
                            # Verify frame still alive
                            _ = frame.IsShown()
                            try:
                                if frame.IsIconized():
                                    frame.Restore()
                            except Exception:
                                pass
                            try:
                                if not frame.IsShown():
                                    frame.Show(True)
                            except Exception:
                                pass
                            try:
                                frame.Raise()
                            except Exception:
                                pass
                            try:
                                frame.SetFocus()
                            except Exception:
                                pass
                            return
                        except Exception:
                            # Dead handle; clear it
                            try:
                                setattr(pcbnew, "_VIBECAD_FRAME", None)
                            except Exception:
                                pass

                except Exception:
                    pass

            # Second: plugin singleton (also persisted on pcbnew).
            if PCBNEW_AVAILABLE:
                try:
                    existing_plugin = getattr(pcbnew, "_VIBECAD_SINGLETON", None)
                    if existing_plugin is not None:
                        _VIBECAD_SINGLETON = existing_plugin
                except Exception:
                    pass

            if _VIBECAD_SINGLETON is None:
                _VIBECAD_SINGLETON = VibeCADPlugin()
                if PCBNEW_AVAILABLE:
                    try:
                        setattr(pcbnew, "_VIBECAD_SINGLETON", _VIBECAD_SINGLETON)
                    except Exception:
                        pass

            _VIBECAD_SINGLETON.Run()
            return
        except Exception:
            logger.exception("VibeCAD failed to start (first attempt)")

        # Recovery path: stale KiCad-cached objects can survive code reloads.
        # Reset all cached handles and try exactly once with a fresh plugin.
        try:
            _clear_cached_state()
            _VIBECAD_SINGLETON = VibeCADPlugin()
            if PCBNEW_AVAILABLE:
                try:
                    setattr(pcbnew, "_VIBECAD_SINGLETON", _VIBECAD_SINGLETON)
                except Exception:
                    pass
            _VIBECAD_SINGLETON.Run()
            return
        except Exception:
            logger.exception("VibeCAD failed to start (recovery attempt)")
            try:
                if WX_AVAILABLE:
                    wx.MessageBox(
                        "VibeCAD could not open. See KiCad's Scripting Console for details.",
                        "VibeCAD Error",
                        wx.OK | wx.ICON_ERROR,
                    )
            except Exception:
                pass


# Register with KiCad
if PCBNEW_AVAILABLE:
    VibeCADAction().register()


# Module-level singleton: KiCad may call ActionPlugin.Run repeatedly.
_VIBECAD_SINGLETON: Optional["VibeCADPlugin"] = None
