"""
VibeCAD Plugin for KiCad 7.

This is the main plugin entry point that integrates with KiCad's
plugin system.
"""

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional, List

from .debug_log import InMemoryLogBuffer, install_debug_log_capture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vibecad')

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
from .checks import (
    Check, CheckResult, 
    MissingBoardOutlineCheck, 
    BoardOutlineOpenCheck,
    ComponentOutsideBoardCheck,
)
from .llm import LLMClient, LLMConfig, IssueExplainer, SuggestionExplainer
from .llm.explainer import Explanation, AnswerResponse
from .config import VibeCADSettings
from .actions import SuggestionManager, Suggestion, ActionResult
from .design.intent_router import decide_route

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
except ImportError:
    DESIGN_AVAILABLE = False
    logger.warning("Design module not available")


class VibeCADPlugin:
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
        self.check_results: List[CheckResult] = []
        self.llm_client: Optional[LLMClient] = None
        # Always set (offline fallback if LLM not configured)
        self.explainer: IssueExplainer = IssueExplainer(None)
        
        # Phase 3: Suggestion manager for assisted design actions
        self.suggestion_manager = SuggestionManager()
        self.suggestion_explainer: Optional[SuggestionExplainer] = None
        
        # Phase 4: Design assistance components
        self.design_agent: Optional['DesignAgent'] = None
        self.library_manager: Optional['LibraryManager'] = None
        self.connection_manager: Optional['ConnectionManager'] = None
        self.bom_exporter: Optional['BOMExporter'] = None
        self._agent_loop: Optional['AgentLoop'] = None
        self._init_design_components()
        
        # Available checks
        self.checks: List[Check] = [
            MissingBoardOutlineCheck(),
            BoardOutlineOpenCheck(),
            ComponentOutsideBoardCheck(),  # Phase 3
        ]
        
        # UI - now uses the new dockable frame
        self.frame = None

        # Host KiCad window captured at activation time (used for docking).
        self._host_frame = None

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
        
        # Persisted user settings (API key, endpoint, model, etc.)
        self.settings = VibeCADSettings.load()

        # Restore verbose setting early so logs during initialization are consistent.
        try:
            self._verbose_enabled = bool(getattr(self.settings, 'verbose', False))
            self.set_verbose(self._verbose_enabled)
        except Exception:
            # Never fail plugin init due to settings issues.
            self._verbose_enabled = False

        # Initialize LLM if configured
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client from environment + persisted settings."""
        def _set_llm(client):
            """Apply LLM client (or None) to all subsystems."""
            self.llm_client = client
            self.explainer = IssueExplainer(client)
            self.suggestion_explainer = SuggestionExplainer(client)
            self.suggestion_manager.set_suggestion_explainer(self.suggestion_explainer)
            if DESIGN_AVAILABLE and self.design_agent:
                try:
                    self.design_agent.set_llm_client(client)
                except Exception:
                    logger.exception("Failed to update DesignAgent LLM client")

        try:
            config = LLMConfig.from_environment()

            # Apply persisted settings as overrides
            try:
                overrides = self.settings.to_llm_overrides()
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            except Exception:
                logger.exception("Failed to apply persisted settings")

            config.timeout = 30

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
            logger.info("Design components not available")
            return
        
        try:
            # Initialize library manager for symbol/footprint downloads
            # Enable keyless GitHub search by default in the KiCad plugin runtime.
            # Curated repo downloads are left empty to avoid large/slow downloads.
            self.library_manager = LibraryManager(
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
            
            logger.info("Phase 4 design components initialized")
        except Exception as e:
            logger.exception(f"Failed to initialize design components: {e}")
            self.design_agent = None
    
    @property
    def llm_configured(self) -> bool:
        """Check if LLM is configured and available."""
        return self.explainer is not None and self.explainer.is_available
    
    def Run(self):
        """KiCad plugin entry point - called when user activates the plugin."""
        logger.info("VibeCAD plugin activated")
        
        if WX_AVAILABLE:
            self._show_frame()
        else:
            logger.error("wxPython not available - cannot show UI")

    def set_verbose(self, verbose: bool):
        """Enable/disable verbose logging for debugging."""
        self._verbose_enabled = bool(verbose)

        # Persist verbose toggle like other settings (API key, model, etc.).
        try:
            if hasattr(self, 'settings') and self.settings is not None:
                setattr(self.settings, 'verbose', bool(verbose))
                self.settings.save()
        except Exception:
            logger.exception("Failed to persist verbose setting")

        level = logging.DEBUG if verbose else logging.INFO
        logger.setLevel(level)
        logging.getLogger('vibecad').setLevel(level)
        # Ensure LLM client logs are visible when verbose is enabled.
        logging.getLogger('vibecad.llm').setLevel(level)
        logging.getLogger('vibecad.llm.client').setLevel(level)
        for handler in logging.getLogger().handlers:
            handler.setLevel(level)
        logger.info(f"Verbose mode {'enabled' if verbose else 'disabled'}")

    def _frame_alive(self) -> bool:
        """Return True if self.frame exists and hasn't been destroyed."""
        if self.frame is None:
            return False

    def _wx_window_alive(self, win) -> bool:
        if win is None or not WX_AVAILABLE:
            return False
        try:
            _ = win.IsShown()
            return True
        except Exception:
            return False
        try:
            # Accessing any wx method will raise if the C++ object is gone.
            _ = self.frame.IsShown()
            return True
        except Exception:
            self.frame = None
            return False

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
        
        self.frame = VibeCADFrame(
            parent,
            title="VibeCAD Design Review",
            on_run_checks=self._on_run_checks,
            on_explain=self._on_explain,
            on_ask=self._on_ask,
            on_set_verbose=self.set_verbose,
            on_toggle_dock=self._on_toggle_dock,
            on_open_settings=self._on_open_settings,
            # Phase 3: Suggestion callbacks
            on_preview_suggestion=self._on_preview_suggestion,
            on_apply_suggestion=self._on_apply_suggestion,
            on_dismiss_suggestion=self._on_dismiss_suggestion,
            on_explain_suggestion=self._on_explain_suggestion,
            on_hide_all_previews=self._on_hide_all_previews,
            # Phase 4: Design agent callbacks
            on_design_message=self._on_design_message,
            on_approve_action=self._on_approve_design_action,
            on_reject_action=self._on_reject_design_action,
            on_new_chat=self._on_new_chat,
            initial_verbose=bool(getattr(self, '_verbose_enabled', False)),
            # Debug tab callbacks
            on_get_debug_text=self.get_debug_text,
            on_clear_debug=self.clear_debug_text,
        )
        
        # Update LLM status indicator
        self.frame.set_llm_status(self.llm_configured)

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
        self.frame.set_pause_callback(self._on_pause_agent)
        
        self.frame.Show()

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
                    self.frame.set_llm_status(self.llm_configured)
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
        def worker():
            try:
                self._load_pcb_data()

                if self.pcb_data is None:
                    wx.CallAfter(self.frame.set_error, "No PCB loaded. Open a PCB in KiCad first.")
                    return

                self.check_results = []
                for check in self.checks:
                    result = check.run(pcb_data=self.pcb_data)
                    self.check_results.append(result)
                    logger.info(f"Check {check.check_id}: {'PASS' if result.passed else 'FAIL'}")

                # Phase 3: Generate suggestions for findings
                suggestions = self.suggestion_manager.generate_suggestions(
                    self.check_results, self.pcb_data
                )
                logger.info(f"Generated {len(suggestions)} suggestions")

                wx.CallAfter(self.frame.set_results, self.check_results)

                # Phase 3: Always notify the frame, even if empty, so the UI can
                # update its empty-state messaging.
                if hasattr(self.frame, 'set_suggestions'):
                    wx.CallAfter(self.frame.set_suggestions, suggestions)
                
                wx.CallAfter(self.frame.set_status, 
                           f"Completed {len(self.checks)} checks, {len(suggestions)} suggestions", "")
            except Exception as e:
                logger.exception("Error running checks")
                wx.CallAfter(self.frame.set_error, str(e))

        if not WX_AVAILABLE:
            return
        threading.Thread(target=worker, daemon=True).start()
    
    # === Phase 3: Suggestion Action Handlers ===
    
    def _on_preview_suggestion(self, suggestion: Suggestion):
        """Show preview overlay for a suggestion."""
        try:
            success = self.suggestion_manager.show_preview(suggestion)
            if not success:
                logger.warning(f"Could not show preview for {suggestion.suggestion_id}")
        except Exception as e:
            logger.exception(f"Error showing preview: {e}")
    
    def _on_apply_suggestion(self, suggestion: Suggestion) -> ActionResult:
        """Apply a suggestion after user approval."""
        try:
            board = None
            if PCBNEW_AVAILABLE:
                board = pcbnew.GetBoard()
            
            result = self.suggestion_manager.apply_suggestion(suggestion, board)
            
            if result.success:
                logger.info(f"Applied suggestion: {result.message}")
            else:
                logger.warning(f"Failed to apply suggestion: {result.error}")
            
            return result
        except Exception as e:
            logger.exception(f"Error applying suggestion: {e}")
            return ActionResult(
                success=False,
                suggestion_id=suggestion.suggestion_id,
                message=f"Error: {e}",
                error=str(e)
            )
    
    def _on_dismiss_suggestion(self, suggestion: Suggestion):
        """Dismiss a suggestion."""
        try:
            pcb_filename = ""
            if PCBNEW_AVAILABLE:
                board = pcbnew.GetBoard()
                if board:
                    pcb_filename = board.GetFileName() or ""
            
            self.suggestion_manager.dismiss_suggestion(suggestion, pcb_filename)
            logger.info(f"Dismissed suggestion: {suggestion.suggestion_id}")
        except Exception as e:
            logger.exception(f"Error dismissing suggestion: {e}")
    
    def _on_explain_suggestion(self, suggestion: Suggestion) -> str:
        """Get LLM explanation for a suggestion."""
        try:
            return self.suggestion_manager.get_explanation(suggestion)
        except Exception as e:
            logger.exception(f"Error getting suggestion explanation: {e}")
            return f"Error: {e}"
    
    def _on_hide_all_previews(self):
        """Hide all preview overlays."""
        try:
            self.suggestion_manager.hide_all_previews()
        except Exception as e:
            logger.exception(f"Error hiding previews: {e}")
    
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
        logger.info("New chat session started")

    def _on_design_message(self, message: str):
        """Handle design message from the copilot-style UI.

        If an agent loop is already running and awaiting input (clarifying
        question), feed the answer back to the loop.  Otherwise, start a
        new agent loop for the user's goal.
        """
        if not DESIGN_AVAILABLE or not self.design_agent:
            if self.frame:
                self.frame.add_design_response(
                    "\u274c Design assistance is not available. "
                    "Please check the installation."
                )
            return False

        # If the agent loop is awaiting user input, feed the answer
        if (self._agent_loop and
                self._agent_loop.state == AgentState.AWAITING_INPUT):
            self._agent_loop.resume(message)
            return True

        # If paused, resume with the new message
        if (self._agent_loop and
                self._agent_loop.state == AgentState.PAUSED):
            self._agent_loop.resume(message)
            if self.frame:
                wx.CallAfter(self.frame.set_agent_running, True)
            return True

        # New message: use LLM intent routing (in background) so the UI stays responsive.
        self._route_design_message_async(message)
        return False

    def _route_design_message_async(self, message: str) -> None:
        """Route message to Q&A or AgentLoop using the LLM (non-blocking)."""
        if not WX_AVAILABLE:
            # In non-wx environments, keep deterministic fallback behavior.
            decision = decide_route(self.llm_client, message)
            if decision.route == 'qa':
                self._answer_design_question_async(message)
            else:
                self._start_agent_loop(message)
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

                # Agent route: start the loop on the GUI thread (pcbnew safety).
                try:
                    if self.frame:
                        wx.CallAfter(self.frame.set_agent_running, True)
                except Exception:
                    pass
                wx.CallAfter(self._start_agent_loop, message)
            except Exception:
                logger.exception("Failed to route design message")
                # Fail safe: start agent
                try:
                    wx.CallAfter(self._start_agent_loop, message)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    def _answer_design_question_async(self, question: str) -> None:
        """Answer a user question in the Design tab without starting AgentLoop."""
        if not WX_AVAILABLE or not self.frame:
            return

        # Load PCB/schematic data NOW on the GUI thread (pcbnew not thread-safe).
        try:
            self._load_pcb_data()
            self._load_schematic_data()
        except Exception:
            logger.debug("Pre-load board data failed for Q&A", exc_info=True)

        def _offline_answer(q: str) -> str:
            ql = (q or "").strip().lower()
            if "resistor" in ql and ("r" in ql or "r1" in ql or "value" in ql or "recommend" in ql):
                return (
                    "It depends what R1 does. Quick defaults:\n\n"
                    "- **Pull-up / pull-down**: 4.7k–10k is typical (lower = stronger, more current).\n"
                    "- **LED series**: $R = (V_{supply} - V_f)/I$ (e.g., 3.3V→2.0V @ 5mA → ~260Ω → 270Ω).\n"
                    "- **I2C pull-ups**: often 2.2k–10k depending on bus capacitance/speed.\n\n"
                    "If you tell me R1’s role (pull-up? LED? divider?), voltage, and target current, I can narrow it down."
                )
            return (
                "I can answer, but I need a bit more context (what is the part/net, voltage levels, currents, and what you’re trying to achieve?)."
            )

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

                if self.llm_client and self.llm_client.is_available:
                    try:
                        from .llm.client import LLMMessage

                        system_prompt = (
                            "You are VibeCAD's Q&A assistant. The user is asking an informational question "
                            "about their PCB/schematic design. You have real-time access to their board state "
                            "(component values, net connections, footprints) provided below.\n\n"
                            "Answer directly and concisely. Reference specific component designators (R1, U3, etc.) "
                            "and net names from the board data when relevant. "
                            "If critical context is missing, ask 1-3 clarifying questions "
                            "and provide typical ranges or rules of thumb. "
                            "Do NOT propose tool/actions or start multi-step planning."
                        )

                        prompt_parts = [question.strip()]
                        if circuit_context_str:
                            prompt_parts.append(f"\n\n--- CURRENT BOARD STATE ---\n{circuit_context_str}")
                        if web_context_str:
                            prompt_parts.append(f"\n\n--- COMPONENT WEB DATA ---{web_context_str}")
                        user_prompt = "\n".join(prompt_parts)

                        resp = self.llm_client.chat([LLMMessage(role='user', content=user_prompt)], system_prompt=system_prompt)
                        answer = (resp.content or "").strip() or _offline_answer(question)
                    except Exception:
                        logger.exception("Design Q&A LLM call failed; falling back offline")
                        answer = _offline_answer(question)
                else:
                    answer = _offline_answer(question)

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

    def _start_agent_loop(self, message: str):
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
            'verbose': bool(getattr(self, '_verbose_enabled', False)),
            'project_dir': project_dir,
        }

        config = AgentLoopConfig(
            max_iterations=50,
            max_drc_retries=5,
            auto_approve_readonly=True,
            yolo_auto_apply=bool(getattr(self.settings, 'yolo_auto_apply', False)),
        )
        self._agent_loop = AgentLoop(self.design_agent, config)

        # Wire callbacks (all use wx.CallAfter -- loop runs on background thread)
        self._agent_loop.set_ui_message_callback(
            lambda text: wx.CallAfter(self.frame.add_design_response, text)
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
            lambda text: wx.CallAfter(self.frame.add_design_response, text)
        )

        def on_state_change(new_state):
            def _handle():
                if not self.frame:
                    return
                if new_state in (AgentState.DONE, AgentState.ERROR, AgentState.PAUSED):
                    self.frame.set_agent_running(False)
                    self.frame.set_design_thinking(False)
                elif new_state == AgentState.AWAITING_INPUT:
                    self.frame.set_agent_awaiting_input(True)
                elif new_state == AgentState.AWAITING_APPROVAL:
                    self.frame.set_design_thinking(False)
                elif new_state in (AgentState.PLANNING, AgentState.EXECUTING,
                                   AgentState.OBSERVING):
                    self.frame.set_design_thinking(True)
            wx.CallAfter(_handle)

        self._agent_loop.set_state_change_callback(on_state_change)

        def refresh_context():
            return {
                'active_editor': self._detect_active_editor(),
                'pcb_data': self.pcb_data,
                'verbose': bool(getattr(self, '_verbose_enabled', False)),
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
        self._agent_loop.run(message, initial_context)

    # -- Agent-loop helpers -----------------------------------------------

    def _on_pause_agent(self):
        """Pause the running agent loop (kill-switch)."""
        if self._agent_loop and self._agent_loop.is_running:
            self._agent_loop.pause()
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
                    'verbose': bool(getattr(self, '_verbose_enabled', False)),
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
                    if PCBNEW_AVAILABLE and board:
                        fn = board.GetFileName() or ''
                        if fn:
                            pcbnew.SaveBoard(fn, board)
                except Exception:
                    logger.debug("pcbnew.SaveBoard() failed (non-fatal)", exc_info=True)

                self._safe_pcbnew_refresh()

                # Let the wx event loop breathe so the UI stays responsive
                # during rapid-fire action sequences.
                try:
                    wx.SafeYield()
                except Exception:
                    pass

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
                            best = item
                            break
                    if best is None:
                        best = results[0]
                    fp_path = getattr(best, 'local_footprint_path', None)
                    if not fp_path and getattr(best, 'footprint_url', None):
                        dl = self.library_manager.download_item(
                            best, install=True, project_dir=project_dir
                        )
                        if dl.success and dl.footprint_path:
                            fp_path = dl.footprint_path
                    if fp_path:
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
            'verbose': bool(getattr(self, '_verbose_enabled', False)),
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
        if not EESCHEMA_AVAILABLE:
            self.schematic_data = None
            return
        
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
        if self.pcb_data is None and PCBNEW_AVAILABLE:
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
        
        self.schematic_data = None
    
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
        
        if self.explainer and self.explainer.is_available:
            return self.explainer.explain(request)
        else:
            # Offline explanation
            explainer = IssueExplainer(None)
            return explainer._generate_offline_explanation(request)
    
    # CLI interface for standalone usage
    def run_checks_on_file(self, pcb_path: str) -> List[CheckResult]:
        """Run checks on a PCB file (for CLI/testing).
        
        Args:
            pcb_path: Path to .kicad_pcb file
        
        Returns:
            List of CheckResult objects
        """
        parser = PCBParser(pcb_path)
        self.pcb_data = parser.parse()
        
        self.check_results = []
        for check in self.checks:
            result = check.run(pcb_data=self.pcb_data)
            self.check_results.append(result)
        
        return self.check_results
    
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
        self.icon_file_name = ""  # Optional: path to icon
    
    def Run(self):
        global _VIBECAD_SINGLETON
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
        except Exception:
            logger.exception("VibeCAD failed to start")


# Register with KiCad
if PCBNEW_AVAILABLE:
    VibeCADAction().register()


# Module-level singleton: KiCad may call ActionPlugin.Run repeatedly.
_VIBECAD_SINGLETON: Optional["VibeCADPlugin"] = None
