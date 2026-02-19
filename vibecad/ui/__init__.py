"""
KiCad UI components for VibeCAD.

This module provides the user interface for the plugin, including
the dockable panel and dialogs.
"""

from .panel import VibeCADPanel
from .results_view import ResultsView
from .dockable_frame import VibeCADFrame
from .settings_dialog import SettingsDialog
from .design_panel import DesignPanel

__all__ = ['VibeCADPanel', 'ResultsView', 'VibeCADFrame', 'SettingsDialog', 'DesignPanel']
