#!/usr/bin/env python3
"""
VibeCAD - KiCad 7 Plugin Entry Point

This file should be placed in your KiCad plugins directory.
It bootstraps the VibeCAD plugin.
"""

# Standard library imports
import os
import sys
from pathlib import Path

# Add the vibecad package to the path
plugin_dir = Path(__file__).parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

# Import and register the plugin
try:
    from vibecad import register
    plugin = register()
except ImportError as e:
    import logging
    logging.error(f"Failed to load VibeCAD plugin: {e}")
    raise


# KiCad looks for this function
def register_plugin():
    """KiCad plugin registration function."""
    return plugin
