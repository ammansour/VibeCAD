"""
VibeCAD - KiCad 7 LLM-Assisted Design Review Plugin

This plugin provides deterministic design rule checking with LLM-powered
explanations. The LLM never modifies designs - it only explains findings.
"""

__version__ = "0.1.0"
__author__ = "VibeCAD Team"

from .plugin import VibeCADPlugin

# KiCad plugin registration
def register():
    """Register the plugin with KiCad."""
    return VibeCADPlugin()
