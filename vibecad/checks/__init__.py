"""
Deterministic design rule checks for KiCad projects.

All checks in this module are purely deterministic - they analyze
parsed data and return structured findings without any LLM involvement.
"""

from .base import Check, CheckResult, Severity, Finding
from .board_outline import MissingBoardOutlineCheck, BoardOutlineOpenCheck
from .component_position import ComponentOutsideBoardCheck

__all__ = [
    'Check',
    'CheckResult', 
    'Severity',
    'Finding',
    'MissingBoardOutlineCheck',
    'BoardOutlineOpenCheck',
    'ComponentOutsideBoardCheck',
]
