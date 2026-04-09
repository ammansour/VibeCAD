"""
LLM integration for VibeCAD.

This module provides a provider-agnostic API wrapper for LLM interactions.
The LLM is used ONLY for:
1. Explaining detected issues in plain English
2. Answering user questions about existing design data
3. Summarizing review results for documentation
4. Explaining suggested fixes (Phase 3)

The LLM must NEVER:
- Create or modify nets, footprints, or layouts
- Infer electrical specs not present in the project files
- Fetch or assume online datasheet data
- Generate geometry or design modifications
"""

from .client import LLMClient, LLMConfig, LLMError
from .explainer import IssueExplainer, Explanation, AnswerResponse, ExplanationRequest

__all__ = [
    'LLMClient', 
    'LLMConfig', 
    'LLMError',
    'IssueExplainer', 
    'Explanation',
    'AnswerResponse',
    'ExplanationRequest',
]
