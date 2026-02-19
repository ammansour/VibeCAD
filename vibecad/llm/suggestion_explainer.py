"""
LLM-powered explanation for suggestions.

This module extends the explainer to provide explanations for
suggested fixes. The LLM explains:
- What the suggestion will change
- Why this fix is reasonable
- What assumptions are being made

The LLM NEVER generates the fix itself - it only explains
deterministically generated suggestions.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .client import LLMClient, LLMError
from ..actions.base import Suggestion

logger = logging.getLogger(__name__)


@dataclass 
class SuggestionExplanation:
    """LLM-generated explanation of a suggestion."""
    what_will_change: str
    why_reasonable: str
    assumptions_explained: str
    risks_and_notes: str
    raw_response: Optional[str] = None


class SuggestionExplainer:
    """Explains suggestions using LLM.
    
    The LLM is only used to explain suggestions that were generated
    deterministically. It does not generate geometry or modifications.
    """
    
    SUGGESTION_EXPLANATION_TEMPLATE = """You are a PCB design assistant explaining a suggested fix to a user.

## Suggestion Details
```json
{suggestion_json}
```

## Instructions
Explain this suggested fix to help the user decide whether to apply it.

Please provide:

1. **What Will Change**
   - Describe exactly what the suggestion will modify
   - Reference specific coordinates, components, and layers
   - Be precise about the geometry that will be added/modified

2. **Why This Fix Is Reasonable**
   - Explain why this approach addresses the finding
   - Reference the rule ID and what it checks
   - Note any standard practices this follows

3. **Assumptions Being Made**
   - Explain each assumption listed in the suggestion
   - Note any simplifications in the approach
   - Highlight anything the user should verify

4. **Risks and Notes**
   - Any potential issues to be aware of
   - Things the user should check after applying
   - Whether additional manual adjustments may be needed

Remember:
- You are ONLY explaining, not generating the fix
- The geometry was calculated deterministically by Python code
- Be specific about coordinates and references
- Help the user make an informed decision"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the explainer.
        
        Args:
            llm_client: LLM client to use. If None, provides offline explanations.
        """
        self.llm_client = llm_client
    
    @property
    def is_available(self) -> bool:
        """Check if LLM explanation is available."""
        return self.llm_client is not None and self.llm_client.is_available
    
    def explain_suggestion(self, suggestion: Suggestion) -> SuggestionExplanation:
        """Generate an explanation for a suggestion.
        
        Args:
            suggestion: The Suggestion to explain
        
        Returns:
            SuggestionExplanation with detailed explanation
        """
        # Get context for LLM
        context = suggestion.get_llm_context()
        suggestion_json = json.dumps(context, indent=2)
        
        prompt = self.SUGGESTION_EXPLANATION_TEMPLATE.format(
            suggestion_json=suggestion_json
        )
        
        if self.llm_client and self.llm_client.is_available:
            try:
                response = self.llm_client.explain_simple(prompt)
                return self._parse_explanation(response)
            except LLMError as e:
                logger.warning(f"LLM explanation failed: {e}")
                return self._generate_offline_explanation(suggestion)
        else:
            return self._generate_offline_explanation(suggestion)
    
    def _parse_explanation(self, response: str) -> SuggestionExplanation:
        """Parse LLM response into structured explanation."""
        # Extract sections from the response
        what_will_change = self._extract_section(response, "What Will Change")
        why_reasonable = self._extract_section(response, "Why This Fix Is Reasonable")
        assumptions = self._extract_section(response, "Assumptions Being Made")
        risks = self._extract_section(response, "Risks and Notes")
        
        return SuggestionExplanation(
            what_will_change=what_will_change or "See full explanation below.",
            why_reasonable=why_reasonable or "This addresses the reported finding.",
            assumptions_explained=assumptions or "See assumptions listed in the suggestion.",
            risks_and_notes=risks or "Review the changes carefully before applying.",
            raw_response=response
        )
    
    def _extract_section(self, text: str, header: str) -> Optional[str]:
        """Extract a section from the LLM response."""
        lines = text.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            # Check for section header
            if header.lower() in line.lower() and ('**' in line or '##' in line):
                in_section = True
                continue
            
            # Check for next section
            if in_section and line.strip() and (line.startswith('**') or line.startswith('##')):
                break
            
            if in_section:
                section_lines.append(line)
        
        if section_lines:
            return '\n'.join(section_lines).strip()
        return None
    
    def _generate_offline_explanation(self, suggestion: Suggestion) -> SuggestionExplanation:
        """Generate explanation without LLM."""
        # Build what will change from geometry changes
        changes_desc = []
        for change in suggestion.geometry_changes:
            if change.description:
                changes_desc.append(f"• {change.description}")
            else:
                changes_desc.append(f"• {change.change_type} on {change.layer}")
        
        what_will_change = '\n'.join(changes_desc) if changes_desc else "No changes specified."
        
        # Build why reasonable from rule ID and description
        why_reasonable = (
            f"This suggestion addresses {suggestion.rule_id}.\n"
            f"{suggestion.description}"
        )
        
        # Format assumptions
        assumptions_text = '\n'.join(f"• {a}" for a in suggestion.assumptions) if suggestion.assumptions else "No assumptions documented."
        
        # Generic risks
        risks = (
            "• Review the suggested changes before applying\n"
            "• Use Edit > Undo (Ctrl+Z) to revert if needed\n"
            "• Run design rule checks after applying\n"
            "• This is a deterministic suggestion, not AI-generated geometry"
        )
        
        return SuggestionExplanation(
            what_will_change=what_will_change,
            why_reasonable=why_reasonable,
            assumptions_explained=assumptions_text,
            risks_and_notes=risks,
            raw_response=None
        )
    
    def format_explanation(self, explanation: SuggestionExplanation) -> str:
        """Format explanation for display in UI."""
        parts = []
        
        parts.append("━━━ WHAT WILL CHANGE ━━━")
        parts.append(explanation.what_will_change)
        parts.append("")
        
        parts.append("━━━ WHY THIS IS REASONABLE ━━━")
        parts.append(explanation.why_reasonable)
        parts.append("")
        
        parts.append("━━━ ASSUMPTIONS MADE ━━━")
        parts.append(explanation.assumptions_explained)
        parts.append("")
        
        parts.append("━━━ RISKS & NOTES ━━━")
        parts.append(explanation.risks_and_notes)
        
        return '\n'.join(parts)
