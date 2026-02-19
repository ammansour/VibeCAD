"""
LLM-powered issue explainer for VibeCAD.

This module takes structured check results and uses the LLM to generate
human-readable explanations. The LLM only explains - it never modifies.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .client import LLMClient, LLMMessage, LLMConfig, LLMError
from ..checks.base import CheckResult, Finding


logger = logging.getLogger(__name__)


@dataclass
class ExplanationRequest:
    """Request for LLM explanation of check results."""
    check_results: List[CheckResult]
    user_question: Optional[str] = None
    project_context: Optional[Dict[str, Any]] = None


@dataclass
class Explanation:
    """LLM-generated explanation of check results."""
    summary: str
    detailed_explanations: List[str]
    suggested_checks: List[str]
    raw_response: Optional[str] = None


@dataclass
class AnswerResponse:
    """LLM response to a user question."""
    question: str
    answer: str
    referenced_components: List[str]
    referenced_nets: List[str]
    referenced_rules: List[str]
    raw_response: Optional[str] = None


class IssueExplainer:
    """Uses LLM to explain detected issues in plain English.
    
    The explainer:
    - Takes structured JSON describing detected issues
    - Sends to LLM for natural language explanation
    - Returns explanations that reference specific facts
    
    The explainer NEVER:
    - Asks the LLM to generate design modifications
    - Passes unstructured design data to the LLM
    - Allows the LLM to infer specifications
    """
    
    EXPLANATION_PROMPT_TEMPLATE = """Analyze the following PCB design check results and provide a clear explanation.

## Check Results (JSON)
```json
{check_results_json}
```

{user_question_section}

{project_context_section}

Please provide:
1. A brief summary of all findings (1-2 sentences)
2. For each finding with severity "error" or "warning":
   - Reference the specific rule ID
   - Explain what was detected and where
   - Explain why this matters for manufacturing or functionality
3. List 2-3 follow-up checks the user should consider running

Remember: Only explain the findings. Do not suggest design modifications."""

    USER_QUESTION_TEMPLATE = """## User Question
The user has asked: {question}

Please address this question in your explanation, using only the facts from the check results above."""

    PROJECT_CONTEXT_TEMPLATE = """## Project Context
Additional context about the project:
```json
{context_json}
```"""

    QUESTION_PROMPT_TEMPLATE = """You are an expert PCB design assistant for KiCad. Answer the user's question based ONLY on the design data provided below.

## Current Design Check Results (JSON)
```json
{check_results_json}
```

## Design Context
{design_context}

## User Question
{question}

## Response Requirements
1. Answer ONLY based on the data provided above
2. Reference specific component refs (e.g., R1, U3, C5) when relevant
3. Reference specific net names when relevant
4. Reference specific rule IDs from the check results when relevant
5. If the answer cannot be determined from the provided data, say so clearly
6. Do NOT:
   - Infer electrical specifications not in the data
   - Reference online datasheets or external sources
   - Suggest design modifications (only explain current state)

Provide a clear, concise answer:"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the explainer.
        
        Args:
            llm_client: LLM client to use. If None, runs in offline mode.
        """
        self.llm_client = llm_client  # None means offline mode
    
    @property
    def is_available(self) -> bool:
        """Check if the explainer is available (LLM configured)."""
        return self.llm_client is not None and self.llm_client.is_available
    
    def explain(self, request: ExplanationRequest) -> Explanation:
        """Generate explanation for check results.
        
        Args:
            request: ExplanationRequest with check results and optional context
        
        Returns:
            Explanation with human-readable text
        
        Raises:
            LLMError: If LLM request fails
            ValueError: If no check results provided
        """
        if not request.check_results:
            raise ValueError("No check results provided for explanation")
        
        llm_client = self.llm_client
        if llm_client is None or not llm_client.is_available:
            return self._generate_offline_explanation(request)
        
        # Build the prompt
        prompt = self._build_prompt(request)
        
        # Get LLM explanation
        try:
            response = llm_client.explain_simple(prompt)
            return self._parse_explanation(response, request)
        except LLMError:
            # Fall back to offline explanation if LLM fails
            return self._generate_offline_explanation(request)
    
    def explain_single_check(self, result: CheckResult, 
                             user_question: Optional[str] = None) -> Explanation:
        """Convenience method to explain a single check result.
        
        Args:
            result: Single CheckResult to explain
            user_question: Optional user question
        
        Returns:
            Explanation
        """
        request = ExplanationRequest(
            check_results=[result],
            user_question=user_question
        )
        return self.explain(request)
    
    def answer_question(self, question: str, 
                        check_results: List[CheckResult],
                        design_context: Optional[Dict[str, Any]] = None) -> AnswerResponse:
        """Answer a user question based on the current design data.
        
        The LLM will only reference facts from the provided data.
        It will NOT infer specs, fetch datasheets, or suggest modifications.
        
        Args:
            question: User's question in natural language
            check_results: Current check results for context
            design_context: Optional additional design info (components, nets, etc.)
        
        Returns:
            AnswerResponse with the answer and referenced elements
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        # Build context JSON
        results_data = [r.to_dict() for r in check_results] if check_results else []
        check_results_json = json.dumps(results_data, indent=2)
        
        design_context_str = ""
        if design_context:
            design_context_str = json.dumps(design_context, indent=2)
        else:
            design_context_str = "No additional design context provided."
        
        prompt = self.QUESTION_PROMPT_TEMPLATE.format(
            check_results_json=check_results_json,
            design_context=design_context_str,
            question=question
        )
        
        llm_client = self.llm_client
        if llm_client is None or not llm_client.is_available:
            return self._generate_offline_answer(question, check_results)
        
        try:
            response = llm_client.explain_simple(prompt)
            return self._parse_answer(question, response, check_results)
        except LLMError as e:
            logger.warning(f"LLM failed, using offline answer: {e}")
            return self._generate_offline_answer(
                question,
                check_results,
                llm_error=str(e),
                attempted_llm=True,
            )
    
    def _parse_answer(self, question: str, response: str, 
                      check_results: List[CheckResult]) -> AnswerResponse:
        """Parse LLM answer and extract references."""
        # Extract component references mentioned (patterns like R1, U3, C5, etc.)
        import re
        component_pattern = r'\b([A-Z]+\d+)\b'
        components = list(set(re.findall(component_pattern, response)))
        
        # Extract net names (usually in quotes or specific patterns)
        net_pattern = r'(?:net\s+)?["\']([^"\']+)["\']|(?:net\s+)(\w+)'
        net_matches = re.findall(net_pattern, response, re.IGNORECASE)
        nets = list(set(n[0] or n[1] for n in net_matches if n[0] or n[1]))
        
        # Extract rule IDs (pattern like BOARD_OUTLINE_001)
        rule_pattern = r'\b([A-Z_]+_\d{3})\b'
        rules = list(set(re.findall(rule_pattern, response)))
        
        return AnswerResponse(
            question=question,
            answer=response,
            referenced_components=components,
            referenced_nets=nets,
            referenced_rules=rules,
            raw_response=response
        )
    
    def _generate_offline_answer(
        self,
        question: str,
        check_results: List[CheckResult],
        llm_error: Optional[str] = None,
        attempted_llm: bool = False,
    ) -> AnswerResponse:
        """Generate an offline answer.

        This is used when:
        - LLM is not configured, or
        - LLM was configured but the request failed (auth/network/etc)
        """
        if attempted_llm and llm_error:
            lower_err = llm_error.lower() if isinstance(llm_error, str) else ""
            rate_limit_hint = None
            if "http 429" in lower_err or "too many requests" in lower_err:
                rate_limit_hint = (
                    "This looks like rate limiting. Wait a bit (30–120s) and try again, "
                    "and avoid rapidly clicking Ask/Explain."
                )

            answer_parts = [
                "I couldn't get a response from the configured LLM endpoint.",
                "",
                f"Error: {llm_error}",
                "",
                "Check your Settings (API key, endpoint, model) and try again.",
                (rate_limit_hint or ""),
                "If this is a TLS/certificate error on macOS, try unchecking 'Verify TLS certificates' or set a CA bundle path.",
                "If you're using GitHub Models, typical values are:",
                "- API base: https://models.github.ai/inference",
                "- Model: openai/gpt-5",
                "",
                "Based on the current check results:",
            ]

            # Remove empty hint line if not applicable
            answer_parts = [p for p in answer_parts if p != ""]
        else:
            answer_parts = [
                "I cannot provide a detailed answer because the LLM is not configured.",
                "",
                "Open ⚙ Settings in the VibeCAD window to set the API key/endpoint/model.",
                "(Environment variables also work: VIBECAD_API_KEY / VIBECAD_API_BASE / VIBECAD_MODEL or GITHUB_TOKEN.)",
                "",
                "Based on the current check results:",
            ]
        
        rules_found = []
        for result in check_results:
            status = "passed" if result.passed else "failed"
            answer_parts.append(f"- {result.check_name}: {status}")
            for finding in result.findings:
                rules_found.append(finding.rule_id)
                answer_parts.append(f"  • [{finding.rule_id}] {finding.message}")

        if not attempted_llm:
            answer_parts.append("")
            answer_parts.append(
                "To get detailed answers, configure the LLM in ⚙ Settings (or set VIBECAD_API_KEY / GITHUB_TOKEN)."
            )
        
        return AnswerResponse(
            question=question,
            answer="\n".join(answer_parts),
            referenced_components=[],
            referenced_nets=[],
            referenced_rules=rules_found,
            raw_response=None
        )

    def _build_prompt(self, request: ExplanationRequest) -> str:
        """Build the prompt for the LLM."""
        # Serialize check results to JSON
        results_data = [r.to_dict() for r in request.check_results]
        check_results_json = json.dumps(results_data, indent=2)
        
        # Build optional sections
        user_question_section = ""
        if request.user_question:
            user_question_section = self.USER_QUESTION_TEMPLATE.format(
                question=request.user_question
            )
        
        project_context_section = ""
        if request.project_context:
            project_context_section = self.PROJECT_CONTEXT_TEMPLATE.format(
                context_json=json.dumps(request.project_context, indent=2)
            )
        
        return self.EXPLANATION_PROMPT_TEMPLATE.format(
            check_results_json=check_results_json,
            user_question_section=user_question_section,
            project_context_section=project_context_section
        )
    
    def _parse_explanation(self, response: str, 
                           request: ExplanationRequest) -> Explanation:
        """Parse LLM response into structured Explanation."""
        # For now, return the full response as the summary
        # A more sophisticated implementation could parse structured sections
        
        # Extract suggested checks (look for numbered lists at the end)
        lines = response.strip().split('\n')
        suggested_checks = []
        detailed = []
        
        in_suggestions = False
        for line in lines:
            line_stripped = line.strip()
            
            # Detect suggestion section
            if 'follow-up' in line_stripped.lower() or 'suggested check' in line_stripped.lower():
                in_suggestions = True
                continue
            
            if in_suggestions and line_stripped:
                # Clean up list markers
                if line_stripped[0].isdigit() or line_stripped.startswith('-'):
                    clean = line_stripped.lstrip('0123456789.-) ').strip()
                    if clean:
                        suggested_checks.append(clean)
        
        return Explanation(
            summary=self._extract_summary(response),
            detailed_explanations=[response],  # Full response as detailed explanation
            suggested_checks=suggested_checks if suggested_checks else [
                "Review board outline continuity",
                "Check design rule compliance",
                "Verify component placement"
            ],
            raw_response=response
        )
    
    def _extract_summary(self, response: str) -> str:
        """Extract a brief summary from the response."""
        # Get first paragraph or first few sentences
        paragraphs = response.strip().split('\n\n')
        if paragraphs:
            first = paragraphs[0].strip()
            # Limit to first 2-3 sentences
            sentences = first.split('. ')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
            return first
        return response[:500] if len(response) > 500 else response
    
    def _generate_offline_explanation(self, request: ExplanationRequest) -> Explanation:
        """Generate a basic explanation without LLM.
        
        Used when LLM is not available or fails.
        """
        summaries = []
        details = []
        
        for result in request.check_results:
            if result.passed:
                summaries.append(f"✓ {result.check_name}: Passed")
            else:
                error_count = result.error_count
                warning_count = result.warning_count
                
                status_parts = []
                if error_count:
                    status_parts.append(f"{error_count} error(s)")
                if warning_count:
                    status_parts.append(f"{warning_count} warning(s)")
                
                summaries.append(f"✗ {result.check_name}: {', '.join(status_parts)}")
                
                # Add finding details
                for finding in result.findings:
                    detail = f"[{finding.rule_id}] {finding.message}"
                    if finding.layer:
                        detail += f" (Layer: {finding.layer})"
                    if finding.location_x is not None and finding.location_y is not None:
                        detail += f" at ({finding.location_x:.2f}, {finding.location_y:.2f}) mm"
                    details.append(detail)
        
        summary = "Design Review Results:\n" + "\n".join(summaries)
        
        return Explanation(
            summary=summary,
            detailed_explanations=details if details else ["No issues detected."],
            suggested_checks=[
                "Run additional design rule checks",
                "Review board outline for manufacturing requirements",
                "Verify all components are properly placed"
            ],
            raw_response=None
        )


def explain_check_results(results: List[CheckResult], 
                          user_question: Optional[str] = None,
                          config: Optional[LLMConfig] = None) -> Explanation:
    """Convenience function to explain check results.
    
    Args:
        results: List of CheckResult objects
        user_question: Optional user question to address
        config: Optional LLM configuration
    
    Returns:
        Explanation object
    """
    client = LLMClient(config) if config else None
    explainer = IssueExplainer(client)
    
    request = ExplanationRequest(
        check_results=results,
        user_question=user_question
    )
    
    return explainer.explain(request)
