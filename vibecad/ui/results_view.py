"""
Results view widget for displaying check results and explanations.

This is a pure Python implementation that can work with wx.
"""

from typing import List, Optional, Callable
from dataclasses import dataclass
import json

from .markdown_utils import markdown_to_html_fragment, render_basic_latex

from ..checks.base import CheckResult, Finding, Severity
from ..llm.explainer import Explanation


@dataclass
class ResultsViewModel:
    """View model for results display."""
    check_results: List[CheckResult]
    explanation: Optional[Explanation] = None
    is_loading: bool = False
    error_message: Optional[str] = None
    verbose: bool = False
    
    @property
    def total_findings(self) -> int:
        return sum(len(r.findings) for r in self.check_results)
    
    @property
    def total_errors(self) -> int:
        return sum(r.error_count for r in self.check_results)
    
    @property
    def total_warnings(self) -> int:
        return sum(r.warning_count for r in self.check_results)
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.check_results)


class ResultsView:
    """Platform-agnostic results view logic.
    
    This class manages the display logic for check results.
    It can be used with any UI framework (wx, Qt, etc.).
    """
    
    def __init__(self):
        self.model = ResultsViewModel(check_results=[])
        self._on_update_callbacks: List[Callable] = []
    
    def add_update_callback(self, callback: Callable):
        """Add a callback to be called when the model updates."""
        self._on_update_callbacks.append(callback)
    
    def _notify_update(self):
        """Notify all callbacks that the model has updated."""
        for callback in self._on_update_callbacks:
            callback(self.model)
    
    def set_loading(self, loading: bool):
        """Set the loading state."""
        self.model.is_loading = loading
        self._notify_update()

    def set_verbose(self, verbose: bool):
        """Enable/disable verbose output (debug details)."""
        self.model.verbose = verbose
        self._notify_update()
    
    def set_results(self, results: List[CheckResult]):
        """Set the check results to display."""
        self.model.check_results = results
        self.model.is_loading = False
        self.model.error_message = None
        self.model.explanation = None
        self._notify_update()
    
    def set_explanation(self, explanation: Explanation):
        """Set the LLM explanation."""
        self.model.explanation = explanation
        self.model.is_loading = False
        self._notify_update()
    
    def set_error(self, message: str):
        """Set an error message."""
        self.model.error_message = message
        self.model.is_loading = False
        self._notify_update()
    
    def clear(self):
        """Clear all results."""
        verbose = self.model.verbose
        self.model = ResultsViewModel(check_results=[], verbose=verbose)
        self._notify_update()
    
    def format_summary_text(self) -> str:
        """Format the results summary as text."""
        if self.model.is_loading:
            return "Running checks..."
        
        if self.model.error_message:
            return f"Error: {self.model.error_message}"
        
        if not self.model.check_results:
            return "No checks have been run yet.\n\nClick 'Run Checks' to analyze your design."
        
        lines = ["═" * 50]
        lines.append("DESIGN REVIEW RESULTS")
        lines.append("═" * 50)
        lines.append("")
        
        if self.model.all_passed:
            lines.append("✓ All checks passed!")
        else:
            lines.append(f"Found {self.model.total_errors} error(s), {self.model.total_warnings} warning(s)")
        
        lines.append("")
        lines.append("─" * 50)
        
        for result in self.model.check_results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            lines.append(f"\n{status} | {result.check_name}")
            lines.append(f"       {result.description[:60]}...")
            
            if result.findings:
                for finding in result.findings:
                    severity_icon = self._get_severity_icon(finding.severity)
                    lines.append(f"       {severity_icon} [{finding.rule_id}] {finding.message}")

                    if self.model.verbose and finding.details:
                        details_json = json.dumps(finding.details, indent=2, sort_keys=True)
                        for dl in details_json.splitlines():
                            lines.append(f"           details: {dl}" if dl == "{" else f"                    {dl}")

            if self.model.verbose and result.context:
                context_json = json.dumps(result.context, indent=2, sort_keys=True)
                lines.append("       context:")
                for cl in context_json.splitlines():
                    lines.append(f"         {cl}")
        
        lines.append("")
        lines.append("─" * 50)
        
        # Add explanation if available
        if self.model.explanation:
            lines.append("")
            lines.append("LLM EXPLANATION")
            lines.append("─" * 50)
            lines.append(render_basic_latex(self.model.explanation.summary))
            
            if self.model.explanation.suggested_checks:
                lines.append("")
                lines.append("Suggested follow-up checks:")
                for i, check in enumerate(self.model.explanation.suggested_checks, 1):
                    lines.append(f"  {i}. {check}")
        
        return "\n".join(lines)
    
    def format_html(self) -> str:
        """Format the results as HTML for rich display."""
        if self.model.is_loading:
            return "<p><em>Running checks...</em></p>"
        
        if self.model.error_message:
            return f'<p style="color: red;"><strong>Error:</strong> {self.model.error_message}</p>'
        
        if not self.model.check_results:
            return """
            <div style="padding: 20px; text-align: center;">
                <h3>No checks have been run yet</h3>
                <p>Click <strong>Run Checks</strong> to analyze your design.</p>
            </div>
            """
        
        html_parts = ['<div style="font-family: sans-serif; font-size: 13px; padding: 10px;">']
        
        # Header
        if self.model.all_passed:
            html_parts.append('<h2 style="color: green;">✓ All Checks Passed</h2>')
        else:
            html_parts.append(f'''
                <h2 style="color: #c00;">Design Issues Found</h2>
                <p><strong>{self.model.total_errors}</strong> error(s), 
                   <strong>{self.model.total_warnings}</strong> warning(s)</p>
            ''')
        
        # Results
        for result in self.model.check_results:
            color = "green" if result.passed else "#c00"
            status = "✓ PASS" if result.passed else "✗ FAIL"
            
            html_parts.append(f'''
                <div style="border: 1px solid #ccc; margin: 10px 0; padding: 10px; border-radius: 5px;">
                    <h3 style="color: {color}; margin: 0;">{status} {result.check_name}</h3>
                    <p style="color: #666; font-size: 0.9em;">{result.description}</p>
            ''')
            
            if result.findings:
                html_parts.append('<ul style="margin: 5px 0;">')
                for finding in result.findings:
                    sev_color = self._get_severity_color(finding.severity)
                    html_parts.append(f'''
                        <li style="color: {sev_color};">
                            <strong>[{finding.rule_id}]</strong> {finding.message}
                        </li>
                    ''')
                html_parts.append('</ul>')
            
            html_parts.append('</div>')
        
        # Explanation
        if self.model.explanation:
            html_parts.append('''
                <div style="background: #f5f5f5; padding: 15px; margin-top: 20px; border-radius: 5px;">
                    <h3 style="margin-top: 0;">💡 LLM Explanation</h3>
            ''')
            
            # Render Markdown (bold, tables, etc.) from the LLM.
            summary_html = markdown_to_html_fragment(self.model.explanation.summary or "")
            html_parts.append(f'<div>{summary_html}</div>')
            
            if self.model.explanation.suggested_checks:
                html_parts.append('<h4>Suggested Follow-up Checks:</h4><ol>')
                for check in self.model.explanation.suggested_checks:
                    html_parts.append(f'<li>{check}</li>')
                html_parts.append('</ol>')
            
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _get_severity_icon(self, severity: Severity) -> str:
        """Get icon for severity level."""
        icons = {
            Severity.ERROR: "❌",
            Severity.WARNING: "⚠️",
            Severity.INFO: "ℹ️",
        }
        return icons.get(severity, "•")
    
    def _get_severity_color(self, severity: Severity) -> str:
        """Get color for severity level."""
        colors = {
            Severity.ERROR: "#c00",
            Severity.WARNING: "#f90",
            Severity.INFO: "#06c",
        }
        return colors.get(severity, "#000")
