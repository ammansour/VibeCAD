"""
Command-line interface for VibeCAD.

This allows running design checks outside of KiCad for testing
and automation purposes.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="VibeCAD - LLM-assisted KiCad design review"
    )
    
    parser.add_argument(
        'pcb_file',
        type=str,
        help='Path to .kicad_pcb file to analyze'
    )
    
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Get LLM explanation of results (requires API key)'
    )
    
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Ask a specific question with the explanation (use with --explain)'
    )
    
    parser.add_argument(
        '--ask',
        type=str,
        help='Ask a standalone question about the design (no checks required)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Check file exists
    pcb_path = Path(args.pcb_file)
    if not pcb_path.exists():
        print(f"Error: File not found: {pcb_path}", file=sys.stderr)
        sys.exit(1)
    
    if not pcb_path.suffix == '.kicad_pcb':
        print(f"Warning: File does not have .kicad_pcb extension", file=sys.stderr)
    
    # Import and run
    from vibecad.plugin import VibeCADPlugin
    
    plugin = VibeCADPlugin()
    
    if args.verbose:
        print(f"Analyzing: {pcb_path}")
    
    # Handle standalone question (--ask)
    if args.ask:
        if args.verbose:
            print(f"Question: {args.ask}")
        
        # Load the PCB for context but skip checks
        plugin.run_checks_on_file(str(pcb_path))
        design_context = plugin._build_design_context()
        
        from vibecad.llm import IssueExplainer
        explainer = plugin.explainer or IssueExplainer(None)
        answer = explainer.answer_question(
            question=args.ask,
            check_results=plugin.check_results,
            design_context=design_context
        )
        
        if args.json:
            output = {
                'file': str(pcb_path),
                'question': args.ask,
                'answer': answer.answer,
                'referenced_components': answer.referenced_components,
                'referenced_nets': answer.referenced_nets,
                'referenced_rules': answer.referenced_rules
            }
            print(json.dumps(output, indent=2))
        else:
            print("=" * 60)
            print("VibeCAD Q&A")
            print("=" * 60)
            print(f"Q: {args.ask}")
            print()
            print(f"A: {answer.answer}")
            
            if args.verbose:
                if answer.referenced_components:
                    print(f"\nReferenced components: {', '.join(answer.referenced_components)}")
                if answer.referenced_nets:
                    print(f"Referenced nets: {', '.join(answer.referenced_nets)}")
                if answer.referenced_rules:
                    print(f"Referenced rules: {', '.join(answer.referenced_rules)}")
        
        sys.exit(0)
    
    # Run checks
    results = plugin.run_checks_on_file(str(pcb_path))
    all_passed = all(r.passed for r in results)
    
    # Output results
    if args.json:
        output = {
            'file': str(pcb_path),
            'results': [r.to_dict() for r in results]
        }
        
        if args.explain:
            explanation = plugin.explain_results(args.question)
            output['explanation'] = {
                'summary': explanation.summary,
                'suggested_checks': explanation.suggested_checks
            }
        
        print(json.dumps(output, indent=2))
    else:
        print("=" * 60)
        print("VibeCAD Design Review Results")
        print("=" * 60)
        print(f"File: {pcb_path}")
        print()
        
        if all_passed:
            print("✓ All checks passed!")
        else:
            total_errors = sum(r.error_count for r in results)
            total_warnings = sum(r.warning_count for r in results)
            print(f"Found {total_errors} error(s), {total_warnings} warning(s)")
        
        print()
        print("-" * 60)
        
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"\n{status} | {result.check_name}")
            print(f"        {result.description}")
            
            for finding in result.findings:
                severity = str(finding.severity).upper()
                print(f"        [{severity}] {finding.rule_id}: {finding.message}")
                
                if args.verbose and finding.details:
                    for key, value in finding.details.items():
                        print(f"               {key}: {value}")
        
        # Get explanation if requested
        if args.explain:
            print()
            print("-" * 60)
            print("LLM EXPLANATION")
            print("-" * 60)
            
            explanation = plugin.explain_results(args.question)
            print(explanation.summary)
            
            if explanation.suggested_checks:
                print("\nSuggested follow-up checks:")
                for i, check in enumerate(explanation.suggested_checks, 1):
                    print(f"  {i}. {check}")
    
    # Exit with error code if any checks failed
    if not all_passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
