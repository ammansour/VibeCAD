"""Minimal Markdown -> HTML renderer used by the wx UI.

We intentionally avoid external dependencies because KiCad ships its own
Python runtime and does not include common Markdown libraries.

Supported features (best-effort):
- Bold/italic, inline code
- Fenced code blocks (```)
- Unordered/ordered lists
- Headings (#, ##, ###)
- GitHub-style pipe tables

This is not a complete Markdown implementation; it is designed to render
LLM responses readably (bold, tables, etc.) inside wx.html2 WebViews.
"""

from __future__ import annotations

import re
from html import escape as _html_escape
from typing import List, Optional, Tuple


# --- Basic LaTeX rendering -------------------------------------------------
#
# KiCad's embedded Python/runtime + wx HTML controls make it impractical to
# depend on MathJax or heavy renderers. For our UI we want the *same font and
# size* as the surrounding text, but with common electrical/math symbols
# rendered nicely.
#
# Strategy: convert a small subset of LaTeX inside math delimiters ($...$,
# $$...$$, \(...\), \[...\]) into Unicode characters.


_latex_block_re = re.compile(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", re.DOTALL)
_latex_display_re = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_latex_inline_re = re.compile(r"(?<!\\)\$(?!\s)(.+?)(?<!\s)\$(?!\$)", re.DOTALL)


_LATEX_COMMAND_MAP = {
    # Common electrical units/symbols
    r"\Omega": "Ω",
    r"\ohm": "Ω",
    r"\mu": "µ",
    r"\micro": "µ",
    r"\degree": "°",

    # Operators / relations
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\mp": "∓",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\neq": "≠",
    r"\approx": "≈",
    r"\propto": "∝",
    r"\infty": "∞",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",

    # Greek (subset)
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\phi": "φ",
    r"\omega": "ω",
    r"\Delta": "Δ",
    r"\Omega": "Ω",
}


# Some LLM responses include unit commands without wrapping them in $...$.
# Convert only a very small, electrical-focused subset outside math.
_LATEX_OUTSIDE_MATH_MAP = {
    r"\Omega": "Ω",
    r"\ohm": "Ω",
    r"\mu": "µ",
    r"\micro": "µ",
    r"\degree": "°",
}


_SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
    "y": "ᵧ",
}


_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
}


def _strip_tex_wrappers(expr: str) -> str:
    out = expr or ""
    # Remove sizing/spacing wrappers that don't change meaning in our context.
    out = re.sub(r"\\(left|right)\b", "", out)
    out = re.sub(r"\\,|\\;|\\:|\\!", " ", out)
    out = re.sub(r"\\displaystyle\b", "", out)
    return out


def _latex_unwrap_text_commands(expr: str) -> str:
    # 	ext{...}, \mathrm{...}, \operatorname{...} -> ...
    out = expr
    for _ in range(4):
        changed = False
        for cmd in ("text", "mathrm", "operatorname"):
            pat = re.compile(rf"\\{cmd}\{{([^{{}}]*)\}}")
            new = pat.sub(r"\1", out)
            if new != out:
                changed = True
                out = new
        if not changed:
            break
    return out


def _map_scripts_or_fallback(text: str, mapping: dict, *, prefix: str, fallback: str) -> str:
    """Map characters to unicode scripts if all are representable.

    When a script cannot be represented in Unicode, we either:
    - return the original notation (fallback='literal'): `_f` stays `_f`
    - emit a placeholder (fallback='placeholder'): `[[SUB:f]]` / `[[SUP:f]]`
      which is later converted into HTML <sub>/<sup> tags.
    """

    if not text:
        return ""

    out_chars: List[str] = []
    for ch in text:
        repl = mapping.get(ch)
        if repl is None:
            if fallback == "placeholder":
                if prefix == "_":
                    return f"[[SUB:{text}]]"
                if prefix == "^":
                    return f"[[SUP:{text}]]"
            return prefix + text
        out_chars.append(repl)
    return "".join(out_chars)


def _apply_sub_sup(expr: str, *, fallback: str) -> str:
    # Convert _{...} / ^{...} and single-char _x/^2.
    # Keep it conservative: if we can't convert chars, we leave them as-is.
    out = expr

    def sub_repl(m: re.Match) -> str:
        content = m.group(1) or m.group(2) or ""
        return _map_scripts_or_fallback(content, _SUBSCRIPT_MAP, prefix="_", fallback=fallback)

    def sup_repl(m: re.Match) -> str:
        content = m.group(1) or m.group(2) or ""
        return _map_scripts_or_fallback(content, _SUPERSCRIPT_MAP, prefix="^", fallback=fallback)

    out = re.sub(r"_\{([^{}]+)\}|_([A-Za-z0-9()+\-=])", sub_repl, out)
    out = re.sub(r"\^\{([^{}]+)\}|\^([A-Za-z0-9()+\-=])", sup_repl, out)
    return out


def _apply_fractions(expr: str) -> str:
    # \frac{a}{b} -> (a)/(b), best-effort (no nested braces support beyond one level).
    out = expr
    frac_re = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    for _ in range(6):
        new = frac_re.sub(r"(\1)/(\2)", out)
        if new == out:
            break
        out = new
    return out


def _latex_to_unicode(expr: str, *, script_fallback: str) -> str:
    out = _strip_tex_wrappers(expr)
    out = _latex_unwrap_text_commands(out)
    out = _apply_fractions(out)

    # Command map replacements.
    for k, v in _LATEX_COMMAND_MAP.items():
        out = out.replace(k, v)

    # A few single-char escapes.
    out = out.replace(r"\_", "_")
    out = out.replace(r"\%", "%")
    out = out.replace(r"\$", "$")

    # Scripts should be processed before we drop braces.
    out = _apply_sub_sup(out, fallback=script_fallback)

    # Remove remaining grouping braces (after handling scripts and frac).
    out = out.replace("{", "").replace("}", "")
    # Tidy whitespace.
    out = re.sub(r"\s+", " ", out).strip()
    return out


def render_basic_latex(markdown_text: str, *, script_fallback: str = "literal") -> str:
    r"""Render a small LaTeX subset in markdown text as Unicode.

    Only content inside math delimiters is converted:
    - $...$ and $$...$$
    - \(...\) and \[...\]

    Code fences (``` blocks) are skipped.
    """

    text = markdown_text or ""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    out_lines: List[str] = []
    in_code = False
    fence_re = re.compile(r"^\s*```")

    for line in lines:
        if fence_re.match(line):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue

        # Skip inline code spans while processing LaTeX.
        parts = re.split(r"(`[^`]*`)", line)
        new_parts: List[str] = []
        for part in parts:
            if part.startswith("`") and part.endswith("`"):
                new_parts.append(part)
                continue

            def repl_display(m: re.Match) -> str:
                return _latex_to_unicode(m.group(1) or "", script_fallback=script_fallback)

            def repl_inline(m: re.Match) -> str:
                return _latex_to_unicode(m.group(1) or "", script_fallback=script_fallback)

            def repl_bracket(m: re.Match) -> str:
                expr = m.group(1) or m.group(2) or ""
                return _latex_to_unicode(expr, script_fallback=script_fallback)

            p = part
            p = _latex_block_re.sub(repl_bracket, p)
            p = _latex_display_re.sub(repl_display, p)
            p = _latex_inline_re.sub(repl_inline, p)

            # Also render a tiny subset of common unit commands outside $...$.
            for k, v in _LATEX_OUTSIDE_MATH_MAP.items():
                p = p.replace(k, v)
            new_parts.append(p)

        out_lines.append("".join(new_parts))

    return "\n".join(out_lines)


def escape_html(text: str) -> str:
    return _html_escape(text or "", quote=True)


def _split_table_row(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _is_table_sep(line: str) -> bool:
    # e.g. | --- | :---: | ---: |
    parts = _split_table_row(line)
    if len(parts) < 2:
        return False
    for p in parts:
        p = p.strip()
        if not p:
            return False
        if not re.fullmatch(r":?-{3,}:?", p):
            return False
    return True


_inline_code_re = re.compile(r"`([^`]+)`")
_bold_re = re.compile(r"(\*\*|__)(.+?)\1")
_italic_re = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)|(?<!_)_(?!\s)(.+?)(?<!\s)_(?!_)")
_link_re = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")

# Placeholders emitted by the LaTeX renderer for scripts that can't be mapped
# to Unicode. These are converted into HTML <sub>/<sup> tags.
_sub_ph_re = re.compile(r"\[\[SUB:([^\]]+)\]\]")
_sup_ph_re = re.compile(r"\[\[SUP:([^\]]+)\]\]")


def _format_inline(text: str) -> str:
    # Input must already be HTML-escaped.
    def repl_code(m: re.Match) -> str:
        return f"<code>{m.group(1)}</code>"

    def repl_bold(m: re.Match) -> str:
        return f"<strong>{m.group(2)}</strong>"

    def repl_italic(m: re.Match) -> str:
        inner = m.group(1) or m.group(2) or ""
        return f"<em>{inner}</em>"

    def repl_link(m: re.Match) -> str:
        label = m.group(1)
        href = m.group(2)
        # Keep href escaped; it already is if the whole text was escaped.
        return f"<a href=\"{href}\">{label}</a>"

    out = text
    out = _link_re.sub(repl_link, out)
    out = _inline_code_re.sub(repl_code, out)
    out = _bold_re.sub(repl_bold, out)
    out = _italic_re.sub(repl_italic, out)

    out = _sub_ph_re.sub(lambda m: f"<sub>{m.group(1)}</sub>", out)
    out = _sup_ph_re.sub(lambda m: f"<sup>{m.group(1)}</sup>", out)
    return out


def markdown_to_html_fragment(markdown_text: str) -> str:
    """Convert a small Markdown subset to an HTML fragment."""
    text = render_basic_latex(markdown_text or "", script_fallback="placeholder")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    out: List[str] = []

    in_code = False
    code_lang: Optional[str] = None
    code_lines: List[str] = []

    def flush_paragraph(par_lines: List[str]) -> None:
        if not par_lines:
            return
        escaped = escape_html("\n".join(par_lines))
        escaped = _format_inline(escaped)
        # Preserve line breaks within the paragraph.
        out.append("<p>" + escaped.replace("\n", "<br>") + "</p>")

    paragraph: List[str] = []

    def flush_code() -> None:
        nonlocal code_lines, code_lang
        code = escape_html("\n".join(code_lines))
        lang_class = f" class=\"lang-{escape_html(code_lang or '')}\"" if code_lang else ""
        out.append(f"<pre><code{lang_class}>" + code + "</code></pre>")
        code_lines = []
        code_lang = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code
        m_fence = re.match(r"^\s*```\s*([A-Za-z0-9_+-]+)?\s*$", line)
        if m_fence:
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph(paragraph)
                paragraph = []
                in_code = True
                code_lang = (m_fence.group(1) or "").strip() or None
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Blank line ends paragraphs/lists/tables blocks
        if not line.strip():
            flush_paragraph(paragraph)
            paragraph = []
            i += 1
            continue

        # Headings
        m_h = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if m_h:
            flush_paragraph(paragraph)
            paragraph = []
            level = len(m_h.group(1))
            content = _format_inline(escape_html(m_h.group(2)))
            out.append(f"<h{level}>" + content + f"</h{level}>")
            i += 1
            continue

        # Tables: header row + separator row
        if "|" in line and (i + 1) < len(lines) and _is_table_sep(lines[i + 1]):
            flush_paragraph(paragraph)
            paragraph = []

            header_cells = [_format_inline(escape_html(c)) for c in _split_table_row(line)]
            i += 2  # skip header + sep

            body_rows: List[List[str]] = []
            while i < len(lines):
                row_line = lines[i]
                if not row_line.strip() or "|" not in row_line:
                    break
                cells = [_format_inline(escape_html(c)) for c in _split_table_row(row_line)]
                body_rows.append(cells)
                i += 1

            thead = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in header_cells) + "</tr></thead>"
            tbody_parts: List[str] = []
            for r in body_rows:
                tbody_parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            tbody = "<tbody>" + "".join(tbody_parts) + "</tbody>"
            out.append("<table>" + thead + tbody + "</table>")
            continue

        # Lists
        m_ul = re.match(r"^\s*([-*•])\s+(.+)$", line)
        m_ol = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if m_ul or m_ol:
            flush_paragraph(paragraph)
            paragraph = []

            is_ordered = bool(m_ol)
            tag = "ol" if is_ordered else "ul"
            items: List[str] = []

            while i < len(lines):
                cur = lines[i]
                m2_ul = re.match(r"^\s*([-*•])\s+(.+)$", cur)
                m2_ol = re.match(r"^\s*(\d+)\.\s+(.+)$", cur)
                if is_ordered:
                    if not m2_ol:
                        break
                    item_text = m2_ol.group(2)
                else:
                    if not m2_ul:
                        break
                    item_text = m2_ul.group(2)

                items.append("<li>" + _format_inline(escape_html(item_text.strip())) + "</li>")
                i += 1

            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            continue

        # Default: part of a paragraph
        paragraph.append(line)
        i += 1

    # Flush any trailing blocks
    if in_code:
        flush_code()
    flush_paragraph(paragraph)

    return "\n".join(out)


def html_document(
    fragment: str,
    *,
    bg_hex: str = "#ffffff",
    fg_hex: str = "#000000",
    border_hex: Optional[str] = None,
) -> str:
    border_hex = border_hex or fg_hex

    # Avoid inner boxes: code blocks and tables inherit background.
    css = f"""
    html, body {{
        margin: 0;
        padding: 0;
        background: {bg_hex};
        color: {fg_hex};
        font-family: sans-serif;
        font-size: 14px;
        line-height: 1.35;
    }}
    p {{ margin: 0 0 8px 0; }}
    h1, h2, h3 {{ margin: 0 0 8px 0; font-size: 15px; }}
    code {{ font-family: monospace; }}
    pre {{
        margin: 6px 0;
        padding: 0;
        background: {bg_hex};
        overflow-x: auto;
        white-space: pre;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 6px 0;
        background: {bg_hex};
    }}
    th, td {{
        border: 1px solid {border_hex};
        padding: 4px 6px;
        vertical-align: top;
        background: {bg_hex};
    }}
    ul, ol {{ margin: 0 0 8px 18px; padding: 0; }}
    a {{ color: {fg_hex}; text-decoration: underline; }}

    /* Keep text size unchanged for scripts; only shift baseline. */
    sub, sup {{ font-size: 1em; line-height: 1; }}
    """

    body = fragment or ""
    return f"<!doctype html><html><head><meta charset=\"utf-8\"><style>{css}</style></head><body>{body}</body></html>"
