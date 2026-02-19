"""KiCad project library table helpers.

We update *project* tables (in the PCB's directory) so newly installed
VibeCAD libraries show up in KiCad choosers without touching global KiCad
configuration.

Files:
- fp-lib-table  (footprints)
- sym-lib-table (symbols)

Both use KiCad S-expression format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from vibecad.parsers.sexpr import SExprNode, parse_sexpr

logger = logging.getLogger(__name__)


def _quote_string(s: str) -> str:
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{s}"'


def _sexpr_to_string(node: SExprNode) -> str:
    parts: List[str] = ['(' + node.name]

    for v in node.values:
        if isinstance(v, SExprNode):
            parts.append(_sexpr_to_string(v))
        elif isinstance(v, (int, float)):
            parts.append(str(v))
        else:
            # Always quote string-ish values in tables for safety.
            parts.append(_quote_string(str(v)))

    for child in node.children:
        parts.append(_sexpr_to_string(child))

    parts.append(')')
    return ' '.join(parts)


def _load_table(path: Path, expected_root: str) -> SExprNode:
    if not path.exists():
        return SExprNode(name=expected_root, values=[], children=[SExprNode(name='version', values=[7], children=[])])

    try:
        text = path.read_text(encoding='utf-8', errors='ignore').strip()
        if not text:
            return SExprNode(name=expected_root, values=[], children=[SExprNode(name='version', values=[7], children=[])])
        root = parse_sexpr(text)
        if root.name != expected_root:
            logger.debug('Unexpected lib table root %s in %s', root.name, path)
            # Wrap or replace with fresh table
            root = SExprNode(name=expected_root, values=[], children=[SExprNode(name='version', values=[7], children=[])])
        return root
    except Exception:
        return SExprNode(name=expected_root, values=[], children=[SExprNode(name='version', values=[7], children=[])])


def _find_lib_nodes(root: SExprNode) -> List[SExprNode]:
    return [c for c in root.children if c.name == 'lib']


def _get_child_value(node: SExprNode, child_name: str) -> Optional[str]:
    c = node.get_child(child_name)
    if c is None:
        return None
    v = c.get_value(0)
    return str(v) if v is not None else None


def _set_child_value(node: SExprNode, child_name: str, value: str) -> None:
    c = node.get_child(child_name)
    if c is None:
        node.children.append(SExprNode(name=child_name, values=[value], children=[]))
    else:
        if c.values:
            c.values[0] = value
        else:
            c.values = [value]


def ensure_fp_lib(project_dir: Path, name: str, uri: str, descr: str = '') -> Path:
    """Ensure a footprint library entry exists in the project's fp-lib-table."""
    table_path = project_dir / 'fp-lib-table'
    root = _load_table(table_path, expected_root='fp_lib_table')

    existing = None
    for lib in _find_lib_nodes(root):
        if (_get_child_value(lib, 'name') or '') == name:
            existing = lib
            break

    if existing is None:
        lib = SExprNode(name='lib', values=[], children=[])
        _set_child_value(lib, 'name', name)
        _set_child_value(lib, 'type', 'KiCad')
        _set_child_value(lib, 'uri', uri)
        _set_child_value(lib, 'options', '')
        _set_child_value(lib, 'descr', descr)
        root.children.append(lib)
    else:
        _set_child_value(existing, 'type', 'KiCad')
        _set_child_value(existing, 'uri', uri)
        if descr:
            _set_child_value(existing, 'descr', descr)

    table_path.write_text(_sexpr_to_string(root) + '\n', encoding='utf-8')
    return table_path


def ensure_sym_lib(project_dir: Path, name: str, uri: str, descr: str = '') -> Path:
    """Ensure a symbol library entry exists in the project's sym-lib-table."""
    table_path = project_dir / 'sym-lib-table'
    root = _load_table(table_path, expected_root='sym_lib_table')

    existing = None
    for lib in _find_lib_nodes(root):
        if (_get_child_value(lib, 'name') or '') == name:
            existing = lib
            break

    if existing is None:
        lib = SExprNode(name='lib', values=[], children=[])
        _set_child_value(lib, 'name', name)
        _set_child_value(lib, 'type', 'KiCad')
        _set_child_value(lib, 'uri', uri)
        _set_child_value(lib, 'options', '')
        _set_child_value(lib, 'descr', descr)
        root.children.append(lib)
    else:
        _set_child_value(existing, 'type', 'KiCad')
        _set_child_value(existing, 'uri', uri)
        if descr:
            _set_child_value(existing, 'descr', descr)

    table_path.write_text(_sexpr_to_string(root) + '\n', encoding='utf-8')
    return table_path


def ensure_project_tables(project_dir: str, footprint_lib_dir: str, symbol_lib_paths: List[str]) -> None:
    """Update project tables so VibeCAD-installed libs are selectable in KiCad."""
    pd = Path(project_dir).expanduser().resolve()
    if not pd.is_dir():
        raise ValueError(f'project_dir is not a directory: {project_dir}')

    # Footprints: one shared .pretty library
    fp_dir = Path(footprint_lib_dir).expanduser().resolve()
    if fp_dir.is_dir():
        ensure_fp_lib(pd, name='VibeCAD', uri=str(fp_dir), descr='VibeCAD-managed footprints')

    # Symbols: each .kicad_sym file is a library
    for p in symbol_lib_paths or []:
        sp = Path(p).expanduser().resolve()
        if not sp.exists() or sp.suffix.lower() != '.kicad_sym':
            continue
        # Keep names short but unique
        lib_name = sp.stem
        ensure_sym_lib(pd, name=lib_name, uri=str(sp), descr='VibeCAD-managed symbols')
