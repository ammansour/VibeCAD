"""
S-Expression parser for KiCad file formats.

KiCad uses a Lisp-like S-expression format for its files.
This module provides a generic parser for that format.
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional, Any
import re


@dataclass
class SExprNode:
    """Represents a node in an S-expression tree."""
    name: str
    values: List[Union[str, float, int, 'SExprNode']] = field(default_factory=list)
    children: List['SExprNode'] = field(default_factory=list)
    
    def get_value(self, index: int = 0, default: Any = None) -> Any:
        """Get a value at the specified index."""
        if index < len(self.values):
            return self.values[index]
        return default
    
    def get_child(self, name: str) -> Optional['SExprNode']:
        """Get the first child with the given name."""
        for child in self.children:
            if child.name == name:
                return child
        return None
    
    def get_children(self, name: str) -> List['SExprNode']:
        """Get all children with the given name."""
        return [child for child in self.children if child.name == name]
    
    def get_nested_value(self, *path: str, default: Any = None) -> Any:
        """Navigate through nested children and get the final value."""
        node = self
        for name in path:
            node = node.get_child(name)
            if node is None:
                return default
        return node.get_value(0, default)


class SExprParser:
    """Parser for KiCad S-expression format."""
    
    # Token patterns
    TOKEN_PATTERN = re.compile(
        r'''
        (?P<LPAREN>\()|
        (?P<RPAREN>\))|
        (?P<STRING>"(?:[^"\\]|\\.)*")|
        (?P<NUMBER>-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|
        (?P<SYMBOL>[^\s()"]+)|
        (?P<WHITESPACE>\s+)
        ''',
        re.VERBOSE
    )
    
    def __init__(self, content: str):
        self.content = content
        self.tokens = self._tokenize()
        self.pos = 0
    
    def _tokenize(self) -> List[tuple]:
        """Tokenize the input content."""
        tokens = []
        for match in self.TOKEN_PATTERN.finditer(self.content):
            kind = match.lastgroup
            value = match.group()
            if kind != 'WHITESPACE':
                tokens.append((kind, value))
        return tokens
    
    def _current_token(self) -> Optional[tuple]:
        """Get the current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None
    
    def _consume(self, expected_kind: Optional[str] = None) -> tuple:
        """Consume and return the current token."""
        token = self._current_token()
        if token is None:
            raise SExprParseError("Unexpected end of input")
        if expected_kind and token[0] != expected_kind:
            raise SExprParseError(f"Expected {expected_kind}, got {token[0]}")
        self.pos += 1
        return token
    
    def parse(self) -> SExprNode:
        """Parse the S-expression and return the root node."""
        return self._parse_node()
    
    def _parse_node(self) -> SExprNode:
        """Parse a single S-expression node."""
        self._consume('LPAREN')
        
        # First element should be the node name (symbol, string, or number for layer defs)
        name_token = self._consume()
        if name_token[0] not in ('SYMBOL', 'STRING', 'NUMBER'):
            raise SExprParseError(f"Expected symbol/number for node name, got {name_token[0]}")
        
        if name_token[0] == 'STRING':
            name = self._unquote(name_token[1])
        else:
            name = str(name_token[1])
        
        node = SExprNode(name=name)
        
        # Parse values and children
        while True:
            token = self._current_token()
            if token is None:
                raise SExprParseError("Unexpected end of input in node")
            
            if token[0] == 'RPAREN':
                self._consume()
                break
            elif token[0] == 'LPAREN':
                node.children.append(self._parse_node())
            elif token[0] == 'NUMBER':
                self._consume()
                # Try to parse as int or float
                try:
                    if '.' in token[1] or 'e' in token[1].lower():
                        node.values.append(float(token[1]))
                    else:
                        node.values.append(int(token[1]))
                except ValueError:
                    node.values.append(token[1])
            elif token[0] == 'STRING':
                self._consume()
                node.values.append(self._unquote(token[1]))
            elif token[0] == 'SYMBOL':
                self._consume()
                node.values.append(token[1])
        
        return node
    
    def _unquote(self, s: str) -> str:
        """Remove quotes and unescape a string."""
        if s.startswith('"') and s.endswith('"'):
            s = s[1:-1]
            # Handle escape sequences
            s = s.replace('\\"', '"')
            s = s.replace('\\\\', '\\')
            s = s.replace('\\n', '\n')
            s = s.replace('\\t', '\t')
        return s


class SExprParseError(Exception):
    """Exception raised for S-expression parsing errors."""
    pass


def parse_sexpr(content: str) -> SExprNode:
    """Parse S-expression content and return the root node."""
    parser = SExprParser(content)
    return parser.parse()
