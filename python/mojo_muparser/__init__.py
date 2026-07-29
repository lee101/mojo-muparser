"""Mojo port of muparser's expression evaluation core."""

from .parser import Parser, ParserError, Program, compile_expression, evaluate

__version__ = "0.1.0"
__all__ = [
    "Parser",
    "ParserError",
    "Program",
    "compile_expression",
    "evaluate",
]
