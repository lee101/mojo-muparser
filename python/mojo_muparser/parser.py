"""muparser-compatible tokenization, bytecode compilation, and Python API."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np

from ._lib import lib

LE, GE, NE, EQ, LT, GT, ADD, SUB, MUL, DIV, POW, LAND, LOR, ASSIGN = range(14)
VAR, VAL, VARPOW2, VARPOW3, VARPOW4, VARMUL = range(20, 26)
NEG, POS = 30, 31
(
    SIN,
    COS,
    TAN,
    COT,
    ASIN,
    ACOS,
    ATAN,
    ATAN2,
    SINH,
    COSH,
    TANH,
    COTH,
    ASINH,
    ACOSH,
    ATANH,
    LOG2,
    LOG10,
    LOG,
    EXP,
    SQRT,
    SIGN,
    RINT,
    ABS,
    SUM,
    AVG,
    MIN,
    MAX,
) = range(40, 67)
SELECT, SEQUENCE, DROP = 70, 71, 72
CUSTOM_BASE = 100

_FUNCTIONS = {
    "sin": (SIN, 1),
    "cos": (COS, 1),
    "tan": (TAN, 1),
    "cot": (COT, 1),
    "asin": (ASIN, 1),
    "acos": (ACOS, 1),
    "atan": (ATAN, 1),
    "atan2": (ATAN2, 2),
    "sinh": (SINH, 1),
    "cosh": (COSH, 1),
    "tanh": (TANH, 1),
    "coth": (COTH, 1),
    "asinh": (ASINH, 1),
    "acosh": (ACOSH, 1),
    "atanh": (ATANH, 1),
    "log2": (LOG2, 1),
    "log10": (LOG10, 1),
    "log": (LOG, 1),
    "ln": (LOG, 1),
    "exp": (EXP, 1),
    "sqrt": (SQRT, 1),
    "sign": (SIGN, 1),
    "rint": (RINT, 1),
    "abs": (ABS, 1),
    "sum": (SUM, None),
    "avg": (AVG, None),
    "min": (MIN, None),
    "max": (MAX, None),
}
_OPERATORS = {
    "<=": (LE, 5, "left"),
    ">=": (GE, 5, "left"),
    "!=": (NE, 5, "left"),
    "==": (EQ, 5, "left"),
    "<": (LT, 5, "left"),
    ">": (GT, 5, "left"),
    "+": (ADD, 6, "left"),
    "-": (SUB, 6, "left"),
    "*": (MUL, 7, "left"),
    "/": (DIV, 7, "left"),
    "^": (POW, 8, "right"),
    "&&": (LAND, 2, "left"),
    "||": (LOR, 1, "left"),
    "=": (ASSIGN, -1, "left"),
}
_NUMBER = re.compile(
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
)
_NAME = re.compile(r"[0-9_A-Za-z]+")
_VALID_IDENTIFIER = re.compile(r"[A-Za-z_][0-9_A-Za-z]*\Z")
_PI = 3.141592653589


class ParserError(ValueError):
    def __init__(self, message: str, expression: str = "", position: int = -1):
        self.expression = expression
        self.position = position
        suffix = f" at position {position}" if position >= 0 else ""
        super().__init__(message + suffix)


@dataclass(frozen=True)
class Function:
    name: str
    opcode: int
    argc: int | None
    callback: Callable[..., Any] | None
    optimizable: bool = True


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    position: int
    value: float | None = None
    variable: int = -1
    function: Function | None = None
    assign_variable: int = -1


@dataclass(frozen=True)
class Program:
    expression: str
    variable_names: tuple[str, ...]
    code: np.ndarray
    constants: np.ndarray
    functions: tuple[Function, ...]
    has_assignment: bool

    @property
    def max_stack(self) -> int:
        depth = maximum = 0
        for op, _, argc in self.code:
            op = int(op)
            if op in (VAR, VAL, VARPOW2, VARPOW3, VARPOW4, VARMUL):
                depth += 1
            elif op in range(14) or op == SEQUENCE:
                depth -= 1
            elif op == SELECT:
                depth -= 2
            elif op == DROP:
                depth -= 1
            elif op >= 40:
                depth -= int(argc) - 1
            maximum = max(maximum, depth)
        return maximum


def _validate_name(name: str) -> None:
    if len(name) > 100:
        raise ParserError("identifier is too long")
    if not _VALID_IDENTIFIER.fullmatch(name):
        raise ParserError(f"invalid identifier {name!r}")


# muparser: src/muParserTokenReader.cpp ParserTokenReader::ReadNextToken
def _tokenize(
    expression: str,
    variables: dict[str, int],
    constants: dict[str, float],
    functions: dict[str, Function],
    auto_variables: bool,
) -> tuple[list[Token], dict[str, int]]:
    tokens: list[Token] = []
    position = 0
    size = len(expression)
    while position < size:
        char = expression[position]
        if ord(char) <= 0x20:
            if 14 <= ord(char) <= 31:
                raise ParserError(
                    "invalid control character", expression, position
                )
            position += 1
            continue
        number = _NUMBER.match(expression, position)
        if number:
            text = number.group()
            tokens.append(Token("value", text, position, float(text)))
            position = number.end()
            continue
        name_match = _NAME.match(expression, position)
        if name_match:
            text = name_match.group()
            end = name_match.end()
            if text in functions and end < size and expression[end] == "(":
                tokens.append(
                    Token(
                        "function",
                        text,
                        position,
                        function=functions[text],
                    )
                )
            elif text in constants:
                tokens.append(
                    Token("value", text, position, float(constants[text]))
                )
            elif text in variables:
                tokens.append(
                    Token(
                        "variable",
                        text,
                        position,
                        variable=variables[text],
                    )
                )
            elif auto_variables and _VALID_IDENTIFIER.fullmatch(text):
                variables[text] = len(variables)
                tokens.append(
                    Token(
                        "variable",
                        text,
                        position,
                        variable=variables[text],
                    )
                )
            else:
                raise ParserError(
                    f"unassignable token {text!r}", expression, position
                )
            position = end
            continue
        matched = False
        for operator in (
            "<=",
            ">=",
            "!=",
            "==",
            "&&",
            "||",
            "+",
            "-",
            "*",
            "/",
            "^",
            "=",
            "<",
            ">",
        ):
            if expression.startswith(operator, position):
                tokens.append(Token("operator", operator, position))
                position += len(operator)
                matched = True
                break
        if matched:
            continue
        punctuation = {
            "(": "left",
            ")": "right",
            ",": "comma",
            "?": "question",
            ":": "colon",
        }
        if char in punctuation:
            tokens.append(Token(punctuation[char], char, position))
            position += 1
            continue
        raise ParserError(
            f"unassignable token {expression[position:]!r}",
            expression,
            position,
        )
    if not tokens:
        raise ParserError("unexpected end of expression", expression, 0)
    return tokens, variables


def _emit_operator(output: list[tuple[int, int, int]], token: Token) -> None:
    if token.kind == "unary":
        output.append((NEG if token.text == "-" else POS, 0, 1))
    elif token.kind == "colon":
        output.append((SELECT, 0, 3))
    elif token.kind == "operator":
        opcode = _OPERATORS[token.text][0]
        output.append((opcode, token.assign_variable, 2))
    else:
        raise RuntimeError(f"cannot emit {token.kind}")


# muparser: src/muParserBase.cpp ParserBase::CreateRPN
def _shunting_yard(
    expression: str, tokens: list[Token]
) -> list[tuple[int, int, int]]:
    output: list[tuple[int, int, int]] = []
    operators: list[Token] = []
    frames: list[dict[str, Any]] = []
    sequence_count = 0
    expect_value = True
    previous: Token | None = None

    def pop_operator() -> None:
        token = operators.pop()
        _emit_operator(output, token)

    for token in tokens:
        if token.kind in ("value", "variable"):
            if not expect_value:
                raise ParserError(
                    f"unexpected value {token.text!r}",
                    expression,
                    token.position,
                )
            if token.kind == "value":
                output.append((VAL, -1, 0))
            else:
                output.append((VAR, token.variable, 0))
            expect_value = False
        elif token.kind == "function":
            if not expect_value:
                raise ParserError(
                    f"unexpected function {token.text!r}",
                    expression,
                    token.position,
                )
            if previous and previous.kind == "unary" and previous.text == "+":
                raise ParserError(
                    "unary plus in front of function",
                    expression,
                    token.position,
                )
            operators.append(token)
        elif token.kind == "left":
            if not expect_value and not (
                previous and previous.kind == "function"
            ):
                raise ParserError(
                    "unexpected opening parenthesis",
                    expression,
                    token.position,
                )
            is_function = bool(
                previous and previous.kind == "function"
            )
            operators.append(token)
            frames.append({"function": is_function, "commas": 0})
            expect_value = True
        elif token.kind == "right":
            if not frames:
                raise ParserError(
                    "unexpected closing parenthesis",
                    expression,
                    token.position,
                )
            empty = bool(previous and previous.kind == "left")
            if expect_value and not empty:
                raise ParserError(
                    "unexpected closing parenthesis",
                    expression,
                    token.position,
                )
            while operators and operators[-1].kind != "left":
                if operators[-1].kind == "question":
                    raise ParserError(
                        "missing colon", expression, token.position
                    )
                pop_operator()
            if not operators:
                raise ParserError(
                    "missing opening parenthesis",
                    expression,
                    token.position,
                )
            operators.pop()
            frame = frames.pop()
            if frame["function"]:
                if not operators or operators[-1].kind != "function":
                    raise ParserError(
                        "internal function stack error",
                        expression,
                        token.position,
                    )
                function_token = operators.pop()
                function = function_token.function
                assert function is not None
                argc = 0 if empty else frame["commas"] + 1
                if function.argc is None:
                    if argc == 0:
                        raise ParserError(
                            f"too few parameters for {function.name}",
                            expression,
                            token.position,
                        )
                elif argc < function.argc:
                    raise ParserError(
                        f"too few parameters for {function.name}",
                        expression,
                        token.position,
                    )
                elif argc > function.argc:
                    raise ParserError(
                        f"too many parameters for {function.name}",
                        expression,
                        token.position,
                    )
                output.append((function.opcode, 0, argc))
            elif empty:
                raise ParserError(
                    "empty parentheses", expression, token.position
                )
            expect_value = False
        elif token.kind == "comma":
            if expect_value:
                raise ParserError(
                    "unexpected argument separator",
                    expression,
                    token.position,
                )
            if frames:
                while operators and operators[-1].kind != "left":
                    pop_operator()
                if not frames[-1]["function"]:
                    raise ParserError(
                        "unexpected argument separator",
                        expression,
                        token.position,
                    )
                frames[-1]["commas"] += 1
            else:
                while operators:
                    pop_operator()
                sequence_count += 1
            expect_value = True
        elif token.kind == "question":
            if expect_value:
                raise ParserError(
                    "unexpected conditional", expression, token.position
                )
            while operators and operators[-1].kind not in (
                "left",
                "question",
                "colon",
            ):
                pop_operator()
            operators.append(token)
            expect_value = True
        elif token.kind == "colon":
            if expect_value:
                raise ParserError(
                    "misplaced colon", expression, token.position
                )
            while operators and operators[-1].kind != "question":
                if operators[-1].kind == "left":
                    raise ParserError(
                        "misplaced colon", expression, token.position
                    )
                pop_operator()
            if not operators:
                raise ParserError(
                    "misplaced colon", expression, token.position
                )
            operators[-1] = Token("colon", ":", token.position)
            expect_value = True
        elif token.kind == "operator":
            if expect_value:
                if token.text not in ("+", "-") or (
                    previous and previous.kind == "unary"
                ):
                    raise ParserError(
                        f"unexpected operator {token.text!r}",
                        expression,
                        token.position,
                    )
                if previous and previous.kind == "operator" and (
                    token.text == "+" or previous.text == "+"
                ):
                    raise ParserError(
                        f"unexpected operator {token.text!r}",
                        expression,
                        token.position,
                    )
                unary = Token("unary", token.text, token.position)
                operators.append(unary)
                previous = unary
                continue
            opcode, precedence, associativity = _OPERATORS[token.text]
            assign_variable = -1
            if opcode == ASSIGN:
                if previous is None or previous.kind != "variable":
                    raise ParserError(
                        "assignment target must be a variable",
                        expression,
                        token.position,
                    )
                assign_variable = previous.variable
                token = Token(
                    "operator",
                    token.text,
                    token.position,
                    assign_variable=assign_variable,
                )
            while operators and operators[-1].kind in (
                "operator",
                "unary",
            ):
                top = operators[-1]
                if top.kind == "unary":
                    top_precedence = 7
                else:
                    top_precedence = _OPERATORS[top.text][1]
                should_pop = (
                    top_precedence > precedence
                    or (
                        top_precedence == precedence
                        and associativity == "left"
                    )
                )
                if not should_pop:
                    break
                pop_operator()
            operators.append(token)
            expect_value = True
        previous = token

    if expect_value:
        raise ParserError(
            "unexpected end of expression", expression, len(expression)
        )
    if frames:
        raise ParserError(
            "missing closing parenthesis", expression, len(expression)
        )
    while operators:
        if operators[-1].kind in ("left", "question"):
            raise ParserError(
                "incomplete expression", expression, len(expression)
            )
        pop_operator()
    for _ in range(sequence_count):
        output.append((SEQUENCE, 0, 2))
    return output


# muparser: include/muParserTemplateMagic.h MathImpl
def _constant_function(opcode: int, args: list[float]) -> float:
    with np.errstate(all="ignore"):
        value = np.float64(args[0]) if args else np.float64(0.0)
        if opcode == SIN:
            return float(np.sin(value))
        if opcode == COS:
            return float(np.cos(value))
        if opcode == TAN:
            return float(np.tan(value))
        if opcode == COT:
            return float(1.0 / np.tan(value))
        if opcode == ASIN:
            return float(np.arcsin(value))
        if opcode == ACOS:
            return float(np.arccos(value))
        if opcode == ATAN:
            return float(np.arctan(value))
        if opcode == ATAN2:
            return float(np.arctan2(args[0], args[1]))
        if opcode == SINH:
            return float(np.sinh(value))
        if opcode == COSH:
            return float(np.cosh(value))
        if opcode == TANH:
            return float(np.tanh(value))
        if opcode == COTH:
            return float(1.0 / np.tanh(value))
        if opcode == ASINH:
            return float(np.log(value + np.sqrt(value * value + 1.0)))
        if opcode == ACOSH:
            return float(np.log(value + np.sqrt(value * value - 1.0)))
        if opcode == ATANH:
            return float(0.5 * np.log((1.0 + value) / (1.0 - value)))
        if opcode == LOG2:
            return float(np.log(value) / np.log(np.float64(2.0)))
        if opcode == LOG10:
            return float(np.log10(value))
        if opcode == LOG:
            return float(np.log(value))
        if opcode == EXP:
            return float(np.exp(value))
        if opcode == SQRT:
            return float(np.sqrt(value))
        if opcode == SIGN:
            return float(-1 if value < 0 else 1 if value > 0 else 0)
        if opcode == RINT:
            return float(np.floor(value + 0.5))
        if opcode == ABS:
            return float(value if value >= 0 else -value)
        if opcode == SUM:
            return float(sum(args))
        if opcode == AVG:
            return float(sum(args) / len(args))
        if opcode == MIN:
            result = args[0]
            for candidate in args:
                result = candidate if candidate < result else result
            return float(result)
        if opcode == MAX:
            result = args[0]
            for candidate in args:
                result = candidate if candidate > result else result
            return float(result)
    raise RuntimeError(f"unknown function opcode {opcode}")


def _constant_binary(opcode: int, left: float, right: float) -> float | None:
    if opcode == DIV and right == 0:
        return None
    with np.errstate(all="ignore"):
        a, b = np.float64(left), np.float64(right)
        if opcode == LE:
            return float(a <= b)
        if opcode == GE:
            return float(a >= b)
        if opcode == NE:
            return float(a != b)
        if opcode == EQ:
            return float(a == b)
        if opcode == LT:
            return float(a < b)
        if opcode == GT:
            return float(a > b)
        if opcode == ADD:
            return float(a + b)
        if opcode == SUB:
            return float(a - b)
        if opcode == MUL:
            return float(a * b)
        if opcode == DIV:
            return float(a / b)
        if opcode == POW:
            return float(np.power(a, b))
        if opcode == LAND:
            return float(bool(int(a)) and bool(int(b)))
        if opcode == LOR:
            return float(bool(int(a)) or bool(int(b)))
    return None


@dataclass
class _Node:
    instructions: list[tuple[int, int, int]]
    constant: float | None = None


# muparser: src/muParserBytecode.cpp ParserByteCode::ConstantFolding
def _fold_constants(
    raw: list[tuple[int, int, int]], literal_values: list[float]
) -> tuple[list[tuple[int, int, int]], list[float]]:
    stack: list[_Node] = []
    constants: list[float] = []
    literal_index = 0

    def literal(value: float) -> _Node:
        index = len(constants)
        constants.append(value)
        return _Node([(VAL, index, 0)], value)

    for op, argument, argc in raw:
        if op == VAL:
            stack.append(literal(literal_values[literal_index]))
            literal_index += 1
        elif op in (VAR, VARPOW2, VARPOW3, VARPOW4, VARMUL):
            stack.append(_Node([(op, argument, argc)]))
        elif op in (NEG, POS):
            node = stack.pop()
            if node.constant is not None:
                stack.append(literal(-node.constant if op == NEG else node.constant))
            else:
                stack.append(_Node(node.instructions + [(op, 0, 1)]))
        elif 0 <= op <= 12:
            right, left = stack.pop(), stack.pop()
            result = None
            if left.constant is not None and right.constant is not None:
                result = _constant_binary(op, left.constant, right.constant)
            if result is not None:
                stack.append(literal(result))
            elif (
                op == POW
                and right.constant in (0.0, 1.0, 2.0, 3.0, 4.0)
                and len(left.instructions) == 1
                and left.instructions[0][0] == VAR
            ):
                variable = left.instructions[0][1]
                if right.constant == 0.0:
                    stack.append(literal(1.0))
                elif right.constant == 1.0:
                    stack.append(left)
                else:
                    stack.append(
                        _Node(
                            [
                                (
                                    {
                                        2.0: VARPOW2,
                                        3.0: VARPOW3,
                                        4.0: VARPOW4,
                                    }[right.constant],
                                    variable,
                                    0,
                                )
                            ]
                        )
                    )
            elif (
                op == MUL
                and len(left.instructions) == 1
                and len(right.instructions) == 1
                and left.instructions[0][0] == VAR
                and right.instructions[0] == left.instructions[0]
            ):
                stack.append(
                    _Node([(VARPOW2, left.instructions[0][1], 0)])
                )
            else:
                stack.append(
                    _Node(left.instructions + right.instructions + [(op, 0, 2)])
                )
        elif op == SEQUENCE:
            right, left = stack.pop(), stack.pop()
            stack.append(
                _Node(
                    left.instructions
                    + right.instructions
                    + [(SEQUENCE, 0, 2)],
                    right.constant,
                )
            )
        elif op == ASSIGN:
            right, left = stack.pop(), stack.pop()
            stack.append(
                _Node(
                    left.instructions
                    + right.instructions
                    + [(op, argument, 2)]
                )
            )
        elif op == SELECT:
            false_node, true_node, condition = (
                stack.pop(),
                stack.pop(),
                stack.pop(),
            )
            if any(
                instruction[0] == ASSIGN
                for node in (true_node, false_node)
                for instruction in node.instructions
            ):
                raise ParserError(
                    "assignments inside conditional branches are not supported"
                )
            if condition.constant is not None:
                stack.append(true_node if condition.constant != 0 else false_node)
            else:
                stack.append(
                    _Node(
                        condition.instructions
                        + true_node.instructions
                        + false_node.instructions
                        + [(SELECT, 0, 3)]
                    )
                )
        elif op >= 40:
            args = [] if argc == 0 else stack[-argc:]
            if argc:
                del stack[-argc:]
            if op < CUSTOM_BASE and all(
                node.constant is not None for node in args
            ):
                stack.append(
                    literal(
                        _constant_function(
                            op, [float(node.constant) for node in args]
                        )
                    )
                )
            else:
                instructions = [
                    instruction
                    for node in args
                    for instruction in node.instructions
                ]
                stack.append(_Node(instructions + [(op, argument, argc)]))
        else:
            raise RuntimeError(f"unknown opcode {op}")
    instructions = [item for node in stack for item in node.instructions]
    if len(stack) != 1:
        raise ParserError("expression did not reduce to one result")
    return instructions, constants


def compile_expression(
    expression: str,
    variable_names: tuple[str, ...],
    constants: dict[str, float],
    user_functions: dict[str, Function],
    *,
    auto_variables: bool = False,
) -> Program:
    variables = {name: index for index, name in enumerate(variable_names)}
    functions = {
        name: Function(name, opcode, argc, None)
        for name, (opcode, argc) in _FUNCTIONS.items()
    }
    functions.update(user_functions)
    tokens, variables = _tokenize(
        expression,
        variables,
        constants,
        functions,
        auto_variables,
    )
    literal_values = [
        float(token.value) for token in tokens if token.kind == "value"
    ]
    raw = _shunting_yard(expression, tokens)
    instructions, values = _fold_constants(raw, literal_values)
    code = np.ascontiguousarray(instructions, dtype=np.int64).reshape(-1, 3)
    constant_array = np.ascontiguousarray(values or [0.0], dtype=np.float64)
    ordered_names = tuple(
        name for name, _ in sorted(variables.items(), key=lambda item: item[1])
    )
    program = Program(
        expression,
        ordered_names,
        code,
        constant_array,
        tuple(
            function
            for function in user_functions.values()
            if np.any(code[:, 0] == function.opcode)
        ),
        any(int(op) == ASSIGN for op in code[:, 0]),
    )
    if len(code) > 256:
        raise ParserError("expression exceeds the 256-instruction limit")
    if program.max_stack > 64:
        raise ParserError("expression exceeds the 64-value stack limit")
    return program


def _array_function(opcode: int, args: list[Any]) -> Any:
    if opcode < CUSTOM_BASE:
        scalar = all(np.ndim(arg) == 0 for arg in args)
        result = _constant_function(opcode, [float(arg) for arg in args]) if scalar else None
        if scalar:
            return result
        value = args[0]
        operations = {
            SIN: np.sin,
            COS: np.cos,
            TAN: np.tan,
            COT: lambda x: 1.0 / np.tan(x),
            ASIN: np.arcsin,
            ACOS: np.arccos,
            ATAN: np.arctan,
            SINH: np.sinh,
            COSH: np.cosh,
            TANH: np.tanh,
            COTH: lambda x: 1.0 / np.tanh(x),
            ASINH: lambda x: np.log(x + np.sqrt(x * x + 1.0)),
            ACOSH: lambda x: np.log(x + np.sqrt(x * x - 1.0)),
            ATANH: lambda x: 0.5 * np.log((1.0 + x) / (1.0 - x)),
            LOG2: lambda x: np.log(x) / np.log(2.0),
            LOG10: np.log10,
            LOG: np.log,
            EXP: np.exp,
            SQRT: np.sqrt,
            SIGN: np.sign,
            RINT: lambda x: np.floor(x + 0.5),
            ABS: np.abs,
        }
        if opcode == ATAN2:
            return np.arctan2(args[0], args[1])
        if opcode == SUM:
            return sum(args)
        if opcode == AVG:
            return sum(args) / len(args)
        if opcode == MIN:
            result = args[0]
            for candidate in args[1:]:
                result = np.where(candidate < result, candidate, result)
            return result
        if opcode == MAX:
            result = args[0]
            for candidate in args[1:]:
                result = np.where(candidate > result, candidate, result)
            return result
        return operations[opcode](value)
    raise RuntimeError("custom function requires its callback")


def _execute_python(
    program: Program,
    values: dict[str, np.ndarray],
    custom_by_opcode: dict[int, Function],
) -> Any:
    stack: list[Any] = []
    variables = [values[name] for name in program.variable_names]
    for op_value, argument_value, argc_value in program.code:
        op, argument, argc = int(op_value), int(argument_value), int(argc_value)
        if op == VAR:
            stack.append(variables[argument])
        elif op == VARPOW2:
            value = variables[argument]
            stack.append(value * value)
        elif op == VARPOW3:
            value = variables[argument]
            stack.append(value * value * value)
        elif op == VARPOW4:
            value = variables[argument]
            stack.append(value * value * value * value)
        elif op == VARMUL:
            stack.append(
                variables[argument] * program.constants[argc]
                + program.constants[argc + 1]
            )
        elif op == VAL:
            stack.append(program.constants[argument])
        elif 0 <= op <= 12:
            right, left = stack.pop(), stack.pop()
            with np.errstate(all="ignore"):
                operations = {
                    LE: lambda: left <= right,
                    GE: lambda: left >= right,
                    NE: lambda: left != right,
                    EQ: lambda: left == right,
                    LT: lambda: left < right,
                    GT: lambda: left > right,
                    ADD: lambda: left + right,
                    SUB: lambda: left - right,
                    MUL: lambda: left * right,
                    DIV: lambda: left / right,
                    POW: lambda: np.power(left, right),
                    LAND: lambda: np.logical_and(left != 0, right != 0),
                    LOR: lambda: np.logical_or(left != 0, right != 0),
                }
                stack.append(np.asarray(operations[op](), dtype=np.float64))
        elif op == ASSIGN:
            right = stack.pop()
            stack.pop()
            variables[argument][...] = right
            stack.append(variables[argument])
        elif op == NEG:
            stack[-1] = -stack[-1]
        elif op == POS:
            pass
        elif op == SELECT:
            false_value, true_value, condition = (
                stack.pop(),
                stack.pop(),
                stack.pop(),
            )
            stack.append(np.where(condition != 0, true_value, false_value))
        elif op == DROP:
            stack.pop()
        elif op == SEQUENCE:
            right = stack.pop()
            stack.pop()
            stack.append(right)
        elif op >= 40:
            args = [] if argc == 0 else stack[-argc:]
            if argc:
                del stack[-argc:]
            if op >= CUSTOM_BASE:
                stack.append(custom_by_opcode[op].callback(*args))
            else:
                with np.errstate(all="ignore"):
                    stack.append(_array_function(op, args))
    return stack[-1]


def _prepare_values(
    program: Program, bindings: dict[str, Any]
) -> tuple[dict[str, np.ndarray], tuple[int, ...], bool]:
    missing = [name for name in program.variable_names if name not in bindings]
    if missing:
        raise ParserError(f"undefined variable {missing[0]!r}")
    shape: tuple[int, ...] | None = None
    prepared: dict[str, np.ndarray] = {}
    all_scalar = True
    for name in program.variable_names:
        source = bindings[name]
        original = source if isinstance(source, np.ndarray) else None
        unconverted = np.asarray(source)
        if unconverted.dtype.kind in "fc" and unconverted.dtype.itemsize > 8:
            raise TypeError(
                f"variable {name!r} has dtype {unconverted.dtype}; "
                "values must be representable as float64"
            )
        if unconverted.dtype.kind in "iu":
            converted = unconverted.astype(np.float64)
            if not np.array_equal(
                unconverted, converted.astype(unconverted.dtype)
            ):
                raise ValueError(
                    f"variable {name!r} contains integers that cannot be "
                    "represented exactly as float64"
                )
            array = converted
        else:
            try:
                array = np.asarray(source, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"variable {name!r} must contain real numeric values"
                ) from error
        if array.ndim:
            all_scalar = False
            if shape is None:
                shape = array.shape
            elif array.shape != shape:
                raise ValueError("all array variables must have the same shape")
            if (
                original is not None
                and original.dtype == np.float64
                and original.flags.c_contiguous
            ):
                array = original
            else:
                array = np.ascontiguousarray(array)
        else:
            array = np.ascontiguousarray(array.reshape(1))
        prepared[name] = array
    return prepared, shape or (), all_scalar


def execute(
    program: Program,
    bindings: dict[str, Any],
    *,
    workers: int | None = None,
    out: np.ndarray | None = None,
) -> float | np.ndarray:
    if workers is not None and (
        isinstance(workers, bool) or not isinstance(workers, int) or workers < 1
    ):
        raise ValueError("workers must be a positive integer")
    prepared, shape, scalar = _prepare_values(program, bindings)
    n = int(np.prod(shape)) if shape else 1
    if program.has_assignment:
        for op, argument, _ in program.code:
            if int(op) == ASSIGN:
                name = program.variable_names[int(argument)]
                target = prepared[name]
                if n > 1 and target.size == 1:
                    raise ValueError(
                        "cannot assign a broadcast scalar during bulk evaluation"
                    )
                if not target.flags.writeable:
                    raise ValueError("assignment target is not writable")
                source = bindings[name]
                if not isinstance(source, np.ndarray) or not np.shares_memory(
                    target, source
                ):
                    raise ValueError(
                        "assignment target must be a writable C-contiguous "
                        "float64 NumPy array"
                    )
    custom = {function.opcode: function for function in program.functions}
    if custom:
        result = _execute_python(program, prepared, custom)
        array = np.asarray(result, dtype=np.float64)
        if out is not None:
            if (
                out.shape != array.shape
                or out.dtype != np.float64
                or not out.flags.c_contiguous
                or not out.flags.writeable
            ):
                raise ValueError(
                    "out must be a writable C-contiguous float64 array "
                    "with the result shape"
                )
            out[...] = array
            return out
        return float(array.reshape(-1)[0]) if scalar else array

    addresses = np.ascontiguousarray(
        [prepared[name].ctypes.data for name in program.variable_names],
        dtype=np.int64,
    )
    strides = np.ascontiguousarray(
        [0 if prepared[name].size == 1 else 1 for name in program.variable_names],
        dtype=np.int64,
    )
    if out is None:
        destination = np.empty(n, dtype=np.float64)
    else:
        if (
            out.dtype != np.float64
            or out.shape != shape
            or not out.flags.c_contiguous
            or not out.flags.writeable
        ):
            raise ValueError(
                "out must be a writable C-contiguous float64 array "
                "with the result shape"
            )
        destination = out.reshape(-1)
    requested = workers if workers is not None else min(os.cpu_count() or 1, 16)
    status = lib().mmup_evaluate_f64(
        program.code.ctypes.data,
        len(program.code),
        program.constants.ctypes.data,
        addresses.ctypes.data,
        strides.ctypes.data,
        destination.ctypes.data,
        n,
        requested,
    )
    if status:
        raise RuntimeError(f"Mojo evaluator rejected bytecode (status {status})")
    if scalar:
        return float(destination[0])
    if out is not None:
        return out
    return destination.reshape(shape)


class Parser:
    """Stateful parser following muparser's define/set/evaluate workflow."""

    def __init__(self, expression: str | None = None):
        self._expression = ""
        self._variables: dict[str, np.ndarray] = {}
        self._constants = {"_pi": _PI, "_e": math.e}
        self._functions: dict[str, Function] = {}
        self._program: Program | None = None
        if expression is not None:
            self.set_expression(expression)

    @property
    def expression(self) -> str:
        return self._expression

    @property
    def variables(self) -> dict[str, np.ndarray]:
        return dict(self._variables)

    def define_var(self, name: str, value: Any = 0.0) -> "Parser":
        _validate_name(name)
        if name in self._constants:
            raise ParserError(f"name conflict for {name!r}")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 0:
            array = np.array(value, dtype=np.float64)
        elif not (
            isinstance(value, np.ndarray)
            and value.dtype == np.float64
            and value.flags.c_contiguous
        ):
            array = np.ascontiguousarray(array)
        self._variables[name] = array
        self._program = None
        return self

    def define_const(self, name: str, value: float) -> "Parser":
        _validate_name(name)
        if name in self._variables:
            raise ParserError(f"name conflict for {name!r}")
        self._constants[name] = float(value)
        self._program = None
        return self

    def define_fun(
        self,
        name: str,
        callback: Callable[..., Any],
        argc: int | None = None,
        *,
        optimizable: bool = True,
    ) -> "Parser":
        _validate_name(name)
        if not callable(callback):
            raise TypeError("callback must be callable")
        if argc is not None and not 0 <= argc <= 10:
            raise ValueError("argc must be between zero and ten, or None")
        opcode = CUSTOM_BASE + len(self._functions)
        self._functions[name] = Function(
            name, opcode, argc, callback, optimizable
        )
        self._program = None
        return self

    def set_expression(self, expression: str) -> "Parser":
        if not isinstance(expression, str):
            raise TypeError("expression must be a string")
        self._expression = expression
        self._program = None
        return self

    set_expr = set_expression

    def compile(self) -> Program:
        if self._program is None:
            self._program = compile_expression(
                self._expression,
                tuple(self._variables),
                self._constants,
                self._functions,
            )
        return self._program

    def evaluate(
        self,
        variables: dict[str, Any] | None = None,
        *,
        workers: int | None = None,
        out: np.ndarray | None = None,
    ) -> float | np.ndarray:
        if variables:
            bindings = dict(self._variables)
            bindings.update(variables)
        else:
            bindings = self._variables
        return execute(self.compile(), bindings, workers=workers, out=out)

    eval = evaluate
    eval_bulk = evaluate

    def used_variables(self) -> tuple[str, ...]:
        return self.compile().variable_names


@lru_cache(maxsize=256)
def _compile_cached(expression: str, names: tuple[str, ...]) -> Program:
    return compile_expression(
        expression,
        names,
        {"_pi": _PI, "_e": math.e},
        {},
        auto_variables=True,
    )


def evaluate(
    expression: str,
    variables: dict[str, Any] | None = None,
    /,
    *,
    workers: int | None = None,
    out: np.ndarray | None = None,
    **kwargs: Any,
) -> float | np.ndarray:
    bindings = dict(variables or {})
    bindings.update(kwargs)
    program = _compile_cached(expression, tuple(bindings))
    return execute(program, bindings, workers=workers, out=out)
