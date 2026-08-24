from __future__ import annotations

import math

import numpy as np
import pytest

import mojo_muparser as mojo


@pytest.mark.parametrize(
    "expression",
    [
        "1+2*3",
        "(1+2)*3",
        "2^2^3",
        "1/2/3",
        "-2^2",
        "2^-2",
        "3--2",
        "3+4*2/(1-5)^2^3",
        "1<2 && 3>=3 || 0",
        "0.5&&1",
        "sin(_pi/2)+cos(0)",
        "exp(ln(7))",
        "10^log10(5)",
        "sum(1,2,3,4)+avg(2,4,6)",
        "min(8,3,-2,4)+max(8,3,-2,4)",
        "1 ? 4 : 9",
        "0 ? 4 : 9",
        "0 ? 1 : 1 ? 2 : 3",
    ],
)
def test_scalar_parity(upstream, expression):
    expected = upstream.scalar(expression)
    actual = mojo.evaluate(expression)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)


@pytest.mark.parametrize(
    "expression",
    [
        "a+b*c-2",
        "a^2 + 2*a*b + b^2",
        "sin(a)+cos(b)+exp(-abs(c))",
        "atan2(a,b)+log2(b)+log10(b)",
        "sinh(a/4)+cosh(c/4)-tanh(b)",
        "asinh(a)+acosh(b+1)+atanh(c/4)",
        "sign(c)+rint(c)",
        "sum(a,b,c,2)/avg(a,b,c)",
        "min(a,b,c)+max(a,b,c)",
        "a<b && b>c || c==0",
        "c>0 ? a*b+c : a/b-c",
    ],
)
def test_bulk_parity(upstream, expression):
    rng = np.random.default_rng(42)
    values = {
        "a": rng.uniform(0.2, 1.2, 4099),
        "b": rng.uniform(1.1, 2.0, 4099),
        "c": rng.uniform(-0.8, 0.8, 4099),
    }
    expected, _ = upstream.bulk(expression, values)
    actual = mojo.evaluate(expression, values)
    assert actual.shape == expected.shape
    tolerance = (
        2.5e-9
        if any(name in expression for name in ("log", "asinh", "acosh", "atanh"))
        else 4e-12
    )
    assert np.allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


@pytest.mark.parametrize("size", [0, 1, 3, 7, 8, 15, 16, 17, 33])
def test_empty_and_simd_tail_sizes(size):
    x = np.linspace(-1.0, 1.0, size)
    actual = mojo.evaluate("sin(x)+x*x", {"x": x})
    expected = np.sin(x) + x * x
    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_multiply_add_fast_path_with_simd_tail(upstream):
    rng = np.random.default_rng(11)
    values = {
        "a": rng.uniform(-2.0, 2.0, 4099),
        "b": rng.uniform(0.5, 1.5, 4099),
        "c": rng.uniform(-1.0, 1.0, 4099),
    }
    expected, _ = upstream.bulk("a*b+c", values)
    actual = mojo.evaluate("a*b+c", values)
    assert np.allclose(actual, expected, rtol=4e-12, atol=4e-12)


@pytest.mark.parametrize("size", [384_611, 384_619])
def test_expression_cost_parallel_threshold(size):
    rng = np.random.default_rng(17)
    a = rng.uniform(0.2, 1.2, size)
    b = rng.uniform(1.1, 2.0, size)
    c = rng.uniform(-0.8, 0.8, size)
    expression = "a+b*c+a*a-b*b+c*c*0.25"
    actual = mojo.evaluate(
        expression,
        {"a": a, "b": b, "c": c},
        workers=4,
    )
    expected = a + b * c + a * a - b * b + c * c * 0.25
    assert np.allclose(actual, expected, rtol=4e-12, atol=4e-12)


def test_parallel_path_matches_serial_with_simd_tail():
    size = 384_619
    rng = np.random.default_rng(23)
    values = {
        "a": rng.uniform(0.2, 1.2, size),
        "b": rng.uniform(1.1, 2.0, size),
        "c": rng.uniform(-0.8, 0.8, size),
    }
    parser = mojo.Parser()
    for name, value in values.items():
        parser.define_var(name, value)
    parser.set_expr("a+b*c+a*a-b*b+c*c*0.25")
    parallel = parser.eval(workers=4)
    serial = parser.eval(workers=1)
    assert np.array_equal(parallel, serial)


def test_bulk_assignment_and_sequence_parity(upstream):
    values = {
        "a": np.arange(1.0, 9.0),
        "b": np.full(8, 2.0),
    }
    expected, expected_bound = upstream.bulk("b=a,b*10,a", values)
    parser = (
        mojo.Parser()
        .define_var("a", values["a"].copy())
        .define_var("b", values["b"].copy())
        .set_expr("b=a,b*10,a")
    )
    actual = parser.eval()
    assert np.array_equal(actual, expected)
    assert np.array_equal(parser.variables["b"], expected_bound["b"])


def test_scalar_assignment_persists(upstream):
    expected = upstream.scalar("a=3,a*10", {"a": 1.0})
    parser = mojo.Parser().define_var("a", 1.0).set_expr("a=3,a*10")
    assert parser.eval() == expected
    assert float(parser.variables["a"]) == 3.0


def test_user_function_scalar_and_bulk_parity(upstream):
    upstream.define_fun1("cube", lambda value: value * value * value)
    expected_scalar = upstream.scalar("cube(x)+1", {"x": 2.5})
    parser = (
        mojo.Parser()
        .define_var("x", 2.5)
        .define_fun("cube", lambda value: value * value * value, 1)
        .set_expr("cube(x)+1")
    )
    assert parser.eval() == pytest.approx(expected_scalar)

    x = np.linspace(-3, 3, 101)
    upstream.clear_vars()
    expected_bulk, _ = upstream.bulk("cube(x)+1", {"x": x})
    parser.define_var("x", x)
    assert np.allclose(parser.eval(), expected_bulk)


def test_variables_constants_and_used_variable_order(upstream):
    expected = upstream.scalar("2*x-1", {"x": 4.0})
    parser = (
        mojo.Parser()
        .define_var("x", 4.0)
        .define_const("scale", 2.0)
        .define_const("offset", -1.0)
        .set_expr("scale*x+offset")
    )
    assert parser.eval() == expected
    assert parser.used_variables() == ("x",)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "2+",
        "(2+3",
        "2(3)",
        "sin()",
        "sin(1,2)",
        "unknown+1",
        "1++2",
        "--2",
        "1 ? 2",
        "1 : 2",
        "sum()",
    ],
)
def test_invalid_syntax_is_rejected(expression):
    with pytest.raises(mojo.ParserError):
        mojo.Parser(expression).compile()


def test_shape_validation_and_scalar_broadcast():
    x = np.arange(12.0).reshape(3, 4)
    result = mojo.evaluate("scale*x+offset", {"x": x, "scale": 2.5, "offset": -1})
    assert np.array_equal(result, 2.5 * x - 1)
    with pytest.raises(ValueError, match="same shape"):
        mojo.evaluate("x+y", {"x": np.ones(3), "y": np.ones(4)})


def test_out_buffer_identity():
    x = np.linspace(-1, 1, 31)
    target = np.empty_like(x)
    returned = mojo.evaluate("x*x+1", {"x": x}, out=target)
    assert returned is target
    assert np.array_equal(target, x * x + 1)


def test_domain_and_ieee_values(upstream):
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected, _ = upstream.bulk("sqrt(x)+log(x)+1/x", {"x": x})
    actual = mojo.evaluate("sqrt(x)+log(x)+1/x", {"x": x})
    assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize(
    "expression",
    ["min(a,b)", "max(a,b)", "a!=b", "a&&b", "a||b", "a?b:2"],
)
def test_nan_predicate_and_reduction_parity(upstream, expression):
    values = {
        "a": np.array([np.nan, 1.0, np.nan, 0.0]),
        "b": np.array([1.0, np.nan, np.nan, 2.0]),
    }
    expected, _ = upstream.bulk(expression, values)
    actual = mojo.evaluate(expression, values)
    assert np.array_equal(actual, expected, equal_nan=True)


def test_gnu_pi_choice_matches_upstream(upstream):
    assert mojo.evaluate("_pi") == upstream.scalar("_pi")
    assert mojo.evaluate("_pi") == 3.141592653589


def test_parser_metadata_and_limits():
    parser = mojo.Parser().define_var("x", 1.0).set_expr("x*x+1")
    program = parser.compile()
    assert program.expression == "x*x+1"
    assert program.variable_names == ("x",)
    assert 1 <= program.max_stack <= 64


def test_low_polynomial_bytecode_optimization():
    parser = mojo.Parser().define_var("x", np.arange(5.0)).set_expr("x^3+x*x")
    program = parser.compile()
    assert 23 in program.code[:, 0]
    assert 22 in program.code[:, 0]
    assert np.array_equal(parser.eval(), np.arange(5.0) ** 3 + np.arange(5.0) ** 2)


def test_zero_argument_user_function():
    parser = mojo.Parser().define_fun("answer", lambda: 42.0, 0).set_expr(
        "answer()+1"
    )
    assert parser.eval() == 43.0


def test_fixed_and_variadic_user_function_arities():
    fixed = (
        mojo.Parser()
        .define_fun("difference", lambda left, right: left - right, 2)
        .set_expr("difference(7,2)")
    )
    assert fixed.eval() == 5.0

    variadic = (
        mojo.Parser()
        .define_fun("product", lambda *values: math.prod(values))
        .set_expr("product(2,3,4)")
    )
    assert variadic.eval() == 24.0


def test_user_function_fallback_handles_optimized_variable_bytecode():
    x = np.linspace(-2.0, 2.0, 17)
    parser = (
        mojo.Parser()
        .define_var("x", x)
        .define_fun("identity", lambda value: value, 1)
        .set_expr("identity(x^2+x^3+x^4)")
    )
    assert np.array_equal(parser.eval(), x**2 + x**3 + x**4)


def test_conditional_branch_assignment_is_rejected():
    parser = (
        mojo.Parser()
        .define_var("a", 0.0)
        .define_var("b", 1.0)
        .set_expr("a ? b=2 : b=3")
    )
    with pytest.raises(mojo.ParserError, match="conditional branches"):
        parser.compile()


@pytest.mark.parametrize(
    ("expression", "value", "expected"),
    [
        ("tan(x)", 0.3, math.tan(0.3)),
        ("cot(x)", 0.3, 1.0 / math.tan(0.3)),
        ("asin(x)", 0.3, math.asin(0.3)),
        ("acos(x)", 0.3, math.acos(0.3)),
        ("atan(x)", 0.3, math.atan(0.3)),
        ("coth(x)", 0.3, 1.0 / math.tanh(0.3)),
        ("log(x)", 1.3, math.log(1.3)),
        ("sqrt(x)", 1.3, math.sqrt(1.3)),
        ("_e+x", 0.3, math.e + 0.3),
    ],
)
def test_remaining_default_functions_and_constants(expression, value, expected):
    actual = mojo.evaluate(expression, {"x": value})
    tolerance = 2.5e-9 if expression.startswith("log") else 2e-12
    assert actual == pytest.approx(expected, rel=tolerance, abs=tolerance)


def test_read_only_scalar_assignment_is_rejected():
    value = np.array(1.0)
    value.flags.writeable = False
    with pytest.raises(ValueError, match="not writable"):
        mojo.evaluate("x=2", {"x": value})


def test_assignment_does_not_silently_write_a_temporary():
    backing = np.arange(8.0)
    view = backing[::2]
    with pytest.raises(ValueError, match="assignment target"):
        mojo.evaluate("x=x+1", {"x": view})
    assert np.array_equal(backing, np.arange(8.0))


def test_custom_fallback_assignment_does_not_write_a_temporary():
    backing = np.arange(8.0)
    view = backing[::2]
    parser = (
        mojo.Parser()
        .define_var("x", np.zeros(4))
        .define_fun("identity", lambda value: value, 1)
        .set_expr("x=identity(x+1)")
    )
    with pytest.raises(ValueError, match="assignment target"):
        parser.eval({"x": view})
    assert np.array_equal(backing, np.arange(8.0))


def test_dtype_narrowing_and_worker_validation():
    with pytest.raises(ValueError, match="represented exactly"):
        mojo.evaluate("x", {"x": np.array([2**53 + 1], dtype=np.int64)})
    with pytest.raises(TypeError, match="float64"):
        mojo.evaluate("x", {"x": np.array([1], dtype=np.longdouble)})
    with pytest.raises(ValueError, match="positive integer"):
        mojo.evaluate("x", {"x": np.array([1.0])}, workers=0)


def test_custom_function_out_requires_safe_buffer():
    x = np.arange(6.0)
    parser = (
        mojo.Parser()
        .define_var("x", x)
        .define_fun("identity", lambda value: value, 1)
        .set_expr("identity(x)")
    )
    target = np.empty(12, dtype=np.float64)[::2]
    assert target.shape == x.shape
    with pytest.raises(ValueError, match="C-contiguous"):
        parser.eval(out=target)
