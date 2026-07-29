# mojo-muparser

`mojo-muparser` is a standalone Mojo port of the compute-heavy core of
[muparser](https://github.com/beltoforion/muparser), the fast C++ mathematical
expression parser. It tokenizes muparser syntax, lowers it with a shunting-yard
compiler to compact postfix bytecode, and evaluates that bytecode over NumPy arrays
in a fused, multithreaded SIMD pass.

This is a derived work of muparser, which is distributed under the BSD 2-Clause
License. The port itself is MIT licensed; [NOTICE](NOTICE) contains the upstream
credit and licence text. The implementation was ported from the actual muparser
2.3.6 source at commit `508cce9`, especially `ParserTokenReader::ReadNextToken`,
`ParserBase::CreateRPN`, `ParserByteCode::ConstantFolding`, and
`ParserBase::ParseCmdCodeBulk`.

## Coverage

The covered language and API are:

- numeric literals, named variables, user constants, `_pi`, and `_e`;
- muparser precedence and associativity for `+`, `-`, `*`, `/`, `^`, comparisons,
  `&&`, `||`, unary signs, parentheses, assignments, comma-separated sequences,
  and `?:`;
- all deterministic default floating-point functions: trigonometric, inverse,
  hyperbolic, logarithmic, `exp`, `sqrt`, `sign`, `rint`, `abs`, and the variadic
  `sum`, `avg`, `min`, and `max`;
- scalar evaluation and equal-shaped `float64` bulk arrays, with scalar broadcast,
  multidimensional shapes, direct `out=` buffers, and assignment back to writable
  C-contiguous `float64` variable arrays;
- fixed-arity and variadic Python user functions. These use a NumPy/Python fallback
  because arbitrary Python callbacks cannot safely execute inside Mojo worker
  threads;
- muparser constant folding and its optimized `x*x` and `x^2` through `x^4`
  bytecodes.

The parity suite calls the installed real muparser 2.3.5 C API for shared behavior;
it does not use a rewritten NumPy parser as its reference. It also draws expression
cases from upstream `muParserTest.cpp`. Every listed default function has a direct
test. Mojo SIMD logarithmic operations differ from the reference scalar libm path,
so those comparisons use a `2.5e-9` tolerance; other covered bulk operations use
`4e-12`.

This is not the complete C++ class and DLL surface. The random `rnd` function,
string constants/functions, locale-specific decimal parsing, custom value
recognizers, custom prefix/postfix/binary operators, the integer parser, bulk
callbacks with row/thread indices, OpenMP configuration controls, and `EvalMulti`
returning every comma-separated intermediate are not implemented. A sequence is
evaluated in order and returns its final value, like ordinary `Eval`. Assignments
inside conditional branches are rejected because the SIMD conditional is a selection,
not a per-lane control-flow jump. Python user callbacks should be pure; both numeric
branch expressions can be computed before conditional selection.

All native inputs are evaluated as `float64`. Lower-precision real inputs are
promoted. Extended-precision values and integers that cannot be represented exactly
as `float64` are rejected instead of silently narrowed. Non-contiguous inputs are
copied for ordinary evaluation, but assignment targets are never copied: they must
be writable C-contiguous `float64` NumPy arrays. `out=` has the same dtype, layout,
shape, and writability requirements.

## Install

The repository pins the tested Mojo nightly. Pixi installs Mojo, NumPy, pytest, and
the real muparser library used by tests and benchmarks.

```bash
pixi install
pixi run build
pixi run test
```

The build creates `dist/libmojo-muparser.so`. Pixi adds `python/` to
`PYTHONPATH`, so an editable package install is not required.

## Usage

```python
import numpy as np
from mojo_muparser import Parser, evaluate

x = np.linspace(-3.0, 3.0, 1_000_000)
y = np.linspace(0.5, 1.5, 1_000_000)

result = evaluate(
    "x > 0 ? sin(x) + x*y : exp(x) - y",
    {"x": x, "y": y},
)
print(result.shape, result.dtype)

parser = (
    Parser()
    .define_var("x", x)
    .define_const("scale", 0.25)
    .set_expr("scale*x^2 + 2*x + 1")
)
target = np.empty_like(x)
assert parser.eval(out=target) is target
```

Stateful assignment and a user function use the same API:

```python
parser = (
    Parser()
    .define_var("a", np.arange(1.0, 6.0))
    .define_var("b", np.zeros(5))
    .define_fun("cube", lambda value: value * value * value, 1)
    .set_expr("b=cube(a), b+1")
)
result = parser.eval()
```

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
x86-64 Linux 6.8.0 with glibc 2.39, using 16 threads and 5,000,000 `float64`
elements. The reference is conda-forge muparser 2.3.5 built with OpenMP. Times are
the best of five warmed calls into preallocated result arrays. Ratio is muparser
time divided by mojo-muparser time.

| expression | mojo-muparser | muparser | ratio |
| --- | ---: | ---: | ---: |
| multiply-add | 13.20 ms | 47.04 ms | 3.56x faster |
| 8-op polynomial | 24.69 ms | 87.82 ms | 3.56x faster |
| transcendental | 41.61 ms | 154.40 ms | 3.71x faster |
| conditional | 11.65 ms | 39.38 ms | 3.38x faster |
| variadic | 22.80 ms | 75.27 ms | 3.30x faster |

These are single-machine microbenchmarks, not general performance claims. The
benchmark validates each result against muparser before timing.

No GPU path is included.

## How it works

Python handles the cold control path. The tokenizer follows muparser's recognition
order and character rules; the compiler preserves the upstream precedence table,
right-associative power, lower-precedence unary signs, function argument counting,
ternaries, assignments, and constant-folding choices. A compiled program is an
`int64` structure-of-arrays-friendly instruction matrix plus a `float64` constant
pool.

Python owns every buffer. ctypes passes instruction, constant, variable-address,
stride, and destination buffers as integer addresses. The non-parametric
`@export(...) ... abi("C")` Mojo wrapper reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values. Variables are represented by flat
address and element-stride arrays; a zero stride broadcasts a scalar without
materializing it. Python validates shapes, dtype, contiguity, writability, and worker
counts before the call, and keeps the prepared arrays and all address tables alive
until the native call returns.

The Mojo evaluator keeps a 64-entry stack of native-width `float64` SIMD vectors.
Each bytecode instruction therefore acts on several independent array elements.
The stack storage is left uninitialized because validated bytecode writes every slot
before reading it, avoiding a repeated 64-slot clear for every SIMD chunk. Common
multiply-add and conditional bytecode shapes bypass interpreter dispatch and reuse
their variable loads. Large arrays are divided among physical cores using an
expression-cost threshold; worker boundaries stay SIMD-aligned, and only the last
worker handles a scalar tail. The whole expression is fused: intermediate operator
arrays are never allocated.
