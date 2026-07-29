"""Benchmark mojo-muparser against the real muparser C bulk API."""

from __future__ import annotations

import ctypes
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

import mojo_muparser as mojo


class Upstream:
    def __init__(self, variables: dict[str, np.ndarray], expression: str):
        library_path = Path(os.environ["CONDA_PREFIX"]) / "lib" / "libmuparser.so"
        self.library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_double)
        self.library.mupCreate.argtypes = [ctypes.c_int]
        self.library.mupCreate.restype = ctypes.c_void_p
        self.library.mupRelease.argtypes = [ctypes.c_void_p]
        self.library.mupDefineBulkVar.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            pointer,
        ]
        self.library.mupSetExpr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.library.mupEvalBulk.argtypes = [
            ctypes.c_void_p,
            pointer,
            ctypes.c_int,
        ]
        self.library.mupGetVersion.argtypes = [ctypes.c_void_p]
        self.library.mupGetVersion.restype = ctypes.c_char_p
        self.handle = self.library.mupCreate(0)
        self.variables = variables
        for name, value in variables.items():
            self.library.mupDefineBulkVar(
                self.handle,
                name.encode(),
                value.ctypes.data_as(pointer),
            )
        self.library.mupSetExpr(self.handle, expression.encode())
        self.result = np.empty(next(iter(variables.values())).size)

    @property
    def version(self) -> str:
        return self.library.mupGetVersion(self.handle).decode()

    def evaluate(self):
        self.library.mupEvalBulk(
            self.handle,
            self.result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.result.size,
        )
        return self.result

    def close(self):
        self.library.mupRelease(self.handle)


def timeit(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def main() -> None:
    workers = min(int(os.environ.get("MMUP_BENCH_THREADS", "16")), os.cpu_count() or 1)
    count = int(os.environ.get("MMUP_BENCH_SIZE", "5000000"))
    rng = np.random.default_rng(7)
    values = {
        "a": np.ascontiguousarray(rng.uniform(0.2, 1.2, count)),
        "b": np.ascontiguousarray(rng.uniform(1.1, 2.0, count)),
        "c": np.ascontiguousarray(rng.uniform(-0.8, 0.8, count)),
    }
    cases = [
        ("multiply-add", "a*b+c"),
        ("8-op polynomial", "a+b*c+a*a-b*b+c*c*0.25"),
        ("transcendental", "sin(a)+cos(b)+exp(-abs(c))"),
        ("conditional", "c>0?a*b+c:a/b-c"),
        ("variadic", "sum(a,b,c,2)/avg(a,b,c)"),
    ]
    rows = []
    version = ""
    for name, expression in cases:
        parser = mojo.Parser()
        for variable, value in values.items():
            parser.define_var(variable, value)
        parser.set_expr(expression)
        mojo_result = np.empty(count)
        upstream = Upstream(values, expression)
        version = upstream.version
        parser.eval(out=mojo_result, workers=workers)
        upstream.evaluate()
        if not np.allclose(
            mojo_result,
            upstream.result,
            rtol=3e-9,
            atol=3e-9,
            equal_nan=True,
        ):
            raise RuntimeError(f"benchmark parity failed for {expression}")
        mojo_time = timeit(
            lambda: parser.eval(out=mojo_result, workers=workers)
        )
        upstream_time = timeit(upstream.evaluate)
        rows.append(
            (name, mojo_time, upstream_time, upstream_time / mojo_time)
        )
        upstream.close()

    print(
        f"Machine: {cpu_model()}, {platform.platform()}, "
        f"{workers} threads, {count:,} float64 elements"
    )
    print(f"Upstream muparser: {version}")
    print()
    print("| expression | mojo-muparser | muparser | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, upstream_time, ratio in rows:
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
