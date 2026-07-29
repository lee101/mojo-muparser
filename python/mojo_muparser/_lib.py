"""ctypes access to the Mojo bytecode evaluator."""

from __future__ import annotations

import ctypes
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_MUPARSER_LIB") or os.path.join(
    ROOT, "dist", "libmojo-muparser.so"
)
I = ctypes.c_int64
_library: ctypes.CDLL | None = None


class BuildError(RuntimeError):
    pass


def build() -> str:
    process = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            build()
        _library = ctypes.CDLL(LIB)
        _library.mmup_evaluate_f64.argtypes = [I] * 8
        _library.mmup_evaluate_f64.restype = I
    return _library
