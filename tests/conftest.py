from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


class UpstreamMuParser:
    """Small ctypes owner for conda-forge's real muparser C API."""

    def __init__(self):
        prefix = Path(os.environ["CONDA_PREFIX"])
        self.library = ctypes.CDLL(str(prefix / "lib" / "libmuparser.so"))
        pointer = ctypes.POINTER(ctypes.c_double)
        self.library.mupCreate.argtypes = [ctypes.c_int]
        self.library.mupCreate.restype = ctypes.c_void_p
        self.library.mupRelease.argtypes = [ctypes.c_void_p]
        self.library.mupSetExpr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.library.mupEval.argtypes = [ctypes.c_void_p]
        self.library.mupEval.restype = ctypes.c_double
        self.library.mupEvalBulk.argtypes = [
            ctypes.c_void_p,
            pointer,
            ctypes.c_int,
        ]
        self.library.mupDefineVar.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            pointer,
        ]
        self.library.mupDefineBulkVar.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            pointer,
        ]
        self.library.mupDefineFun1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        self.library.mupError.argtypes = [ctypes.c_void_p]
        self.library.mupError.restype = ctypes.c_int
        self.library.mupGetErrorMsg.argtypes = [ctypes.c_void_p]
        self.library.mupGetErrorMsg.restype = ctypes.c_char_p
        self.library.mupClearVar.argtypes = [ctypes.c_void_p]
        self.handle = self.library.mupCreate(0)
        if not self.handle:
            raise RuntimeError("mupCreate failed")
        self._arrays: list[np.ndarray] = []
        self._callbacks: list[object] = []

    def close(self):
        if self.handle:
            self.library.mupRelease(self.handle)
            self.handle = None

    def _check(self):
        if self.library.mupError(self.handle):
            message = self.library.mupGetErrorMsg(self.handle)
            raise ValueError(message.decode() if message else "muparser error")

    def define_var(self, name: str, value, *, bulk: bool):
        array = np.ascontiguousarray(value, dtype=np.float64)
        if not array.ndim:
            array = array.reshape(1)
        self._arrays.append(array)
        pointer = array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        function = (
            self.library.mupDefineBulkVar
            if bulk
            else self.library.mupDefineVar
        )
        function(self.handle, name.encode(), pointer)
        self._check()
        return array

    def define_fun1(self, name: str, callback):
        callback_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
        wrapped = callback_type(callback)
        self._callbacks.append(wrapped)
        self.library.mupDefineFun1(
            self.handle,
            name.encode(),
            ctypes.cast(wrapped, ctypes.c_void_p),
            0,
        )
        self._check()

    def clear_vars(self):
        self.library.mupClearVar(self.handle)
        self._arrays.clear()
        self._check()

    def scalar(self, expression: str, variables=None) -> float:
        for name, value in (variables or {}).items():
            self.define_var(name, value, bulk=False)
        self.library.mupSetExpr(self.handle, expression.encode())
        result = self.library.mupEval(self.handle)
        self._check()
        return result

    def bulk(self, expression: str, variables: dict[str, np.ndarray]):
        size = 0
        bound = {}
        for name, value in variables.items():
            array = np.asarray(value, dtype=np.float64)
            size = array.size if not size else size
            assert array.size == size
            bound[name] = self.define_var(name, array, bulk=True)
        result = np.empty(size, dtype=np.float64)
        self.library.mupSetExpr(self.handle, expression.encode())
        self.library.mupEvalBulk(
            self.handle,
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            size,
        )
        self._check()
        return result, bound


@pytest.fixture
def upstream():
    reference = UpstreamMuParser()
    try:
        yield reference
    finally:
        reference.close()
