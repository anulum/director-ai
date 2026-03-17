# Backward-compat shim: director_ai.core.async_streaming -> director_ai.core.runtime.async_streaming
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.runtime.async_streaming")
_sys.modules[__name__] = _real
