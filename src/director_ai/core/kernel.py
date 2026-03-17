# Backward-compat shim: director_ai.core.kernel -> director_ai.core.runtime.kernel
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.runtime.kernel")
_sys.modules[__name__] = _real
