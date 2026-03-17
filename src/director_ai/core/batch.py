# Backward-compat shim: director_ai.core.batch -> director_ai.core.runtime.batch
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.runtime.batch")
_sys.modules[__name__] = _real
