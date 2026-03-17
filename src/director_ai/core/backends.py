# Backward-compat shim: director_ai.core.backends -> director_ai.core.scoring.backends
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.scoring.backends")
_sys.modules[__name__] = _real
