# Backward-compat shim: director_ai.core.sanitizer -> director_ai.core.safety.sanitizer
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.safety.sanitizer")
_sys.modules[__name__] = _real
