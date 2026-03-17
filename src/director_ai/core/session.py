# Backward-compat shim: director_ai.core.session -> director_ai.core.runtime.session
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.runtime.session")
_sys.modules[__name__] = _real
