# Backward-compat shim: director_ai.core.audit -> director_ai.core.safety.audit
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.safety.audit")
_sys.modules[__name__] = _real
