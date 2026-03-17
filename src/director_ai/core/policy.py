# Backward-compat shim: director_ai.core.policy -> director_ai.core.safety.policy
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.safety.policy")
_sys.modules[__name__] = _real
