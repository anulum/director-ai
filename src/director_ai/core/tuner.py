# Backward-compat shim: director_ai.core.tuner -> director_ai.core.training.tuner
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.training.tuner")
_sys.modules[__name__] = _real
