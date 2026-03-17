# Backward-compat shim: director_ai.core.nli -> director_ai.core.scoring.nli
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.scoring.nli")
_sys.modules[__name__] = _real
