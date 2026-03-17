# Backward-compat shim: director_ai.core.lite_scorer -> director_ai.core.scoring.lite_scorer
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.scoring.lite_scorer")
_sys.modules[__name__] = _real
