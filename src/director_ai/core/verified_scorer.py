# Backward-compat shim: director_ai.core.verified_scorer -> director_ai.core.scoring.verified_scorer
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.scoring.verified_scorer")
_sys.modules[__name__] = _real
