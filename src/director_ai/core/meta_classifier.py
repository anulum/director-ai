# Backward-compat shim: director_ai.core.meta_classifier -> director_ai.core.scoring.meta_classifier
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.scoring.meta_classifier")
_sys.modules[__name__] = _real
