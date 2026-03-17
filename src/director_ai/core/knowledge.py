# Backward-compat shim: director_ai.core.knowledge -> director_ai.core.retrieval.knowledge
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.retrieval.knowledge")
_sys.modules[__name__] = _real
