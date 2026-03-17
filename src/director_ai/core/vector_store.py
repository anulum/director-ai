# Backward-compat shim: director_ai.core.vector_store -> director_ai.core.retrieval.vector_store
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.retrieval.vector_store")
_sys.modules[__name__] = _real
