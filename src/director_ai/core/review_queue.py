# Backward-compat shim: director_ai.core.review_queue -> director_ai.core.runtime.review_queue
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.runtime.review_queue")
_sys.modules[__name__] = _real
