# Backward-compat shim: director_ai.core.doc_parser -> director_ai.core.retrieval.doc_parser
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.retrieval.doc_parser")
_sys.modules[__name__] = _real
