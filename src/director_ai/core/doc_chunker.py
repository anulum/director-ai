# Backward-compat shim: director_ai.core.doc_chunker -> director_ai.core.retrieval.doc_chunker
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.retrieval.doc_chunker")
_sys.modules[__name__] = _real
