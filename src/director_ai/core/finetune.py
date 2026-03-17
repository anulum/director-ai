# Backward-compat shim: director_ai.core.finetune -> director_ai.core.training.finetune
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.training.finetune")
_sys.modules[__name__] = _real
