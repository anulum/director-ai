# Backward-compat shim: director_ai.core.finetune_benchmark -> director_ai.core.training.finetune_benchmark
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.training.finetune_benchmark")
_sys.modules[__name__] = _real
