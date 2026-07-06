# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Backward-compat shim: director_ai.core.nli -> director_ai.core.scoring.nli
"""Backward-compatibility re-export of :mod:`director_ai.core.scoring.nli`."""

from __future__ import annotations

import importlib as _il
import sys as _sys

from .scoring._nli_export import _load_onnx_session as _load_onnx_session
from .scoring.nli import (
    NLIScorer as NLIScorer,
)
from .scoring.nli import (
    OnnxDynamicBatcher as OnnxDynamicBatcher,
)
from .scoring.nli import (
    _load_nli_model as _load_nli_model,
)
from .scoring.nli import (
    _probs_to_confidence as _probs_to_confidence,
)
from .scoring.nli import (
    export_onnx as export_onnx,
)
from .scoring.nli import (
    export_tensorrt as export_tensorrt,
)
from .scoring.nli import (
    nli_available as nli_available,
)

__all__ = [
    "NLIScorer",
    "OnnxDynamicBatcher",
    "export_onnx",
    "export_tensorrt",
    "nli_available",
]

_real = _il.import_module("director_ai.core.scoring.nli")
_sys.modules[__name__] = _real
