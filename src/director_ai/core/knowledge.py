# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Backward-compat shim: director_ai.core.knowledge -> director_ai.core.retrieval.knowledge
import importlib as _il
import sys as _sys

_real = _il.import_module("director_ai.core.retrieval.knowledge")
_sys.modules[__name__] = _real
