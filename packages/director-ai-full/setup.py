# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — director-ai-full build shim (paid single-module layering)
"""Full-wheel build shim: layer paid single modules into the full wheel.

All static metadata stays in ``pyproject.toml``; this file exists solely to
plug in the ``build_py`` hook that copies the paid single modules
(``director_ai/server.py``, …) from the shared ``../../src`` tree. See
``packages/tier_build_hooks.py``.
"""

import sys
from pathlib import Path

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tier_build_hooks import (  # noqa: E402  # needs the sys.path bootstrap above
    PaidTierBuildPy,
)

setup(cmdclass={"build_py": PaidTierBuildPy})
