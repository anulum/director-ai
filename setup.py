# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — free-wheel build shim (paid single-module slicing)
"""Free-wheel build shim: keep paid single modules out of the free wheel.

All static metadata stays in ``pyproject.toml``; this file exists solely to
plug in the ``build_py`` hook, because ``packages.find.exclude`` drops whole
packages only. See ``packages/tier_build_hooks.py``.
"""

import sys
from pathlib import Path

from setuptools import setup

sys.path.insert(0, str(Path(__file__).resolve().parent / "packages"))

from tier_build_hooks import (  # noqa: E402  # needs the sys.path bootstrap above
    FreeTierBuildPy,
)

setup(cmdclass={"build_py": FreeTierBuildPy})
