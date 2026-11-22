# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the pinned quality interpreter for notebook profile tests
"""Name the pinned quality interpreter that runs the bound notebook capture.

The notebook profile refuses any interpreter but the verified build, so the
tests that run its capture launch the one ``tools/install_quality_python.sh``
installs. ``DIRECTOR_QUALITY_PYTHON`` only chooses which interpreter a test
launches; the profile's engine check reads no environment variable.
"""

from __future__ import annotations

import os

import pytest


def quality_python() -> str:
    """Return the pinned quality interpreter, or fail the test by name.

    Returns
    -------
    str
        The interpreter ``DIRECTOR_QUALITY_PYTHON`` names.
    """
    value = os.environ.get("DIRECTOR_QUALITY_PYTHON", "")
    if not value:
        pytest.fail(
            "DIRECTOR_QUALITY_PYTHON must name the pinned quality interpreter "
            "that tools/install_quality_python.sh installs"
        )
    return value
