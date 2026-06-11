# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Back-compat shim: enterprise.redactor -> core.redactor

"""Back-compat re-export.

``PIIRedactor`` and friends moved to :mod:`director_ai.core.redactor` (Apache-2.0
core) so the redaction path no longer depends on the BUSL ``enterprise`` package
and the basic PyPI wheel can ship it. Import from ``director_ai.core.redactor``;
this shim keeps the old path working.
"""

from __future__ import annotations

from director_ai.core.redactor import (
    PIIRedactionFinding,
    PIIRedactionReport,
    PIIRedactor,
)

__all__ = ["PIIRedactionFinding", "PIIRedactionReport", "PIIRedactor"]
