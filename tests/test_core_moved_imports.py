# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — core package moved-import guidance tests

from __future__ import annotations

import pytest


def test_moved_to_enterprise_attribute_raises_helpful_import_error():
    import director_ai.core as core

    with pytest.raises(ImportError, match="moved to director_ai.enterprise"):
        _ = core.TenantRouter
