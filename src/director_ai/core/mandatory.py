# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Mandatory execution guards for production acceleration and integrations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def mandatory_execution(
    logger_name: str,
    *,
    component: str = "mandatory component",
) -> Iterator[None]:
    """Propagate mandatory-path failures with audit logging.

    DIRECTOR-AI treats declared accelerators and integrations as required
    production capabilities. This guard records the failure path and then
    re-raises the original exception, preventing silent fallback or degraded
    behaviour.
    """
    try:
        yield
    except Exception:
        logging.getLogger(logger_name).error(
            "%s failed; mandatory production capability is unavailable",
            component,
            exc_info=True,
        )
        raise
