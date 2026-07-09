# SPDX-License-Identifier: Apache-2.0
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
from typing import NoReturn


def require_rust_kernel(kernel: str) -> NoReturn:
    """Raise the actionable kernel-absent error for a mandatory accelerator.

    Import-fallback stubs for kernels that have no bit-exact pure-Python
    equivalent (ADR-0001) call this instead of raising a bare
    ``RuntimeError``, so a base install hitting a Rust-only path is told
    exactly how to fix it.

    Parameters
    ----------
    kernel : str
        Name of the missing ``backfire_kernel`` function.

    Raises
    ------
    RuntimeError
        Always; names the missing kernel and the ``[rust]`` extra that
        provides it.
    """
    raise RuntimeError(
        f"backfire_kernel {kernel} is unavailable and this capability has no "
        "bit-exact pure-Python fallback (ADR-0001). Install the Rust "
        "accelerator: pip install 'director-ai[rust]'."
    )


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
