# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — edge runtime readiness package

"""Edge and mobile runtime readiness contracts."""

from .runtime_profile import (
    EDGE_RUNTIME_READINESS_SCHEMA_VERSION,
    EdgeRuntimeCheck,
    EdgeRuntimeReadiness,
    build_edge_runtime_readiness,
    probe_backfire_kernel_symbols,
)

__all__ = [
    "EDGE_RUNTIME_READINESS_SCHEMA_VERSION",
    "EdgeRuntimeCheck",
    "EdgeRuntimeReadiness",
    "build_edge_runtime_readiness",
    "probe_backfire_kernel_symbols",
]
