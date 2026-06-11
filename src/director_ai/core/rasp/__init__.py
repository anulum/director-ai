# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime application self-protection (RASP)

"""Runtime application self-protection from behavioural anomalies.

The last line of defence once input filters and guardrails are bypassed: the
application watches its own behaviour for an attack in progress — a request-rate
spike, an oversized payload, a halt-rate surge. :class:`RuntimeSelfProtection`
scores each observation per metric with a dependency-free robust (median/MAD)
:class:`StreamingRobustDetector` and reports a tenant-safe :class:`AnomalyVerdict`
(ok / watch / alert) plus a recent-anomaly count, so the host can shed load or
escalate. It scores; the host decides whether to block.
"""

from .detector import AnomalyScore, StreamingRobustDetector
from .monitor import AnomalyVerdict, RuntimeSelfProtection

__all__ = [
    "AnomalyScore",
    "AnomalyVerdict",
    "RuntimeSelfProtection",
    "StreamingRobustDetector",
]
