# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Managed-service control plane for the Director-AI hosted API.

The open-core data plane (``server.py``, ``proxy.py``) scores and guards; this
package adds the multi-tenant control plane that the hosted beta needs:
persistent accounts, issued/revocable API keys, usage metering, and plan quotas.
It is the commercial layer (BUSL-1.1) and is inert unless ``managed_mode`` is on,
so the self-hosted Apache surface keeps its static-key behaviour unchanged.
"""

from __future__ import annotations

from .accounts import Account, AccountStore, APIKey

__all__ = ["Account", "APIKey", "AccountStore"]
