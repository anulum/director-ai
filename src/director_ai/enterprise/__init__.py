# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Enterprise Extensions

"""Enterprise extensions: multi-tenant isolation, policy engine, audit logging.

::

    from director_ai.enterprise import TenantRouter, Policy, AuditLogger
"""

__all__ = [
    "AuditEntry",
    "AuditLogger",
    "Policy",
    "TenantRouter",
    "Violation",
]

_ENTERPRISE_IMPORTS = {
    "TenantRouter": ("..core.tenant", "TenantRouter"),
    "Policy": ("..core.policy", "Policy"),
    "Violation": ("..core.policy", "Violation"),
    "AuditLogger": ("..core.audit", "AuditLogger"),
    "AuditEntry": ("..core.audit", "AuditEntry"),
}


def __getattr__(name: str):
    if name in _ENTERPRISE_IMPORTS:
        import importlib

        module_path, attr = _ENTERPRISE_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
