# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Enterprise Extensions
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Enterprise extensions: multi-tenant isolation, policy engine, audit logging.

::

    from director_ai.enterprise import TenantRouter, Policy, AuditLogger
"""

__all__ = [
    "TenantRouter",
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
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
