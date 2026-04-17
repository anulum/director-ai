# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Per-installation audit salt loader

"""Load a per-installation salt for audit-log fingerprints.

Fingerprints in audit logs hash API keys with this salt so that leaked
logs cannot be directly mapped back to keys. A static salt shared across
all Director-AI installations is a rainbow-table target (VULN-DAI-003);
each deployment should set its own.

Resolution order:

1. ``DIRECTOR_AUDIT_SALT`` environment variable (utf-8 string).
2. ``DIRECTOR_AUDIT_SALT_FILE`` path pointing at a file whose contents
   are stripped and used as the salt.
3. A legacy default with a one-time warning — preserved so that existing
   audit logs remain comparable after the patch lands. Set one of the
   variables above to silence the warning and rotate the salt.
"""

from __future__ import annotations

import logging
import os
import pathlib
import threading

logger = logging.getLogger("DirectorAI.AuditSalt")

_LEGACY_DEFAULT = b"director-ai-audit-v1"
_ENV_VAR = "DIRECTOR_AUDIT_SALT"
_ENV_FILE = "DIRECTOR_AUDIT_SALT_FILE"

_warn_lock = threading.Lock()
_warned = False


def _warn_legacy_once() -> None:
    global _warned
    with _warn_lock:
        if _warned:
            return
        _warned = True
    logger.warning(
        "Using legacy default audit salt. Set %s or %s to a "
        "per-installation secret to mitigate rainbow-table attacks on "
        "leaked audit logs.",
        _ENV_VAR,
        _ENV_FILE,
    )


def reset_warning_for_tests() -> None:
    """Test hook — re-arm the one-shot legacy warning."""
    global _warned
    with _warn_lock:
        _warned = False


def get_audit_salt() -> bytes:
    """Return the audit-log salt for this installation.

    Always returns a non-empty bytes value so call sites never need to
    branch on absence.
    """
    env_value = os.environ.get(_ENV_VAR)
    if env_value:
        return env_value.encode("utf-8")
    env_file = os.environ.get(_ENV_FILE)
    if env_file:
        path = pathlib.Path(env_file)
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"{_ENV_FILE}={env_file!r} cannot be read: {exc}"
            ) from exc
        if not content:
            raise RuntimeError(f"{_ENV_FILE}={env_file!r} is empty")
        return content.encode("utf-8")
    _warn_legacy_once()
    return _LEGACY_DEFAULT


__all__ = ["get_audit_salt", "reset_warning_for_tests"]
