# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Field Checks

"""Field checks shared by the record validators.

Dataclasses do not enforce their annotations, so every validator checks the
runtime type of each field. ``bool`` is refused wherever an ``int`` is
required, because it is an ``int`` subclass.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from datetime import datetime

from .errors import MalformedDigestError, ModelLifecycleError

_HEX64 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z")


def require_sha256(value: object, field: str) -> str:
    """Return ``value`` when it is a SHA-256 hex digest, else refuse."""
    if isinstance(value, str) and _HEX64.fullmatch(value) is not None:
        return value
    raise MalformedDigestError(f"{field} must be 64 lowercase hex characters")


def require_text(value: object, field: str, error: type[ModelLifecycleError]) -> str:
    """Return a non-empty string without surrounding whitespace, else refuse."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise error(f"{field} must be a non-empty string without surrounding space")
    return value


def require_int(
    value: object, field: str, error: type[ModelLifecycleError], *, minimum: int
) -> int:
    """Return an ``int`` (never ``bool``) of at least ``minimum``, else refuse."""
    if type(value) is not int or value < minimum:
        raise error(f"{field} must be an integer >= {minimum}")
    return value


def require_bool(value: object, field: str, error: type[ModelLifecycleError]) -> bool:
    """Return a ``bool``, else refuse."""
    if type(value) is not bool:
        raise error(f"{field} must be a boolean")
    return value


def require_finite(
    value: object, field: str, error: type[ModelLifecycleError]
) -> float:
    """Return a finite ``float``, else refuse."""
    if type(value) is not float or not math.isfinite(value):
        raise error(f"{field} must be a finite float")
    return value


def require_choice(
    value: object,
    field: str,
    choices: Collection[str],
    error: type[ModelLifecycleError],
) -> str:
    """Return ``value`` when it is one of the allowed strings, else refuse."""
    if not isinstance(value, str) or value not in choices:
        raise error(f"{field} must be one of {sorted(choices)}")
    return value


def require_tuple(
    value: object, field: str, error: type[ModelLifecycleError]
) -> tuple[object, ...]:
    """Return ``value`` when it is a tuple, else refuse."""
    if not isinstance(value, tuple):
        raise error(f"{field} must be a tuple")
    return value


def require_unique(
    values: Sequence[object], field: str, error: type[ModelLifecycleError]
) -> None:
    """Refuse a sequence that repeats a member."""
    if len(set(values)) != len(values):
        raise error(f"{field} must not repeat a member")


def require_digests(
    value: object, field: str, error: type[ModelLifecycleError], *, non_empty: bool
) -> tuple[str, ...]:
    """Return a tuple of unique SHA-256 digests, else refuse."""
    items = require_tuple(value, field, error)
    if non_empty and not items:
        raise error(f"{field} must not be empty")
    digests = tuple(require_sha256(item, field) for item in items)
    require_unique(digests, field, error)
    return digests


def require_utc_timestamp(
    value: object, field: str, error: type[ModelLifecycleError]
) -> str:
    """Return an ISO-8601 UTC timestamp ending in ``Z``, else refuse."""
    if not isinstance(value, str) or _UTC.fullmatch(value) is None:
        raise error(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise error(f"{field} is not a real calendar time") from exc
    return value
