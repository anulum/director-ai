# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — environment-variable parsing for configuration

"""String-to-typed coercion for ``DIRECTOR_*`` environment variables.

Split out of ``config.py``: turning the string values of environment variables
into the typed fields ``DirectorConfig.from_env`` expects is a distinct
responsibility from the configuration dataclass itself. Both helpers are
re-exported from ``config`` so the existing private import paths
(``from director_ai.core.config import _coerce`` /
``_parse_api_keys_env``) keep working.
"""

from __future__ import annotations

import json

__all__ = ["coerce_env_value", "parse_api_keys_env"]


def parse_api_keys_env(raw: str) -> list[str]:
    """Parse ``DIRECTOR_API_KEYS`` accepting a JSON array or a comma list.

    Operators reach for both spellings: a JSON array (``["sk-a","sk-b"]``, the
    form the production checklist used to show) and a bare comma-separated list
    (``sk-a,sk-b``). Parsing only one of them silently embeds brackets/quotes
    into the literal key and produces the "auth is configured but the keys never
    match" footgun. A JSON array of strings is honoured when the value parses to
    a list of strings; everything else falls back to comma splitting. Blank and
    whitespace-only entries are dropped.
    """
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and all(isinstance(k, str) for k in parsed):
            return [k.strip() for k in parsed if k.strip()]
    return [k.strip() for k in raw.split(",") if k.strip()]


def coerce_env_value(value: str, type_hint: str) -> object:
    """Coerce a string env var to the target type."""
    if type_hint == "bool":
        low = value.lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(
            f"invalid bool value: {value!r} (expected true/false/1/0/yes/no)",
        )
    if type_hint == "int":
        return int(value)
    if type_hint == "float":
        return float(value)
    if "list" in type_hint:
        items = [s.strip() for s in value.split(",") if s.strip()]
        if "int" in type_hint:
            return [int(x) for x in items]
        if "float" in type_hint:
            return [float(x) for x in items]
        return items
    if "tuple" in type_hint:
        # ``tuple[str, ...]`` fields (e.g. enabled modalities, evidence-firewall
        # sensitivity allowlist) must split on commas like lists. Without this
        # the raw string falls through and downstream ``frozenset(value)`` turns
        # it into a per-character set, silently corrupting the allowlist.
        parts = [s.strip() for s in value.split(",") if s.strip()]
        if "int" in type_hint:
            return tuple(int(x) for x in parts)
        if "float" in type_hint:
            return tuple(float(x) for x in parts)
        return tuple(parts)
    return value
