# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - security-exception register validator
"""Validate the governed security-scan exception register.

Each accepted scan finding must carry an owner, a reason, a compensating
control, and a future expiry. This validator fails (exit 1) on a malformed or
expired entry, so a tolerated finding can never silently become permanent
security debt. It is stdlib-only (tomllib) so it runs in the lint job without
extra dependencies.

Usage::

    python -m tools.check_security_exceptions
    python -m tools.check_security_exceptions --today 2026-12-31  # for testing
"""

from __future__ import annotations

import sys
import tomllib
from datetime import date
from pathlib import Path

_REQUIRED_FIELDS = (
    "id",
    "tool",
    "package",
    "reason",
    "compensating_control",
    "owner",
    "opened",
    "expires",
    "scope",
)
_ALLOWED_TOOLS = frozenset({"pip-audit", "semgrep", "bandit"})
_SCHEMA_VERSION = "director.security_exceptions.v1"
_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent / "requirements" / "security-exceptions.toml"
)


def validate(payload: dict, today: date) -> list[str]:
    """Return a list of problems with the register; empty means valid."""
    problems: list[str] = []
    if payload.get("schema_version") != _SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {_SCHEMA_VERSION!r}, "
            f"got {payload.get('schema_version')!r}"
        )
    exceptions = payload.get("exceptions", [])
    if not isinstance(exceptions, list):
        problems.append("exceptions must be an array")
        return problems
    for index, entry in enumerate(exceptions):
        where = f"exceptions[{index}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be a table")
            continue
        for field in _REQUIRED_FIELDS:
            if not str(entry.get(field, "")).strip():
                problems.append(f"{where}: missing required field {field!r}")
        tool = entry.get("tool")
        if tool is not None and tool not in _ALLOWED_TOOLS:
            problems.append(f"{where}: tool {tool!r} not in {sorted(_ALLOWED_TOOLS)}")
        expires = entry.get("expires")
        if expires:
            try:
                expiry = date.fromisoformat(str(expires))
            except ValueError:
                problems.append(f"{where}: expires {expires!r} is not an ISO date")
            else:
                if expiry < today:
                    problems.append(
                        f"{where}: exception {entry.get('id')!r} expired on "
                        f"{expiry.isoformat()} — re-review or remediate"
                    )
    return problems


def main(argv: list[str]) -> int:
    path = _DEFAULT_PATH
    today = date.today()
    i = 0
    while i < len(argv):
        if argv[i] == "--path" and i + 1 < len(argv):
            path = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--today" and i + 1 < len(argv):
            today = date.fromisoformat(argv[i + 1])
            i += 2
        else:
            i += 1

    if not path.exists():
        print(f"security exception register not found: {path}", file=sys.stderr)
        return 1
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    problems = validate(payload, today)
    if problems:
        print("security exception register INVALID:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    active = payload.get("exceptions", [])
    print(f"security exception register OK ({len(active)} active exception(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
