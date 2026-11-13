# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# DIRECTOR-AI — governed security exception register validation
"""Validate security-scan exceptions and the required complete-lock audit command.

Each accepted scan finding must carry an owner, a reason, a compensating
control, and an expiry no earlier than the validation date. The CLI fails
(exit 1) on a malformed or
expired entry, so a tolerated finding can never silently become permanent
security debt. It is stdlib-only (tomllib) so it runs in the lint job without
extra dependencies. Workflow tests use ``validate_lock_audit`` to enforce the
complete native lock scope and refusal of shell success overrides.

Usage::

    python -m tools.check_security_exceptions
    python -m tools.check_security_exceptions --today 2026-12-31  # for testing
"""

from __future__ import annotations

import shlex
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
_ALLOWED_TOOLS = frozenset({"pip-audit", "uv-audit", "semgrep", "bandit", "dependabot"})
_SCHEMA_VERSION = "director.security_exceptions.v1"
_DEFAULT_PATH = (
    Path(__file__).resolve().parent.parent / "requirements" / "security-exceptions.toml"
)
_LOCK_AUDIT_ARGV = (
    "uv",
    "audit",
    "--frozen",
    "--no-progress",
    "--color",
    "never",
    "--output-format",
    "json",
)


def lock_audit_ignores(payload: dict[str, object], today: date) -> frozenset[str]:
    """Return the advisory ids the complete-lock audit may ignore on ``today``.

    Only ``uv-audit`` entries of the register qualify, and only through their
    expiry date; every other entry, and any entry without an id, gives none.

    Parameters
    ----------
    payload : dict[str, object]
        Decoded register with its exception tables.
    today : datetime.date
        Validation date; an entry is usable through its expiry date.

    Returns
    -------
    frozenset[str]
        The registered, unexpired ``uv-audit`` advisory ids.
    """
    exceptions = payload.get("exceptions", [])
    ignores: set[str] = set()
    for entry in exceptions if isinstance(exceptions, list) else []:
        if not isinstance(entry, dict) or entry.get("tool") != "uv-audit":
            continue
        try:
            expiry = date.fromisoformat(str(entry.get("expires")))
        except ValueError:
            continue
        advisory = str(entry.get("id", "")).strip()
        if advisory and expiry >= today:
            ignores.add(advisory)
    return frozenset(ignores)


def validate_lock_audit(
    command: str,
    *,
    continue_on_error: object = False,
    ignores: frozenset[str] = frozenset(),
) -> list[str]:
    """Check the complete-lock CI command without executing an advisory scan.

    Parameters
    ----------
    command : str
        Shell command read from the maintained workflow step.
    continue_on_error : object, optional
        Decoded step or job failure override; only literal False is accepted.
    ignores : frozenset[str], optional
        Advisory ids the register allows the audit to ignore (see
        :func:`lock_audit_ignores`). After the fixed command, only
        ``--ignore <id>`` pairs naming these ids, each once, are accepted.

    Returns
    -------
    list[str]
        Problems with failure propagation, command syntax or audit scope.
        Shell control operators, extra flags, multiple commands and ignores
        that the register does not hold are refused.
    """
    problems: list[str] = []
    if continue_on_error is not False:
        problems.append("The complete-lock audit must propagate failures")
    try:
        argv = tuple(shlex.split(command.strip()))
    except ValueError:
        problems.append("The complete-lock audit has malformed shell quoting")
    else:
        base, extra = argv[: len(_LOCK_AUDIT_ARGV)], argv[len(_LOCK_AUDIT_ARGV) :]
        malformed_ignores = len(extra) % 2 or any(
            flag != "--ignore" for flag in extra[::2]
        )
        if (
            len(command.strip().splitlines()) != 1
            or base != _LOCK_AUDIT_ARGV
            or malformed_ignores
        ):
            problems.append("The complete-lock audit command or scope has changed")
        else:
            named = extra[1::2]
            if len(set(named)) != len(named) or not set(named) <= ignores:
                problems.append(
                    "The complete-lock audit ignores an advisory the register does not hold"
                )
    return problems


def validate(payload: dict[str, object], today: date) -> list[str]:
    """Return structural, required-field and expiry problems in the register.

    Parameters
    ----------
    payload : dict[str, object]
        Decoded register with its schema version and exception tables.
    today : datetime.date
        Validation date; an exception remains valid through its expiry date.

    Returns
    -------
    list[str]
        Problems with the declared schema, entry fields, tool or expiry; empty
        when the implemented checks pass. Every expiry is parsed, including
        missing and false-valued inputs.
    """
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
    """Read the requested register and print its validation result.

    Parameters
    ----------
    argv : list[str]
        Command arguments; recognised path and ISO validation-date overrides
        follow the --path and --today options.

    Returns
    -------
    int
        Zero for a valid register, or one for a missing file or reported problem.
    """
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
