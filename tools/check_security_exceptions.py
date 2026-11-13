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
    python -m tools.check_security_exceptions --npm-audit report.json
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import cast

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
_ALLOWED_TOOLS = frozenset(
    {"pip-audit", "uv-audit", "npm-audit", "semgrep", "bandit", "dependabot"}
)
_GHSA_ID = re.compile(r"GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}", re.IGNORECASE)
NPM_AUDIT_EMPTY = "npm audit report is empty"
NPM_AUDIT_MALFORMED = "npm audit report is not JSON"
NPM_AUDIT_ERROR = "npm audit report is an error"
NPM_AUDIT_NO_VULNERABILITIES = "npm audit report has no vulnerabilities section"
NPM_AUDIT_BAD_RECORD = "npm audit report has a malformed vulnerability record"
NPM_AUDIT_BAD_VIA = "npm audit report has a malformed via entry"
NPM_AUDIT_UNKNOWN_FOLLOW = "npm audit report follows a package that is not listed"
NPM_AUDIT_NO_GHSA = "npm audit report has an advisory without a GHSA id"
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


def npm_audit_unregistered(advisory: str) -> str:
    """Return the whole refusal for an advisory the register does not hold.

    Parameters
    ----------
    advisory : str
        GHSA id taken from an advisory URL in the npm report.

    Returns
    -------
    str
        The complete failure message for that advisory.
    """
    return f"npm audit advisory {advisory} is not held by an unexpired npm-audit entry"


def npm_audit_expired(advisory: str) -> str:
    """Return the whole refusal for an advisory whose npm-audit entry has lapsed.

    Parameters
    ----------
    advisory : str
        GHSA id taken from an advisory URL in the npm report.

    Returns
    -------
    str
        The complete failure message for that advisory.
    """
    return f"npm audit advisory {advisory} is held only by an expired npm-audit entry"


def _npm_hold(
    payload: dict[str, object], package: str, advisory: str, today: date
) -> str:
    """Classify whether an unexpired npm-audit entry holds one advisory.

    Parameters
    ----------
    payload : dict[str, object]
        Decoded exception register.
    package : str
        Vulnerability name from the npm report.
    advisory : str
        GHSA id, compared without regard to case.
    today : datetime.date
        Validation date; an entry remains usable through its expiry date.

    Returns
    -------
    str
        ``held`` when an unexpired matching entry exists, ``expired`` when
        every match has lapsed or has no ISO expiry, otherwise ``absent``.
    """
    exceptions = payload.get("exceptions", [])
    if not isinstance(exceptions, list):
        return "absent"
    saw_expired = False
    for entry in exceptions:
        if not isinstance(entry, dict):
            continue
        if entry.get("tool") != "npm-audit":
            continue
        if str(entry.get("package", "")).strip() != package:
            continue
        if str(entry.get("id", "")).strip().lower() != advisory.lower():
            continue
        try:
            expiry = date.fromisoformat(str(entry.get("expires")))
        except ValueError:
            saw_expired = True
            continue
        if expiry >= today:
            return "held"
        saw_expired = True
    return "expired" if saw_expired else "absent"


def _npm_findings(
    vulnerabilities: dict[str, object],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Collect GHSA ids and refuse a via entry that is not an advisory or a name.

    A via value that only names another package is followed by reading that
    package's own record. It is not itself an advisory.

    Parameters
    ----------
    vulnerabilities : dict[str, object]
        The report's vulnerabilities object, keyed by package name.

    Returns
    -------
    tuple[list[tuple[str, str]], list[str]]
        Unique ``(package, advisory)`` pairs, then structural problems. A
        non-empty problem list means the pairs must not be judged.
    """
    findings: list[tuple[str, str]] = []
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    for package, record in vulnerabilities.items():
        if not isinstance(record, dict):
            problems.append(NPM_AUDIT_BAD_RECORD)
            continue
        via = record.get("via", [])
        if not isinstance(via, list):
            problems.append(NPM_AUDIT_BAD_VIA)
            continue
        for item in via:
            if isinstance(item, str):
                if item not in vulnerabilities:
                    problems.append(NPM_AUDIT_UNKNOWN_FOLLOW)
                continue
            if not isinstance(item, dict):
                problems.append(NPM_AUDIT_BAD_VIA)
                continue
            url = item.get("url")
            match = _GHSA_ID.search(url) if isinstance(url, str) else None
            if match is None:
                problems.append(NPM_AUDIT_NO_GHSA)
                continue
            identity = (package, "GHSA-" + match.group(0)[5:].lower())
            if identity not in seen:
                seen.add(identity)
                findings.append(identity)
    return findings, problems


def check_npm_audit(report: str, payload: dict[str, object], today: date) -> list[str]:
    """Fail unless every GHSA in an npm audit JSON report is governed.

    Parameters
    ----------
    report : str
        Raw ``npm audit --json`` text. npm's process status is ignored; only
        this text is judged.
    payload : dict[str, object]
        Decoded exception register.
    today : datetime.date
        Validation date for npm-audit expiry.

    Returns
    -------
    list[str]
        Whole failure messages. Empty only when the report is JSON, is not an
        npm error, has a vulnerabilities object, and each advisory URL's GHSA
        is held by an unexpired ``npm-audit`` entry for that package.
    """
    if not report.strip():
        return [NPM_AUDIT_EMPTY]
    try:
        decoded: object = json.loads(report)
    except json.JSONDecodeError:
        return [NPM_AUDIT_MALFORMED]
    if not isinstance(decoded, dict):
        return [NPM_AUDIT_MALFORMED]
    if "error" in decoded:
        return [NPM_AUDIT_ERROR]
    raw_vulnerabilities = decoded.get("vulnerabilities")
    if not isinstance(raw_vulnerabilities, dict):
        return [NPM_AUDIT_NO_VULNERABILITIES]
    findings, problems = _npm_findings(cast(dict[str, object], raw_vulnerabilities))
    if problems:
        return problems
    for package, advisory in sorted(findings, key=lambda item: (item[1], item[0])):
        hold = _npm_hold(payload, package, advisory, today)
        if hold == "held":
            continue
        if hold == "expired":
            problems.append(npm_audit_expired(advisory))
        else:
            problems.append(npm_audit_unregistered(advisory))
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
    report_path: Path | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--path" and i + 1 < len(argv):
            path = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--today" and i + 1 < len(argv):
            today = date.fromisoformat(argv[i + 1])
            i += 2
        elif argv[i] == "--npm-audit" and i + 1 < len(argv):
            report_path = Path(argv[i + 1])
            i += 2
        else:
            i += 1

    if not path.exists():
        print(f"security exception register not found: {path}", file=sys.stderr)
        return 1
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if report_path is not None:
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except OSError:
            problems = [NPM_AUDIT_EMPTY]
        else:
            problems = check_npm_audit(report_text, payload, today)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print("npm audit report OK")
        return 0
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
