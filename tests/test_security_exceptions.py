# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the governed security-exception register validator."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_security_exceptions", _ROOT / "tools" / "check_security_exceptions.py"
)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

_TODAY = date(2026, 6, 7)


def _entry(**overrides: str) -> dict[str, str]:
    base = {
        "id": "GHSA-aaaa-bbbb-cccc",
        "tool": "pip-audit",
        "package": "torch",
        "reason": "no fix; not reachable",
        "compensating_control": "optional extra only",
        "owner": "protoscience@anulum.li",
        "opened": "2026-06-07",
        "expires": "2026-12-31",
        "scope": "ci-dependency-audit",
    }
    base.update(overrides)
    return base


class TestValidate:
    def test_empty_register_valid(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [],
        }
        assert checker.validate(payload, _TODAY) == []

    def test_valid_full_entry(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry()],
        }
        assert checker.validate(payload, _TODAY) == []

    def test_wrong_schema_version(self) -> None:
        payload = {"schema_version": "bad", "exceptions": []}
        problems = checker.validate(payload, _TODAY)
        assert any("schema_version" in p for p in problems)

    def test_missing_required_field(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(owner="")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("missing required field 'owner'" in p for p in problems)

    def test_expired_entry_rejected(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(expires="2026-01-01")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("expired" in p for p in problems)

    def test_bad_tool_rejected(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(tool="nessus")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("not in" in p for p in problems)

    def test_bad_expiry_date(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(expires="not-a-date")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("not an ISO date" in p for p in problems)

    def test_non_list_exceptions(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": {"nope": True},
        }
        problems = checker.validate(payload, _TODAY)
        assert any("must be an array" in p for p in problems)

    def test_entry_not_a_table(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": ["just-a-string"],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("must be a table" in p for p in problems)


class TestCommittedRegister:
    def test_committed_register_is_valid(self) -> None:
        rc = checker.main([])
        assert rc == 0

    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        rc = checker.main(["--path", str(tmp_path / "absent.toml")])
        assert rc == 1

    def test_expired_via_cli_today(self, tmp_path: Path) -> None:
        reg = tmp_path / "sec.toml"
        reg.write_text(
            'schema_version = "director.security_exceptions.v1"\n'
            "exceptions = [\n"
            '  { id = "GHSA-x", tool = "pip-audit", package = "p", '
            'reason = "r", compensating_control = "c", '
            'owner = "o@x", opened = "2026-01-01", expires = "2026-02-01", '
            'scope = "s" },\n'
            "]\n",
            encoding="utf-8",
        )
        # Far-future "today" makes the entry expired -> non-zero.
        rc = checker.main(["--path", str(reg), "--today", "2027-01-01"])
        assert rc == 1


def test_module_runs_as_main() -> None:
    # Smoke: the committed register validates through the real entrypoint.
    assert checker.main([]) == 0


@pytest.mark.parametrize("missing", checker._REQUIRED_FIELDS)
def test_each_required_field_enforced(missing: str) -> None:
    payload = {
        "schema_version": "director.security_exceptions.v1",
        "exceptions": [_entry(**{missing: ""})],
    }
    problems = checker.validate(payload, _TODAY)
    assert any(f"missing required field {missing!r}" in p for p in problems)


class TestDependabotTool:
    def test_dependabot_tool_accepted(self) -> None:
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(tool="dependabot")],
        }
        assert checker.validate(payload, _TODAY) == []


class TestMcpWaiverInvariant:
    """Locks the owner-ruled expiring waiver for the transitive mcp pin.

    Ruled 2026-07-17 (SC-NEUROCORE mcp-waiver precedent): the three HIGH
    advisories against mcp==1.23.3 in the CI SAST lock are waived ONLY while
    (a) mcp stays out of every runtime dependency surface, (b) mcp still
    reaches the repository solely as semgrep's hard-pinned transitive dep in
    requirements/ci-sast.txt, and (c) the expiry has not lapsed. Any of these
    failing means the waiver premise broke — re-review it, do not patch the
    test to stay green.
    """

    WAIVED = {
        "GHSA-vj7q-gjh5-988w",  # CVE-2026-59950
        "GHSA-jpw9-pfvf-9f58",  # CVE-2026-52869
        "GHSA-hvrp-rf83-w775",  # CVE-2026-52870
    }
    EXPIRY = date(2026, 10, 17)
    REGISTER = _ROOT / "requirements" / "security-exceptions.toml"
    CI_SAST_LOCK = _ROOT / "requirements" / "ci-sast.txt"
    RUNTIME_LOCKS = (
        _ROOT / "requirements.txt",
        _ROOT / "requirements" / "docker-server.txt",
    )

    def _entries(self) -> list[dict[str, Any]]:
        payload = tomllib.loads(self.REGISTER.read_text(encoding="utf-8"))
        return [e for e in payload["exceptions"] if e["package"] == "mcp"]

    def test_register_carries_exactly_the_ruled_waivers(self) -> None:
        entries = self._entries()
        assert {e["id"] for e in entries} == self.WAIVED
        for entry in entries:
            assert entry["tool"] == "dependabot"
            assert entry["scope"] == "dependabot-ci-sast-lock"
            assert entry["opened"] == "2026-07-17"

    def test_waiver_has_not_lapsed(self) -> None:
        # Load-bearing lapse guard: after the expiry this test goes red and
        # forces the re-review the ruling requires.
        for entry in self._entries():
            assert date.fromisoformat(str(entry["expires"])) == self.EXPIRY
        assert date.today() <= self.EXPIRY, (
            "mcp waiver expired on 2026-10-17 — re-review it (has semgrep "
            "unpinned mcp yet?); do not extend the date without a new ruling"
        )

    def test_each_entry_states_the_reachability_invariant(self) -> None:
        for entry in self._entries():
            control = entry["compensating_control"]
            assert "semgrep scan" in control
            assert "MCP server" in control
            assert "runtime" in control
            assert "mcp==1.23.3" in entry["reason"]

    def test_mcp_absent_from_runtime_dependency_surfaces(self) -> None:
        pyproject = tomllib.loads(
            (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = list(pyproject["project"].get("dependencies", []))
        for deps in pyproject["project"].get("optional-dependencies", {}).values():
            declared.extend(deps)
        for group in pyproject.get("dependency-groups", {}).values():
            declared.extend(d for d in group if isinstance(d, str))
        names = {
            re.split(r"[<>=!~;\[\] ]", dep, maxsplit=1)[0].lower() for dep in declared
        }
        assert "mcp" not in names
        for lock in self.RUNTIME_LOCKS:
            lines = lock.read_text(encoding="utf-8").splitlines()
            assert not any(line.startswith("mcp==") for line in lines), lock

    def test_mcp_enters_only_as_semgreps_pinned_transitive(self) -> None:
        # If this pin ever moves (upstream unpin or a resolvable bump), the
        # waiver premise is gone — remove the entries instead of updating it.
        lines = self.CI_SAST_LOCK.read_text(encoding="utf-8").splitlines()
        assert any(line.startswith("mcp==1.23.3") for line in lines)
        assert any(line.startswith("semgrep==") for line in lines)
