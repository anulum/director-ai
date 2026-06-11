# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the governed security-exception register validator."""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_security_exceptions", _ROOT / "tools" / "check_security_exceptions.py"
)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

_TODAY = date(2026, 6, 7)


def _entry(**overrides):
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
    def test_empty_register_valid(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [],
        }
        assert checker.validate(payload, _TODAY) == []

    def test_valid_full_entry(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry()],
        }
        assert checker.validate(payload, _TODAY) == []

    def test_wrong_schema_version(self):
        payload = {"schema_version": "bad", "exceptions": []}
        problems = checker.validate(payload, _TODAY)
        assert any("schema_version" in p for p in problems)

    def test_missing_required_field(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(owner="")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("missing required field 'owner'" in p for p in problems)

    def test_expired_entry_rejected(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(expires="2026-01-01")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("expired" in p for p in problems)

    def test_bad_tool_rejected(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(tool="nessus")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("not in" in p for p in problems)

    def test_bad_expiry_date(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(expires="not-a-date")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("not an ISO date" in p for p in problems)

    def test_non_list_exceptions(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": {"nope": True},
        }
        problems = checker.validate(payload, _TODAY)
        assert any("must be an array" in p for p in problems)

    def test_entry_not_a_table(self):
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": ["just-a-string"],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("must be a table" in p for p in problems)


class TestCommittedRegister:
    def test_committed_register_is_valid(self):
        rc = checker.main([])
        assert rc == 0

    def test_missing_file_exits_nonzero(self, tmp_path):
        rc = checker.main(["--path", str(tmp_path / "absent.toml")])
        assert rc == 1

    def test_expired_via_cli_today(self, tmp_path):
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


def test_module_runs_as_main():
    # Smoke: the committed register validates through the real entrypoint.
    assert checker.main([]) == 0


@pytest.mark.parametrize("missing", checker._REQUIRED_FIELDS)
def test_each_required_field_enforced(missing):
    payload = {
        "schema_version": "director.security_exceptions.v1",
        "exceptions": [_entry(**{missing: ""})],
    }
    problems = checker.validate(payload, _TODAY)
    assert any(f"missing required field {missing!r}" in p for p in problems)
