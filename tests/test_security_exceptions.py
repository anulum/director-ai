# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Security exception and remediation guards
"""Tests for the governed security-exception register validator."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

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


class TestMcpRemediationInvariant:
    """Reject resolved MCP waivers and regression below advisory fix floors.

    The 2026-07-17 owner ruling was closed when the native Semgrep lock
    resolved MCP 1.29.0. Patched packages replace reachability exceptions;
    generic register tests above retain malformed-entry and expiry checks.
    """

    RESOLVED = {
        "GHSA-vj7q-gjh5-988w": Version("1.28.1"),
        "GHSA-jpw9-pfvf-9f58": Version("1.27.2"),
        "GHSA-hvrp-rf83-w775": Version("1.27.2"),
    }
    REGISTER = _ROOT / "requirements" / "security-exceptions.toml"
    CI_SAST_LOCK = _ROOT / "requirements" / "ci-sast.txt"
    RUNTIME_LOCKS = (
        _ROOT / "requirements.txt",
        _ROOT / "requirements" / "docker-server.txt",
    )

    def _entries(self) -> list[dict[str, Any]]:
        """Read the actual governed exception register."""
        payload = tomllib.loads(self.REGISTER.read_text(encoding="utf-8"))
        return list(payload["exceptions"])

    def _mcp_version(self) -> Version:
        """Read the unique resolved MCP version from the native SAST lock."""
        pins = re.findall(
            r"^mcp==([^ ;\\]+)",
            self.CI_SAST_LOCK.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        assert len(pins) == 1, "SAST must carry one auditable MCP pin"
        return Version(pins[0])

    def test_register_excludes_exactly_the_resolved_waivers(self) -> None:
        """Resolved advisory IDs cannot remain scanner exceptions."""
        assert not ({entry["id"] for entry in self._entries()} & self.RESOLVED.keys())

    def test_no_mcp_waiver_expiry_is_extended(self) -> None:
        """MCP remediation closes the waiver instead of renewing its expiry."""
        assert not [entry for entry in self._entries() if entry["package"] == "mcp"]

    def test_patched_pin_replaces_the_reachability_exception(self) -> None:
        """Every waived advisory must be fixed by the resolved SAST version."""
        for advisory, fixed in self.RESOLVED.items():
            assert self._mcp_version() >= fixed, advisory

    def test_mcp_absent_from_runtime_dependency_surfaces(self) -> None:
        """The scanner dependency does not become a direct runtime dependency."""
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

    def test_mcp_enters_as_semgreps_patched_transitive(self) -> None:
        """The checked SAST pin remains wired to an installed Semgrep scanner."""
        text = self.CI_SAST_LOCK.read_text(encoding="utf-8")
        assert re.search(r"^semgrep==[^ ;\\]+", text, flags=re.MULTILINE)
        assert self._mcp_version() >= max(self.RESOLVED.values())


class TestPyjwtRemediationInvariant:
    """Require patched PyJWT profiles and closure of the CI-only waiver."""

    RESOLVED = {
        "GHSA-w6j9-cwv2-h6wq",
        "GHSA-9v7f-9g4p-ffgj",
        "GHSA-hxm8-2xgr-2p9m",
        "GHSA-ffc3-869f-jxw9",
        "GHSA-9j54-fg26-wv3r",
        "GHSA-42vr-xj54-vc7v",
        "GHSA-jwrc-g2q2-pq5p",
        "GHSA-r6x4-923q-g947",
        "GHSA-p4g4-x82p-q773",
        "GHSA-2gx3-rcp4-g85q",
        "GHSA-8wjv-2p76-3863",
        "GHSA-w2cx-738m-mc7w",
        "GHSA-gvp8-978c-rx2q",
    }
    FIXED_FLOOR = Version("2.15.0")
    REGISTER = _ROOT / "requirements" / "security-exceptions.toml"
    CI_SAST_LOCK = _ROOT / "requirements" / "ci-sast.txt"

    def _entries(self) -> list[dict[str, Any]]:
        """Read every entry from the actual governed exception register."""
        payload = tomllib.loads(self.REGISTER.read_text(encoding="utf-8"))
        return list(payload["exceptions"])

    def test_register_excludes_the_resolved_advisories(self) -> None:
        """None of the thirteen resolved findings may remain an exception."""
        assert not ({entry["id"] for entry in self._entries()} & self.RESOLVED)

    def test_no_pyjwt_exception_remains(self) -> None:
        """The fixed upstream graph closes rather than broadens the waiver."""
        assert not [
            entry for entry in self._entries() if entry["package"].lower() == "pyjwt"
        ]

    def test_remediated_register_validates_after_the_previous_expiry(self) -> None:
        """The real CLI accepts the fixed register after the old waiver ends."""
        assert checker.main(["--today", "2026-10-18"]) == 0

    def test_scanner_lock_resolves_compatible_patched_transitives(self) -> None:
        """The native scanner lock cannot regress to the vulnerable graph."""
        text = self.CI_SAST_LOCK.read_text(encoding="utf-8")
        for package, floor in (
            ("pyjwt", self.FIXED_FLOOR),
            ("semgrep", Version("1.179.0")),
        ):
            pins = re.findall(rf"^{package}==([^ ;\\]+)", text, flags=re.MULTILINE)
            assert len(pins) == 1, f"SAST must carry one auditable {package} pin"
            assert Version(pins[0]) >= floor

    def test_vulnerable_pin_is_absent_from_other_runtime_and_tool_profiles(
        self,
    ) -> None:
        """Every locked profile containing PyJWT meets the remediation floor."""
        profiles = [
            _ROOT / "requirements.txt",
            _ROOT / "discord-bot" / "requirements.txt",
            *_ROOT.glob("requirements/*.txt"),
            *_ROOT.glob("training/*.txt"),
            *_ROOT.glob("training/*.lock"),
            _ROOT / "tools/offline_license_ceremony/requirements-offline.txt",
        ]
        for profile in profiles:
            pins = re.findall(
                r"^pyjwt==([^ ;\\]+)",
                profile.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            assert all(Version(pin) >= self.FIXED_FLOOR for pin in pins), profile
        pyproject = tomllib.loads(
            (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        declared = [
            Requirement(dependency)
            for dependency in pyproject["project"]["optional-dependencies"][
                "enterprise"
            ]
            if Requirement(dependency).name.lower() == "pyjwt"
        ]
        assert len(declared) == 1, (
            "Enterprise must declare one auditable PyJWT requirement"
        )
        assert any(
            bound.operator in {">=", "==", "~="}
            and Version(bound.version) >= self.FIXED_FLOOR
            for bound in declared[0].specifier
        ), "Enterprise installs must exclude versions below the advisory fix floor"
        locked = tomllib.loads((_ROOT / "uv.lock").read_text(encoding="utf-8"))
        versions = [p["version"] for p in locked["package"] if p["name"] == "pyjwt"]
        assert all(Version(version) >= self.FIXED_FLOOR for version in versions)

    def test_workflows_only_invoke_the_scanner_mode(self) -> None:
        """Actual CI run steps cannot start the excluded Semgrep MCP server."""
        scan_steps: list[str] = []
        for path in (_ROOT / ".github/workflows").glob("*.yml"):
            workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
            for job in workflow.get("jobs", {}).values():
                for step in job.get("steps", []):
                    command = step.get("run", "")
                    if re.search(r"\bsemgrep\b", command):
                        scan_steps.append(command)
                        assert not re.search(r"\bsemgrep\s+(?:mcp|serve)\b", command), (
                            path
                        )
        assert scan_steps
        assert all(re.search(r"\bsemgrep\s+scan\b", command) for command in scan_steps)
