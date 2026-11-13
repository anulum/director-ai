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
import json
import re
import runpy
import sys
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

from tools.audit_ci_workflow_modularity import read_workflow
from tools.check_security_exceptions import lock_audit_ignores, validate_lock_audit
from tools.ci_workflow_inventory import object_mapping

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_security_exceptions", _ROOT / "tools" / "check_security_exceptions.py"
)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

_TODAY = date(2026, 6, 7)


def _entry(**overrides: str) -> dict[str, str]:
    """Build a complete register entry and apply the supplied string overrides."""
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
    """Check register structure, required fields, recognised tools and expiry refusal."""

    def test_empty_register_valid(self) -> None:
        """Accept a correctly versioned register with no active exceptions."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [],
        }
        assert checker.validate(payload, _TODAY) == []

    def test_valid_full_entry(self) -> None:
        """Accept a complete attributed exception whose expiry is still current."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry()],
        }
        assert checker.validate(payload, _TODAY) == []

    def test_wrong_schema_version(self) -> None:
        """Report an unsupported security-exception schema version."""
        payload = {"schema_version": "bad", "exceptions": []}
        problems = checker.validate(payload, _TODAY)
        assert any("schema_version" in p for p in problems)

    def test_missing_required_field(self) -> None:
        """Report a blank required owner field."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(owner="")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("missing required field 'owner'" in p for p in problems)

    def test_expired_entry_rejected(self) -> None:
        """Reject an exception whose expiry precedes the explicit validation date."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(expires="2026-01-01")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("expired" in p for p in problems)

    def test_bad_tool_rejected(self) -> None:
        """Reject a scanner outside the maintained allowlist."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(tool="nessus")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("not in" in p for p in problems)

    def test_bad_expiry_date(self) -> None:
        """Report a non-ISO expiry string."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(expires="not-a-date")],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("not an ISO date" in p for p in problems)

    def test_non_list_exceptions(self) -> None:
        """Reject a register whose exceptions field is not an array."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": {"nope": True},
        }
        problems = checker.validate(payload, _TODAY)
        assert any("must be an array" in p for p in problems)

    def test_entry_not_a_table(self) -> None:
        """Reject a non-table entry in the exceptions array."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": ["just-a-string"],
        }
        problems = checker.validate(payload, _TODAY)
        assert any("must be a table" in p for p in problems)


class TestCommittedRegister:
    """Check the actual register and the validator file/date command boundaries."""

    def test_committed_register_is_valid(self) -> None:
        """Validate the actual governed register through its command entry point."""
        rc = checker.main([])
        assert rc == 0

    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        """Refuse a requested register file that does not exist."""
        rc = checker.main(["--path", str(tmp_path / "absent.toml")])
        assert rc == 1

    def test_expired_via_cli_today(self, tmp_path: Path) -> None:
        """Refuse a real TOML exception using the explicit command validation date."""
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
    """Exercise the command function against the actual governed register."""
    assert checker.main([]) == 0


@pytest.mark.parametrize("missing", checker._REQUIRED_FIELDS)
def test_each_required_field_enforced(missing: str) -> None:
    """Report every blank required metadata field in a complete register entry."""
    payload = {
        "schema_version": "director.security_exceptions.v1",
        "exceptions": [_entry(**{missing: ""})],
    }
    problems = checker.validate(payload, _TODAY)
    assert any(f"missing required field {missing!r}" in p for p in problems)


class TestDependabotTool:
    """Check that Dependabot remains a recognised scanner source."""

    def test_dependabot_tool_accepted(self) -> None:
        """Accept a complete current exception attributed to Dependabot."""
        payload = {
            "schema_version": "director.security_exceptions.v1",
            "exceptions": [_entry(tool="dependabot")],
        }
        assert checker.validate(payload, _TODAY) == []


class TestMcpRemediationInvariant:
    """Reject resolved MCP waivers and regression below advisory fix floors.

    The 2026-07-17 owner ruling stays closed. Semgrep 1.179.0 runs from its
    official image, so the SAST lock no longer carries MCP. Generic register
    tests above retain malformed-entry and expiry checks.
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

    def test_register_excludes_exactly_the_resolved_waivers(self) -> None:
        """Resolved advisory IDs cannot remain scanner exceptions."""
        assert not ({entry["id"] for entry in self._entries()} & self.RESOLVED.keys())

    def test_no_mcp_waiver_expiry_is_extended(self) -> None:
        """MCP remediation closes the waiver instead of renewing its expiry."""
        assert not [entry for entry in self._entries() if entry["package"] == "mcp"]

    def test_sast_lock_does_not_carry_mcp(self) -> None:
        """The image-hosted scanner does not put MCP back into the SAST lock."""
        text = self.CI_SAST_LOCK.read_text(encoding="utf-8")
        assert not re.search(r"^mcp==", text, flags=re.MULTILINE)
        assert not re.search(r"^semgrep==", text, flags=re.MULTILINE)
        assert re.search(r"^bandit==", text, flags=re.MULTILINE)

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

    def test_semgrep_stays_outside_the_locked_profiles(self) -> None:
        """The scanner image is the only Semgrep install, and MCP stays out."""
        text = self.CI_SAST_LOCK.read_text(encoding="utf-8")
        assert not re.search(r"^semgrep==", text, flags=re.MULTILINE)
        assert not re.search(r"^mcp==", text, flags=re.MULTILINE)
        workflow = (_ROOT / ".github/workflows/ci-security.yml").read_text(
            encoding="utf-8"
        )
        assert (
            "semgrep/semgrep@sha256:93963d9295a366f59e4850127b1550400ee7b388f04fe144e4a1f6325d96e01b"
            in workflow
        )
        assert "semgrep scan" in workflow


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
        assert not re.search(r"^semgrep==", text, flags=re.MULTILINE)
        assert not re.search(r"^pyjwt==", text, flags=re.MULTILINE)

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


@pytest.mark.parametrize("expires", [False, 0, 0.0, None, [], {}, ""])
def test_false_valued_expiry_is_rejected(expires: object) -> None:
    """Reject every false-valued expiry through the public register validator."""
    entry: dict[str, object] = dict(_entry())
    entry["expires"] = expires
    payload: dict[str, object] = {
        "schema_version": "director.security_exceptions.v1",
        "exceptions": [entry],
    }

    assert any(
        "not an ISO date" in problem for problem in checker.validate(payload, _TODAY)
    )


@pytest.mark.parametrize(
    "expires",
    ["2026-06-07", date(2026, 6, 7), "2026-12-31", date(2026, 12, 31)],
)
def test_string_and_native_date_expiry_remain_valid(expires: str | date) -> None:
    """Preserve ISO strings, native TOML dates and inclusive expiry-day validity."""
    entry: dict[str, object] = dict(_entry())
    entry["expires"] = expires
    payload: dict[str, object] = {
        "schema_version": "director.security_exceptions.v1",
        "exceptions": [entry],
    }

    assert checker.validate(payload, _TODAY) == []


@pytest.mark.parametrize("raw_expiry", ["0", "false"])
def test_cli_rejects_false_valued_toml_expiry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], raw_expiry: str
) -> None:
    """Reject real integer-zero and Boolean-false TOML expiries through the CLI."""
    register = tmp_path / "false-expiry.toml"
    register.write_text(
        'schema_version = "director.security_exceptions.v1"\n'
        "[[exceptions]]\n"
        'id = "GHSA-aaaa-bbbb-cccc"\n'
        'tool = "pip-audit"\n'
        'package = "probe"\n'
        'reason = "controlled validator regression"\n'
        'compensating_control = "no production exception requested"\n'
        'owner = "protoscience@anulum.li"\n'
        'opened = "2026-10-03"\n'
        'scope = "ci-dependency-audit"\n'
        f"expires = {raw_expiry}\n",
        encoding="utf-8",
    )

    assert checker.main(["--path", str(register), "--today", "2026-10-03"]) == 1
    assert "not an ISO date" in capsys.readouterr().err


@pytest.fixture
def lock_audit_step() -> dict[str, object]:
    """Read the unique complete-lock audit from the real reusable workflow."""
    workflow = read_workflow(_ROOT / ".github/workflows/ci-security.yml")
    jobs = object_mapping(workflow["jobs"])
    security = object_mapping(jobs["security"])
    steps = security["steps"]
    assert isinstance(steps, list)
    matches = [
        step
        for step in map(object_mapping, steps)
        if step.get("name") == "Audit complete Python lock"
    ]
    assert len(matches) == 1
    return matches[0]


_COMMITTED_REGISTER = _ROOT / "requirements" / "security-exceptions.toml"
"""The repository's own register, fixed at import so tests that move ``_ROOT`` keep it."""


def _registered_ignores() -> frozenset[str]:
    """Return the advisory ids the committed register lets the lock audit ignore today."""
    payload = tomllib.loads(_COMMITTED_REGISTER.read_text(encoding="utf-8"))
    return lock_audit_ignores(payload, date.today())


def test_complete_lock_audit_is_unconditional_and_required(
    lock_audit_step: dict[str, object],
) -> None:
    """Bind the real native audit to the required main push and PR security job."""
    command = lock_audit_step["run"]
    assert isinstance(command, str)
    ignores = _registered_ignores()
    assert validate_lock_audit(command, ignores=ignores) == []
    assert set(lock_audit_step) == {"name", "run"}
    called = read_workflow(_ROOT / ".github/workflows/ci-security.yml")
    job = object_mapping(object_mapping(called["jobs"])["security"])
    assert "if" not in job
    assert (
        validate_lock_audit(
            command,
            continue_on_error=job.get("continue-on-error", False),
            ignores=ignores,
        )
        == []
    )
    caller = read_workflow(_ROOT / ".github/workflows/ci.yml")
    for scope in (called, job, caller):
        environment = object_mapping(scope.get("env", {}))
        assert not any(key.startswith("UV_") for key in environment), (
            "UV_ environment variables must not override the complete-lock audit"
        )
    events = object_mapping(caller["on"])
    for event in ("push", "pull_request"):
        assert object_mapping(events[event])["branches"] == ["main"]
    jobs = object_mapping(caller["jobs"])
    security = object_mapping(jobs["security"])
    assert security == {"uses": "./.github/workflows/ci-security.yml"}
    needs = object_mapping(jobs["ci-gate"])["needs"]
    assert isinstance(needs, list)
    assert "security" in needs


@pytest.mark.parametrize("location", ["workflow", "job", "caller"])
@pytest.mark.parametrize(
    ("variable", "prohibited"), [("UV_PROJECT", True), ("CI", False)]
)
def test_complete_lock_guard_checks_inherited_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_audit_step: dict[str, object],
    location: str,
    variable: str,
    prohibited: bool,
) -> None:
    """Reject real workflow environment overrides while allowing unrelated keys."""
    called = read_workflow(_ROOT / ".github/workflows/ci-security.yml")
    caller = read_workflow(_ROOT / ".github/workflows/ci.yml")
    scopes = {
        "workflow": called,
        "job": object_mapping(object_mapping(called["jobs"])["security"]),
        "caller": caller,
    }
    scopes[location]["env"] = {variable: "controlled-environment-probe"}
    jobs = object_mapping(called["jobs"])
    jobs["security"] = scopes["job"]
    called["jobs"] = jobs
    workflows = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    for name, workflow in (("ci-security.yml", called), ("ci.yml", caller)):
        (workflows / name).write_text(yaml.safe_dump(workflow), encoding="utf-8")
    monkeypatch.setitem(globals(), "_ROOT", tmp_path)
    if prohibited:
        with pytest.raises(AssertionError, match="UV_ environment variables"):
            test_complete_lock_audit_is_unconditional_and_required(lock_audit_step)
    else:
        test_complete_lock_audit_is_unconditional_and_required(lock_audit_step)


@pytest.mark.parametrize(
    "change",
    [
        ("--frozen", ""),
        ("audit", "tree"),
        ("uv ", "pip-audit --locked "),
        (" --no-progress", " --package joserfc --no-progress"),
        (" --color never", " --ignore-until-fixed --color never"),
        (" --output-format json", " --output-format json || true"),
        (" --output-format json", " --output-format json; exit 0"),
        ("uv audit", "uv\naudit"),
        ("json", '"json'),
        ("uv audit --frozen --no-progress --color never --output-format json", ""),
    ],
)
def test_complete_lock_policy_refuses_actual_command_mutations(
    lock_audit_step: dict[str, object], change: tuple[str, str]
) -> None:
    """Refuse scope loss and success overrides applied to the real CI command."""
    command = lock_audit_step["run"]
    assert isinstance(command, str)
    original, replacement = change
    assert original in command
    ignores = _registered_ignores()
    assert validate_lock_audit(command, ignores=ignores) == []
    assert validate_lock_audit(command.replace(original, replacement), ignores=ignores)


@pytest.mark.parametrize("override", [True, "${{ true }}", 0, None])
def test_complete_lock_policy_refuses_failure_overrides(
    lock_audit_step: dict[str, object], override: object
) -> None:
    """Fail closed for literal and expression-based job or step overrides."""
    command = lock_audit_step["run"]
    assert isinstance(command, str)
    assert validate_lock_audit(
        command, continue_on_error=override, ignores=_registered_ignores()
    )


def test_complete_lock_setup_preserves_tool_pin_and_requirements_audits() -> None:
    """Retain the existing CI tool version and both required profile audits."""
    workflow = read_workflow(_ROOT / ".github/workflows/ci-security.yml")
    job = object_mapping(object_mapping(workflow["jobs"])["security"])
    steps = job["steps"]
    assert isinstance(steps, list)
    setup = [
        step
        for step in map(object_mapping, steps)
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert len(setup) == 1
    assert setup[0]["uses"] == (
        "astral-sh/setup-uv@68559e99cc807c1eb7e9238fd43a15bba460eec2"
    )
    nightly = read_workflow(_ROOT / ".github/workflows/nightly-red-team.yml")
    nightly_job = object_mapping(object_mapping(nightly["jobs"])["live-red-team"])
    nightly_steps = nightly_job["steps"]
    assert isinstance(nightly_steps, list)
    nightly_pins = [
        object_mapping(step["with"])["version"]
        for step in map(object_mapping, nightly_steps)
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert [object_mapping(setup[0]["with"])["version"]] == nightly_pins == ["0.13.6"]
    audits = [
        step
        for step in map(object_mapping, steps)
        if step.get("name") == "Audit dependencies"
    ]
    assert len(audits) == 1
    assert set(audits[0]) == {"name", "run"}
    assert str(audits[0]["run"]).strip().splitlines() == [
        "pip-audit --strict --desc on -r requirements.txt",
        "pip-audit --strict --desc on -r requirements/docker-server.txt",
    ]


def test_register_script_entrypoint_accepts_the_committed_register(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the maintained script entrypoint against the actual exception file."""
    script = _ROOT / "tools/check_security_exceptions.py"
    monkeypatch.setattr(sys, "argv", [str(script)])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 0
    assert "security exception register OK" in capsys.readouterr().out


def test_register_script_extra_argument_cannot_mask_an_expired_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep expiry refusal effective through the script's argument parser."""
    register = tmp_path / "expired.toml"
    register.write_text(
        'schema_version = "director.security_exceptions.v1"\n'
        "exceptions = [\n"
        '  { id = "GHSA-expiry-regression", tool = "pip-audit", package = "probe", '
        'reason = "validator expiry regression", compensating_control = "test only", '
        'owner = "protoscience@anulum.li", opened = "2026-01-01", '
        'expires = "2026-02-01", scope = "ci-dependency-audit" },\n'
        "]\n",
        encoding="utf-8",
    )
    script = _ROOT / "tools/check_security_exceptions.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "extra-argument",
            "--path",
            str(register),
            "--today",
            "2027-01-01",
        ],
    )
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 1
    assert "expired on 2026-02-01" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("dependency", "prohibited"), [("pytest>=7", False), ("mcp>=1.30", True)]
)
def test_manifest_guard_checks_grouped_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    prohibited: bool,
) -> None:
    """Prove the actual CI guard detects a prohibited new dependency group."""
    manifest = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "dependency-groups" not in tomllib.loads(manifest)
    (tmp_path / "pyproject.toml").write_text(
        manifest
        + "\n[dependency-groups]\n"
        + f"validation = [{dependency!r}, {{ include-group = 'base' }}]\n"
        + "base = []\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(globals(), "_ROOT", tmp_path)
    guard = TestMcpRemediationInvariant()
    if prohibited:
        with pytest.raises(AssertionError, match="mcp"):
            guard.test_mcp_absent_from_runtime_dependency_surfaces()
    else:
        guard.test_mcp_absent_from_runtime_dependency_surfaces()


_IGNORE_REFUSAL = (
    "The complete-lock audit ignores an advisory the register does not hold"
)
_SCOPE_REFUSAL = "The complete-lock audit command or scope has changed"


@pytest.mark.parametrize(
    ("suffix", "message"),
    [
        (" --ignore GHSA-unregistered-probe", _IGNORE_REFUSAL),
        (" --ignore GHSA-f4j7-r4q5-qw2c", _IGNORE_REFUSAL),
        (" --ignore", _SCOPE_REFUSAL),
        (" --ignore-until-fixed GHSA-f4j7-r4q5-qw2c", _SCOPE_REFUSAL),
    ],
)
def test_lock_audit_refuses_an_ignore_the_register_does_not_grant(
    lock_audit_step: dict[str, object], suffix: str, message: str
) -> None:
    """Refuse an unregistered, repeated, dangling or differently spelled ignore."""
    command = lock_audit_step["run"]
    assert isinstance(command, str)
    problems = validate_lock_audit(command + suffix, ignores=_registered_ignores())
    assert problems == [message]


def test_lock_audit_refuses_the_registered_ignores_after_their_expiry(
    lock_audit_step: dict[str, object],
) -> None:
    """An expired register entry no longer lets the real audit skip its advisory."""
    payload = tomllib.loads(_COMMITTED_REGISTER.read_text(encoding="utf-8"))
    expiries = {
        date.fromisoformat(entry["expires"])
        for entry in payload["exceptions"]
        if entry["tool"] == "uv-audit"
    }
    after = max(expiries).toordinal() + 1
    ignores = lock_audit_ignores(payload, date.fromordinal(after))
    command = lock_audit_step["run"]
    assert isinstance(command, str)
    assert ignores == frozenset()
    assert validate_lock_audit(command, ignores=ignores) == [_IGNORE_REFUSAL]


def test_only_unexpired_uv_audit_entries_grant_lock_audit_ignores() -> None:
    """Other tools, bad expiries, empty ids and malformed tables grant nothing."""
    entry = {"tool": "uv-audit", "id": "GHSA-kept", "expires": "2026-06-07"}
    payload: dict[str, object] = {
        "exceptions": [
            entry,
            {**entry, "id": "GHSA-other-tool", "tool": "pip-audit"},
            {**entry, "id": "GHSA-bad-date", "expires": "never"},
            {**entry, "id": "  "},
            {**entry, "id": "GHSA-expired", "expires": "2026-06-06"},
            "not a table",
        ]
    }
    assert lock_audit_ignores(payload, _TODAY) == frozenset({"GHSA-kept"})
    assert lock_audit_ignores({"exceptions": "not a list"}, _TODAY) == frozenset()


_NPM_TODAY = date(2026, 11, 13)
_NPM_ADVISORY = "GHSA-g3hv-75wg-m249"
_NPM_URL = "https://github.com/advisories/GHSA-g3hv-75wg-m249"


def _npm_entry(**overrides: str) -> dict[str, str]:
    """Build one complete npm-audit register entry."""
    entry = {
        "id": _NPM_ADVISORY,
        "tool": "npm-audit",
        "package": "minimatch",
        "reason": "typedoc 0.28.20 requires minimatch ^10.2.5 and fails with 11",
        "compensating_control": "dev-only documentation tool; not in the published package",
        "owner": "protoscience@anulum.li",
        "opened": "2026-11-13",
        "expires": "2026-12-13",
        "scope": "packages/vercel-ai",
    }
    entry.update(overrides)
    return entry


def _npm_report(vulnerabilities: dict[str, object]) -> str:
    """Serialise one npm audit vulnerabilities object."""
    return json.dumps({"auditReportVersion": 2, "vulnerabilities": vulnerabilities})


def _minimatch_via(url: str = _NPM_URL) -> dict[str, object]:
    """Build one advisory via object whose URL carries the governed GHSA."""
    return {"source": 1113512, "name": "minimatch", "url": url, "severity": "high"}


def test_npm_audit_accepts_a_report_covered_by_an_unexpired_entry() -> None:
    """Accept the real advisory shape, including a followed package name and a duplicate URL."""
    report = _npm_report(
        {
            "typedoc": {"via": ["minimatch"]},
            "minimatch": {"via": [_minimatch_via(), _minimatch_via(_NPM_URL.upper())]},
        }
    )
    assert (
        checker.check_npm_audit(report, {"exceptions": [_npm_entry()]}, _NPM_TODAY)
        == []
    )


def test_npm_audit_accepts_a_report_with_no_advisories() -> None:
    """A vulnerabilities object with no via entries is a clean report."""
    assert (
        checker.check_npm_audit(
            _npm_report({"typedoc": {}}), {"exceptions": []}, _NPM_TODAY
        )
        == []
    )


def test_npm_audit_refuses_an_unregistered_advisory_with_its_whole_message() -> None:
    """An advisory with no npm-audit entry fails as that exact sentence."""
    report = _npm_report({"minimatch": {"via": [_minimatch_via()]}})
    assert checker.check_npm_audit(report, {"exceptions": []}, _NPM_TODAY) == [
        checker.npm_audit_unregistered(_NPM_ADVISORY)
    ]
    assert checker.check_npm_audit(
        report, {"exceptions": "not a list"}, _NPM_TODAY
    ) == [checker.npm_audit_unregistered(_NPM_ADVISORY)]
    assert checker.check_npm_audit(
        report,
        {
            "exceptions": [
                "not a table",
                _npm_entry(tool="uv-audit"),
                _npm_entry(package="typedoc"),
                _npm_entry(id="GHSA-other-xxxx-yyyy"),
            ]
        },
        _NPM_TODAY,
    ) == [checker.npm_audit_unregistered(_NPM_ADVISORY)]


def test_npm_audit_refuses_an_expired_entry_with_its_whole_message() -> None:
    """A lapsed or undated npm-audit entry does not hold its advisory."""
    report = _npm_report({"minimatch": {"via": [_minimatch_via()]}})
    expired = checker.check_npm_audit(
        report,
        {"exceptions": [_npm_entry(expires="2026-11-12")]},
        _NPM_TODAY,
    )
    undated = checker.check_npm_audit(
        report,
        {"exceptions": [_npm_entry(expires="never")]},
        _NPM_TODAY,
    )
    assert expired == undated == [checker.npm_audit_expired(_NPM_ADVISORY)]
    held_after_expired = checker.check_npm_audit(
        report,
        {
            "exceptions": [
                _npm_entry(expires="2026-11-12"),
                _npm_entry(),
            ]
        },
        _NPM_TODAY,
    )
    assert held_after_expired == []


@pytest.mark.parametrize(
    ("report", "message"),
    [
        ("", checker.NPM_AUDIT_EMPTY),
        (" \n\t", checker.NPM_AUDIT_EMPTY),
        ("{", checker.NPM_AUDIT_MALFORMED),
        ("null", checker.NPM_AUDIT_MALFORMED),
        ("[]", checker.NPM_AUDIT_MALFORMED),
        (
            json.dumps({"error": {"code": "ENOLOCK"}, "vulnerabilities": {}}),
            checker.NPM_AUDIT_ERROR,
        ),
        ("{}", checker.NPM_AUDIT_NO_VULNERABILITIES),
        (
            json.dumps({"vulnerabilities": []}),
            checker.NPM_AUDIT_NO_VULNERABILITIES,
        ),
        (
            json.dumps({"vulnerabilities": {"minimatch": "nope"}}),
            checker.NPM_AUDIT_BAD_RECORD,
        ),
        (
            json.dumps({"vulnerabilities": {"minimatch": {"via": "minimatch"}}}),
            checker.NPM_AUDIT_BAD_VIA,
        ),
        (
            json.dumps({"vulnerabilities": {"minimatch": {"via": [1]}}}),
            checker.NPM_AUDIT_BAD_VIA,
        ),
        (
            json.dumps({"vulnerabilities": {"minimatch": {"via": ["typedoc"]}}}),
            checker.NPM_AUDIT_UNKNOWN_FOLLOW,
        ),
        (
            json.dumps({"vulnerabilities": {"minimatch": {"via": [{"url": 1}]}}}),
            checker.NPM_AUDIT_NO_GHSA,
        ),
        (
            json.dumps(
                {
                    "vulnerabilities": {
                        "minimatch": {
                            "via": [{"url": "https://example.invalid/advisory"}]
                        }
                    }
                }
            ),
            checker.NPM_AUDIT_NO_GHSA,
        ),
    ],
)
def test_npm_audit_refuses_unusable_reports_with_their_whole_message(
    report: str, message: str
) -> None:
    """Each empty, malformed, error and incomplete report fails as one whole sentence."""
    assert checker.check_npm_audit(
        report, {"exceptions": [_npm_entry()]}, _NPM_TODAY
    ) == [message]


def test_npm_audit_cli_prints_the_whole_refusal_and_accepts_a_covered_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI prints only the checker message, and npm's absence from argv cannot pass a bad report."""
    register = tmp_path / "register.toml"
    register.write_text(
        _COMMITTED_REGISTER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    refused = tmp_path / "refused.json"
    refused.write_text("{", encoding="utf-8")
    assert (
        checker.main(
            [
                "--path",
                str(register),
                "--npm-audit",
                str(refused),
                "--today",
                "2026-11-13",
            ]
        )
        == 1
    )
    assert capsys.readouterr().err == checker.NPM_AUDIT_MALFORMED + "\n"
    missing = tmp_path / "missing.json"
    assert (
        checker.main(
            [
                "--path",
                str(register),
                "--today",
                "2026-11-13",
                "--npm-audit",
                str(missing),
            ]
        )
        == 1
    )
    assert capsys.readouterr().err == checker.NPM_AUDIT_EMPTY + "\n"
    covered = tmp_path / "covered.json"
    covered.write_text(
        _npm_report({"minimatch": {"via": [_minimatch_via()]}}),
        encoding="utf-8",
    )
    assert (
        checker.main(
            [
                "--npm-audit",
                str(covered),
                "--path",
                str(register),
                "--today",
                "2026-11-13",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "npm audit report OK\n"
    assert (
        checker.main(["--path", str(register), "--today", "2026-11-13", "--npm-audit"])
        == 0
    )
    assert "security exception register OK" in capsys.readouterr().out


def test_vercel_middleware_step_uses_the_npm_audit_checker() -> None:
    """The middleware step judges the JSON report; the gateway audit stays a plain level gate."""
    workflow = read_workflow(_ROOT / ".github/workflows/ci-static.yml")
    steps = object_mapping(object_mapping(workflow["jobs"])["lint"])["steps"]
    assert isinstance(steps, list)
    mapped = [object_mapping(step) for step in steps]
    middleware = next(
        step
        for step in mapped
        if step.get("name")
        == "Vercel middleware types, HTTP contracts and API documentation"
    )
    gateway = next(
        step
        for step in mapped
        if step.get("name")
        == "Gateway native JavaScript types, runtime and documentation"
    )
    run = str(middleware["run"])
    assert 'audit="$RUNNER_TEMP/vercel-ai-npm-audit.json"' in run
    assert 'npm audit --json > "$audit" || true' in run
    assert (
        'PYTHONPATH="$GITHUB_WORKSPACE" python -m tools.check_security_exceptions '
        '--npm-audit "$audit"'
    ) in run
    assert "npm audit --audit-level=low" not in run
    assert "npm audit --audit-level=low" in str(gateway["run"])
    assert "--npm-audit" not in str(gateway["run"])
    payload = tomllib.loads(_COMMITTED_REGISTER.read_text(encoding="utf-8"))
    assert (
        checker.check_npm_audit(
            _npm_report(
                {
                    "typedoc": {"via": ["minimatch"]},
                    "minimatch": {"via": [_minimatch_via()]},
                }
            ),
            payload,
            _NPM_TODAY,
        )
        == []
    )
    assert checker.check_npm_audit(
        _npm_report({"minimatch": {"via": [_minimatch_via()]}}),
        payload,
        date(2026, 12, 14),
    ) == [checker.npm_audit_expired(_NPM_ADVISORY)]
