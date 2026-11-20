# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# DIRECTOR-AI — npm lock audit gate with expiring development-only exceptions
"""An npm advisory passes only under an active, development-only, seven-day exception."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tools.npm_audit_gate import (
    AUDIT_COMMAND,
    AdvisoryException,
    AuditGateError,
    judge,
    load_exceptions,
    main,
    parse_exception,
    run_audit,
)

ROOT = Path(__file__).resolve().parents[1]
ADVISORY = "GHSA-fc5q-39xc-vjpf"
URL = f"https://github.com/advisories/{ADVISORY}"
GRANTED = date(2026, 11, 18)
EXPIRES = date(2026, 11, 24)
TODAY = GRANTED
LOCK_SHA256 = "a" * 64


def _exception(package: str = "picomatch", **changes: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "advisory": ADVISORY,
        "package": package,
        "scope": "dev",
        "granted": GRANTED.isoformat(),
        "expires": EXPIRES.isoformat(),
        "reason": "development tooling only",
        "lock_sha256": LOCK_SHA256,
    }
    item.update(changes)
    return item


def _advisory(url: str = URL, dependency: str = "picomatch") -> dict[str, Any]:
    return {
        "source": 1,
        "name": dependency,
        "dependency": dependency,
        "url": url,
        "severity": "high",
    }


def _vulnerabilities() -> dict[str, Any]:
    """Build an audit fixture for the cockpit lock and its Vitest dependency cycle."""
    return {
        "picomatch": {"via": [_advisory()], "nodes": ["node_modules/picomatch"]},
        "fdir": {"via": ["picomatch"], "nodes": ["node_modules/fdir"]},
        "tinyglobby": {
            "via": ["fdir", "picomatch"],
            "nodes": ["node_modules/tinyglobby"],
        },
        "vite": {"via": ["picomatch", "tinyglobby"], "nodes": ["node_modules/vite"]},
        "vitest": {
            "via": ["@vitest/coverage-v8", "vite"],
            "nodes": ["node_modules/vitest"],
        },
        "@vitest/coverage-v8": {
            "via": ["vitest"],
            "nodes": ["node_modules/@vitest/coverage-v8"],
        },
    }


def _report(vulnerabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    listed = _vulnerabilities() if vulnerabilities is None else vulnerabilities
    return {"auditReportVersion": 2, "vulnerabilities": listed, "metadata": {}}


def _lock(report: dict[str, Any], **overrides: dict[str, Any]) -> dict[str, Any]:
    installs: dict[str, Any] = {"": {"name": "client"}}
    for entry in report["vulnerabilities"].values():
        for node in entry["nodes"]:
            installs[node] = {"version": "1.0.0", "dev": True}
    installs.update(overrides)
    return {"lockfileVersion": 3, "packages": installs}


def _granted(**changes: Any) -> tuple[AdvisoryException, ...]:
    return (parse_exception(_exception(**changes)),)


def test_a_dev_only_advisory_under_an_active_exception_passes_through_the_cycle() -> (
    None
):
    """Every package that carries or reaches the excepted advisory is accepted, cycles included."""
    report = _report()
    failures, used = judge(report, _lock(report), _granted(), TODAY)

    assert failures == []
    assert used == {ADVISORY}


def test_an_advisory_without_an_exception_fails() -> None:
    """A second advisory on another package fails although the first one is excepted."""
    vulnerabilities = _vulnerabilities()
    vulnerabilities["vite"]["via"].append(
        _advisory("https://github.com/advisories/GHSA-aaaa-bbbb-cccc", "vite")
    )
    report = _report(vulnerabilities)

    failures, _ = judge(report, _lock(report), _granted(), TODAY)

    assert failures == ["vite: GHSA-aaaa-bbbb-cccc (high) has no active exception"]


def test_an_exception_covers_only_its_own_package() -> None:
    """The same advisory id granted for another package does not cover this one."""
    report = _report()
    failures, used = judge(report, _lock(report), _granted(package="minimatch"), TODAY)

    assert failures == [f"picomatch: {ADVISORY} (high) has no active exception"]
    assert used == set()


def test_an_advisory_without_a_github_advisory_url_fails() -> None:
    """An advisory npm reports without a GitHub advisory id cannot match any exception."""
    report = _report(
        {
            "picomatch": {
                "via": [_advisory(url="https://example.org/x")],
                "nodes": ["node_modules/picomatch"],
            }
        }
    )

    failures, _ = judge(report, _lock(report), _granted(), TODAY)

    assert failures == [
        "picomatch: https://example.org/x (high) has no active exception"
    ]


@pytest.mark.parametrize(
    ("install", "reason"),
    [
        ({"version": "4.0.7"}, "a production install"),
        ({"version": "4.0.7", "dev": False}, "an explicit production install"),
        (
            {"version": "4.0.7", "devOptional": True},
            "an optional install shared with production",
        ),
        (None, "a path the lock does not have"),
    ],
)
def test_an_install_outside_development_dependencies_fails(
    install: dict[str, Any] | None, reason: str
) -> None:
    """The exception never covers a package the lock installs outside development dependencies."""
    report = _report()
    lock = _lock(report)
    if install is None:
        del lock["packages"]["node_modules/vite"]
    else:
        lock["packages"]["node_modules/vite"] = install

    failures, _ = judge(report, lock, _granted(), TODAY)

    assert failures == ["vite: node_modules/vite is not a development-only install"], (
        reason
    )


def test_a_dependency_the_report_does_not_list_fails() -> None:
    """A package reached through an unlisted vulnerable name is not judged as accepted."""
    vulnerabilities = _vulnerabilities()
    vulnerabilities["fdir"]["via"] = ["unknown-package"]
    report = _report(vulnerabilities)

    failures, _ = judge(report, _lock(report), _granted(), TODAY)

    assert failures == [
        "fdir: depends on unknown-package, which the report does not list"
    ]


def test_a_package_without_an_install_path_fails() -> None:
    """The development-only proof needs at least one install path."""
    vulnerabilities = _vulnerabilities()
    vulnerabilities["fdir"]["nodes"] = []
    report = _report(vulnerabilities)

    failures, _ = judge(report, _lock(report), _granted(), TODAY)

    assert failures == ["fdir: the report names no install path"]


def test_the_exception_holds_through_the_day_before_it_expires_and_fails_on_that_day() -> (
    None
):
    """On its expiry day the exception fails by itself, and its advisory fails again."""
    report = _report()
    lock = _lock(report)

    assert judge(report, lock, _granted(), date(2026, 11, 23)) == ([], {ADVISORY})
    failures, used = judge(report, lock, _granted(), EXPIRES)
    assert failures == [
        f"{ADVISORY}: the exception expired on 2026-11-24",
        f"picomatch: {ADVISORY} (high) has no active exception",
    ]
    assert used == set()


def test_an_expired_exception_fails_a_clean_lock_too() -> None:
    """A stale exception must be removed even when no lock reports its advisory any more."""
    report = _report({})

    assert judge(report, _lock(report), _granted(), EXPIRES) == (
        [f"{ADVISORY}: the exception expired on 2026-11-24"],
        set(),
    )


@pytest.mark.parametrize(
    ("report", "lock", "message"),
    [
        (
            {"auditReportVersion": 1, "vulnerabilities": {}},
            {"packages": {}},
            "version 2 report",
        ),
        ({"auditReportVersion": 2}, {"packages": {}}, "version 2 report"),
        (
            {"auditReportVersion": 2, "vulnerabilities": {}},
            {"dependencies": {}},
            "no packages map",
        ),
    ],
)
def test_an_unusable_report_or_lock_is_refused(
    report: dict[str, Any], lock: dict[str, Any], message: str
) -> None:
    """A report or lock in another format is never judged as clean."""
    with pytest.raises(AuditGateError, match=message):
        judge(report, lock, _granted(), TODAY)


@pytest.mark.parametrize(
    ("item", "message"),
    [
        ("GHSA-fc5q-39xc-vjpf", "exactly the fields"),
        (
            {key: value for key, value in _exception().items() if key != "reason"},
            "exactly the fields",
        ),
        ({**_exception(), "approved": "yes"}, "exactly the fields"),
        (_exception(advisory="CVE-2026-1"), "not a GitHub advisory id"),
        (_exception(lock_sha256="not-a-digest"), "lowercase lock SHA-256"),
        (_exception(advisory="GHSA-FC5Q-39XC-VJPF"), "not a GitHub advisory id"),
        (_exception(package=" "), "names its package and its reason"),
        (_exception(reason=""), "names its package and its reason"),
        (_exception(scope="prod"), "development dependencies only"),
        (_exception(expires="2026-11-18"), "1 to 7 days, not 0"),
        (_exception(expires="2026-11-26"), "1 to 7 days, not 8"),
        (_exception(expires="2026-11-11"), "1 to 7 days, not -7"),
        (_exception(granted="17.11.2026"), "Invalid isoformat"),
    ],
)
def test_an_invalid_exception_is_refused(item: object, message: str) -> None:
    """Only a development-scope exception of one to seven days for a named advisory is valid."""
    with pytest.raises(ValueError, match=message):
        parse_exception(item)


def test_a_seven_day_exception_is_the_longest_valid_one() -> None:
    """The boundary span is accepted with its fields as written."""
    expiry = date(2026, 11, 25)
    assert parse_exception(
        _exception(package=" picomatch ", expires=expiry.isoformat())
    ) == AdvisoryException(
        ADVISORY, "picomatch", GRANTED, expiry, "development tooling only", LOCK_SHA256
    )


def test_the_exceptions_file_needs_its_schema(tmp_path: Path) -> None:
    """A file without schema 1 and an exceptions list is refused; a valid one loads."""
    path = tmp_path / "exceptions.json"
    path.write_text(json.dumps({"exceptions": []}), encoding="utf-8")
    with pytest.raises(AuditGateError, match="schema: 1"):
        load_exceptions(path)
    path.write_text(
        json.dumps({"schema": 1, "exceptions": [_exception()]}), encoding="utf-8"
    )
    assert load_exceptions(path) == _granted()


def test_the_repository_exceptions_are_valid() -> None:
    """Bind committed exceptions to the Studio lock and respect the withdrawal."""
    exceptions = load_exceptions(ROOT / ".github/npm-audit-exceptions.json")
    lock = ROOT / "studio-web/package-lock.json"
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    assert ADVISORY not in {exception.advisory for exception in exceptions}
    assert all(exception.lock_sha256 == digest for exception in exceptions)


class _Runner:
    """Record the audit call and answer with a fixed output and exit code."""

    def __init__(self, stdout: str, returncode: int) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, "")


def test_the_audit_runs_npm_lock_only_in_the_lock_directory(tmp_path: Path) -> None:
    """The gate inspects the committed lock without installing, and reads its JSON report."""
    runner = _Runner(json.dumps(_report()), 1)

    assert run_audit(tmp_path, runner) == _report()
    args, kwargs = runner.calls[0]
    assert (
        args
        == list(AUDIT_COMMAND)
        == [
            "npm",
            "audit",
            "--package-lock-only",
            "--json",
            "--include=prod",
            "--include=dev",
            "--include=optional",
            "--include=peer",
        ]
    )
    assert kwargs["cwd"] == tmp_path
    assert kwargs["check"] is False
    assert run_audit(tmp_path, _Runner(json.dumps(_report({})), 0)) == _report({})


@pytest.mark.parametrize(
    ("stdout", "returncode", "message"),
    [
        ("npm ERR! network", 1, "no JSON report"),
        ("[]", 0, "npm audit failed"),
        (json.dumps({"error": {"code": "ENOAUDIT"}}), 1, "npm audit failed"),
        (json.dumps(_report({})), 1, "disagrees"),
        (json.dumps(_report()), 0, "disagrees"),
        (json.dumps(_report()), 2, "disagrees"),
    ],
)
def test_an_audit_that_did_not_run_cleanly_is_refused(
    stdout: str, returncode: int, message: str
) -> None:
    """No output, an error object or an exit code that contradicts the report never passes."""
    with pytest.raises(AuditGateError, match=message):
        run_audit(Path("."), _Runner(stdout, returncode))


def _write_inputs(tmp_path: Path, report: dict[str, Any]) -> tuple[Path, Path]:
    exceptions = tmp_path / "exceptions.json"
    lock = tmp_path / "client" / "package-lock.json"
    lock.parent.mkdir()
    lock.write_text(json.dumps(_lock(report)), encoding="utf-8")
    entry = _exception(lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
    exceptions.write_text(
        json.dumps({"schema": 1, "exceptions": [entry]}), encoding="utf-8"
    )
    return exceptions, lock


def test_the_cli_reports_accepted_and_clean_locks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Accepted advisories and clean locks are printed, and the exit code is 0."""
    exceptions, lock = _write_inputs(tmp_path, _report())
    report = json.dumps(_report())

    assert (
        main(
            ["--exceptions", str(exceptions), str(lock)],
            runner=_Runner(report, 1),
            today=TODAY,
        )
        == 0
    )
    assert (
        main(
            ["--exceptions", str(exceptions), str(lock)],
            runner=_Runner(json.dumps(_report({})), 0),
            today=TODAY,
        )
        == 0
    )

    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"{lock}: {ADVISORY} accepted in development dependencies only;"
        " the exception expires on 2026-11-24",
        f"{lock}: found 0 vulnerabilities",
    ]


def test_the_cli_fails_a_lock_with_any_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failure in one lock fails the run; a missing lock fails as unreadable."""
    exceptions, lock = _write_inputs(tmp_path, _report())
    report = json.dumps(_report())

    status = main(
        [
            "--exceptions",
            str(exceptions),
            str(lock),
            str(tmp_path / "missing" / "package-lock.json"),
        ],
        runner=_Runner(report, 1),
        today=EXPIRES,
    )

    assert status == 1
    err = capsys.readouterr().err
    assert f"{lock}: {ADVISORY}: the exception expired on 2026-11-24" in err
    assert "missing/package-lock.json: [Errno 2]" in err


def test_the_cli_refuses_an_unusable_exceptions_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An invalid exceptions file stops the gate with exit 2 before any audit runs."""
    exceptions = tmp_path / "exceptions.json"
    exceptions.write_text(
        json.dumps({"schema": 1, "exceptions": [_exception(scope="prod")]}),
        encoding="utf-8",
    )
    runner = _Runner("{}", 0)

    assert (
        main(
            ["--exceptions", str(exceptions), "package-lock.json"],
            runner=runner,
            today=TODAY,
        )
        == 2
    )
    assert "development dependencies only" in capsys.readouterr().err
    assert runner.calls == []


def test_unreported_production_install_fails() -> None:
    """A second production path for an affected package remains in scope."""
    report = _report()
    path = "node_modules/example/node_modules/picomatch"
    lock = _lock(report)
    lock["packages"][path] = {"version": "4.0.7"}

    assert judge(report, lock, _granted(), TODAY)[0] == [
        f"picomatch: {path} is not a development-only install",
    ]


def test_another_advisory_on_the_same_package_fails() -> None:
    """The approved package can still carry an unapproved advisory."""
    report = _report()
    report["vulnerabilities"]["picomatch"]["via"].append(
        _advisory("https://github.com/advisories/GHSA-aaaa-bbbb-cccc")
    )
    assert judge(report, _lock(report), _granted(), TODAY)[0] == [
        "picomatch: GHSA-aaaa-bbbb-cccc (high) has no active exception",
    ]


def test_a_changed_lock_does_not_run_npm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A byte change to the authorised lock invalidates the decision."""
    exceptions, lock = _write_inputs(tmp_path, _report())
    lock.write_bytes(lock.read_bytes() + b"\n")
    runner = _Runner(json.dumps(_report()), 1)

    assert (
        main(["--exceptions", str(exceptions), str(lock)], runner=runner, today=TODAY)
        == 1
    )
    assert "the lock SHA-256 differs" in capsys.readouterr().err
    assert not runner.calls


def test_a_lock_changed_during_npm_audit_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A report cannot be paired with a concurrent replacement of its input."""
    exceptions, lock = _write_inputs(tmp_path, _report())

    def changing_runner(
        args: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        lock.write_bytes(lock.read_bytes() + b"\n")
        return subprocess.CompletedProcess(args, 0, json.dumps(_report({})), "")

    assert (
        main(
            ["--exceptions", str(exceptions), str(lock)],
            runner=changing_runner,
            today=TODAY,
        )
        == 1
    )
    assert "changed during npm audit" in capsys.readouterr().err


def test_a_future_exception_does_not_hold() -> None:
    """A clean report cannot activate the decision before its start date."""
    report = _report({})
    assert judge(report, _lock(report), _granted(), date(2026, 11, 17)) == (
        [f"{ADVISORY}: the exception starts on 2026-11-18"],
        set(),
    )


def test_an_advisory_free_dependency_cycle_is_refused() -> None:
    """Each reported package has to reach an actual advisory."""
    report = _report()
    report["vulnerabilities"]["nominal"] = {
        "via": ["nominal"],
        "nodes": ["node_modules/nominal"],
    }
    with pytest.raises(AuditGateError, match="does not reach a direct advisory"):
        judge(report, _lock(report), _granted(), TODAY)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"via": []}, "no valid advisory links"),
        ({"via": [42]}, "invalid advisory record"),
        ({"nodes": [12]}, "invalid install paths"),
    ],
)
def test_malformed_report_records_fail(change: dict[str, object], message: str) -> None:
    """Malformed records are errors even when the advisory itself is approved."""
    report = _report()
    lock = _lock(report)
    report["vulnerabilities"]["picomatch"].update(change)
    with pytest.raises(AuditGateError, match=message):
        judge(report, lock, _granted(), TODAY)


def test_studio_uses_the_gate_before_installing() -> None:
    """The sole explicit Studio audit runs the gate and cannot be bypassed."""
    from tools.audit_ci_workflow_modularity import read_workflow
    from tools.ci_workflow_inventory import object_mapping

    workflow = ROOT / ".github/workflows/ci-rust.yml"
    job = object_mapping(object_mapping(read_workflow(workflow)["jobs"])["rust"])
    steps = job["steps"]
    assert isinstance(steps, list)
    (gate,) = (
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Audit the locked Studio dependencies"
    )
    (setup,) = (
        i
        for i, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("actions/setup-node@")
    )
    (install,) = (
        i
        for i, step in enumerate(steps)
        if step.get("name") == "Install and build Studio from its lock"
    )
    assert setup < gate < install
    assert steps[gate] == {
        "name": "Audit the locked Studio dependencies",
        "working-directory": ".",
        "run": (
            "python3 -m tools.npm_audit_gate --exceptions .github/npm-audit-exceptions.json "
            "studio-web/package-lock.json"
        ),
    }
    assert "continue-on-error" not in job
    assert "--no-audit" in steps[install]["run"]
    workflows = ROOT / ".github/workflows"
    for location in sorted(
        set(workflows.glob("*.yml")) | set(workflows.glob("*.yaml"))
    ):
        data = location.read_text(encoding="utf-8")
        if "npm audit" not in data:
            continue
        assert location.name == "ci-static.yml"
        jobs = object_mapping(read_workflow(location)["jobs"])
        for raw in jobs.values():
            other = object_mapping(raw)
            other_steps = other.get("steps", [])
            assert isinstance(other_steps, list)
            for step in other_steps:
                run = str(step.get("run", ""))
                if "npm audit" in run:
                    assert step.get("working-directory") in {
                        "packages/vercel-ai",
                        "gateway/go/bench",
                    }
                    assert "studio-web" not in run


def test_duplicate_exception_stops_before_npm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ambiguous pair of grants cannot pass the gate."""
    path = tmp_path / "exceptions.json"
    item = _exception()
    path.write_text(json.dumps({"schema": 1, "exceptions": [item, item]}))
    runner = _Runner("{}", 0)

    assert main(["--exceptions", str(path), "package-lock.json"], runner=runner) == 2
    assert "repeats an advisory and package" in capsys.readouterr().err
    assert not runner.calls


def test_invalid_package_record_is_refused() -> None:
    """An npm report needs an object for every affected package."""
    report = _report()
    lock = _lock(report)
    report["vulnerabilities"]["picomatch"] = None
    with pytest.raises(AuditGateError, match="invalid package record"):
        judge(report, lock, _granted(), TODAY)


def test_a_non_object_lock_fails_before_npm(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid digest cannot authorise npm to audit a non-object lock."""
    report = _report()
    exceptions, lock = _write_inputs(tmp_path, report)
    lock.write_bytes(b"[]")
    entry = _exception(lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
    exceptions.write_text(json.dumps({"schema": 1, "exceptions": [entry]}))
    runner = _Runner(json.dumps(report), 1)

    status = main(
        ["--exceptions", str(exceptions), str(lock)], runner=runner, today=TODAY
    )

    assert status == 1
    assert "the lock is not a JSON object" in capsys.readouterr().err
    assert not runner.calls
