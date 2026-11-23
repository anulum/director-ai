# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the installed support-contract probes
"""Run the probes on Core, Core+Pro and Core+Full trees, as the package job does.

Each tree is the package source sliced by the same manifest the wheels use:
Core without the paid packages and modules, Pro without the labs packages,
Full complete. The script runs in a subprocess from outside the checkout, Pro
first so that it emits the packet the Core run verifies. The receipts must
then support every claimed installation. The failure branches run in process.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import director_ai
import director_ai.cli as cli_module
import director_ai.support_contract as reader
from director_ai.support_contract_catalog import CONTRACT
from tools.support_contract_receipts import check_receipts

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_support_contract.py"
COMMIT = "0" * 40
TREE = "7" * 40


def _load_script() -> ModuleType:
    """Import the probe script by path, as it is not a package module."""
    spec = importlib.util.spec_from_file_location("probe_support_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROBE = _load_script()


def _tree(base: Path, *, paid: bool, labs: bool) -> Path:
    """Return a site directory holding the package as one wheel set ships it."""
    site = base / "site"
    shutil.copytree(
        ROOT / "src" / "director_ai",
        site / "director_ai",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ROOT / "packages" / "paid_tier_manifest.json").read_text(encoding="utf-8")
    )
    packages = project["tool"]["setuptools"]["packages"]["find"]["exclude"]
    dropped = list(manifest["labs_packages"]) if not labs else []
    if not paid:
        dropped = list(packages)
        for module in manifest["paid_modules"]:
            (site / module).unlink(missing_ok=True)
    for pattern in dropped:
        path = site / pattern.rstrip("*").replace(".", "/")
        if path.is_dir():
            shutil.rmtree(path)
    _record(site)
    return site


def _record(site: Path) -> None:
    """Record every package file under ``director-ai``, as the Core wheel's RECORD does."""
    info = site / "director_ai-0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: director-ai\nVersion: 0\n", encoding="utf-8"
    )
    lines = []
    for path in sorted((site / "director_ai").rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).digest()
            encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
            relative = path.relative_to(site).as_posix()
            lines.append(f"{relative},sha256={encoded},{path.stat().st_size}")
    lines.append("director_ai-0.dist-info/RECORD,,")
    (info / "RECORD").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Run the script on the three trees and return their results.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session temporary directories.

    Returns
    -------
    dict[str, object]
        The receipt directory, the packet directory and each run's exit code.
    """
    base = tmp_path_factory.mktemp("support-probes")
    receipts, declared = base / "receipts", base / "declared"
    packets, wheels = base / "packets", base / "wheels"
    for directory in (receipts, declared, packets, wheels):
        directory.mkdir()
    (wheels / "director_ai-0-py3-none-any.whl").write_bytes(b"stand-in wheel")
    codes: dict[str, int] = {}
    sites: dict[str, Path] = {}
    for installation, paid, labs, host in (
        ("core+pro", True, False, "hosted-ci"),
        ("core", False, False, "hosted-ci"),
        ("core+full", True, True, "hosted-ci"),
        ("core", False, False, "declared"),
    ):
        if installation not in sites:
            sites[installation] = _tree(base / installation, paid=paid, labs=labs)
        site = sites[installation]
        out = (declared if host == "declared" else receipts) / f"{installation}.json"
        arguments = [
            sys.executable,
            "-P",
            str(SCRIPT),
            "--installation",
            installation,
            "--host",
            host,
            "--commit",
            COMMIT,
            "--tree",
            TREE,
            "--staged",
            "--machine",
            "test-machine",
            "--wheels",
            str(wheels),
            "--out",
            str(out),
            "--checkout",
            str(ROOT),
            "--packets",
            str(packets),
            "--model-revision",
            "stand-in",
            "--model-digest",
            "d" * 64,
        ]
        result = subprocess.run(
            arguments,
            cwd=base,
            env={**_environment(), "PYTHONPATH": str(site)},
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        codes[f"{host} {installation}"] = result.returncode
        assert result.returncode == 0, result.stderr[-2000:]
    return {"receipts": receipts, "declared": declared, "codes": codes}


def _environment() -> dict[str, str]:
    """Return the test environment without a Python path of its own."""
    import os

    return {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}


def _receipt(
    runs: dict[str, object], installation: str, where: str = "receipts"
) -> dict[str, object]:
    """Return one run's decoded receipt."""
    directory = runs[where]
    assert isinstance(directory, Path)
    value = json.loads((directory / f"{installation}.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_every_installation_passes_its_probes(runs: dict[str, object]) -> None:
    """Pass every probe and negative probe on each installed combination."""
    assert runs["codes"] == {
        "hosted-ci core+pro": 0,
        "hosted-ci core": 0,
        "hosted-ci core+full": 0,
        "declared core": 0,
    }
    core = _receipt(runs, "core")
    negatives, probes = core["negatives"], core["probes"]
    assert isinstance(negatives, dict) and isinstance(probes, dict)
    assert set(negatives) == {
        "evidence-emission-refused",
        "overlay-authority-refused",
        "paid-import-boundary",
        "tampered-packet-refused",
    }
    assert "rest-server-transport" not in probes
    assert probes["evidence-verification"]["passed"] is True
    assert probes["pii-redactor"]["detail"] == "8 PII classes redacted"
    assert probes["pii-redactor"]["conditions"] == {
        "strict-trail-refusal": "not_applicable"
    }
    for paid in ("core+pro", "core+full"):
        paid_probes = _receipt(runs, paid)["probes"]
        assert isinstance(paid_probes, dict)
        assert paid_probes["evidence-emission"]["passed"] is True
        redactor = paid_probes["pii-redactor"]
        assert "refused a missing HMAC secret and accepted" in redactor["detail"]
        assert redactor["conditions"] == {"strict-trail-refusal": "passed"}
    assert _receipt(runs, "core+full")["model"] is None
    assert (core["tree"], core["staged"], core["machine"]) == (
        TREE,
        True,
        "test-machine",
    )
    declared = _receipt(runs, "core", "declared")
    assert declared["model"] == {"revision": "stand-in", "digest": "d" * 64}
    digest = hashlib.sha256(b"stand-in wheel").hexdigest()
    assert declared["wheels"] == {"director_ai-0-py3-none-any.whl": digest}


def test_the_receipts_support_every_claimed_installation(
    runs: dict[str, object],
) -> None:
    """Bind the shipped contract's availability to these receipts."""
    directory = runs["receipts"]
    assert isinstance(directory, Path)
    assert check_receipts(CONTRACT, [directory], tree=TREE) == ()


def test_a_failed_requirement_raises_the_probe_error() -> None:
    """Raise the probe's error with the message when a condition is false."""
    PROBE._require(True, "unused")
    with pytest.raises(PROBE.ProbeError, match="broken"):
        PROBE._require(False, "broken")


def test_a_failing_probe_is_recorded_not_raised() -> None:
    """Record a probe's exception as a failed result."""

    def broken() -> str:
        raise ValueError("bad state")

    assert PROBE._run(broken) == {"passed": False, "detail": "ValueError: bad state"}
    assert PROBE._run(lambda: "fine") == {"passed": True, "detail": "fine"}


@pytest.mark.parametrize(
    ("code", "status"),
    [(None, 0), (3, 3), ("refused here", 1)],
)
def test_the_cli_exit_status_is_normalised(
    monkeypatch: pytest.MonkeyPatch, code: object, status: int
) -> None:
    """Map a SystemExit of None, an int or a message to an exit status."""

    def exit_with(arguments: list[str]) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(cli_module, "main", exit_with)
    result, output = PROBE._cli(["anything"])
    assert result == status
    if isinstance(code, str):
        assert code in output


def test_a_clean_cli_run_exits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report status 0 and the output of a CLI call that returns."""
    monkeypatch.setattr(cli_module, "main", lambda arguments: print("done"))
    assert PROBE._cli([]) == (0, "done\n")


def test_a_resolving_paid_facade_is_refused() -> None:
    """Return the ImportError of an absent facade; refuse one that resolves."""

    class Absent:
        """A module whose facade raises the tier diagnostic."""

        def __getattr__(self, name: str) -> object:
            """Raise as an absent paid facade does."""
            raise ImportError(f"{name} requires the advanced tier")

    assert PROBE.paid_facade_refusal(Absent(), "LLMGenerator") == (
        "LLMGenerator requires the advanced tier"
    )
    with pytest.raises(PROBE.ProbeError, match="resolved in Core"):
        PROBE.paid_facade_refusal(SimpleNamespace(LLMGenerator=1), "LLMGenerator")


def test_a_guard_built_without_a_model_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse when a guard that requires model-backed NLI is built anyway."""
    monkeypatch.setattr(director_ai, "ProductionGuard", lambda config: object())
    with pytest.raises(PROBE.ProbeError, match="built without a model"):
        PROBE.probe_guard_fail_closed()


def test_a_server_started_without_a_key_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse when production mode starts without an API key."""
    real: Callable[..., object] = PROBE._posture_config
    monkeypatch.setattr(
        PROBE,
        "_posture_config",
        lambda **values: real(
            **{k: v for k, v in values.items() if k != "production_mode"}
        ),
    )
    with pytest.raises(PROBE.ProbeError, match="started without an API key"):
        PROBE.probe_rest_server_transport()


def test_an_accepted_overlay_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse when the loader accepts an overlay that ships the contract."""
    monkeypatch.setattr(reader, "verify_single_authority", lambda *args: True)
    monkeypatch.setattr(reader, "load_contract", lambda distributions=None: CONTRACT)
    with pytest.raises(PROBE.ProbeError, match="overlay that ships the contract"):
        PROBE.negative_overlay_authority(None)


def test_an_unrecorded_contract_fails_the_authority_probe() -> None:
    """Fail when no distribution records the contract, as in this checkout."""
    with pytest.raises(
        PROBE.ProbeError,
        match="director-ai does not record director_ai.support_contract with",
    ):
        PROBE.negative_overlay_authority(None)


def test_emission_without_a_packet_directory_uses_a_temporary_one() -> None:
    """Emit and verify a packet in a temporary directory when none is given."""
    assert PROBE.probe_evidence_emission(None) == "packet emitted and verified"


def test_verification_needs_the_emitted_packet(tmp_path: Path) -> None:
    """Refuse to verify without a packet directory or without its packet."""
    with pytest.raises(PROBE.ProbeError, match="no packet directory"):
        PROBE.probe_evidence_verification(None)
    with pytest.raises(PROBE.ProbeError, match="no emitted packet"):
        PROBE.probe_evidence_verification(tmp_path)


def test_a_strict_trail_built_without_a_secret_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse when the strict compliance trail builds without its HMAC secret."""
    import director_ai.compliance.audit_log as audit_module

    monkeypatch.setenv("DIRECTOR_AUDIT_HMAC_SECRET", "kept-by-the-probe")
    monkeypatch.setattr(audit_module, "AuditLog", lambda *args, **kwargs: object())
    with pytest.raises(PROBE.ProbeError, match="without a durable HMAC secret"):
        PROBE.probe_pii_redactor()
    import os

    assert os.environ["DIRECTOR_AUDIT_HMAC_SECRET"] == "kept-by-the-probe"


def _refuse_always(message: str) -> Callable[..., object]:
    """Return a stand-in trail constructor that always refuses with *message*."""

    def construct(*args: object, **kwargs: object) -> object:
        """Refuse, whatever the secret."""
        raise ValueError(message)

    return construct


def test_the_strict_trail_must_refuse_for_the_secret_and_accept_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail on another strict refusal, and on a refusal that a set secret keeps."""
    import os

    import director_ai.compliance.audit_log as audit_module

    monkeypatch.delenv("DIRECTOR_AUDIT_HMAC_SECRET", raising=False)
    outcome = PROBE.probe_pii_redactor()
    assert outcome.conditions == {"strict-trail-refusal": "passed"}
    assert "DIRECTOR_AUDIT_HMAC_SECRET" not in os.environ
    monkeypatch.setattr(
        audit_module, "AuditLog", _refuse_always("strict_mode requires a redactor")
    )
    with pytest.raises(PROBE.ProbeError, match="refused for another reason"):
        PROBE.probe_pii_redactor()
    monkeypatch.setattr(
        audit_module, "AuditLog", _refuse_always(f"{PROBE.HMAC_REFUSAL} (stand-in)")
    )
    with pytest.raises(PROBE.ProbeError, match="refused a durable HMAC secret"):
        PROBE.probe_pii_redactor()
    assert "DIRECTOR_AUDIT_HMAC_SECRET" not in os.environ


def test_the_installation_decides_where_the_trail_must_be(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a paid install without the trail and Core with it, never skip."""
    assert PROBE.installed_combination() != "core"
    monkeypatch.setattr(PROBE, "_trail_installed", lambda: False)
    with pytest.raises(PROBE.ProbeError, match="trail is missing from core"):
        PROBE.probe_pii_redactor()
    monkeypatch.setattr(PROBE, "installed_combination", lambda: "core")
    assert PROBE.probe_pii_redactor() == PROBE.Outcome(
        "8 PII classes redacted", {"strict-trail-refusal": "not_applicable"}
    )
    monkeypatch.setattr(PROBE, "_trail_installed", lambda: True)
    with pytest.raises(PROBE.ProbeError, match="trail is installed in Core"):
        PROBE.probe_pii_redactor()


def test_a_probe_outcome_records_its_conditions() -> None:
    """Record a passed outcome's detail and conditions, and a plain detail alone."""
    outcome = PROBE.Outcome("done", {"named": "passed"})
    assert PROBE._run(lambda: outcome) == {
        "passed": True,
        "detail": "done",
        "conditions": {"named": "passed"},
    }
    assert PROBE._run(lambda: "plain") == {"passed": True, "detail": "plain"}


def test_every_identifier_the_redactor_row_names_is_probed() -> None:
    """Sample each PHI identifier and class that the redactor row's note names."""
    note = CONTRACT.operation("pii-redactor").note
    assert f"{len(PROBE.PII_SAMPLES)} classes" in note
    for identifier in ("MRN", "DOB", "NHS"):
        assert identifier in note
        assert any(sample.startswith(identifier) for sample in PROBE.PII_SAMPLES["phi"])


def test_a_root_facade_without_the_diagnostic_fails_the_boundary_probe(
    tmp_path: Path,
) -> None:
    """Fail the paid-boundary probe when the root facade lets the bare error out."""
    site = _tree(tmp_path / "core", paid=False, labs=False)
    root = site / "director_ai" / "__init__.py"
    text = root.read_text(encoding="utf-8")
    narrowed = "            if not absent_from_tier(error.name, target):\n                raise\n"
    assert text.count(narrowed) == 1
    root.write_text(text.replace(narrowed, "            raise\n"), encoding="utf-8")
    code = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('probe', sys.argv[1])\n"
        "probe = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(probe)\n"
        "print(probe._run(lambda: probe.negative_paid_boundary(None)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-P", "-c", code, str(SCRIPT)],
        cwd=tmp_path,
        env={**_environment(), "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "the root facade error is not the tier diagnostic" in result.stdout
    assert "'passed': False" in result.stdout


def test_the_tamper_probe_needs_the_pro_packet(tmp_path: Path) -> None:
    """Refuse without a packet directory or without the emitted packet."""
    with pytest.raises(PROBE.ProbeError, match="no packet directory"):
        PROBE.negative_tampered_packet(None)
    with pytest.raises(PROBE.ProbeError, match="left no packet"):
        PROBE.negative_tampered_packet(tmp_path)


def test_the_script_refuses_the_checkout_and_a_wrong_installation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Return 2 when imported from the checkout or for another combination.

    The checkout case names the combination this environment really holds, so
    only the checkout guard can refuse it.
    """
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME"):
        monkeypatch.setenv(name, "unset-by-test")
    common = ["--host", "hosted-ci", "--commit", COMMIT, "--tree", TREE]
    common += ["--wheels", str(tmp_path)]
    out = ["--out", str(tmp_path / "r.json")]
    checkout = Path(director_ai.__file__ or "").resolve().parents[1]
    combination = PROBE.installed_combination()
    assert (
        PROBE.main(
            ["--installation", combination, *common, *out, "--checkout", str(checkout)]
        )
        == 2
    )
    assert "imported from the checkout" in capsys.readouterr().err
    other = tmp_path / "elsewhere"
    other.mkdir()
    wrong = "core" if combination != "core" else "core+full"
    assert (
        PROBE.main(["--installation", wrong, *common, *out, "--checkout", str(other)])
        == 2
    )
    assert f"The environment holds {combination}" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_the_stand_in_overlay_records_only_the_catalog() -> None:
    """Expose only the stand-in's METADATA and RECORD."""
    overlay = PROBE._OverlayDistribution()
    assert overlay.read_text("METADATA") is not None
    assert overlay.read_text("RECORD") == "director_ai/support_contract_catalog.py,,\n"
    assert overlay.read_text("entry_points.txt") is None


def test_a_failing_negative_probe_fails_the_run(tmp_path: Path) -> None:
    """Exit 1 and name the failure when Core has no Pro packet to verify."""
    site = _tree(tmp_path / "core", paid=False, labs=False)
    empty, wheels = tmp_path / "packets", tmp_path / "wheels"
    empty.mkdir()
    wheels.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-P",
            str(SCRIPT),
            "--installation",
            "core",
            "--host",
            "hosted-ci",
            "--commit",
            COMMIT,
            "--tree",
            TREE,
            "--wheels",
            str(wheels),
            "--out",
            str(tmp_path / "core.json"),
            "--checkout",
            str(ROOT),
            "--packets",
            str(empty),
        ],
        cwd=tmp_path,
        env={**_environment(), "PYTHONPATH": str(site)},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 1
    assert "FAILED: tampered-packet-refused" in result.stderr
    receipt = json.loads((tmp_path / "core.json").read_text(encoding="utf-8"))
    assert receipt["negatives"]["tampered-packet-refused"]["passed"] is False
    assert (receipt["staged"], receipt["machine"]) == (False, socket.gethostname())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rate_limit_rpm": 0}, "refused nothing"),
        ({"metrics_require_auth": False}, "served without authentication"),
        ({"api_keys": []}, "an unauthenticated request passed"),
    ],
)
def test_a_weakened_server_posture_fails_the_probe(
    monkeypatch: pytest.MonkeyPatch, change: dict[str, object], message: str
) -> None:
    """Fail the transport probe when the server loses part of its posture."""
    real: Callable[..., object] = PROBE._posture_config

    def weakened(**values: object) -> object:
        """Weaken every posture except the fail-closed start-up check."""
        return (
            real(**values)
            if "production_mode" in values
            else real(**{**values, **change})
        )

    monkeypatch.setattr(PROBE, "_posture_config", weakened)
    with pytest.raises(PROBE.ProbeError, match=message):
        PROBE.probe_rest_server_transport()


def test_a_proxy_without_its_key_fails_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail the proxy probe when the proxy stops checking the API key."""
    import director_ai.proxy as proxy_module

    real = proxy_module.create_proxy_app
    monkeypatch.setattr(
        proxy_module,
        "create_proxy_app",
        lambda **values: real(**{**values, "api_keys": None}),
    )
    with pytest.raises(PROBE.ProbeError, match="an unauthenticated request passed"):
        PROBE.probe_proxy_transport()


def test_an_unredacted_class_fails_the_redactor_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail the redactor probe for any one class left in, or the profile turned off."""
    from director_ai.core.config_profiles import PROFILE_DEFINITIONS
    from director_ai.core.redactor import PIIRedactor

    real = PIIRedactor.redact
    monkeypatch.setattr(
        PIIRedactor,
        "redact",
        lambda self, text: text if "192.168.1.20" in text else real(self, text),
    )
    with pytest.raises(PROBE.ProbeError, match="the ipv4 sample was not redacted"):
        PROBE.probe_pii_redactor()
    monkeypatch.setattr(
        PIIRedactor,
        "redact",
        lambda self, text: text if "DOB 1980-01-31" in text else real(self, text),
    )
    with pytest.raises(
        PROBE.ProbeError, match="the phi sample was not redacted: DOB 1980-01-31"
    ):
        PROBE.probe_pii_redactor()
    monkeypatch.setitem(PROFILE_DEFINITIONS["production"], "redact_pii", False)
    with pytest.raises(PROBE.ProbeError, match="does not turn the redactor on"):
        PROBE.probe_pii_redactor()


def test_a_refused_emission_that_leaves_output_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail when Core refuses emission but leaves its output path behind."""

    def refuse_but_write(arguments: list[str]) -> tuple[int, str]:
        """Create the target, then refuse as the advanced tier does."""
        Path(arguments[-1]).mkdir(parents=True)
        return 1, "this subcommand requires the advanced tier"

    monkeypatch.setattr(PROBE, "_cli", refuse_but_write)
    with pytest.raises(PROBE.ProbeError, match="left an output path"):
        PROBE.negative_evidence_emission(None)


def test_an_accepted_tampered_or_damaged_packet_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail when a tampered packet, a changed byte or a missing member is accepted."""
    assert PROBE.probe_evidence_emission(tmp_path) == "packet emitted and verified"
    monkeypatch.setattr(
        PROBE, "_cli", lambda arguments: (0, "Evidence packet VERIFIED")
    )
    with pytest.raises(PROBE.ProbeError, match="accepted a tampered packet"):
        PROBE.negative_tampered_packet(tmp_path)
    original = (tmp_path / "evidence_packet.json").read_text(encoding="utf-8")

    def accepts_changed_bytes(text: str) -> tuple[int, str]:
        """Accept every packet, as a broken verifier would."""
        return 0, "Evidence packet VERIFIED"

    monkeypatch.setattr(PROBE, "_verifies", accepts_changed_bytes)
    with pytest.raises(PROBE.ProbeError, match="a one-byte change was accepted"):
        PROBE.probe_evidence_verification(tmp_path)

    def accepts_missing_members(text: str) -> tuple[int, str]:
        """Refuse a changed byte but accept a packet without its integrity member."""
        if '"integrity"' in text and text != original:
            return 1, "Evidence packet INVALID (digest_mismatch)"
        return 0, "Evidence packet VERIFIED"

    monkeypatch.setattr(PROBE, "_verifies", accepts_missing_members)
    with pytest.raises(PROBE.ProbeError, match="missing a member was accepted"):
        PROBE.probe_evidence_verification(tmp_path)
