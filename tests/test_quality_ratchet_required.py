# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — required all-profile authority, recording and check tests
"""Verify the all-profile gate in process, with real Git and substituted captures.

The nine native collectors are replaced by fixed snapshots so that the gate's
own authority, scope, recording, comparison and transition rules are tested
without running every native tool.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

import tools.check_quality_ratchet as cli
import tools.quality_ratchet_gate as gate
from tests.quality_ratchet_succession_support import (
    ENGINE,
    NEW_IDENTITY,
    OLD_IDENTITY,
    RULES,
    entry,
    file_bytes,
)
from tests.test_quality_ratchet_succession_load import commit
from tools.quality_ratchet_baseline import Baseline, parse_baseline, resolve_commit
from tools.quality_ratchet_canaries import CANARY_RUNNERS, CanaryRun
from tools.quality_ratchet_contracts import Finding, Snapshot
from tools.quality_ratchet_profiles import REQUIRED_PROFILES
from tools.quality_ratchet_succession import TRANSITIONS_PATH
from tools.quality_ratchet_succession_parse import parse_transition

SOURCE = "src.py"
SOURCE_BYTES = b"x = 1\n"
DIGEST = hashlib.sha256(SOURCE_BYTES).hexdigest()


def snapshots(
    ts_identity: str = OLD_IDENTITY, *findings: Finding
) -> dict[str, Snapshot]:
    """Return one complete snapshot per required profile over ``src.py``.

    Parameters
    ----------
    ts_identity : str
        The typescript-docs identity; every other profile is fixed.
    *findings : Finding
        Findings for the python-docs profile.

    Returns
    -------
    dict[str, Snapshot]
        Snapshots keyed by profile.
    """
    result = {}
    for index, name in enumerate(REQUIRED_PROFILES):
        identity = ts_identity if name == "typescript-docs" else f"{index + 1:x}" * 64
        result[name] = Snapshot(
            name,
            identity,
            {SOURCE: DIGEST},
            findings if name == "python-docs" else (),
            True,
        )
    return result


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a repository at its policy origin, with a small configuration set.

    Parameters
    ----------
    tmp_path : Path
        Pytest's temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to narrow the configuration inventory.

    Returns
    -------
    Path
        The repository root, with an untracked policy naming its first commit.
    """
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / SOURCE).write_bytes(SOURCE_BYTES)
    (root / "cfg.toml").write_text("a = 1\n", encoding="utf-8")
    monkeypatch.setattr(gate, "PROFILE_CONFIGURATIONS", {"python-docs": ("cfg.toml",)})
    origin = commit(root)
    policy = root / gate.POLICY_PATH
    policy.parent.mkdir(parents=True)
    profiles = ", ".join(f'"{name}"' for name in REQUIRED_PROFILES)
    policy.write_text(
        f'schema = "{gate.POLICY_SCHEMA}"\norigin_commit = "{origin}"\n'
        f"profiles = [{profiles}]\n",
        encoding="utf-8",
    )
    return root


def use_captures(
    monkeypatch: pytest.MonkeyPatch,
    produce: Callable[[], dict[str, Snapshot]],
) -> None:
    """Replace the nine native collectors by a fixed producer.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used for the replacement.
    produce : Callable[[], dict[str, Snapshot]]
        Returns the snapshots of one capture.
    """
    monkeypatch.setattr(cli, "capture_required_profiles", lambda root, args: produce())


def run(root: Path, command: str, output: Path | None = None) -> int:
    """Run the all-profile CLI in process.

    Parameters
    ----------
    root : Path
        Repository root.
    command : str
        ``capture``, ``record`` or ``check``.
    output : Path | None
        Output path for capture and record.

    Returns
    -------
    int
        The CLI's exit status.
    """
    arguments = [command, "--all", "--root", str(root)]
    if output is not None:
        arguments += ["--output", str(output)]
    return cli.main(arguments)


def adopt(root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Record the initial baseline at the origin and commit it with the policy.

    Parameters
    ----------
    root : Path
        Repository root at its origin.
    monkeypatch : pytest.MonkeyPatch
        Fixture for the substituted captures.
    tmp_path : Path
        Directory for the recorded output.

    Returns
    -------
    str
        The adoption commit.
    """
    use_captures(monkeypatch, snapshots)
    output = tmp_path / "recorded.json"
    assert run(root, "record", output) == 0
    (root / "tools" / "quality_ratchet_baseline.json").write_bytes(output.read_bytes())
    return commit(root)


def test_capture_record_and_check_pass_on_unchanged_findings(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Capture, adopt at the origin, then check and re-record without refusals."""
    use_captures(monkeypatch, snapshots)
    captured = tmp_path / "captured.json"
    assert run(repository, "capture", captured) == 0
    assert json.loads(captured.read_text())["schema"] == "quality-ratchet.capture.v1"
    adopt(repository, monkeypatch, tmp_path)
    assert run(repository, "check") == 0
    assert run(repository, "record", tmp_path / "again.json") == 0


def test_check_refuses_a_gained_finding(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse a finding on an unchanged source, and write no record."""
    adopt(repository, monkeypatch, tmp_path)
    gained = Finding(SOURCE, "D103", 1, 1, "Missing documentation")
    use_captures(monkeypatch, lambda: snapshots(OLD_IDENTITY, gained))
    assert run(repository, "check") == 1
    output = tmp_path / "refused.json"
    assert run(repository, "record", output) == 1
    assert not output.exists()


def test_declared_transition_passes_the_gate_with_its_canary(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Apply a committed transition and its canary; refuse without the canary."""
    adopt(repository, monkeypatch, tmp_path)
    (repository / TRANSITIONS_PATH).write_bytes(file_bytes(entry()))
    use_captures(monkeypatch, lambda: snapshots(NEW_IDENTITY))
    output = tmp_path / "transition.json"
    assert run(repository, "record", output) == 1
    keys = parse_transition(entry()).canary.new_findings
    run_ = CanaryRun(RULES, "4" * 64, "linux-x64", ENGINE, keys)
    monkeypatch.setitem(CANARY_RUNNERS, "typescript-docs", lambda root, args: run_)
    assert run(repository, "record", output) == 0
    recorded = parse_baseline(output.read_bytes())
    assert recorded.profiles["typescript-docs"].identity == NEW_IDENTITY


def test_changes_during_analysis_are_refused(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse a policy or candidate baseline that changes while collecting."""
    adopt(repository, monkeypatch, tmp_path)
    baseline = repository / "tools" / "quality_ratchet_baseline.json"

    def touch_baseline() -> dict[str, Snapshot]:
        """Rewrite the candidate baseline's content during the capture."""
        data = json.loads(baseline.read_bytes())
        data["profiles"]["go-docs"]["identity"] = "f" * 64
        baseline.write_text(json.dumps(data), encoding="utf-8")
        return snapshots()

    use_captures(monkeypatch, touch_baseline)
    assert run(repository, "check") == 2
    policy = repository / gate.POLICY_PATH

    def touch_policy() -> dict[str, Snapshot]:
        """Rewrite the policy during the capture."""
        policy.write_bytes(policy.read_bytes() + b"# changed\n")
        return snapshots()

    use_captures(monkeypatch, touch_policy)
    assert run(repository, "capture", tmp_path / "c.json") == 2


def test_authority_refuses_drift_and_unqualified_first_records(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse each way the policy, base or checkout can lack authority."""
    use_captures(monkeypatch, snapshots)
    origin = resolve_commit(repository, "HEAD")
    with pytest.raises(ValueError, match="no trusted Git authority"):
        gate.required_authority(repository, origin, recording=False)
    (repository / "extra.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean historical checkout"):
        gate.required_authority(repository, origin, recording=True)
    (repository / "extra.txt").unlink()
    later = commit(repository)
    with pytest.raises(ValueError, match="no trusted Git authority"):
        gate.required_authority(repository, later, recording=True)
    with pytest.raises(ValueError, match="no trusted Git authority"):
        gate.required_authority(repository, origin, recording=True)


def test_policy_must_match_its_committed_copy(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse a working policy that differs from the trusted committed one."""
    adopt(repository, monkeypatch, tmp_path)
    policy = repository / gate.POLICY_PATH
    policy.write_bytes(policy.read_bytes() + b"# drift\n")
    base = resolve_commit(repository, "HEAD")
    with pytest.raises(ValueError, match="differs from the trusted base"):
        gate.required_authority(repository, base, recording=False)


def test_initial_policy_committed_at_the_origin_must_not_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse an initial record whose committed origin policy was edited."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    policy = root / gate.POLICY_PATH
    policy.parent.mkdir(parents=True)
    policy.write_text("placeholder\n", encoding="utf-8")
    origin = commit(root)
    profiles = ", ".join(f'"{name}"' for name in REQUIRED_PROFILES)
    policy.write_text(
        f'schema = "{gate.POLICY_SCHEMA}"\norigin_commit = "{origin}"\n'
        f"profiles = [{profiles}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differs from its origin"):
        gate.required_authority(root, origin, recording=True)


def test_policy_carrier_must_be_exact() -> None:
    """Refuse a policy with another schema or profile sequence."""
    with pytest.raises(ValueError, match="incomplete required quality policy"):
        gate.policy_origin(b'schema = "other"\norigin_commit = "x"\nprofiles = []\n')


def test_required_scope_refuses_partial_carriers() -> None:
    """Refuse another origin, a missing profile, an incomplete or misnamed one."""
    origin = "0" * 40
    full = snapshots()
    with pytest.raises(ValueError, match="authority or profile set"):
        gate.required_baseline_scope(Baseline("1" * 40, full), origin)
    partial = dict(full)
    partial.pop("go-docs")
    with pytest.raises(ValueError, match="authority or profile set"):
        gate.required_baseline_scope(Baseline(origin, partial), origin)
    broken = dict(full)
    old = broken["go-docs"]
    broken["go-docs"] = Snapshot(old.profile, old.identity, old.sources, (), False)
    with pytest.raises(ValueError, match="incomplete profile"):
        gate.required_baseline_scope(Baseline(origin, broken), origin)
    broken["go-docs"] = Snapshot("rust-docs", old.identity, old.sources, (), True)
    with pytest.raises(ValueError, match="incomplete profile"):
        gate.required_baseline_scope(Baseline(origin, broken), origin)


def test_first_record_qualifies_against_the_origin_sources(
    repository: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Refuse a first record over sources that differ from the origin commit."""
    origin = resolve_commit(repository, "HEAD")
    wrong = {
        name: Snapshot(name, s.identity, {SOURCE: "0" * 64}, (), True)
        for name, s in snapshots().items()
    }
    with pytest.raises(ValueError, match="does not match original source"):
        gate.qualify_required_record(repository, origin, Baseline(origin, wrong))
    absent = {
        name: Snapshot(name, s.identity, {"absent.py": DIGEST}, (), True)
        for name, s in snapshots().items()
    }
    with pytest.raises(ValueError, match="does not match original source"):
        gate.qualify_required_record(repository, origin, Baseline(origin, absent))
    base = adopt(repository, monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="no trusted Git authority|adopted during"):
        gate.qualify_required_record(repository, base, Baseline(origin, snapshots()))


def test_capture_consistency_is_verified(repository: Path) -> None:
    """Refuse an inventory change or source bytes that differ between profiles."""
    origin = resolve_commit(repository, "HEAD")
    before = gate.required_input_state(repository)
    gate.verify_required_captures(repository, Baseline(origin, snapshots()), before)
    (repository / "new.txt").write_text("n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during analysis"):
        gate.verify_required_captures(repository, Baseline(origin, snapshots()), before)
    after = gate.required_input_state(repository)
    wrong = {
        name: Snapshot(name, s.identity, {SOURCE: "0" * 64}, (), True)
        for name, s in snapshots().items()
    }
    with pytest.raises(ValueError, match="changed between native profiles"):
        gate.verify_required_captures(repository, Baseline(origin, wrong), after)
