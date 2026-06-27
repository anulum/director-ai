# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — provenance evidence packet tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pytest import MonkeyPatch

from benchmarks import provenance_evidence
from benchmarks.provenance_evidence import (
    main,
    run_kb_lineage_probe,
    run_provenance_chain_probe,
    run_provenance_evidence,
)


def test_git_commit_falls_back_when_git_is_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify provenance evidence handles missing and failing git clients."""

    module = cast(Any, provenance_evidence)
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    assert module._git_commit() == "unknown"

    monkeypatch.setattr(module.shutil, "which", lambda _name: "git")

    def raise_subprocess(*_args: object, **_kwargs: object) -> None:
        raise module.subprocess.SubprocessError()

    monkeypatch.setattr(module.subprocess, "run", raise_subprocess)
    assert module._git_commit() == "unknown"

    class Completed:
        stdout = "abc123\n"

    def complete_subprocess(*_args: object, **_kwargs: object) -> Completed:
        return Completed()

    monkeypatch.setattr(module.subprocess, "run", complete_subprocess)
    assert module._git_commit() == "abc123"


def test_kb_lineage_probe_reports_root_evolution_and_signed_conflict() -> None:
    """Verify the KB lineage probe records tenant-safe root evolution."""

    packet = run_kb_lineage_probe()
    serialised = json.dumps(packet, sort_keys=True)

    assert packet["passed"] is True
    assert packet["roots_changed"] is True
    assert packet["tenant_scoped"] is True
    assert packet["conflict"] == {
        "conflict_type": "signed_fact",
        "claim_id": "dose-claim",
        "signed_fact_id": "signed-dose-v1",
        "reason": "new fact differs from protected claim state",
    }
    assert len(packet["audit_record"]["merkle_root"]) == 64
    assert packet["raw_fact_payload_included"] is False
    assert "Refunds are available" not in serialised
    assert "Dose is" not in serialised


def test_provenance_chain_probe_verifies_proofs_and_detects_tamper() -> None:
    """Verify the provenance chain probe accepts proofs and rejects tampering."""

    packet = run_provenance_chain_probe(fact_count=3)

    assert packet["passed"] is True
    assert packet["fact_count"] == 3
    assert packet["proofs_verified"] is True
    assert packet["healthy_all_ok"] is True
    assert packet["tamper_detected"] is True
    assert packet["tamper_failure_reason_present"] is True
    assert packet["chain_ok"] is True
    assert packet["first_bad_index"] is None
    assert packet["chain_entries"] == 2
    assert packet["raw_fact_payload_included"] is False


def test_provenance_chain_probe_rejects_empty_fact_sets() -> None:
    """Verify the provenance chain probe validates the fact count."""

    try:
        run_provenance_chain_probe(fact_count=0)
    except ValueError as exc:
        assert "fact_count" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_provenance_evidence_payload_has_acceptance_summary() -> None:
    """Verify the aggregate R9 packet exposes acceptance and limit metadata."""

    packet = run_provenance_evidence(fact_count=2)

    assert packet["schema_version"] == "director-ai.provenance-evidence.v1"
    assert packet["acceptance"] == {
        "passed": True,
        "checks": {
            "kb_lineage": True,
            "provenance_chain": True,
        },
        "limits": {
            "local_only": True,
            "external_operator_signoff_included": False,
        },
    }
    assert set(packet["probes"]) == {"kb_lineage", "provenance_chain"}


def test_main_writes_requested_output_path(tmp_path: Path) -> None:
    """Verify the CLI writes the requested provenance evidence artifact."""

    output = tmp_path / "provenance.json"

    exit_code = main(["--fact-count", "2", "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True


def test_main_uses_default_results_path(monkeypatch: MonkeyPatch) -> None:
    """Verify the R9 CLI saves to the default benchmark results path."""

    saved: list[str] = []

    def save_results(payload: object, filename: str) -> None:
        saved.append(filename)

    monkeypatch.setattr(provenance_evidence, "save_results", save_results)
    monkeypatch.setattr(provenance_evidence, "_git_commit", lambda: "abc123")

    assert main(["--fact-count", "2"]) == 0
    assert len(saved) == 1
    assert saved[0].startswith("provenance_evidence_")
