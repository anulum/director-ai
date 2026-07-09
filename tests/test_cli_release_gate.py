# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Release-gate + Model-activation CLI Tests

"""Real-surface tests for the release-gate and model-activation CLI (WCC-3).

The assemble path runs against the full manifest set produced by the real
fixture generator; activation flips the persisted designation in a real
SQLite-backed job store and is proven durable by reloading the store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai import cli
from director_ai.finetune_jobs import _JobStore
from tools.generate_customer_model_factory_fixture import main as fixture_main

_EVIDENCE_FLAGS = {
    "runtime-package": "runtime_package.json",
    "evidence-pack": "evidence_pack.json",
    "monitoring-manifest": "monitoring_manifest.json",
    "risk-register": "risk_register.json",
    "observability-operations-evidence": "observability_operations_evidence.json",
    "provenance-lineage-evidence": "provenance_lineage_evidence.json",
    "conformal-routing-evidence": "conformal_routing_evidence.json",
    "trajectory-rollback-evidence": "trajectory_rollback_evidence.json",
    "multimodal-temporal-evidence": "multimodal_temporal_evidence.json",
    "federated-privacy-evidence": "federated_privacy_evidence.json",
    "edge-mobile-evidence": "edge_mobile_evidence.json",
    "auto-redteam-defence-evidence": "auto_redteam_defence_evidence.json",
    "formal-symbolic-evidence": "formal_symbolic_evidence.json",
    "deployment-hardening-evidence": "deployment_hardening_evidence.json",
}


def _assemble_args(fixtures: Path, output: Path, *, ready: bool = True) -> list[str]:
    enterprise = fixtures / "enterprise_readiness.json"
    enterprise.write_text(
        json.dumps({"ready": ready, "blocking_debt_ids": [] if ready else ["DEBT-1"]}),
        encoding="utf-8",
    )
    args = [
        "release-gate",
        "assemble",
        "--release-id",
        "cmf-customer-alpha-test",
        "--generated-at",
        "2026-05-18T18:45:00Z",
        "--enterprise-readiness",
        str(enterprise),
        "--output",
        str(output),
    ]
    for flag, filename in _EVIDENCE_FLAGS.items():
        args.extend([f"--{flag}", str(fixtures / filename)])
    return args


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    """Generate the full Customer Model Factory manifest set once."""
    out = tmp_path_factory.mktemp("cmf-fixtures")
    assert fixture_main(["--output-dir", str(out)]) == 0
    return out


class TestReleaseGateAssemble:
    def test_assemble_writes_manifest_via_the_cli(self, fixture_dir, tmp_path):
        output = tmp_path / "release_gate.json"

        with pytest.raises(SystemExit) as exc_info:
            cli.main(_assemble_args(fixture_dir, output))

        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["release_id"] == "cmf-customer-alpha-test"
        # Exit code mirrors promotion_allowed: 0 allowed / 1 blocked.
        assert exc_info.value.code == (0 if manifest["promotion_allowed"] else 1)

    def test_enterprise_not_ready_blocks_promotion(self, fixture_dir, tmp_path):
        output = tmp_path / "release_gate.json"

        with pytest.raises(SystemExit) as exc_info:
            cli.main(_assemble_args(fixture_dir, output, ready=False))

        assert exc_info.value.code == 1
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert manifest["promotion_allowed"] is False

    def test_help_prints_usage_without_error(self, capsys):
        cli.main(["release-gate", "--help"])
        assert "release-gate assemble" in capsys.readouterr().out

    def test_missing_subcommand_exits_one(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["release-gate"])
        assert exc_info.value.code == 1

    def test_unknown_subcommand_exits_one(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["release-gate", "publish"])
        assert exc_info.value.code == 1
        assert "Unknown release-gate subcommand" in capsys.readouterr().out


def _seed_job(models_dir: Path, *, state: str = "completed") -> str:
    store = _JobStore(models_dir / "finetune_jobs.sqlite3")
    job = store.create({"epochs": 1})
    job.state = state
    job.model_path = str(models_dir / job.job_id)
    store.save(job)
    return job.job_id


class TestModelActivation:
    def test_activate_persists_across_store_reloads(self, tmp_path, capsys):
        job_id = _seed_job(tmp_path)

        cli.main(["model-activate", job_id, "--models-dir", str(tmp_path)])

        assert "marked active" in capsys.readouterr().out
        reloaded = _JobStore(tmp_path / "finetune_jobs.sqlite3")
        job = reloaded.get(job_id)
        assert job is not None
        assert job.activated is True

    def test_rollback_clears_the_persisted_designation(self, tmp_path, capsys):
        job_id = _seed_job(tmp_path)
        cli.main(["model-activate", job_id, "--models-dir", str(tmp_path)])

        cli.main(["model-rollback", job_id, "--models-dir", str(tmp_path)])

        assert "rolled back" in capsys.readouterr().out
        reloaded = _JobStore(tmp_path / "finetune_jobs.sqlite3")
        job = reloaded.get(job_id)
        assert job is not None
        assert job.activated is False

    def test_activate_unknown_job_exits_one(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["model-activate", "no-such-job", "--models-dir", str(tmp_path)])
        assert exc_info.value.code == 1
        assert "not found" in capsys.readouterr().out

    def test_activate_incomplete_job_exits_one(self, tmp_path, capsys):
        job_id = _seed_job(tmp_path, state="training")

        with pytest.raises(SystemExit) as exc_info:
            cli.main(["model-activate", job_id, "--models-dir", str(tmp_path)])

        assert exc_info.value.code == 1
        assert "not completed" in capsys.readouterr().out

    def test_rollback_unknown_job_exits_one(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["model-rollback", "no-such-job", "--models-dir", str(tmp_path)])
        assert exc_info.value.code == 1


def test_tools_wrapper_still_delegates(fixture_dir, tmp_path):
    """The historical tools/ entry point keeps working after the move."""
    from tools.assemble_customer_model_factory_release import main as wrapper_main

    output = tmp_path / "release_gate.json"
    args = _assemble_args(fixture_dir, output)
    assert wrapper_main(args[2:]) in (0, 1)
    assert output.exists()
