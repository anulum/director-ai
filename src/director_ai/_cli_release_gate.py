# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Release-gate and model-activation CLI commands

"""CLI commands for the Customer Model Factory release gate and activation.

``director-ai release-gate assemble`` builds the final release-gate manifest
from the per-stage manifest JSON files (WCC-3 — previously reachable only via
``tools/assemble_customer_model_factory_release.py``, which now delegates
here). ``director-ai model-activate`` / ``model-rollback`` flip the persisted
activation designation in the fine-tuning job store (BUG-2): the flag is
durable and protects the model from deletion; a RUNNING server applies it
after a restart — use the REST ``/v1/finetune/{job_id}/activate`` endpoint for
a live hot-swap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT_MODELS_DIR = "./director-models"
_JOB_DB_FILENAME = "finetune_jobs.sqlite3"


def _read_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def assemble_release_gate(argv: list[str] | None = None) -> int:
    """Assemble the final release-gate manifest from manifest JSON files.

    Returns ``0`` when the assembled gate allows promotion, ``1`` otherwise.
    """
    from director_ai.core.customer_model_factory.evidence_pack import (
        CustomerEvidencePackManifest,
    )
    from director_ai.core.customer_model_factory.monitoring_manifest import (
        CustomerMonitoringManifest,
    )
    from director_ai.core.customer_model_factory.release_gate import (
        AutoRedteamDefenceEvidence,
        ConformalRoutingEvidence,
        DeploymentHardeningEvidence,
        EdgeMobileEvidence,
        FederatedPrivacyEvidence,
        FormalSymbolicEvidence,
        MultimodalTemporalEvidence,
        ObservabilityOperationsEvidence,
        ProvenanceLineageEvidence,
        TrajectoryRollbackEvidence,
        build_release_gate_manifest,
    )
    from director_ai.core.customer_model_factory.risk_register import (
        CustomerRiskRegister,
    )
    from director_ai.core.customer_model_factory.runtime_package import (
        CustomerRuntimePackage,
    )

    parser = argparse.ArgumentParser(
        prog="director-ai release-gate assemble",
        description="Assemble the Customer Model Factory release-gate manifest.",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--enterprise-readiness", type=Path, required=True)
    parser.add_argument("--runtime-package", type=Path, required=True)
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--monitoring-manifest", type=Path, required=True)
    parser.add_argument("--risk-register", type=Path, required=True)
    parser.add_argument("--observability-operations-evidence", type=Path, required=True)
    parser.add_argument("--provenance-lineage-evidence", type=Path, required=True)
    parser.add_argument("--conformal-routing-evidence", type=Path, required=True)
    parser.add_argument("--trajectory-rollback-evidence", type=Path, required=True)
    parser.add_argument("--multimodal-temporal-evidence", type=Path, required=True)
    parser.add_argument("--federated-privacy-evidence", type=Path, required=True)
    parser.add_argument("--edge-mobile-evidence", type=Path, required=True)
    parser.add_argument("--auto-redteam-defence-evidence", type=Path, required=True)
    parser.add_argument("--formal-symbolic-evidence", type=Path, required=True)
    parser.add_argument("--deployment-hardening-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    enterprise = _read_json(args.enterprise_readiness)
    release_gate = build_release_gate_manifest(
        release_id=args.release_id,
        enterprise_ready=bool(enterprise.get("ready")),
        enterprise_blocking_debt_ids=tuple(enterprise.get("blocking_debt_ids", ())),
        runtime_package=CustomerRuntimePackage.from_dict(
            _read_json(args.runtime_package)
        ),
        evidence_pack=CustomerEvidencePackManifest.from_dict(
            _read_json(args.evidence_pack)
        ),
        monitoring_manifest=CustomerMonitoringManifest.from_dict(
            _read_json(args.monitoring_manifest)
        ),
        risk_register=CustomerRiskRegister.from_dict(_read_json(args.risk_register)),
        observability_operations_evidence=ObservabilityOperationsEvidence.from_dict(
            _read_json(args.observability_operations_evidence)
        ),
        provenance_lineage_evidence=ProvenanceLineageEvidence.from_dict(
            _read_json(args.provenance_lineage_evidence)
        ),
        conformal_routing_evidence=ConformalRoutingEvidence.from_dict(
            _read_json(args.conformal_routing_evidence)
        ),
        trajectory_rollback_evidence=TrajectoryRollbackEvidence.from_dict(
            _read_json(args.trajectory_rollback_evidence)
        ),
        multimodal_temporal_evidence=MultimodalTemporalEvidence.from_dict(
            _read_json(args.multimodal_temporal_evidence)
        ),
        federated_privacy_evidence=FederatedPrivacyEvidence.from_dict(
            _read_json(args.federated_privacy_evidence)
        ),
        edge_mobile_evidence=EdgeMobileEvidence.from_dict(
            _read_json(args.edge_mobile_evidence)
        ),
        auto_redteam_defence_evidence=AutoRedteamDefenceEvidence.from_dict(
            _read_json(args.auto_redteam_defence_evidence)
        ),
        formal_symbolic_evidence=FormalSymbolicEvidence.from_dict(
            _read_json(args.formal_symbolic_evidence)
        ),
        deployment_hardening_evidence=DeploymentHardeningEvidence.from_dict(
            _read_json(args.deployment_hardening_evidence)
        ),
        generated_at=args.generated_at,
    )
    release_gate.write_json(args.output)
    return 0 if release_gate.promotion_allowed else 1


def _print_release_gate_help() -> None:
    print(
        "Usage: director-ai release-gate assemble --release-id ID "
        "--generated-at TS --enterprise-readiness F ... --output F\n"
        "Assembles the Customer Model Factory release-gate manifest; "
        "exits 1 when promotion is blocked."
    )


def _cmd_release_gate(args: list[str]) -> None:
    """Release-gate commands: assemble the promotion manifest."""
    if args and args[0] in ("-h", "--help", "help"):
        _print_release_gate_help()
        return
    if not args:
        _print_release_gate_help()
        sys.exit(1)
    if args[0] != "assemble":
        print(f"Unknown release-gate subcommand: {args[0]}")
        sys.exit(1)
    sys.exit(assemble_release_gate(args[1:]))


def _activation_store(models_dir: str) -> Any:
    from director_ai.finetune_jobs import _JobStore

    return _JobStore(Path(models_dir) / _JOB_DB_FILENAME)


def _parse_activation_args(prog: str, args: list[str]) -> tuple[str, str]:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("job_id")
    parser.add_argument(
        "--models-dir",
        default=_DEFAULT_MODELS_DIR,
        help="Fine-tuning models directory holding the persistent job store.",
    )
    parsed = parser.parse_args(args)
    return parsed.job_id, parsed.models_dir


def _cmd_model_activate(args: list[str]) -> None:
    """Persist the activation designation for a completed fine-tune job."""
    job_id, models_dir = _parse_activation_args("director-ai model-activate", args)
    store = _activation_store(models_dir)
    job = store.get(job_id)
    if job is None:
        print(f"Error: job {job_id} not found in {models_dir}")
        sys.exit(1)
    if job.state != "completed":
        print(f"Error: job {job_id} is not completed (state={job.state})")
        sys.exit(1)
    job.activated = True
    store.save(job)
    print(f"Model {job_id} marked active: {job.model_path}")
    print(
        "Designation persisted; it protects the model from deletion. A running "
        "server applies it after restart — use the REST activate endpoint for "
        "a live hot-swap."
    )


def _cmd_model_rollback(args: list[str]) -> None:
    """Clear the persisted activation designation for a fine-tune job."""
    job_id, models_dir = _parse_activation_args("director-ai model-rollback", args)
    store = _activation_store(models_dir)
    job = store.get(job_id)
    if job is None:
        print(f"Error: job {job_id} not found in {models_dir}")
        sys.exit(1)
    job.activated = False
    store.save(job)
    print(f"Model {job_id} rolled back (activation designation cleared).")
