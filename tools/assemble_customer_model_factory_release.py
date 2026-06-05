# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release assembler

"""Assemble the final Customer Model Factory release gate from manifest JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from director_ai.core.customer_model_factory.evidence_pack import (
    CustomerEvidencePackManifest,
)
from director_ai.core.customer_model_factory.monitoring_manifest import (
    CustomerMonitoringManifest,
)
from director_ai.core.customer_model_factory.release_gate import (
    DeploymentHardeningEvidence,
    ObservabilityOperationsEvidence,
    build_release_gate_manifest,
)
from director_ai.core.customer_model_factory.risk_register import CustomerRiskRegister
from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)


def main(argv: list[str] | None = None) -> int:
    """Run the release-gate assembler."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--enterprise-readiness", type=Path, required=True)
    parser.add_argument("--runtime-package", type=Path, required=True)
    parser.add_argument("--evidence-pack", type=Path, required=True)
    parser.add_argument("--monitoring-manifest", type=Path, required=True)
    parser.add_argument("--risk-register", type=Path, required=True)
    parser.add_argument("--observability-operations-evidence", type=Path, required=True)
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
        deployment_hardening_evidence=DeploymentHardeningEvidence.from_dict(
            _read_json(args.deployment_hardening_evidence)
        ),
        generated_at=args.generated_at,
    )
    release_gate.write_json(args.output)
    return 0 if release_gate.promotion_allowed else 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
