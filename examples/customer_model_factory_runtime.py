# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory runtime example

"""Load a Customer Model Factory runtime package without network side effects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)


def load_runtime_package(path: Path) -> CustomerRuntimePackage:
    """Load a generated runtime package manifest from local JSON."""

    return CustomerRuntimePackage.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def build_score_request(
    package: CustomerRuntimePackage,
    *,
    prompt: str,
    response: str,
    source_refs: list[str],
) -> dict[str, Any]:
    """Build a local scorer request from a runtime package.

    The returned dictionary is transport-neutral. A customer can pass it to an
    in-process scorer, a private REST endpoint, or an on-prem queue without this
    helper opening a network connection.
    """

    config = package.runtime_config
    return {
        "runtime_id": package.runtime_id,
        "customer_id": package.customer_id,
        "workspace_id": package.workspace_id,
        "tenant_id": package.tenant_id,
        "deployment_id": package.deployment_id,
        "selected_model_artifact_uri": config["selected_model_artifact_uri"],
        "prompt": prompt,
        "response": response,
        "source_refs": list(source_refs),
        "thresholds": {
            "approve": config["threshold"],
            "abstain": config["abstention_threshold"],
            "escalate": config["escalation_threshold"],
        },
        "require_citations": config["require_citations"],
        "audit_log_uri": config["audit_log_uri"],
        "evidence_pack_uri": config["evidence_pack_uri"],
        "telemetry_mode": config["telemetry_mode"],
        "external_callbacks_allowed": config["external_callbacks_allowed"],
    }


def build_audit_metadata(package: CustomerRuntimePackage) -> dict[str, str]:
    """Return the immutable identifiers a customer should log per decision."""

    return {
        "runtime_id": package.runtime_id,
        "runtime_hash": package.runtime_hash,
        "deployment_id": package.deployment_id,
        "deployment_hash": str(package.runtime_config["deployment_hash"]),
        "evidence_hash": package.evidence_hash,
        "tenant_id": package.tenant_id,
    }
