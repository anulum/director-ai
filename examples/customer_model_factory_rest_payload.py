# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory REST payload example

"""Build a REST scoring payload from a Customer Model Factory runtime package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)


def build_rest_payload(
    runtime_package_path: Path,
    *,
    prompt: str,
    response: str,
    source_refs: list[str],
) -> dict[str, Any]:
    """Build a REST request payload without sending it."""

    package = CustomerRuntimePackage.from_dict(
        json.loads(runtime_package_path.read_text(encoding="utf-8"))
    )
    score_request = _build_score_request(
        package,
        prompt=prompt,
        response=response,
        source_refs=source_refs,
    )
    return {
        "method": "POST",
        "path": "/v1/score",
        "headers": {
            "Content-Type": "application/json",
            "X-Director-Customer": package.customer_id,
            "X-Director-Workspace": package.workspace_id,
            "X-Director-Tenant": package.tenant_id,
        },
        "json": score_request,
    }


def _build_score_request(
    package: CustomerRuntimePackage,
    *,
    prompt: str,
    response: str,
    source_refs: list[str],
) -> dict[str, Any]:
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
