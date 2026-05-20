# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory customer examples tests

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_example(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime_package(tmp_path: Path) -> Path:
    package = CustomerRuntimePackage(
        schema_version="1.0.0",
        runtime_id="runtime-customer-alpha-20260518",
        ready=True,
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        deployment_id="customer-alpha-prod-20260518",
        evidence_hash="a" * 64,
        runtime_mode="offline_private",
        runtime_config={
            "customer_id": "customer-alpha",
            "workspace_id": "customer-alpha-prod",
            "tenant_id": "customer-alpha-tenant",
            "deployment_id": "customer-alpha-prod-20260518",
            "deployment_hash": "b" * 64,
            "evidence_hash": "a" * 64,
            "selected_benchmark_id": "customer-alpha-private-v1",
            "selected_model_artifact_uri": "gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha",
            "threshold": 0.72,
            "abstention_threshold": 0.58,
            "escalation_threshold": 0.40,
            "require_citations": True,
            "audit_log_uri": "gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
            "evidence_pack_uri": "gs://customer-artifacts/customer-alpha/evidence/pack",
            "rollback_package_uri": "gs://customer-artifacts/customer-alpha/deployments/previous.json",
            "retention_days": 365,
            "telemetry_mode": "customer_controlled",
            "external_callbacks_allowed": False,
            "callback_endpoints": [],
        },
        findings=(),
        runtime_hash="c" * 64,
    )
    return package.write_json(tmp_path / "runtime_package.json")


def test_python_runtime_example_builds_local_score_request(tmp_path: Path):
    example = _load_example(ROOT / "examples" / "customer_model_factory_runtime.py")
    package_path = _runtime_package(tmp_path)

    package = example.load_runtime_package(package_path)
    request = example.build_score_request(
        package,
        prompt="Can I promise this mortgage rate?",
        response="Escalate the rate promise to compliance before sending.",
        source_refs=["policy://customer-alpha/mortgage-rates"],
    )

    assert package.runtime_id == "runtime-customer-alpha-20260518"
    assert request["tenant_id"] == "customer-alpha-tenant"
    assert request["selected_model_artifact_uri"].endswith("cmf-customer-alpha")
    assert request["thresholds"] == {
        "approve": 0.72,
        "abstain": 0.58,
        "escalate": 0.4,
    }
    assert request["external_callbacks_allowed"] is False
    assert request["source_refs"] == ["policy://customer-alpha/mortgage-rates"]


def test_rest_payload_example_builds_customer_scoring_payload(tmp_path: Path):
    example = _load_example(
        ROOT / "examples" / "customer_model_factory_rest_payload.py"
    )
    package_path = _runtime_package(tmp_path)

    payload = example.build_rest_payload(
        package_path,
        prompt="Can I say the fee is waived forever?",
        response="Do not claim the fee is waived forever without policy evidence.",
        source_refs=["policy://customer-alpha/fees"],
    )

    assert payload["method"] == "POST"
    assert payload["path"] == "/v1/score"
    assert payload["headers"]["X-Director-Customer"] == "customer-alpha"
    assert payload["headers"]["X-Director-Tenant"] == "customer-alpha-tenant"
    assert payload["json"]["runtime_id"] == "runtime-customer-alpha-20260518"
    assert payload["json"]["telemetry_mode"] == "customer_controlled"
    assert payload["json"]["external_callbacks_allowed"] is False
    assert payload["json"]["source_refs"] == ["policy://customer-alpha/fees"]


def test_customer_examples_do_not_import_network_clients():
    blocked_import_tokens = {"requests", "httpx", "urllib", "socket", "aiohttp"}
    for path in (
        ROOT / "examples" / "customer_model_factory_runtime.py",
        ROOT / "examples" / "customer_model_factory_rest_payload.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert not any(f"import {token}" in source for token in blocked_import_tokens)
        assert not any(f"from {token}" in source for token in blocked_import_tokens)


def test_readme_advertises_only_implemented_customer_model_factory_surface():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "Customer Model Factory" in readme
    assert "customer_model_factory_runtime.py" in readme
    assert "zero silent unsafe passes" in readme
    assert "100%" + " accuracy" not in readme
    assert (
        json.loads(
            (
                ROOT / "schemas" / "customer-model-factory-runtime-package.schema.json"
            ).read_text(encoding="utf-8")
        )["title"]
        == "DIRECTOR-AI Customer Model Factory Runtime Package"
    )
