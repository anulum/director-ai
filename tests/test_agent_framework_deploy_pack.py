# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — agent framework deploy pack tests

"""Regression tests for the agent-framework integration examples and templates."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from examples.agent_framework_guardrails import run_all_smokes

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deploy" / "agent-frameworks"


def test_agent_framework_smokes_are_json_serializable() -> None:
    """The executable examples return stable, JSON-safe contracts."""

    packet = run_all_smokes()

    json.dumps(packet, sort_keys=True)
    assert packet["langgraph"]["route"] == "ship_response"
    assert packet["langgraph"]["director_ai_approved"] is True
    assert packet["crewai"]["tool_name"] == "director_ai_fact_check"
    assert "APPROVED" in packet["crewai"]["tool_output"]
    assert packet["llamaindex"]["kept_nodes"] == 1
    assert packet["llamaindex"]["response_approved"] is True


def test_cloud_run_template_wires_runtime_secrets_and_limits() -> None:
    """The Cloud Run template exposes the server without committing secrets."""

    service = yaml.safe_load((DEPLOY_DIR / "cloud-run-service.yaml").read_text())
    container = service["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}

    assert service["kind"] == "Service"
    assert container["ports"][0]["containerPort"] == 8080
    assert container["resources"]["limits"] == {"cpu": "2", "memory": "2Gi"}
    assert env["DIRECTOR_API_KEYS"]["valueFrom"]["secretKeyRef"]["name"] == (
        "director-api-keys"
    )
    assert env["DIRECTOR_KB_SIGNING_KEY"]["valueFrom"]["secretKeyRef"]["name"] == (
        "director-kb-signing-key"
    )
    assert "sk-" not in (DEPLOY_DIR / "cloud-run-service.yaml").read_text()


def test_vercel_template_targets_remote_director_service() -> None:
    """The Vercel template delegates review to Cloud Run via environment keys."""

    template = json.loads((DEPLOY_DIR / "vercel.json").read_text())

    assert template["env"]["DIRECTOR_AI_ENDPOINT"] == "@director_ai_endpoint"
    assert template["env"]["DIRECTOR_API_KEY"] == "@director_api_key"
    assert template["headers"][0]["headers"][0] == {
        "key": "Cache-Control",
        "value": "no-store",
    }


def test_integration_docs_link_the_deploy_pack() -> None:
    """The public integration docs point users to the shared deploy pack."""

    docs = [
        ROOT / "docs-site" / "integrations" / "agent-framework-deploy.md",
        ROOT / "docs-site" / "integrations" / "langgraph.md",
        ROOT / "docs-site" / "integrations" / "crewai.md",
        ROOT / "docs-site" / "integrations" / "llamaindex.md",
        ROOT / "docs-site" / "integrations" / "vercel-ai.md",
    ]

    for path in docs:
        assert "deploy/agent-frameworks" in path.read_text(encoding="utf-8")
