# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - public endpoint exposure policy tests
"""Static policy guards for public endpoint exposure documentation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from director_ai.core.config import DirectorConfig
from director_ai.server import _AUTH_EXEMPT_PATHS_BASE

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements/public_endpoint_exposure_policy.toml"


def _load_policy() -> dict[str, object]:
    """Return the static public endpoint exposure policy."""
    return tomllib.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _policy_endpoints(policy: dict[str, object]) -> list[dict[str, object]]:
    """Return endpoint entries from the static policy file."""
    return cast(list[dict[str, object]], policy["endpoints"])


def _policy_defaults(policy: dict[str, object]) -> dict[str, object]:
    """Return deployment default entries from the static policy file."""
    return cast(dict[str, object], policy["defaults"])


def _read(path: str) -> str:
    """Read a repository-relative text file."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_static_policy_matches_server_exempt_paths() -> None:
    """The static policy should match the server's auth-exempt base set."""
    policy = _load_policy()
    endpoints = _policy_endpoints(policy)
    server = _read(str(policy["server_module"]))

    exempt = {
        endpoint["path"]
        for endpoint in endpoints
        if endpoint["default_auth"] == "exempt"
    }

    assert exempt == _AUTH_EXEMPT_PATHS_BASE
    assert "/v1/metrics/prometheus" not in _AUTH_EXEMPT_PATHS_BASE
    assert "if cfg.metrics_require_auth" in server
    assert '_AUTH_EXEMPT_PATHS_BASE | {"/v1/metrics/prometheus"}' in server


def test_static_policy_matches_config_defaults() -> None:
    """The static policy should match documented security-sensitive defaults."""
    policy = _load_policy()
    defaults = _policy_defaults(policy)

    cfg = DirectorConfig()
    assert defaults["metrics_require_auth"] is cfg.metrics_require_auth
    assert defaults["source_endpoint_enabled"] is cfg.source_endpoint_enabled
    assert "private scrape" in _read(str(policy["deployment_doc"]))


def test_docs_cover_each_endpoint_and_control() -> None:
    """The deployment guide should cover every endpoint and control."""
    policy = _load_policy()
    doc = _read(str(policy["deployment_doc"]))
    nav = _read("mkdocs.yml")

    for endpoint in _policy_endpoints(policy):
        assert str(endpoint["path"]) in doc
        assert str(endpoint["internet_policy"]) in doc

    for control in _policy_defaults(policy):
        assert str(control) in doc

    assert "deployment/public-endpoints.md" in nav


def test_server_documentation_links_to_exposure_page() -> None:
    """The API reference should link operators to public exposure rules."""
    doc = _read("docs-site/api/server.md")
    assert "deployment/public-endpoints.md" in doc
    assert "/v1/health" in doc
    assert "/v1/ready" in doc
    assert "/v1/source" in doc
    assert "/v1/metrics/prometheus" in doc
