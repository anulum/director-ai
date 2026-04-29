# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - CORS reverse proxy policy tests

import tomllib
from pathlib import Path

from director_ai.core.config import DirectorConfig

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements/cors_reverse_proxy_policy.toml"


def _load_policy() -> dict[str, object]:
    return tomllib.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _flatten_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _flatten_strings(entry)]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _flatten_strings(entry)]
    return []


def test_cors_policy_matches_application_defaults() -> None:
    policy = _load_policy()
    app = policy["application"]
    assert isinstance(app, dict)

    cfg = DirectorConfig()
    assert app["setting"] == "cors_origins"
    assert app["default"] == cfg.cors_origins
    assert cfg.cors_origins == ""


def test_cors_header_contract_matches_server_middleware() -> None:
    policy = _load_policy()
    headers = policy["headers"]
    server = _read(str(policy["server_module"]))
    assert isinstance(headers, dict)

    for method in headers["allow_methods"]:
        assert f'"{method}"' in server

    for header in headers["allow_headers"]:
        assert f'"{header}"' in server

    assert headers["allow_credentials"] is False


def test_docs_cover_each_proxy_example() -> None:
    policy = _load_policy()
    doc = _read(str(policy["deployment_doc"]))
    nav = _read("mkdocs.yml")

    doc_lower = doc.lower()
    for example in policy["examples"]:
        assert isinstance(example, dict)
        assert str(example["name"]).lower() in doc_lower
        assert str(example["rule"]).split(" with ")[0] in doc

    assert "deployment/cors-reverse-proxy.md" in nav


def test_cors_examples_do_not_use_origin_wildcards() -> None:
    policy = _load_policy()
    doc = _read(str(policy["deployment_doc"]))
    strings = [
        value
        for key, value in policy.items()
        if key != "forbidden"
        for value in _flatten_strings(value)
    ]

    assert '"*"' not in doc
    assert "'*'" not in doc
    assert "*;" not in doc
    assert all(value.strip() != "*" for value in strings)
    assert "https://app.example.com" in doc
    assert "https://admin.example.com" in doc
