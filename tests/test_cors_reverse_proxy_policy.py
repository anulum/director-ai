# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - CORS reverse proxy policy tests

"""Check maintained CORS defaults, middleware declarations and proxy guidance."""

import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from director_ai.core.config import DirectorConfig
from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements/cors_reverse_proxy_policy.toml"


def _load_policy() -> dict[str, object]:
    """Read the maintained CORS policy without erasing decoded value types.

    Returns
    -------
    dict[str, object]
        String-keyed policy root preserving all decoded TOML values.

    Raises
    ------
    OSError
        If the policy cannot be read.
    tomllib.TOMLDecodeError
        If the policy contains invalid TOML.
    ValueError
        If its decoded root is not a string-keyed mapping.
    """
    return object_mapping(tomllib.loads(POLICY_PATH.read_text(encoding="utf-8")))


def _read(path: str) -> str:
    """Read a repository-relative policy or documentation file.

    Parameters
    ----------
    path : str
        Repository-relative path used by a maintained policy.

    Returns
    -------
    str
        UTF-8 text of the file.

    Raises
    ------
    OSError
        If the file cannot be read.
    UnicodeError
        If its content is not valid UTF-8.
    """
    return (ROOT / path).read_text(encoding="utf-8")


def _flatten_strings(value: object) -> list[str]:
    """Collect string leaves from nested policy values.

    Parameters
    ----------
    value : object
        String, list, mapping or other decoded policy value.

    Returns
    -------
    list[str]
        String leaves in traversal order; other scalar values are ignored.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for entry in value for item in _flatten_strings(entry)]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _flatten_strings(entry)]
    return []


def _origin_tuple(value: str) -> tuple[str, str]:
    """Extract scheme and network location from an origin string.

    Parameters
    ----------
    value : str
        Origin or documented URL to parse.

    Returns
    -------
    tuple[str, str]
        Parsed scheme and network location without path or query components.

    Raises
    ------
    ValueError
        If URL parsing rejects an invalid authority, such as malformed IPv6.
    """
    parsed = urlsplit(value)
    return parsed.scheme, parsed.netloc


def test_cors_policy_matches_application_defaults() -> None:
    """Match the policy's browser-origin setting to public config defaults."""
    policy = _load_policy()
    app = object_mapping(policy["application"])

    cfg = DirectorConfig()
    assert app["setting"] == "cors_origins"
    assert app["default"] == cfg.cors_origins
    assert cfg.cors_origins == ""


def test_cors_header_contract_matches_server_middleware() -> None:
    """Match declared methods and headers to maintained middleware source."""
    policy = _load_policy()
    headers = object_mapping(policy["headers"])
    server = _read(str(policy["server_module"]))

    allow_methods = headers["allow_methods"]
    assert isinstance(allow_methods, list)
    for method in allow_methods:
        assert isinstance(method, str)
        assert f'"{method}"' in server

    allow_headers = headers["allow_headers"]
    assert isinstance(allow_headers, list)
    for header in allow_headers:
        assert isinstance(header, str)
        assert f'"{header}"' in server

    assert headers["allow_credentials"] is False


def test_docs_cover_each_proxy_example() -> None:
    """Require the deployment guide to cover each declared proxy rule."""
    policy = _load_policy()
    doc = _read(str(policy["deployment_doc"]))
    nav = _read("mkdocs.yml")

    doc_lower = doc.lower()
    examples = policy["examples"]
    assert isinstance(examples, list)
    for record in examples:
        example = object_mapping(record)
        assert str(example["name"]).lower() in doc_lower
        assert str(example["rule"]).split(" with ")[0] in doc

    assert "deployment/cors-reverse-proxy.md" in nav


def test_cors_examples_do_not_use_origin_wildcards() -> None:
    """Reject wildcard examples and require every configured browser origin."""
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
    application = object_mapping(policy["application"])
    expected_origins = {
        _origin_tuple(origin.strip())
        for origin in str(application["example"]).split(",")
    }
    documented_origins = {
        _origin_tuple(value.strip('"` ,;'))
        for value in doc.split()
        if value.startswith("https://")
    }
    assert expected_origins <= documented_origins
