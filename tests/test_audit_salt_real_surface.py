# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - audit salt real-surface tests
"""Real server-surface coverage for audit salt enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from httpx import Response

pytest.importorskip("fastapi", reason="fastapi required for server route tests")

from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.server import create_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

_PRIMARY_KEY = "audit-salt-alpha-key"
_SECONDARY_KEY = "audit-salt-beta-key"
_KB_WRITE_KEYS = '{"kid-1":"audit-salt-hmac-material-for-route-tests-32"}'


@pytest.fixture(autouse=True)
def _audit_salt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep audit salt settings isolated from the operator shell."""
    monkeypatch.delenv("DIRECTOR_AUDIT_SALT", raising=False)
    monkeypatch.delenv("DIRECTOR_AUDIT_SALT_FILE", raising=False)
    monkeypatch.delenv("DIRECTOR_AUDIT_SALT_STRICT", raising=False)
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")


def _production_config() -> DirectorConfig:
    """Return a production-mode server config that stays fully local in tests."""
    return DirectorConfig(
        mode="general",
        production_mode=True,
        llm_provider="local",
        llm_api_url="https://llm.internal.example/v1",
        scorer_backend="lite",
        use_nli=False,
        coherence_threshold=0.0,
        hard_limit=0.0,
        soft_limit=0.0,
        adaptive_threshold_enabled=False,
        api_keys=[_PRIMARY_KEY, _SECONDARY_KEY],
        knowledge_write_hmac_keys=_KB_WRITE_KEYS,
        hybrid_retrieval=False,
        reranker_enabled=False,
        retrieval_abstention_threshold=0.0,
    )


def _review_payload(session_id: str) -> dict[str, str]:
    """Return a minimal public review payload for session ownership checks."""
    return {
        "prompt": "Which control is required for audit logs?",
        "response": "Production audit logs require per-installation salt.",
        "session_id": session_id,
    }


def _post_review(client: TestClient, *, key: str, session_id: str) -> Response:
    """Post to the real review route with a caller-selected API key."""
    return cast(
        Response,
        client.post(
            "/v1/review",
            headers={"X-API-Key": key},
            json=_review_payload(session_id),
        ),
    )


def test_audit_salt_unit_guard_declares_this_companion() -> None:
    """The legacy audit-salt unit guard should declare this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_audit_salt.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_audit_salt_real_surface.py" in reason


def test_production_app_refuses_missing_audit_salt() -> None:
    """Production app creation should fail before serving without a salt."""
    with pytest.raises(RuntimeError, match="audit salt"):
        create_app(_production_config())


def test_public_review_uses_salted_api_key_session_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public review should keep sessions bound to the creating API key."""
    monkeypatch.setenv("DIRECTOR_AUDIT_SALT", "audit-salt-real-surface-env")

    with TestClient(create_app(_production_config())) as client:
        first = _post_review(
            client,
            key=_PRIMARY_KEY,
            session_id="audit-salt-session",
        )
        repeat = _post_review(
            client,
            key=_PRIMARY_KEY,
            session_id="audit-salt-session",
        )
        rejected = _post_review(
            client,
            key=_SECONDARY_KEY,
            session_id="audit-salt-session",
        )

    assert first.status_code == 200, first.text
    first_payload = cast(dict[str, object], first.json())
    assert isinstance(first_payload["approved"], bool)
    assert isinstance(first_payload["coherence"], float)

    assert repeat.status_code == 200, repeat.text
    assert rejected.status_code == 403
    assert rejected.json() == {"detail": "Session belongs to a different API key"}


def test_public_review_accepts_file_backed_audit_salt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Production app creation should accept a readable salt file."""
    salt_path = tmp_path / "audit-salt.txt"
    salt_path.write_text("audit-salt-real-surface-file\n", encoding="utf-8")
    monkeypatch.setenv("DIRECTOR_AUDIT_SALT_FILE", str(salt_path))

    with TestClient(create_app(_production_config())) as client:
        response = _post_review(
            client,
            key=_PRIMARY_KEY,
            session_id="audit-salt-file-session",
        )

    assert response.status_code == 200, response.text
    payload = cast(dict[str, object], response.json())
    assert isinstance(payload["approved"], bool)
