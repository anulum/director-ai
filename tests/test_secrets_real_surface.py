# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for Director managed-secret environment hydration."""

from __future__ import annotations

import os
from collections.abc import Iterable

import pytest

from director_ai.core.secrets import (
    EnvSecretsBackend,
    SecretsProvider,
    build_backend_from_env,
)


def _clear_env(monkeypatch: pytest.MonkeyPatch, names: Iterable[str]) -> None:
    """Remove ``names`` from the process environment for an isolated test."""
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_prefixed_env_provider_hydrates_only_present_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefixed env backend hydrates real process env vars without fakes."""
    managed_names = (
        "DIRECTOR_ADMIN_KEY",
        "DIRECTOR_API_KEYS",
        "OPENAI_API_KEY",
    )
    _clear_env(monkeypatch, managed_names)
    _clear_env(monkeypatch, (f"PROD_{name}" for name in managed_names))
    monkeypatch.setenv("PROD_DIRECTOR_ADMIN_KEY", "admin-secret")
    monkeypatch.setenv("PROD_DIRECTOR_API_KEYS", "api-key-a,api-key-b")
    monkeypatch.setenv("PROD_OPENAI_API_KEY", "")

    provider = SecretsProvider(
        EnvSecretsBackend(prefix="PROD_"),
        cache_ttl_seconds=0,
    )

    loaded = provider.hydrate_environ(managed_names)

    assert loaded == ["DIRECTOR_ADMIN_KEY", "DIRECTOR_API_KEYS"]
    assert os.environ["DIRECTOR_ADMIN_KEY"] == "admin-secret"
    assert os.environ["DIRECTOR_API_KEYS"] == "api-key-a,api-key-b"
    assert "OPENAI_API_KEY" not in os.environ


def test_hydration_preserves_existing_env_unless_overwrite_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing env values remain authoritative unless overwrite is explicit."""
    _clear_env(monkeypatch, ("DIRECTOR_ADMIN_KEY", "PROD_DIRECTOR_ADMIN_KEY"))
    monkeypatch.setenv("DIRECTOR_ADMIN_KEY", "existing-admin")
    monkeypatch.setenv("PROD_DIRECTOR_ADMIN_KEY", "managed-admin")
    provider = SecretsProvider(EnvSecretsBackend(prefix="PROD_"))

    assert provider.hydrate_environ(("DIRECTOR_ADMIN_KEY",)) == []
    assert os.environ["DIRECTOR_ADMIN_KEY"] == "existing-admin"

    loaded = provider.hydrate_environ(("DIRECTOR_ADMIN_KEY",), overwrite=True)

    assert loaded == ["DIRECTOR_ADMIN_KEY"]
    assert os.environ["DIRECTOR_ADMIN_KEY"] == "managed-admin"


def test_build_backend_from_env_selects_prefixed_env_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env backend factory honors the configured managed-secret prefix."""
    _clear_env(
        monkeypatch,
        (
            "DIRECTOR_SECRETS_BACKEND",
            "DIRECTOR_SECRETS_PREFIX",
            "MANAGED_DIRECTOR_LICENSE_SIGNING_KEY",
        ),
    )
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", " env ")
    monkeypatch.setenv("DIRECTOR_SECRETS_PREFIX", "MANAGED_")
    monkeypatch.setenv("MANAGED_DIRECTOR_LICENSE_SIGNING_KEY", "license-secret")

    backend = build_backend_from_env()

    assert isinstance(backend, EnvSecretsBackend)
    assert backend.get_secret("DIRECTOR_LICENSE_SIGNING_KEY") == "license-secret"


def test_env_provider_refetches_rotated_values_after_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalidation refreshes a cached environment-backed managed secret."""
    _clear_env(monkeypatch, ("DIRECTOR_API_KEYS",))
    monkeypatch.setenv("DIRECTOR_API_KEYS", "old-key")
    provider = SecretsProvider(EnvSecretsBackend(), cache_ttl_seconds=300)

    assert provider.get("DIRECTOR_API_KEYS") == "old-key"

    monkeypatch.setenv("DIRECTOR_API_KEYS", "new-key")

    assert provider.get("DIRECTOR_API_KEYS") == "old-key"

    provider.invalidate("DIRECTOR_API_KEYS")

    assert provider.get("DIRECTOR_API_KEYS") == "new-key"
