# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — secrets backends (env / Vault / AWS / Azure) + rotation

"""Resolve named secrets from a managed backend instead of bare ``os.environ``.

Director reads a handful of process secrets directly from the environment —
``DIRECTOR_AUDIT_HMAC_SECRET``, ``DIRECTOR_LICENSE_SIGNING_KEY``,
``DIRECTOR_LICENSE_KEY``, ``DIRECTOR_API_KEYS``,
``DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS``, ``DIRECTOR_ADMIN_KEY``,
``DIRECTOR_AI_POLAR_WEBHOOK_SECRET``, ``DIRECTOR_AI_POLAR_ACCESS_TOKEN`` and the
``*_API_KEY`` provider keys. In a managed deployment those values live in a
secrets manager, not in the environment of every pod.

This module bridges the two. A :class:`SecretsBackend` resolves a secret by
name; :class:`EnvSecretsBackend` is the default (reads ``os.environ``), and
:class:`VaultSecretsBackend`, :class:`AWSSecretsManagerBackend` and
:class:`AzureKeyVaultBackend` pull from HashiCorp Vault, AWS Secrets Manager and
Azure Key Vault respectively (their SDKs are imported lazily, so none is a hard
dependency). :class:`SecretsProvider` wraps a backend with TTL caching and a
:meth:`SecretsProvider.hydrate_environ` step that loads the managed secrets into
``os.environ`` at startup — so every existing ``os.environ.get(...)`` call site
transparently resolves from the backend without being rewritten.

Key rotation
------------
TTL caching is what makes rotation work without a restart: set
``cache_ttl_seconds`` to a value shorter than your rotation interval and the
provider re-fetches each secret after the TTL lapses, so a rotated value is
picked up automatically. The supported rotation pattern is **overlapping
windows** — issue the new secret while the old one is still accepted, let both
validate for one TTL (plus a safety margin), then retire the old one. For the
HMAC audit key and the knowledge-write HMAC keys this is first-class:
``DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS`` already accepts a comma-separated list, so
rotation is "prepend the new key, drop the oldest after the window".
:func:`rotation_guidance` returns the per-secret guidance as structured data.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "SecretNotFoundError",
    "SecretsBackend",
    "EnvSecretsBackend",
    "VaultSecretsBackend",
    "AWSSecretsManagerBackend",
    "AzureKeyVaultBackend",
    "SecretsProvider",
    "build_backend_from_env",
    "hydrate_managed_secrets",
    "MANAGED_SECRET_NAMES",
    "rotation_guidance",
]

# The process secrets Director resolves from the environment. hydrate_environ()
# defaults to this set so an operator pointing at a managed backend gets every
# call site (audit HMAC, license keys, API keys, Polar) bridged at once.
MANAGED_SECRET_NAMES: tuple[str, ...] = (
    "DIRECTOR_AUDIT_HMAC_SECRET",
    "DIRECTOR_LICENSE_SIGNING_KEY",
    "DIRECTOR_LICENSE_KEY",
    "DIRECTOR_API_KEYS",
    "DIRECTOR_API_KEY_TENANT_MAP",
    "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS",
    "DIRECTOR_ADMIN_KEY",
    "DIRECTOR_AI_POLAR_WEBHOOK_SECRET",
    "DIRECTOR_AI_POLAR_ACCESS_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)


class SecretNotFoundError(KeyError):
    """A required secret was absent from the backend."""


@runtime_checkable
class SecretsBackend(Protocol):
    """Resolve a secret value by name. Returns ``None`` when absent."""

    def get_secret(self, name: str) -> str | None:
        """Return ``name`` from the backend, or ``None`` when absent."""
        ...


class EnvSecretsBackend:
    """Resolve secrets from ``os.environ`` (the default backend).

    An optional ``prefix`` is prepended to every lookup, so a deployment can
    namespace its secrets (e.g. ``prefix="PROD_"`` resolves ``DIRECTOR_ADMIN_KEY``
    from ``PROD_DIRECTOR_ADMIN_KEY``).
    """

    def __init__(self, *, prefix: str = "") -> None:
        self._prefix = prefix

    def get_secret(self, name: str) -> str | None:
        """Return ``name`` from ``os.environ`` with the configured prefix."""
        value = os.environ.get(self._prefix + name)
        return value if value else None


class VaultSecretsBackend:
    """Resolve secrets from HashiCorp Vault (KV v2).

    Requires the ``hvac`` package (imported lazily). Each Director secret name
    maps to a field inside a Vault secret at ``{mount}/{path}`` — by default a
    single secret holds all fields, keyed by the Director secret name. Supply
    ``field_map`` to remap a Director name to a different Vault field, or
    ``path_map`` to split secrets across multiple Vault paths.

    Parameters
    ----------
    url, token:
        Vault address and token. Fall back to ``VAULT_ADDR`` / ``VAULT_TOKEN``.
    mount, path:
        KV v2 mount point and secret path.
    field_map:
        Optional ``{director_name: vault_field}`` overrides.
    path_map:
        Optional ``{director_name: vault_path}`` overrides for split layouts.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        token: str | None = None,
        mount: str = "secret",
        path: str = "director-ai",
        field_map: Mapping[str, str] | None = None,
        path_map: Mapping[str, str] | None = None,
    ) -> None:
        self._url = url or os.environ.get("VAULT_ADDR", "")
        self._token = token or os.environ.get("VAULT_TOKEN", "")
        if not self._url or not self._token:
            raise ValueError("Vault url/token required (VAULT_ADDR / VAULT_TOKEN)")
        self._mount = mount
        self._path = path
        self._field_map = dict(field_map or {})
        self._path_map = dict(path_map or {})
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import hvac
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "VaultSecretsBackend requires 'hvac' "
                    "(pip install director-ai[vault])"
                ) from exc
            self._client = hvac.Client(url=self._url, token=self._token)
        return self._client

    def get_secret(self, name: str) -> str | None:
        """Return ``name`` from Vault KV v2 using field/path overrides."""
        client = self._get_client()
        path = self._path_map.get(name, self._path)
        field = self._field_map.get(name, name)
        read = client.secrets.kv.v2.read_secret_version  # type: ignore[attr-defined]
        resp = read(path=path, mount_point=self._mount, raise_on_deleted_version=True)
        data = resp.get("data", {}).get("data", {})
        value = data.get(field)
        return str(value) if value not in (None, "") else None


class AWSSecretsManagerBackend:
    """Resolve secrets from AWS Secrets Manager.

    Requires ``boto3`` (imported lazily). A secret may be a flat string (mapped
    one-to-one with a Director name) or a JSON blob whose keys are Director
    secret names. ``secret_id_map`` overrides the AWS SecretId for a given name;
    otherwise ``secret_id`` is read as a JSON blob.

    Parameters
    ----------
    region_name:
        AWS region; falls back to ``AWS_REGION`` / ``AWS_DEFAULT_REGION``.
    secret_id:
        SecretId of a JSON blob holding all fields (when ``secret_id_map`` is
        not used for a name).
    secret_id_map:
        Optional ``{director_name: aws_secret_id}`` for one-secret-per-value.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        secret_id: str = "director-ai",
        secret_id_map: Mapping[str, str] | None = None,
        client: object | None = None,
    ) -> None:
        self._region = (
            region_name
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        self._secret_id = secret_id
        self._secret_id_map = dict(secret_id_map or {})
        self._client = client

    def _get_client(self) -> object:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "AWSSecretsManagerBackend requires 'boto3' "
                    "(pip install director-ai[aws])"
                ) from exc
            self._client = boto3.client("secretsmanager", region_name=self._region)
        return self._client

    def get_secret(self, name: str) -> str | None:
        """Return ``name`` from AWS Secrets Manager."""
        import json

        client = self._get_client()
        if name in self._secret_id_map:
            resp = client.get_secret_value(  # type: ignore[attr-defined]
                SecretId=self._secret_id_map[name]
            )
            value = resp.get("SecretString")
            return str(value) if value else None
        resp = client.get_secret_value(SecretId=self._secret_id)  # type: ignore[attr-defined]
        blob = resp.get("SecretString")
        if not blob:
            return None
        try:
            data = json.loads(blob)
        except (ValueError, TypeError):
            return None
        value = data.get(name)
        return str(value) if value not in (None, "") else None


class AzureKeyVaultBackend:
    """Resolve secrets from Azure Key Vault.

    Requires ``azure-keyvault-secrets`` and ``azure-identity`` (imported
    lazily). Azure secret names cannot contain underscores, so Director names are
    mapped by replacing ``_`` with ``-`` and lowercasing (overridable via
    ``name_map``).

    Parameters
    ----------
    vault_url:
        Key Vault URL, e.g. ``https://my-vault.vault.azure.net``; falls back to
        ``AZURE_KEY_VAULT_URL``.
    credential:
        Optional pre-built Azure credential; defaults to
        ``DefaultAzureCredential``.
    name_map:
        Optional ``{director_name: azure_secret_name}`` overrides.
    """

    def __init__(
        self,
        *,
        vault_url: str | None = None,
        credential: object | None = None,
        name_map: Mapping[str, str] | None = None,
        client: object | None = None,
    ) -> None:
        self._vault_url = vault_url or os.environ.get("AZURE_KEY_VAULT_URL", "")
        if not self._vault_url and client is None:
            raise ValueError("Azure vault_url required (AZURE_KEY_VAULT_URL)")
        self._credential = credential
        self._name_map = dict(name_map or {})
        self._client = client

    @staticmethod
    def _azurify(name: str) -> str:
        return name.replace("_", "-").lower()

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from azure.identity import DefaultAzureCredential
                from azure.keyvault.secrets import SecretClient
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "AzureKeyVaultBackend requires 'azure-keyvault-secrets' and "
                    "'azure-identity' (pip install director-ai[azure])"
                ) from exc
            credential = self._credential or DefaultAzureCredential()
            self._client = SecretClient(
                vault_url=self._vault_url, credential=credential
            )
        return self._client

    def get_secret(self, name: str) -> str | None:
        """Return ``name`` from Azure Key Vault."""
        client = self._get_client()
        azure_name = self._name_map.get(name) or self._azurify(name)
        try:
            secret = client.get_secret(azure_name)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - SDK raises ResourceNotFoundError
            return None
        value = getattr(secret, "value", None)
        return str(value) if value not in (None, "") else None


@dataclass(frozen=True)
class _CacheEntry:
    value: str | None
    fetched_at: float


class SecretsProvider:
    """TTL-cached access to a :class:`SecretsBackend`, with environ hydration.

    The TTL is what enables rotation without a restart: after
    ``cache_ttl_seconds`` a cached secret is re-fetched, so a rotated value is
    picked up on the next access. Set the TTL shorter than your rotation
    interval.

    Parameters
    ----------
    backend:
        The resolving backend. Defaults to :class:`EnvSecretsBackend`.
    cache_ttl_seconds:
        Cache lifetime; ``0`` disables caching (every access hits the backend).
    """

    def __init__(
        self,
        backend: SecretsBackend | None = None,
        *,
        cache_ttl_seconds: float = 300.0,
    ) -> None:
        self._backend: SecretsBackend = backend or EnvSecretsBackend()
        self._ttl = max(0.0, float(cache_ttl_seconds))
        self._cache: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    @property
    def backend(self) -> SecretsBackend:
        """Return the backend used for uncached secret resolution."""
        return self._backend

    def _now(self) -> float:
        return time.monotonic()

    def get(
        self, name: str, default: str | None = None, *, required: bool = False
    ) -> str | None:
        """Return the secret ``name``; ``default`` when absent.

        Raises :class:`SecretNotFoundError` when ``required`` and the secret is
        absent.
        """
        with self._lock:
            entry = self._cache.get(name)
            if (
                entry is not None
                and self._ttl > 0
                and self._now() - entry.fetched_at < self._ttl
            ):
                value = entry.value
                if value is None and required:
                    raise SecretNotFoundError(name)
                return value if value is not None else default
            value = self._backend.get_secret(name)
            self._cache[name] = _CacheEntry(value=value, fetched_at=self._now())
        if value is None:
            if required:
                raise SecretNotFoundError(name)
            return default
        return value

    def invalidate(self, name: str | None = None) -> None:
        """Drop ``name`` (or the whole cache) so the next access re-fetches."""
        with self._lock:
            if name is None:
                self._cache.clear()
            else:
                self._cache.pop(name, None)

    def hydrate_environ(
        self,
        names: Iterable[str] = MANAGED_SECRET_NAMES,
        *,
        overwrite: bool = False,
    ) -> list[str]:
        """Load secrets from the backend into ``os.environ``.

        Call this once at startup so existing ``os.environ.get(...)`` call sites
        resolve from the managed backend. By default an env var already present
        is left untouched (``overwrite=False``); pass ``overwrite=True`` to let
        the backend win. Returns the names that were set.
        """
        loaded: list[str] = []
        for name in names:
            if not overwrite and os.environ.get(name):
                continue
            value = self.get(name)
            if value:
                os.environ[name] = value
                loaded.append(name)
        return loaded


def build_backend_from_env() -> SecretsBackend:
    """Construct a backend from ``DIRECTOR_SECRETS_BACKEND``.

    Values: ``env`` (default), ``vault``, ``aws``, ``azure``. Each backend reads
    its own connection settings from the environment (``VAULT_ADDR`` etc.), so
    selecting a managed backend is one env var plus that backend's standard
    connection variables.
    """
    kind = os.environ.get("DIRECTOR_SECRETS_BACKEND", "env").strip().lower()
    if kind in ("", "env"):
        return EnvSecretsBackend(prefix=os.environ.get("DIRECTOR_SECRETS_PREFIX", ""))
    if kind == "vault":
        return VaultSecretsBackend(
            mount=os.environ.get("DIRECTOR_VAULT_MOUNT", "secret"),
            path=os.environ.get("DIRECTOR_VAULT_PATH", "director-ai"),
        )
    if kind == "aws":
        return AWSSecretsManagerBackend(
            secret_id=os.environ.get("DIRECTOR_AWS_SECRET_ID", "director-ai"),
        )
    if kind == "azure":
        return AzureKeyVaultBackend()
    raise ValueError(f"unknown DIRECTOR_SECRETS_BACKEND: {kind!r}")


def hydrate_managed_secrets(
    names: Iterable[str] = MANAGED_SECRET_NAMES,
    *,
    overwrite: bool = False,
) -> list[str]:
    """Bridge managed secrets into ``os.environ`` at startup.

    A no-op when ``DIRECTOR_SECRETS_BACKEND`` is unset or ``env`` (the default),
    so a plain environment-variable deployment is unaffected. Otherwise builds
    the configured backend, loads the managed secrets into ``os.environ`` (only
    those not already set unless ``overwrite``), and returns the names loaded.
    Call once, before config/license reads, so every downstream
    ``os.environ.get(...)`` resolves from the backend.
    """
    kind = os.environ.get("DIRECTOR_SECRETS_BACKEND", "env").strip().lower()
    if kind in ("", "env"):
        return []
    provider = SecretsProvider(build_backend_from_env())
    return provider.hydrate_environ(names, overwrite=overwrite)


@dataclass(frozen=True)
class _RotationRule:
    secret: str
    overlap_supported: bool
    note: str


def rotation_guidance() -> list[dict[str, object]]:
    """Per-secret rotation guidance as structured records.

    Each record documents whether the secret supports overlapping-window
    rotation (old and new valid simultaneously) and how to rotate it. Tooling
    and docs render this; it is data, not prose, so it stays in sync with the
    managed-secret set.
    """
    rules = (
        _RotationRule(
            "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS",
            True,
            "Comma-separated list: prepend the new key, keep the old one for one "
            "TTL window, then drop the oldest. Zero-downtime.",
        ),
        _RotationRule(
            "DIRECTOR_AUDIT_HMAC_SECRET",
            False,
            "Single key. Rotating changes the audit hash chain; rotate at a "
            "segment boundary and archive the prior segment for verification.",
        ),
        _RotationRule(
            "DIRECTOR_API_KEYS",
            True,
            "Comma-separated list: issue the new key, distribute, then retire the "
            "old key after clients have migrated.",
        ),
        _RotationRule(
            "DIRECTOR_LICENSE_SIGNING_KEY",
            False,
            "Signing key. Re-sign issued licenses or run dual-verify during the "
            "overlap, then retire the old public key.",
        ),
        _RotationRule(
            "DIRECTOR_ADMIN_KEY",
            False,
            "Single admin credential. Set the new value in the backend; with a "
            "TTL shorter than the rotation interval it is picked up live.",
        ),
        _RotationRule(
            "DIRECTOR_AI_POLAR_WEBHOOK_SECRET",
            False,
            "Provider-issued. Rotate in the Polar dashboard, then update the "
            "backend; brief overlap accepted by re-trying verification.",
        ),
    )
    return [
        {
            "secret": r.secret,
            "overlap_supported": r.overlap_supported,
            "note": r.note,
        }
        for r in rules
    ]
