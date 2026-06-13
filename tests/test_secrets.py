# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — secrets backend tests (offline, SDKs mocked)

from __future__ import annotations

import json
import types

import pytest

from director_ai.core import secrets as sec
from director_ai.core.secrets import (
    MANAGED_SECRET_NAMES,
    AWSSecretsManagerBackend,
    AzureKeyVaultBackend,
    EnvSecretsBackend,
    SecretNotFoundError,
    SecretsBackend,
    SecretsProvider,
    VaultSecretsBackend,
    build_backend_from_env,
    rotation_guidance,
)


# --------------------------------------------------------------------------- #
# EnvSecretsBackend
# --------------------------------------------------------------------------- #
def test_env_backend_reads_environ(monkeypatch):
    monkeypatch.setenv("DIRECTOR_ADMIN_KEY", "topsecret")
    assert EnvSecretsBackend().get_secret("DIRECTOR_ADMIN_KEY") == "topsecret"


def test_env_backend_empty_is_none(monkeypatch):
    monkeypatch.setenv("DIRECTOR_ADMIN_KEY", "")
    assert EnvSecretsBackend().get_secret("DIRECTOR_ADMIN_KEY") is None
    monkeypatch.delenv("DIRECTOR_ADMIN_KEY", raising=False)
    assert EnvSecretsBackend().get_secret("DIRECTOR_ADMIN_KEY") is None


def test_env_backend_prefix(monkeypatch):
    monkeypatch.setenv("PROD_DIRECTOR_ADMIN_KEY", "p")
    assert EnvSecretsBackend(prefix="PROD_").get_secret("DIRECTOR_ADMIN_KEY") == "p"


def test_env_backend_satisfies_protocol():
    assert isinstance(EnvSecretsBackend(), SecretsBackend)


# --------------------------------------------------------------------------- #
# VaultSecretsBackend (fake hvac injected via sys.modules)
# --------------------------------------------------------------------------- #
class _FakeKVv2:
    def __init__(self, store):
        self._store = store

    def read_secret_version(self, *, path, mount_point, raise_on_deleted_version):
        data = self._store.get((mount_point, path))
        if data is None:
            raise KeyError("no such path")
        return {"data": {"data": data}}


class _FakeVaultClient:
    def __init__(self, url, token, store):
        self.url = url
        self.token = token
        self.secrets = types.SimpleNamespace(
            kv=types.SimpleNamespace(v2=_FakeKVv2(store))
        )


def _install_fake_hvac(monkeypatch, store):
    captured = {}

    def Client(url, token):  # noqa: N802 - mimic hvac.Client
        client = _FakeVaultClient(url, token, store)
        captured["client"] = client
        return client

    fake = types.ModuleType("hvac")
    fake.Client = Client
    monkeypatch.setitem(__import__("sys").modules, "hvac", fake)
    return captured


def test_vault_requires_url_and_token(monkeypatch):
    monkeypatch.delenv("VAULT_ADDR", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    with pytest.raises(ValueError):
        VaultSecretsBackend()


def test_vault_reads_field(monkeypatch):
    store = {("secret", "director-ai"): {"DIRECTOR_ADMIN_KEY": "vaulted"}}
    cap = _install_fake_hvac(monkeypatch, store)
    be = VaultSecretsBackend(url="http://v:8200", token="t")
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "vaulted"
    # client built lazily with the supplied connection params
    assert cap["client"].url == "http://v:8200"
    assert cap["client"].token == "t"


def test_vault_missing_field_is_none(monkeypatch):
    store = {("secret", "director-ai"): {"OTHER": "x"}}
    _install_fake_hvac(monkeypatch, store)
    be = VaultSecretsBackend(url="http://v", token="t")
    assert be.get_secret("DIRECTOR_ADMIN_KEY") is None


def test_vault_field_and_path_map(monkeypatch):
    store = {("kv", "licenses"): {"signing": "sig-value"}}
    _install_fake_hvac(monkeypatch, store)
    be = VaultSecretsBackend(
        url="http://v",
        token="t",
        mount="kv",
        path="director-ai",
        field_map={"DIRECTOR_LICENSE_SIGNING_KEY": "signing"},
        path_map={"DIRECTOR_LICENSE_SIGNING_KEY": "licenses"},
    )
    assert be.get_secret("DIRECTOR_LICENSE_SIGNING_KEY") == "sig-value"


def test_vault_env_fallback_for_url_token(monkeypatch):
    monkeypatch.setenv("VAULT_ADDR", "http://env-vault")
    monkeypatch.setenv("VAULT_TOKEN", "env-tok")
    store = {("secret", "director-ai"): {"DIRECTOR_ADMIN_KEY": "ok"}}
    cap = _install_fake_hvac(monkeypatch, store)
    be = VaultSecretsBackend()
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "ok"
    assert cap["client"].url == "http://env-vault"


def test_vault_client_built_once(monkeypatch):
    store = {("secret", "director-ai"): {"A": "1", "B": "2"}}
    cap = _install_fake_hvac(monkeypatch, store)
    be = VaultSecretsBackend(url="http://v", token="t")
    first = be.get_secret("A")
    second = be.get_secret("B")  # reuses the cached client (no rebuild)
    assert first == "1"
    assert second == "2"
    assert cap["client"] is be._get_client()


# --------------------------------------------------------------------------- #
# AWSSecretsManagerBackend (fake boto3 client injected)
# --------------------------------------------------------------------------- #
class _FakeAWSClient:
    def __init__(self, blobs):
        self._blobs = blobs

    def get_secret_value(self, *, SecretId):  # noqa: N803 - boto3 kwarg
        if SecretId not in self._blobs:
            raise KeyError(SecretId)
        return {"SecretString": self._blobs[SecretId]}


def test_aws_json_blob(monkeypatch):
    client = _FakeAWSClient(
        {"director-ai": json.dumps({"DIRECTOR_ADMIN_KEY": "aws-secret"})}
    )
    be = AWSSecretsManagerBackend(client=client)
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "aws-secret"
    assert be.get_secret("MISSING") is None


def test_aws_secret_id_map(monkeypatch):
    client = _FakeAWSClient({"admin-arn": "flat-secret"})
    be = AWSSecretsManagerBackend(
        client=client, secret_id_map={"DIRECTOR_ADMIN_KEY": "admin-arn"}
    )
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "flat-secret"


def test_aws_invalid_json_blob_is_none(monkeypatch):
    client = _FakeAWSClient({"director-ai": "not-json{{"})
    be = AWSSecretsManagerBackend(client=client)
    assert be.get_secret("DIRECTOR_ADMIN_KEY") is None


def test_aws_empty_string_is_none(monkeypatch):
    client = _FakeAWSClient({"director-ai": json.dumps({"DIRECTOR_ADMIN_KEY": ""})})
    be = AWSSecretsManagerBackend(client=client)
    assert be.get_secret("DIRECTOR_ADMIN_KEY") is None


def test_aws_empty_secret_string_is_none(monkeypatch):
    # SecretString itself empty -> short-circuit before JSON parse
    client = _FakeAWSClient({"director-ai": ""})
    be = AWSSecretsManagerBackend(client=client)
    assert be.get_secret("DIRECTOR_ADMIN_KEY") is None


def test_aws_lazy_boto3_import(monkeypatch):
    # Backend with no injected client builds one lazily from a stubbed boto3.
    built = {}

    def fake_boto3_client(service, region_name=None):
        built["service"] = service
        built["region"] = region_name
        return _FakeAWSClient(
            {"director-ai": json.dumps({"DIRECTOR_ADMIN_KEY": "lazy"})}
        )

    fake = types.ModuleType("boto3")
    fake.client = fake_boto3_client
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake)
    be = AWSSecretsManagerBackend(region_name="eu-central-1")
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "lazy"
    assert built == {"service": "secretsmanager", "region": "eu-central-1"}


# --------------------------------------------------------------------------- #
# AzureKeyVaultBackend (fake SecretClient injected)
# --------------------------------------------------------------------------- #
class _FakeAzureClient:
    def __init__(self, store):
        self._store = store

    def get_secret(self, name):
        if name not in self._store:
            raise RuntimeError("ResourceNotFound")
        return types.SimpleNamespace(value=self._store[name])


def test_azure_name_mapping(monkeypatch):
    # DIRECTOR_ADMIN_KEY -> director-admin-key
    client = _FakeAzureClient({"director-admin-key": "azure-secret"})
    be = AzureKeyVaultBackend(vault_url="https://v.vault.azure.net", client=client)
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "azure-secret"


def test_azure_explicit_name_map(monkeypatch):
    client = _FakeAzureClient({"adminkey": "x"})
    be = AzureKeyVaultBackend(
        client=client, name_map={"DIRECTOR_ADMIN_KEY": "adminkey"}
    )
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "x"


def test_azure_missing_returns_none(monkeypatch):
    client = _FakeAzureClient({})
    be = AzureKeyVaultBackend(client=client)
    assert be.get_secret("DIRECTOR_ADMIN_KEY") is None


def test_azure_requires_url_without_client(monkeypatch):
    monkeypatch.delenv("AZURE_KEY_VAULT_URL", raising=False)
    with pytest.raises(ValueError):
        AzureKeyVaultBackend()


def test_azure_lazy_sdk_import(monkeypatch):
    # Backend with no injected client builds one lazily from stubbed Azure SDKs.
    built = {}

    class _Cred:
        pass

    def _SecretClient(*, vault_url, credential):  # noqa: N802 - mimic SDK class
        built["vault_url"] = vault_url
        built["credential"] = credential
        return _FakeAzureClient({"director-admin-key": "lazy-azure"})

    sysmods = __import__("sys").modules
    azure_pkg = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = _Cred
    kv_pkg = types.ModuleType("azure.keyvault")
    secrets_mod = types.ModuleType("azure.keyvault.secrets")
    secrets_mod.SecretClient = _SecretClient
    monkeypatch.setitem(sysmods, "azure", azure_pkg)
    monkeypatch.setitem(sysmods, "azure.identity", identity_mod)
    monkeypatch.setitem(sysmods, "azure.keyvault", kv_pkg)
    monkeypatch.setitem(sysmods, "azure.keyvault.secrets", secrets_mod)

    be = AzureKeyVaultBackend(vault_url="https://v.vault.azure.net")
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "lazy-azure"
    assert built["vault_url"] == "https://v.vault.azure.net"
    assert isinstance(built["credential"], _Cred)


# --------------------------------------------------------------------------- #
# SecretsProvider — caching, TTL, required, invalidate, hydrate
# --------------------------------------------------------------------------- #
class _CountingBackend:
    def __init__(self, values):
        self.values = dict(values)
        self.calls = 0

    def get_secret(self, name):
        self.calls += 1
        return self.values.get(name)


def test_provider_default_is_env():
    assert isinstance(SecretsProvider().backend, EnvSecretsBackend)


def test_provider_get_and_default():
    be = _CountingBackend({"A": "1"})
    p = SecretsProvider(be)
    assert p.get("A") == "1"
    assert p.get("B") is None
    assert p.get("B", "fallback") == "fallback"


def test_provider_required_raises():
    p = SecretsProvider(_CountingBackend({}))
    with pytest.raises(SecretNotFoundError):
        p.get("MISSING", required=True)


def test_provider_caches_within_ttl():
    be = _CountingBackend({"A": "1"})
    p = SecretsProvider(be, cache_ttl_seconds=1000)
    p.get("A")
    p.get("A")
    assert be.calls == 1  # second served from cache


def test_provider_ttl_zero_disables_cache():
    be = _CountingBackend({"A": "1"})
    p = SecretsProvider(be, cache_ttl_seconds=0)
    p.get("A")
    p.get("A")
    assert be.calls == 2


def test_provider_ttl_expiry_refetches(monkeypatch):
    be = _CountingBackend({"A": "old"})
    p = SecretsProvider(be, cache_ttl_seconds=100)
    clock = {"t": 1000.0}
    monkeypatch.setattr(p, "_now", lambda: clock["t"])
    assert p.get("A") == "old"
    be.values["A"] = "new"  # rotate
    clock["t"] = 1050.0
    assert p.get("A") == "old"  # still cached
    clock["t"] = 1200.0
    assert p.get("A") == "new"  # TTL lapsed -> rotated value picked up


def test_provider_required_on_cached_none():
    be = _CountingBackend({})
    p = SecretsProvider(be, cache_ttl_seconds=1000)
    assert p.get("X") is None  # caches the None
    with pytest.raises(SecretNotFoundError):
        p.get("X", required=True)  # cached None + required -> raises


def test_provider_invalidate_one_and_all():
    be = _CountingBackend({"A": "1", "B": "2"})
    p = SecretsProvider(be, cache_ttl_seconds=1000)
    p.get("A")
    p.get("B")
    p.invalidate("A")
    p.get("A")
    assert be.calls == 3
    p.invalidate()
    p.get("A")
    p.get("B")
    assert be.calls == 5


def test_hydrate_environ_sets_missing(monkeypatch):
    for n in MANAGED_SECRET_NAMES:
        monkeypatch.delenv(n, raising=False)
    be = _CountingBackend({"DIRECTOR_ADMIN_KEY": "k", "OPENAI_API_KEY": "o"})
    loaded = SecretsProvider(be).hydrate_environ()
    assert set(loaded) == {"DIRECTOR_ADMIN_KEY", "OPENAI_API_KEY"}
    import os

    assert os.environ["DIRECTOR_ADMIN_KEY"] == "k"


def test_hydrate_environ_respects_existing(monkeypatch):
    monkeypatch.setenv("DIRECTOR_ADMIN_KEY", "from-env")
    be = _CountingBackend({"DIRECTOR_ADMIN_KEY": "from-backend"})
    loaded = SecretsProvider(be).hydrate_environ(["DIRECTOR_ADMIN_KEY"])
    assert loaded == []
    import os

    assert os.environ["DIRECTOR_ADMIN_KEY"] == "from-env"


def test_hydrate_environ_overwrite(monkeypatch):
    monkeypatch.setenv("DIRECTOR_ADMIN_KEY", "from-env")
    be = _CountingBackend({"DIRECTOR_ADMIN_KEY": "from-backend"})
    loaded = SecretsProvider(be).hydrate_environ(["DIRECTOR_ADMIN_KEY"], overwrite=True)
    assert loaded == ["DIRECTOR_ADMIN_KEY"]
    import os

    assert os.environ["DIRECTOR_ADMIN_KEY"] == "from-backend"


# --------------------------------------------------------------------------- #
# build_backend_from_env + rotation_guidance
# --------------------------------------------------------------------------- #
def test_build_default_env(monkeypatch):
    monkeypatch.delenv("DIRECTOR_SECRETS_BACKEND", raising=False)
    assert isinstance(build_backend_from_env(), EnvSecretsBackend)


def test_build_env_explicit_with_prefix(monkeypatch):
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "env")
    monkeypatch.setenv("DIRECTOR_SECRETS_PREFIX", "P_")
    be = build_backend_from_env()
    assert isinstance(be, EnvSecretsBackend)
    monkeypatch.setenv("P_DIRECTOR_ADMIN_KEY", "v")
    assert be.get_secret("DIRECTOR_ADMIN_KEY") == "v"


def test_build_vault(monkeypatch):
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "vault")
    monkeypatch.setenv("VAULT_ADDR", "http://v")
    monkeypatch.setenv("VAULT_TOKEN", "t")
    assert isinstance(build_backend_from_env(), VaultSecretsBackend)


def test_build_aws(monkeypatch):
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "aws")
    assert isinstance(build_backend_from_env(), AWSSecretsManagerBackend)


def test_build_azure(monkeypatch):
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "azure")
    monkeypatch.setenv("AZURE_KEY_VAULT_URL", "https://v.vault.azure.net")
    assert isinstance(build_backend_from_env(), AzureKeyVaultBackend)


def test_build_unknown_raises(monkeypatch):
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "bogus")
    with pytest.raises(ValueError):
        build_backend_from_env()


def test_rotation_guidance_covers_managed_secrets():
    guidance = rotation_guidance()
    assert guidance
    secrets_listed = {g["secret"] for g in guidance}
    # the rotatable, list-valued secrets must advertise overlap support
    overlap = {g["secret"] for g in guidance if g["overlap_supported"]}
    assert "DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS" in overlap
    assert "DIRECTOR_API_KEYS" in overlap
    # every documented secret is a known managed secret
    assert secrets_listed.issubset(set(MANAGED_SECRET_NAMES))
    # every record carries a non-empty note
    assert all(g["note"] for g in guidance)


def test_module_exports_complete():
    for name in sec.__all__:
        assert hasattr(sec, name)


# --------------------------------------------------------------------------- #
# hydrate_managed_secrets (startup bridge)
# --------------------------------------------------------------------------- #
def test_hydrate_managed_secrets_noop_for_env(monkeypatch):
    monkeypatch.delenv("DIRECTOR_SECRETS_BACKEND", raising=False)
    assert sec.hydrate_managed_secrets() == []
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "env")
    assert sec.hydrate_managed_secrets() == []


def test_hydrate_managed_secrets_loads_from_backend(monkeypatch):
    for n in MANAGED_SECRET_NAMES:
        monkeypatch.delenv(n, raising=False)
    blob = json.dumps({"DIRECTOR_ADMIN_KEY": "from-aws", "OPENAI_API_KEY": "k"})
    fake = types.ModuleType("boto3")
    fake.client = lambda service, region_name=None: _FakeAWSClient(
        {"director-ai": blob}
    )
    monkeypatch.setitem(__import__("sys").modules, "boto3", fake)
    monkeypatch.setenv("DIRECTOR_SECRETS_BACKEND", "aws")
    loaded = sec.hydrate_managed_secrets()
    assert set(loaded) == {"DIRECTOR_ADMIN_KEY", "OPENAI_API_KEY"}
    import os

    assert os.environ["DIRECTOR_ADMIN_KEY"] == "from-aws"
