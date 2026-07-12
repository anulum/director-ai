# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Focused serve/proxy CLI path tests."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from director_ai import _cli_serve


class FakeDirectorConfig:
    from_env_calls = 0
    from_profile_calls: list[str] = []

    def __init__(self, **kwargs):
        self.profile = kwargs.pop("profile", "default")
        self.mode = kwargs.pop("mode", "general")
        self.server_host = kwargs.pop("server_host", "")
        self.server_port = kwargs.pop("server_port", 0)
        self.cors_origins = kwargs.pop("cors_origins", "")
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_env(cls):
        cls.from_env_calls += 1
        return cls(profile="env-profile")

    @classmethod
    def from_profile(cls, profile):
        cls.from_profile_calls.append(profile)
        return cls(profile=profile)


def _install_fake_config(monkeypatch):
    FakeDirectorConfig.from_env_calls = 0
    FakeDirectorConfig.from_profile_calls = []
    fake_config = ModuleType("director_ai.core.config")
    fake_config.DirectorConfig = FakeDirectorConfig
    monkeypatch.setitem(sys.modules, "director_ai.core.config", fake_config)


def test_serve_rejects_invalid_mode(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_serve._cmd_serve(["--mode", "unsafe"])

    assert exc_info.value.code == 1
    assert "general" in capsys.readouterr().out


def test_serve_http_applies_mode_cors_and_single_worker(monkeypatch, capsys):
    _install_fake_config(monkeypatch)
    uvicorn_calls: list[dict[str, object]] = []
    created_configs: list[FakeDirectorConfig] = []
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: uvicorn_calls.append(
        {"app": app, **kwargs},
    )
    fake_server = ModuleType("director_ai.server")

    def fake_create_app(config):
        created_configs.append(config)
        return {"profile": config.profile, "mode": config.mode}

    fake_server.create_app = fake_create_app
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "director_ai.server", fake_server)

    _cli_serve._cmd_serve(
        [
            "--mode",
            "grounded",
            "--host",
            "127.0.0.1",
            "--port",
            "9099",
            "--cors-origins",
            "https://console.example",
            "--ignored-token",
        ],
    )

    assert FakeDirectorConfig.from_env_calls == 1
    config = created_configs[0]
    assert config.mode == "grounded"
    assert config.server_host == "127.0.0.1"
    assert config.server_port == 9099
    assert config.cors_origins == "https://console.example"
    assert uvicorn_calls == [
        {
            "app": {"profile": "env-profile", "mode": "grounded"},
            "host": "127.0.0.1",
            "port": 9099,
        },
    ]
    assert "env-profile" in capsys.readouterr().out


def test_serve_dev_flag_forces_dataclass_config_to_dev(monkeypatch):
    @dataclass
    class DataclassConfig:
        profile: str = "dataclass-profile"
        mode: str = "general"
        server_host: str = ""
        server_port: int = 0
        cors_origins: str = ""
        production_mode: bool = True

        @classmethod
        def from_env(cls):
            return cls()

    fake_config = ModuleType("director_ai.core.config")
    fake_config.DirectorConfig = DataclassConfig
    fake_uvicorn = ModuleType("uvicorn")
    uvicorn_calls: list[dict[str, object]] = []
    fake_uvicorn.run = lambda app, **kwargs: uvicorn_calls.append(
        {"app": app, **kwargs},
    )
    fake_server = ModuleType("director_ai.server")
    fake_server.create_app = lambda config: {
        "production_mode": config.production_mode,
        "host": config.server_host,
    }
    monkeypatch.setitem(sys.modules, "director_ai.core.config", fake_config)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "director_ai.server", fake_server)

    _cli_serve._cmd_serve(["--dev"])

    assert uvicorn_calls == [
        {
            "app": {"production_mode": False, "host": "127.0.0.1"},
            "host": "127.0.0.1",
            "port": 8080,
        },
    ]


def test_serve_production_flag_refuses_unhardened_config(monkeypatch, capsys):
    _install_fake_config(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        _cli_serve._cmd_serve(["--production"])

    assert exc_info.value.code == 1
    assert "--production requires a hardened config" in capsys.readouterr().out


def test_serve_rejects_invalid_port_worker_and_transport(capsys):
    with pytest.raises(SystemExit) as port_exc:
        _cli_serve._cmd_serve(["--port", "not-a-port"])

    assert port_exc.value.code == 1
    assert "invalid port number: not-a-port" in capsys.readouterr().out

    with pytest.raises(SystemExit) as worker_exc:
        _cli_serve._cmd_serve(["--workers", "0"])

    assert worker_exc.value.code == 1
    assert "invalid worker count: 0" in capsys.readouterr().out

    with pytest.raises(SystemExit) as transport_exc:
        _cli_serve._cmd_serve(["--transport", "udp"])

    assert transport_exc.value.code == 1
    assert "http" in capsys.readouterr().out


def test_serve_multi_worker_sets_environment_and_factory_target(monkeypatch):
    _install_fake_config(monkeypatch)
    calls: list[dict[str, object]] = []
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: calls.append({"app": app, **kwargs})
    fake_server = ModuleType("director_ai.server")
    fake_server.create_app = lambda config: {"config": config}
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "director_ai.server", fake_server)
    monkeypatch.delenv("DIRECTOR_PROFILE", raising=False)

    _cli_serve._cmd_serve(
        [
            "--profile",
            "fast",
            "--host",
            "localhost",
            "--port",
            "9100",
            "--workers",
            "3",
        ],
    )

    assert FakeDirectorConfig.from_profile_calls == ["fast"]
    assert calls == [
        {
            "app": "director_ai.server:create_app",
            "factory": True,
            "host": "localhost",
            "port": 9100,
            "workers": 3,
        },
    ]
    assert sys.modules["os"].environ["DIRECTOR_PROFILE"] == "fast"
    assert sys.modules["os"].environ["DIRECTOR_SERVER_HOST"] == "localhost"
    assert sys.modules["os"].environ["DIRECTOR_SERVER_PORT"] == "9100"


def test_serve_grpc_starts_and_waits(monkeypatch, capsys):
    _install_fake_config(monkeypatch)
    events: list[str] = []
    fake_grpc = ModuleType("director_ai.grpc_server")

    class FakeServer:
        def start(self):
            events.append("start")

        def wait_for_termination(self):
            events.append("wait")

    fake_grpc.create_grpc_server = lambda config, max_workers, port: (
        events.append(f"{config.profile}:{max_workers}:{port}") or FakeServer()
    )
    monkeypatch.setitem(sys.modules, "director_ai.grpc_server", fake_grpc)

    _cli_serve._cmd_serve(["--transport", "grpc", "--workers", "2", "--port", "50051"])

    assert events == ["env-profile:2:50051", "start", "wait"]
    assert "gRPC server" in capsys.readouterr().out


def test_proxy_rejects_unknown_failure_mode(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_serve._cmd_proxy(["--on-fail", "panic"])

    assert exc_info.value.code == 1
    assert "reject" in capsys.readouterr().out


def test_proxy_builds_app_from_flags_and_config_env(monkeypatch, capsys):
    _install_fake_config(monkeypatch)
    proxy_calls: list[dict[str, object]] = []
    uvicorn_calls: list[dict[str, object]] = []
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kwargs: uvicorn_calls.append(
        {"app": app, **kwargs},
    )
    fake_proxy = ModuleType("director_ai.proxy")
    fake_proxy.create_proxy_app = lambda **kwargs: (
        proxy_calls.append(kwargs)
        or {
            "proxy": True,
        }
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "director_ai.proxy", fake_proxy)

    _cli_serve._cmd_proxy(
        [
            "--port",
            "8088",
            "--threshold",
            "0.42",
            "--facts",
            "facts.jsonl",
            "--facts-root",
            "/kb",
            "--upstream-url",
            "http://upstream.local",
            "--on-fail",
            "warn",
            "--api-keys",
            "k1, k2 ,,",
            "--allow-http-upstream",
            "--audit-db",
            "audit.sqlite",
            "--config-env",
        ],
    )

    assert proxy_calls == [
        {
            "threshold": 0.42,
            "facts_path": "facts.jsonl",
            "facts_root": "/kb",
            "upstream_url": "http://upstream.local",
            "on_fail": "warn",
            "api_keys": ["k1", "k2"],
            "allow_http_upstream": True,
            "audit_db": "audit.sqlite",
            "config": proxy_calls[0]["config"],
            "moderations": "local",
        },
    ]
    assert isinstance(proxy_calls[0]["config"], FakeDirectorConfig)
    assert uvicorn_calls == [
        {"app": {"proxy": True}, "host": "0.0.0.0", "port": 8088},
    ]
    out = capsys.readouterr().out
    assert "threshold=0.42" in out


def test_proxy_rejects_unknown_moderations_mode(capsys):
    with pytest.raises(SystemExit) as exc_info:
        _cli_serve._cmd_proxy(["--moderations", "panic"])

    assert exc_info.value.code == 1
    assert "local" in capsys.readouterr().out


def test_proxy_moderations_flag_is_forwarded(monkeypatch):
    _install_fake_config(monkeypatch)
    proxy_calls: list[dict[str, object]] = []
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda *_args, **_kwargs: None
    fake_proxy = ModuleType("director_ai.proxy")
    fake_proxy.create_proxy_app = lambda **kwargs: (
        proxy_calls.append(kwargs)
        or {
            "proxy": True,
        }
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "director_ai.proxy", fake_proxy)

    _cli_serve._cmd_proxy(["--moderations", "upstream"])

    assert proxy_calls[0]["moderations"] == "upstream"


def test_proxy_defaults_do_not_load_environment_config(monkeypatch):
    _install_fake_config(monkeypatch)
    proxy_calls: list[dict[str, object]] = []
    fake_uvicorn = ModuleType("uvicorn")
    fake_uvicorn.run = lambda *_args, **_kwargs: None
    fake_proxy = ModuleType("director_ai.proxy")
    fake_proxy.create_proxy_app = lambda **kwargs: (
        proxy_calls.append(kwargs)
        or {
            "proxy": True,
        }
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "director_ai.proxy", fake_proxy)

    _cli_serve._cmd_proxy(["--ignored-token"])

    assert FakeDirectorConfig.from_env_calls == 0
    assert proxy_calls[0]["config"] is None
    assert proxy_calls[0]["api_keys"] is None
    assert proxy_calls[0]["on_fail"] == "reject"


def test_stress_test_json_uses_streaming_kernel_and_reports_halts(
    monkeypatch,
    capsys,
):
    calls: list[list[str]] = []
    fake_streaming = ModuleType("director_ai.core.runtime.streaming")

    class FakeStreamingKernel:
        def stream_tokens(self, tokens, coherence_cb):
            calls.append(tokens)
            score = coherence_cb(tokens[0])
            return SimpleNamespace(
                halted=len(calls) == 1 and score > 0,
                token_count=len(tokens),
            )

    fake_streaming.StreamingKernel = FakeStreamingKernel
    monkeypatch.setitem(
        sys.modules, "director_ai.core.runtime.streaming", fake_streaming
    )

    _cli_serve._cmd_stress_test(
        [
            "--streams",
            "2",
            "--tokens-per-stream",
            "3",
            "--concurrency",
            "1",
            "--json",
            "--ignored-token",
        ],
    )

    assert calls == [["tok0", "tok1", "tok2"], ["tok0", "tok1", "tok2"]]
    out = capsys.readouterr().out
    assert '"streams": 2' in out
    assert '"tokens_per_stream": 3' in out
    assert '"halt_rate": 0.5' in out


def test_stress_test_text_report(monkeypatch, capsys):
    fake_streaming = ModuleType("director_ai.core.runtime.streaming")

    class FakeStreamingKernel:
        def stream_tokens(self, tokens, _coherence_cb):
            return SimpleNamespace(halted=False, token_count=len(tokens))

    fake_streaming.StreamingKernel = FakeStreamingKernel
    monkeypatch.setitem(
        sys.modules, "director_ai.core.runtime.streaming", fake_streaming
    )

    _cli_serve._cmd_stress_test(
        ["--streams", "1", "--tokens-per-stream", "2", "--concurrency", "1"],
    )

    out = capsys.readouterr().out
    assert "Streams:     1" in out
    assert "Halt rate:   0.00%" in out
    assert "Latency p95:" in out


class TestRunMode:
    """Explicit --dev/--production run-mode enforcement (real DirectorConfig)."""

    def _hardened_kwargs(self):
        return {
            "api_keys": '["sk-test"]',
            "llm_api_url": "https://llm.internal.example/v1",
            "knowledge_write_hmac_keys": (
                '{"kid-1":"signing-secret-at-least-32-chars-xx"}'
            ),
        }

    def test_dev_forces_production_off(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(production_mode=True, **self._hardened_kwargs())
        resolved = _cli_serve._apply_run_mode(cfg, "dev")
        assert resolved.production_mode is False

    def test_production_with_keys(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(**self._hardened_kwargs())
        resolved = _cli_serve._apply_run_mode(cfg, "production")
        assert resolved.production_mode is True

    def test_production_without_keys_exits(self, capsys):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig()
        with pytest.raises(SystemExit) as exc:
            _cli_serve._apply_run_mode(cfg, "production")
        assert exc.value.code == 1
        assert "--production requires a hardened config" in capsys.readouterr().out

    def test_implicit_production_refused(self, capsys):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(production_mode=True, **self._hardened_kwargs())
        with pytest.raises(SystemExit) as exc:
            _cli_serve._apply_run_mode(cfg, "")
        assert exc.value.code == 1
        assert "Refusing to start in production implicitly" in capsys.readouterr().out

    def test_dev_config_passes_through(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig()
        resolved = _cli_serve._apply_run_mode(cfg, "")
        assert resolved.production_mode is False


class TestResolveBindHost:
    """The dev server binds to loopback unless a host is given or it is prod."""

    @staticmethod
    def _hardened_kwargs():
        return {
            "api_keys": ["writer-key"],
            "llm_api_url": "https://llm.internal.example/v1",
            "knowledge_write_hmac_keys": (
                '{"kid-1":"signing-secret-at-least-32-chars-xx"}'
            ),
        }

    def test_dev_defaults_to_loopback(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.delenv("DIRECTOR_SERVER_HOST", raising=False)
        cfg = DirectorConfig()  # production_mode False, secure loopback default
        assert _cli_serve._resolve_bind_host(cfg, "", False) == "127.0.0.1"

    def test_explicit_host_wins_in_dev(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.delenv("DIRECTOR_SERVER_HOST", raising=False)
        cfg = DirectorConfig()
        assert (
            _cli_serve._resolve_bind_host(cfg, "0.0.0.0", True) == "0.0.0.0"  # noqa: S104
        )

    def test_explicit_host_wins_for_custom_address(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.delenv("DIRECTOR_SERVER_HOST", raising=False)
        cfg = DirectorConfig()
        assert _cli_serve._resolve_bind_host(cfg, "192.0.2.10", True) == "192.0.2.10"

    def test_env_host_honoured_in_dev(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        # The container image sets DIRECTOR_SERVER_HOST=0.0.0.0; a dev-mode server
        # must honour it (a container binds all interfaces; exposure is the port
        # mapping's job) rather than overriding it with loopback.
        monkeypatch.setenv("DIRECTOR_SERVER_HOST", "0.0.0.0")  # noqa: S104
        cfg = DirectorConfig()
        assert _cli_serve._resolve_bind_host(cfg, "", False) == "0.0.0.0"  # noqa: S104

    def test_explicit_host_overrides_env(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.setenv("DIRECTOR_SERVER_HOST", "0.0.0.0")  # noqa: S104
        cfg = DirectorConfig()
        assert _cli_serve._resolve_bind_host(cfg, "10.0.0.9", True) == "10.0.0.9"

    def test_production_defaults_to_all_interfaces(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.delenv("DIRECTOR_SERVER_HOST", raising=False)
        cfg = DirectorConfig(production_mode=True, **self._hardened_kwargs())
        # With the secure loopback default config, production still exposes all
        # interfaces for a reverse proxy.
        assert _cli_serve._resolve_bind_host(cfg, "", False) == "0.0.0.0"  # noqa: S104

    def test_production_honours_explicitly_configured_host(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.delenv("DIRECTOR_SERVER_HOST", raising=False)
        cfg = DirectorConfig(
            production_mode=True,
            server_host="10.0.0.7",
            **self._hardened_kwargs(),
        )
        assert _cli_serve._resolve_bind_host(cfg, "", False) == "10.0.0.7"

    def test_production_respects_explicit_host(self, monkeypatch):
        from director_ai.core.config import DirectorConfig

        monkeypatch.delenv("DIRECTOR_SERVER_HOST", raising=False)
        cfg = DirectorConfig(production_mode=True, **self._hardened_kwargs())
        assert _cli_serve._resolve_bind_host(cfg, "10.0.0.5", True) == "10.0.0.5"
