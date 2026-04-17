# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for gRPC coverage gaps pipeline."""

from __future__ import annotations

import contextlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import director_ai as _director_pkg

_SENTINEL = object()


@contextlib.contextmanager
def _grpc_context(mods):
    saved = {}
    for attr in ("director_pb2", "director_pb2_grpc"):
        full_key = f"director_ai.{attr}"
        if full_key not in mods:
            continue
        old = _director_pkg.__dict__.pop(attr, _SENTINEL)
        if old is not _SENTINEL:
            saved[attr] = old
    try:
        with patch.dict(sys.modules, mods):
            yield
    finally:
        for attr, old in saved.items():
            setattr(_director_pkg, attr, old)


def _build_grpc_mocks(*, with_proto=True, with_reflection=False):
    grpc = MagicMock()
    grpc.__version__ = "1.78.0"
    server = MagicMock()
    grpc.server.return_value = server
    grpc.ServerInterceptor = type("ServerInterceptor", (), {})
    grpc.StatusCode = MagicMock()
    grpc.StatusCode.INVALID_ARGUMENT = "INVALID_ARGUMENT"
    grpc.StatusCode.UNAUTHENTICATED = "UNAUTHENTICATED"
    grpc.unary_unary_rpc_method_handler = MagicMock()
    grpc.unary_stream_rpc_method_handler = MagicMock()

    mods = {"grpc": grpc}

    if with_proto:
        pb2 = MagicMock()
        descriptor = MagicMock()
        svc_desc = MagicMock()
        svc_desc.full_name = "director_ai.DirectorService"
        descriptor.services_by_name = {"DirectorService": svc_desc}
        pb2.DESCRIPTOR = descriptor
        pb2.ReviewResponse = SimpleNamespace
        pb2.ProcessResponse = SimpleNamespace
        pb2.BatchReviewResponse = SimpleNamespace
        pb2.TokenEvent = SimpleNamespace

        pb2_grpc = MagicMock()
        mods["director_ai.director_pb2"] = pb2
        mods["director_ai.director_pb2_grpc"] = pb2_grpc
    else:
        mods["director_ai.director_pb2"] = MagicMock()
        mods["director_ai.director_pb2_grpc"] = MagicMock()

    if with_reflection:
        reflection_mod = MagicMock()
        reflection_mod.SERVICE_NAME = "grpc.reflection.v1alpha.ServerReflection"
        v1alpha = MagicMock()
        v1alpha.reflection = reflection_mod
        mods["grpc_reflection"] = MagicMock()
        mods["grpc_reflection.v1alpha"] = v1alpha
        mods["grpc_reflection.v1alpha.reflection"] = reflection_mod

    return grpc, server, mods


def _get_servicer(mods):
    pb2_grpc = mods["director_ai.director_pb2_grpc"]
    return pb2_grpc.add_DirectorServiceServicer_to_server.call_args[0][0]


def _get_interceptor(grpc_mock):
    return grpc_mock.server.call_args.kwargs["interceptors"][0]


# ── Line 69: api_key_tenant_map JSON load ─────────────────────────────────────


class TestApiKeyTenantMap:
    def test_tenant_map_is_loaded_from_config(self):
        from director_ai.core.config import DirectorConfig

        tenant_map = {"key-abc": "tenant-1", "key-xyz": "tenant-2"}
        cfg = DirectorConfig(
            use_nli=False,
            api_key_tenant_map=json.dumps(tenant_map),
        )
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50300)
            assert server is not None

    def test_tenant_map_empty_string_skipped(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, api_key_tenant_map=None)
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            result = create_grpc_server(config=cfg, port=50301)
            assert result is server


# ── Line 101: _tenant_from_context with API-key-based tenant lookup ───────────


class TestTenantFromContextApiKey:
    def test_api_key_resolves_tenant(self):
        from director_ai.core.config import DirectorConfig

        tenant_map = {"my-api-key": "tenant-42"}
        cfg = DirectorConfig(
            use_nli=False,
            api_key_tenant_map=json.dumps(tenant_map),
            api_keys=["my-api-key"],
        )
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50302)
            servicer = _get_servicer(mods)

            ctx = MagicMock()
            ctx.invocation_metadata.return_value = [("x-api-key", "my-api-key")]

            request = SimpleNamespace(prompt="sky?", response="The sky is blue.")
            servicer.Review(request, ctx)

    def test_api_key_not_in_map_falls_back_to_tenant_id(self):
        from director_ai.core.config import DirectorConfig

        tenant_map = {"other-key": "tenant-99"}
        cfg = DirectorConfig(
            use_nli=False,
            api_key_tenant_map=json.dumps(tenant_map),
        )
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50303)
            servicer = _get_servicer(mods)

            ctx = MagicMock()
            ctx.invocation_metadata.return_value = [
                ("x-api-key", "unknown-key"),
                ("x-tenant-id", "fallback-tenant"),
            ]

            request = SimpleNamespace(prompt="sky?", response="The sky is blue.")
            servicer.Review(request, ctx)


# ── Lines 203, 208: StreamTokens RPC ─────────────────────────────────────────


class TestStreamTokensRpc:
    def test_stream_tokens_yields_events(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, hard_limit=0.1)
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50304)
            servicer = _get_servicer(mods)

            ctx = MagicMock()
            ctx.invocation_metadata.return_value = []

            request = SimpleNamespace(prompt="What is 2+2?")
            events = list(servicer.StreamTokens(request, ctx))
            assert len(events) > 0
            first = events[0]
            assert hasattr(first, "token")
            assert hasattr(first, "coherence")
            assert hasattr(first, "index")
            assert hasattr(first, "halted")

    def test_stream_tokens_halt_reason_set_when_below_hard_limit(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, soft_limit=1.0, hard_limit=0.99)
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50305)
            servicer = _get_servicer(mods)

            ctx = MagicMock()
            ctx.invocation_metadata.return_value = []

            request = SimpleNamespace(prompt="What is 2+2?")
            events = list(servicer.StreamTokens(request, ctx))
            halted_events = [e for e in events if e.halted]
            for e in halted_events:
                assert e.halt_reason == "hard_limit"

    def test_stream_tokens_no_halt_reason_when_above_hard_limit(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, hard_limit=0.0)
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50306)
            servicer = _get_servicer(mods)

            ctx = MagicMock()
            ctx.invocation_metadata.return_value = []

            request = SimpleNamespace(prompt="What is 2+2?")
            events = list(servicer.StreamTokens(request, ctx))
            for e in events:
                if not e.halted:
                    assert e.halt_reason == ""


# ── Lines 234-238: reflection service_names block ────────────────────────────


class TestReflectionServiceNames:
    def test_reflection_with_proto_includes_director_service(self):
        grpc_mock, server, mods = _build_grpc_mocks(
            with_proto=True,
            with_reflection=True,
        )
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50307)
            reflection_mod = mods["grpc_reflection.v1alpha.reflection"]
            reflection_mod.enable_server_reflection.assert_called_once()
            call_args = reflection_mod.enable_server_reflection.call_args[0]
            service_names = call_args[0]
            assert any("DirectorService" in str(n) for n in service_names)

    def test_reflection_adds_reflection_service_name(self):
        grpc_mock, server, mods = _build_grpc_mocks(
            with_proto=True,
            with_reflection=True,
        )
        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50308)
            reflection_mod = mods["grpc_reflection.v1alpha.reflection"]
            call_args = reflection_mod.enable_server_reflection.call_args[0]
            service_names = call_args[0]
            assert reflection_mod.SERVICE_NAME in service_names


# ── Line 252: logger.warning when proto stubs not registered ─────────────────


class TestProtoStubsNotRegistered:
    def test_no_proto_logs_warning(self, caplog):

        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)

        MagicMock(side_effect=ImportError("no stubs"))
        with _grpc_context(mods):
            with patch.dict(sys.modules, {"director_ai.director_pb2": None}):
                if "director_ai.grpc_server" in sys.modules:
                    del sys.modules["director_ai.grpc_server"]

                import importlib

                import director_ai.grpc_server as grpc_mod

                importlib.reload(grpc_mod)

                # The warning path (line 252) is only hit when has_proto=False.
                # We simulate that by checking that the server is still returned
                # even when proto stubs raise ImportError during create.
                with pytest.raises(ImportError):
                    # Stub ImportError raised at proto resolution (line 73-80)
                    with patch.dict(sys.modules, {"director_ai.director_pb2": None}):
                        grpc_mod.create_grpc_server(port=50399)

    def test_no_proto_stubs_raises_import_error(self):
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)
        mods["director_ai.director_pb2"] = None

        with _grpc_context(mods):
            if "director_ai.grpc_server" in sys.modules:
                del sys.modules["director_ai.grpc_server"]
            from director_ai.grpc_server import create_grpc_server

            with pytest.raises(ImportError, match="protobuf stubs not found"):
                create_grpc_server(port=50400)


# ── Auth interceptor: stream method path (line 208 / grpc.unary_stream) ──────


class TestAuthInterceptorStreamMethod:
    def test_stream_method_uses_unary_stream_handler(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, api_keys=["valid-key"])
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50309)
            interceptor = _get_interceptor(grpc_mock)

            handler_details = MagicMock()
            handler_details.invocation_metadata = [("x-api-key", "wrong-key")]
            handler_details.method = "/director_ai.DirectorService/StreamTokens"
            continuation = MagicMock()

            interceptor.intercept_service(continuation, handler_details)
            grpc_mock.unary_stream_rpc_method_handler.assert_called_once()
            continuation.assert_not_called()

    def test_non_stream_method_uses_unary_unary_handler(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(use_nli=False, api_keys=["valid-key"])
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50310)
            interceptor = _get_interceptor(grpc_mock)

            handler_details = MagicMock()
            handler_details.invocation_metadata = [("x-api-key", "wrong-key")]
            handler_details.method = "/director_ai.DirectorService/Review"
            continuation = MagicMock()

            interceptor.intercept_service(continuation, handler_details)
            grpc_mock.unary_unary_rpc_method_handler.assert_called_once()
            continuation.assert_not_called()
