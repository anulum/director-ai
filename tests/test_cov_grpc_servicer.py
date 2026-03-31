# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle coverage for gRPC servicer RPCs pipeline (STRONG)."""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import director_ai as _director_pkg

_SENTINEL = object()


@contextlib.contextmanager
def _grpc_context(mods):
    """Patch sys.modules AND clear director_ai package-level submodule caches.

    Python caches submodules as attributes on the parent package. ``from . import
    director_pb2_grpc`` checks ``getattr(director_ai, 'director_pb2_grpc')``
    before ``sys.modules``, so we must clear those attrs for the mock to take
    effect. Only clears attrs for modules actually present in *mods*.
    """
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


def _build_grpc_mocks(*, with_proto=False, with_reflection=False):
    grpc = MagicMock()
    grpc.__version__ = "1.78.0"
    server = MagicMock()
    grpc.server.return_value = server
    grpc.ServerInterceptor = type("ServerInterceptor", (), {})

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
        # Prevent real proto modules from importing with mocked grpc
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


class TestServicerReview:
    def test_review_rpc(self):
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50199)
            servicer = _get_servicer(mods)

            request = SimpleNamespace(prompt="sky?", response="The sky is blue.")
            resp = servicer.Review(request, MagicMock())
            assert hasattr(resp, "coherence")

    def test_process_rpc(self):
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50200)
            servicer = _get_servicer(mods)

            request = SimpleNamespace(prompt="What is 2+2?")
            resp = servicer.Process(request, MagicMock())
            assert hasattr(resp, "output")

    def test_review_batch_rpc(self):
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50201)
            servicer = _get_servicer(mods)

            req1 = SimpleNamespace(prompt="sky?", response="The sky is blue.")
            req2 = SimpleNamespace(prompt="sun?", response="The sun is hot.")
            request = SimpleNamespace(requests=[req1, req2])
            resp = servicer.ReviewBatch(request, MagicMock())
            assert hasattr(resp, "responses")

    def test_review_batch_too_large(self):
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50202)
            servicer = _get_servicer(mods)

            request = SimpleNamespace(requests=[MagicMock()] * 1001)
            ctx = MagicMock()
            servicer.ReviewBatch(request, ctx)
            ctx.abort.assert_called_once()


class TestProtoFallback:
    def test_proto_stubs_present(self):
        grpc_mock, server, mods = _build_grpc_mocks(with_proto=True)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            result = create_grpc_server(port=50203)
            assert result is server


class TestReflection:
    def test_reflection_enabled(self):
        grpc_mock, server, mods = _build_grpc_mocks(
            with_proto=True,
            with_reflection=True,
        )

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(port=50204)
            reflection_mod = mods["grpc_reflection.v1alpha.reflection"]
            reflection_mod.enable_server_reflection.assert_called_once()


class TestTLSPort:
    def test_tls_credentials(self, tmp_path):
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_bytes(b"CERT")
        key.write_bytes(b"KEY")

        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(
                port=50205,
                tls_cert_path=str(cert),
                tls_key_path=str(key),
            )
            server.add_secure_port.assert_called_once()
            server.add_insecure_port.assert_not_called()


class TestAuthInterceptor:
    def test_auth_no_keys_passes(self):
        from director_ai.core.config import DirectorConfig

        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)
        cfg = DirectorConfig(use_nli=False, api_keys=[])

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50206)
            interceptor = _get_interceptor(grpc_mock)

            handler_details = MagicMock()
            handler_details.invocation_metadata = []
            continuation = MagicMock()

            interceptor.intercept_service(continuation, handler_details)
            continuation.assert_called_once()

    def test_auth_valid_key_passes(self):
        from director_ai.core.config import DirectorConfig

        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)
        cfg = DirectorConfig(use_nli=False, api_keys=["secret-key-123"])

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50207)
            interceptor = _get_interceptor(grpc_mock)

            handler_details = MagicMock()
            handler_details.invocation_metadata = [("x-api-key", "secret-key-123")]
            continuation = MagicMock()

            interceptor.intercept_service(continuation, handler_details)
            continuation.assert_called_once()

    def test_auth_invalid_key_rejects(self):
        from director_ai.core.config import DirectorConfig

        grpc_mock, server, mods = _build_grpc_mocks(with_proto=False)
        cfg = DirectorConfig(use_nli=False, api_keys=["secret-key-123"])

        with _grpc_context(mods):
            from director_ai.grpc_server import create_grpc_server

            create_grpc_server(config=cfg, port=50208)
            interceptor = _get_interceptor(grpc_mock)

            handler_details = MagicMock()
            handler_details.invocation_metadata = [("x-api-key", "wrong")]
            continuation = MagicMock()

            interceptor.intercept_service(continuation, handler_details)
            continuation.assert_not_called()
