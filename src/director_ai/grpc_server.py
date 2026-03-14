# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — gRPC Transport
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""gRPC server for Director-Class AI.

Usage::

    from director_ai.grpc_server import create_grpc_server
    server = create_grpc_server(config)
    server.start()
    server.wait_for_termination()

Requires ``pip install director-ai[grpc]``.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading

from .core.config import DirectorConfig

logger = logging.getLogger("DirectorAI.gRPC")


def create_grpc_server(
    config: DirectorConfig | None = None,
    max_workers: int = 4,
    port: int = 50051,
    tls_cert_path: str | None = None,
    tls_key_path: str | None = None,
):
    """Create and return a gRPC server (not yet started).

    Raises ImportError with install instructions if grpcio is missing.

    When *tls_cert_path* and *tls_key_path* are provided, the server
    binds a secure port with TLS.  Otherwise it falls back to an
    insecure port.
    """
    try:
        from concurrent import futures

        import grpc
    except ImportError as exc:
        raise ImportError(
            "gRPC transport requires grpcio. "
            "Install with: pip install director-ai[grpc]",
        ) from exc

    cfg = config or DirectorConfig.from_env()

    from .core.agent import CoherenceAgent

    store = cfg.build_store()
    scorer = cfg.build_scorer(store=store)
    agent = CoherenceAgent(_scorer=scorer, _store=store)

    # Resolve proto message factories
    try:
        from . import director_pb2
    except ImportError as exc:
        raise ImportError(
            "gRPC protobuf stubs not found. "
            "Run: python -m grpc_tools.protoc -Iproto "
            "--python_out=src/director_ai --grpc_python_out=src/director_ai "
            "proto/director.proto",
        ) from exc

    review_resp = director_pb2.ReviewResponse  # type: ignore[attr-defined]
    process_resp = director_pb2.ProcessResponse  # type: ignore[attr-defined]
    batch_resp = director_pb2.BatchReviewResponse  # type: ignore[attr-defined]
    token_evt = director_pb2.TokenEvent  # type: ignore[attr-defined]
    has_proto = True

    # Shared background event loop for async streaming RPCs
    _bg_loop = asyncio.new_event_loop()
    _bg_thread = threading.Thread(
        target=_bg_loop.run_forever,
        daemon=True,
        name="grpc-async",
    )
    _bg_thread.start()

    class DirectorServicer:
        """Implements the DirectorService RPC methods."""

        def Review(self, request, context):  # noqa: N802
            approved, score = scorer.review(request.prompt, request.response)
            return review_resp(
                approved=approved,
                coherence=score.score,
                h_logical=score.h_logical,
                h_factual=score.h_factual,
                warning=score.warning,
            )

        def Process(self, request, context):  # noqa: N802
            result = agent.process(request.prompt)
            return process_resp(
                output=result.output,
                coherence=result.coherence.score if result.coherence else 0.0,
                halted=result.halted,
                candidates_evaluated=result.candidates_evaluated,
                warning=result.coherence.warning if result.coherence else False,
                fallback_used=result.fallback_used,
            )

        def ReviewBatch(self, request, context):  # noqa: N802
            if len(request.requests) > 1000:
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"batch too large: {len(request.requests)} > 1000",
                )
                return batch_resp(responses=[])
            responses = []
            for req in request.requests:
                approved, score = scorer.review(req.prompt, req.response)
                responses.append(
                    review_resp(
                        approved=approved,
                        coherence=score.score,
                        h_logical=score.h_logical,
                        h_factual=score.h_factual,
                        warning=score.warning,
                    ),
                )
            return batch_resp(responses=responses)

        def StreamTokens(self, request, context):  # noqa: N802
            import queue

            q: queue.Queue[tuple[str, float] | None] = queue.Queue()

            async def _produce():
                try:
                    async for tok, coh in agent.stream(request.prompt):
                        q.put((tok, coh))
                finally:
                    q.put(None)

            future = asyncio.run_coroutine_threadsafe(_produce(), _bg_loop)

            i = 0
            while True:
                item = q.get()
                if item is None:
                    break
                tok, coh = item
                halted = coh < cfg.hard_limit
                yield token_evt(
                    token=tok,
                    coherence=round(coh, 4),
                    index=i,
                    halted=halted,
                    halt_reason="hard_limit" if halted else "",
                )
                i += 1

            future.result()

    # Auth interceptor
    class _AuthInterceptor(grpc.ServerInterceptor):
        def intercept_service(self, continuation, handler_call_details):
            if not cfg.api_keys:
                return continuation(handler_call_details)
            metadata = dict(handler_call_details.invocation_metadata)
            provided = metadata.get("x-api-key", "")
            if not any(hmac.compare_digest(provided, k) for k in cfg.api_keys):

                def _abort(req, ctx):
                    ctx.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid API key")

                # Detect RPC type from method name to return correct handler
                method = handler_call_details.method or ""
                if "Stream" in method.rsplit("/", 1)[-1]:
                    return grpc.unary_stream_rpc_method_handler(
                        lambda req, ctx: iter([_abort(req, ctx)]),
                    )
                return grpc.unary_unary_rpc_method_handler(_abort)
            return continuation(handler_call_details)

    _mb = 1024 * 1024
    server_options = [
        ("grpc.max_receive_message_length", cfg.grpc_max_message_mb * _mb),
        ("grpc.max_send_message_length", 4 * cfg.grpc_max_message_mb * _mb),
        ("grpc.keepalive_time_ms", 30_000),
        ("grpc.keepalive_timeout_ms", 10_000),
    ]

    interceptors = [_AuthInterceptor()]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
        options=server_options,
    )

    # Optional server reflection
    try:
        from grpc_reflection.v1alpha import reflection

        service_names = []
        if has_proto:
            service_names.append(
                director_pb2.DESCRIPTOR.services_by_name["DirectorService"].full_name,
            )
        service_names.append(reflection.SERVICE_NAME)
        reflection.enable_server_reflection(service_names, server)
        logger.debug("gRPC reflection enabled")
    except (ImportError, AttributeError, KeyError):
        pass

    if has_proto:
        from . import director_pb2_grpc

        director_pb2_grpc.add_DirectorServiceServicer_to_server(
            DirectorServicer(),
            server,
        )
    else:
        logger.warning(
            "Proto stubs not found — run scripts/gen_proto.sh. "
            "Server created but service not registered.",
        )

    if tls_cert_path and tls_key_path:
        with open(tls_cert_path, "rb") as cf, open(tls_key_path, "rb") as kf:
            creds = grpc.ssl_server_credentials([(kf.read(), cf.read())])
        server.add_secure_port(f"[::]:{port}", creds)
        logger.info(
            "gRPC server configured on port %d (TLS, workers=%d)",
            port,
            max_workers,
        )
    else:
        server.add_insecure_port(f"[::]:{port}")
        logger.info(
            "gRPC server configured on port %d (insecure, workers=%d)",
            port,
            max_workers,
        )

    return server
