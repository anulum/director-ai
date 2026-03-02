# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — gRPC Transport
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
gRPC server for Director-Class AI.

Usage::

    from director_ai.grpc_server import create_grpc_server
    server = create_grpc_server(config)
    server.start()
    server.wait_for_termination()

Requires ``pip install director-ai[grpc]``.
"""

from __future__ import annotations

import hmac
import logging
from types import SimpleNamespace

from .core.config import DirectorConfig

logger = logging.getLogger("DirectorAI.gRPC")


def _ns(**kw):
    return SimpleNamespace(**kw)


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
            "Install with: pip install director-ai[grpc]"
        ) from exc

    cfg = config or DirectorConfig.from_env()

    from .core.agent import CoherenceAgent
    from .core.streaming import StreamingKernel

    scorer = cfg.build_scorer()
    agent = CoherenceAgent()

    # Resolve proto message factories
    try:
        from . import director_pb2

        review_resp = director_pb2.ReviewResponse
        process_resp = director_pb2.ProcessResponse
        batch_resp = director_pb2.BatchReviewResponse
        token_evt = director_pb2.TokenEvent
        has_proto = True
    except ImportError:
        review_resp = _ns  # type: ignore[assignment]
        process_resp = _ns  # type: ignore[assignment]
        batch_resp = _ns  # type: ignore[assignment]
        token_evt = _ns  # type: ignore[assignment]
        has_proto = False

    class DirectorServicer:  # noqa: N801
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
                    )
                )
            return batch_resp(responses=responses)

        def StreamTokens(self, request, context):  # noqa: N802
            # Replay mode: generate full result then stream with scoring
            result = agent.process(request.prompt)
            tokens = result.output.split()
            kernel = StreamingKernel(
                hard_limit=cfg.hard_limit,
                soft_limit=cfg.soft_limit,
            )

            def _make_cb(sc, pr):
                acc = []

                def cb(token):
                    acc.append(token)
                    text = " ".join(acc)
                    _, s = sc.review(pr, text)
                    return s.score

                return cb

            session = kernel.stream_tokens(
                iter(tokens), _make_cb(scorer, request.prompt)
            )
            for event in session.events:
                yield token_evt(
                    token=event.token,
                    coherence=round(event.coherence, 4),
                    index=event.index,
                    halted=event.halted,
                    halt_reason=session.halt_reason if event.halted else "",
                )

    # Auth interceptor
    class _AuthInterceptor(grpc.ServerInterceptor):
        def intercept_service(self, continuation, handler_call_details):
            if not cfg.api_keys:
                return continuation(handler_call_details)
            metadata = dict(handler_call_details.invocation_metadata)
            provided = metadata.get("x-api-key", "")
            if not any(hmac.compare_digest(provided, k) for k in cfg.api_keys):
                return grpc.unary_unary_rpc_method_handler(
                    lambda req, ctx: ctx.abort(
                        grpc.StatusCode.UNAUTHENTICATED, "invalid API key"
                    )
                )
            return continuation(handler_call_details)

    interceptors = [_AuthInterceptor()]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
    )

    if has_proto:
        from . import director_pb2_grpc

        director_pb2_grpc.add_DirectorServiceServicer_to_server(
            DirectorServicer(),
            server,
        )
    else:
        logger.warning(
            "Proto stubs not found — run scripts/gen_proto.sh. "
            "Server created but service not registered."
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
