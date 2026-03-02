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

import logging
import math
from types import SimpleNamespace

from .core.config import DirectorConfig

logger = logging.getLogger("DirectorAI.gRPC")


def _ns(**kw):
    return SimpleNamespace(**kw)


def create_grpc_server(
    config: DirectorConfig | None = None,
    max_workers: int = 4,
    port: int = 50051,
):
    """Create and return a gRPC server (not yet started).

    Raises ImportError with install instructions if grpcio is missing.
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
    from .core.scorer import CoherenceScorer
    from .core.streaming import StreamingKernel

    scorer = CoherenceScorer(
        threshold=cfg.coherence_threshold,
        use_nli=cfg.use_nli,
    )
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
            result = agent.process(request.prompt)
            tokens = result.output.split()
            kernel = StreamingKernel(
                hard_limit=cfg.hard_limit,
                soft_limit=cfg.soft_limit,
            )

            def _coherence_cb(token):
                h = hash(token) & 0xFFFFFFFF
                return 0.8 + 0.1 * math.sin(h)

            session = kernel.stream_tokens(iter(tokens), _coherence_cb)
            for event in session.events:
                yield token_evt(
                    token=event.token,
                    coherence=round(event.coherence, 4),
                    index=event.index,
                    halted=event.halted,
                    halt_reason=session.halt_reason if event.halted else "",
                )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))

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

    server.add_insecure_port(f"[::]:{port}")
    logger.info("gRPC server configured on port %d (workers=%d)", port, max_workers)
    return server
