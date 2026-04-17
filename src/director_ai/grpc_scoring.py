# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CoherenceScoring gRPC server (director.v1)

"""gRPC entrypoint for the ``director.v1.CoherenceScoring`` service.

Sits in front of :class:`director_ai.core.scoring.scorer.CoherenceScorer`
and exposes it over the schema declared in
``schemas/proto/director/v1/director.proto``. The Go gateway (Phase 2)
dials this server on port ``50052`` by default.

Two RPCs:

* ``ScoreClaim`` — unary. Client sends one claim plus supporting
  documents, server returns a :class:`CoherenceVerdict`.
* ``ScoreStream`` — bidirectional. Client sends one
  :class:`ScoreTokenRequest` per candidate token; server returns
  one verdict per request. First halted verdict closes the stream.

A deliberately separate module from the legacy
:mod:`director_ai.grpc_server`, which serves the Python-only
``DirectorService`` schema.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent import futures
from typing import TYPE_CHECKING

import grpc

from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.proto.converters import verdict_to_proto
from director_ai.proto.director.v1 import director_pb2 as pb
from director_ai.proto.director.v1 import director_pb2_grpc as rpc

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from director_ai.core.types import CoherenceScore

__all__ = ["CoherenceScoringService", "serve"]

logger = logging.getLogger("DirectorAI.gRPC.Scoring")


class CoherenceScoringService(rpc.CoherenceScoringServicer):
    """Implementation of ``director.v1.CoherenceScoring``.

    The service delegates scoring to an injected
    :class:`CoherenceScorer`. Tests may pass a stub scorer through
    ``scorer=`` to avoid loading NLI models.
    """

    def __init__(
        self,
        *,
        scorer: CoherenceScorer | None = None,
        store: GroundTruthStore | None = None,
        threshold: float = 0.5,
        use_nli: bool | None = False,
    ) -> None:
        self._store = store if store is not None else GroundTruthStore()
        self._scorer = scorer if scorer is not None else CoherenceScorer(
            threshold=threshold,
            ground_truth_store=self._store,
            use_nli=use_nli,
        )
        self._threshold = threshold

    # Unary -----------------------------------------------------

    def ScoreClaim(
        self,
        request: pb.ScoreClaimRequest,
        context: grpc.ServicerContext,
    ) -> pb.ScoreClaimResponse:
        if not request.claim:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "claim is required")

        self._push_documents(list(request.documents))
        threshold = (
            request.threshold if request.threshold > 0 else self._threshold
        )

        start = time.perf_counter()
        approved, score = self._scorer.review(
            prompt=_prompt_from_documents(list(request.documents), request.claim),
            action=request.claim,
            tenant_id=request.tenant_id,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        verdict = _score_to_verdict(
            score=score,
            approved=approved,
            hard_limit=threshold,
        )
        return pb.ScoreClaimResponse(verdict=verdict, latency_ms=latency_ms)

    # Streaming -------------------------------------------------

    def ScoreStream(
        self,
        request_iterator: Iterable[pb.ScoreTokenRequest],
        context: grpc.ServicerContext,
    ) -> Iterator[pb.ScoreTokenResponse]:
        documents_seen: set[str] = set()
        for req in request_iterator:
            if not context.is_active():  # pragma: no cover — cancel path
                return
            if req.documents:
                new_docs = [d for d in req.documents if d not in documents_seen]
                self._push_documents(new_docs)
                documents_seen.update(new_docs)
            accumulated = req.accumulated_text + (req.next_token or "")
            if not accumulated.strip():
                continue
            approved, score = self._scorer.review(
                prompt=_prompt_from_documents(
                    list(req.documents), req.next_token or ""
                ),
                action=accumulated,
                tenant_id=req.tenant_id,
            )
            yield pb.ScoreTokenResponse(
                verdict=_score_to_verdict(
                    score=score,
                    approved=approved,
                    hard_limit=self._threshold,
                )
            )
            if not approved:
                # Client must stop on halt; we stop yielding too so a
                # misbehaving caller cannot keep polling past it.
                return

    # Internal --------------------------------------------------

    def _push_documents(self, documents: list[str]) -> None:
        for i, doc in enumerate(documents):
            if doc:
                self._store.add(f"grpc-doc-{i}", doc)


def _score_to_verdict(
    *,
    score: CoherenceScore,
    approved: bool,
    hard_limit: float,
) -> pb.CoherenceVerdict:
    return verdict_to_proto(
        score=getattr(score, "score", 0.0),
        halted=not approved,
        halt_reason="coherence" if not approved else "none",
        hard_limit=hard_limit,
        message="Coherence below threshold" if not approved else "ok",
    )


def _prompt_from_documents(documents: list[str], claim: str) -> str:
    if not documents:
        return claim
    joined = "\n".join(documents)
    return f"{joined}\n\nClaim: {claim}"


def serve(
    *,
    listen_addr: str = "[::]:50052",
    max_workers: int = 8,
    service: CoherenceScoringService | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[grpc.Server, int]:
    """Start a gRPC server on ``listen_addr`` and return ``(server, port)``.

    The default port 50052 leaves 50051 free for the legacy
    :mod:`grpc_server` (``DirectorService``). Tests typically pass
    ``listen_addr="[::]:0"`` to get an ephemeral port and rely on the
    returned ``port`` to dial the server.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=(
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
        ),
    )
    rpc.add_CoherenceScoringServicer_to_server(
        service or CoherenceScoringService(),
        server,
    )
    port = server.add_insecure_port(listen_addr)
    server.start()
    logger.info(
        "CoherenceScoring gRPC server listening on %s (port %d)",
        listen_addr, port,
    )
    if stop_event is not None:
        threading.Thread(
            target=_stop_on_event,
            args=(server, stop_event),
            daemon=True,
        ).start()
    return server, port


def _stop_on_event(server: grpc.Server, event: threading.Event) -> None:
    event.wait()
    server.stop(0)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="director-ai-grpc-scoring")
    parser.add_argument("--listen", default="[::]:50052")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-nli", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    service = CoherenceScoringService(
        threshold=args.threshold,
        use_nli=args.use_nli,
    )
    server, _port = serve(
        listen_addr=args.listen,
        max_workers=args.max_workers,
        service=service,
    )
    server.wait_for_termination()
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
