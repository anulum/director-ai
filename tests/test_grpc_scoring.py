# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — tests for director.v1 CoherenceScoring gRPC server

"""Multi-angle coverage for the gRPC scoring service: unary happy
path, argument validation, document push into the knowledge store,
bidirectional streaming, halt-on-first-flagged-verdict, threshold
override, stop-event lifecycle."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

import grpc
import pytest

from director_ai.grpc_scoring import (
    CoherenceScoringService,
    _prompt_from_documents,
    _score_to_verdict,
    serve,
)
from director_ai.proto.director.v1 import director_pb2 as pb
from director_ai.proto.director.v1 import director_pb2_grpc as rpc


class FakeScore(NamedTuple):
    score: float
    h_logical: float = 0.0
    h_factual: float = 0.0
    warning: bool = False


class FakeScorer:
    """Deterministic stand-in for CoherenceScorer, used to keep tests
    fast and NLI-free."""

    def __init__(self, *, score: float = 0.9, approved: bool = True) -> None:
        self.score_value = score
        self.approved = approved
        self.calls: list[tuple[str, str, str]] = []

    def review(self, prompt: str, action: str, tenant_id: str = ""):
        self.calls.append((prompt, action, tenant_id))
        return self.approved, FakeScore(self.score_value)


@contextmanager
def _server_with(service: CoherenceScoringService) -> Iterator[str]:
    stop = threading.Event()
    server, port = serve(
        listen_addr="[::]:0",
        service=service,
        stop_event=stop,
        max_workers=4,
    )
    try:
        yield f"localhost:{port}"
    finally:
        stop.set()
        server.stop(0).wait()


class TestUnary:
    def test_score_claim_returns_verdict(self):
        scorer = FakeScorer(score=0.82, approved=True)
        svc = CoherenceScoringService(scorer=scorer)
        req = pb.ScoreClaimRequest(
            claim="Paris is the capital of France.",
            documents=["Paris is the capital of France."],
            tenant_id="t-1",
            request_id="r-1",
        )
        ctx = _ServicerContext()
        resp = svc.ScoreClaim(req, ctx)
        assert resp.verdict.score == pytest.approx(0.82)
        assert resp.verdict.halted is False
        assert resp.verdict.halt_reason == pb.HALT_REASON_NONE
        assert resp.latency_ms >= 0
        assert len(scorer.calls) == 1
        prompt, action, tid = scorer.calls[0]
        assert "Paris is the capital of France." in prompt
        assert action == "Paris is the capital of France."
        assert tid == "t-1"

    def test_halted_verdict_when_not_approved(self):
        svc = CoherenceScoringService(scorer=FakeScorer(score=0.2, approved=False))
        req = pb.ScoreClaimRequest(claim="Unsupported claim.")
        resp = svc.ScoreClaim(req, _ServicerContext())
        assert resp.verdict.halted is True
        assert resp.verdict.halt_reason == pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD

    def test_empty_claim_is_invalid(self):
        svc = CoherenceScoringService(scorer=FakeScorer())
        ctx = _ServicerContext()
        with pytest.raises(_AbortError) as excinfo:
            svc.ScoreClaim(pb.ScoreClaimRequest(), ctx)
        assert excinfo.value.code == grpc.StatusCode.INVALID_ARGUMENT

    def test_threshold_override_flows_through(self):
        svc = CoherenceScoringService(scorer=FakeScorer(score=0.4, approved=True), threshold=0.3)
        req = pb.ScoreClaimRequest(claim="x", threshold=0.7)
        resp = svc.ScoreClaim(req, _ServicerContext())
        assert resp.verdict.hard_limit == pytest.approx(0.7)


class TestStreaming:
    def test_stream_yields_verdict_per_token(self):
        svc = CoherenceScoringService(scorer=FakeScorer(score=0.8, approved=True))
        reqs = [
            pb.ScoreTokenRequest(
                accumulated_text="",
                next_token="Paris",
                documents=["Paris is the capital of France."],
            ),
            pb.ScoreTokenRequest(
                accumulated_text="Paris",
                next_token=" is",
            ),
            pb.ScoreTokenRequest(
                accumulated_text="Paris is",
                next_token=" the capital",
            ),
        ]
        out = list(svc.ScoreStream(iter(reqs), _ServicerContext()))
        assert len(out) == 3
        assert all(v.verdict.halted is False for v in out)

    def test_stream_halts_on_first_flagged_verdict(self):
        flip = iter([(True, 0.8), (False, 0.1), (True, 0.8)])

        class FlipScorer:
            def __init__(self):
                self.calls = 0

            def review(self, prompt, action, tenant_id=""):
                self.calls += 1
                approved, value = next(flip)
                return approved, FakeScore(value)

        svc = CoherenceScoringService(scorer=FlipScorer())
        reqs = [
            pb.ScoreTokenRequest(accumulated_text="", next_token="a"),
            pb.ScoreTokenRequest(accumulated_text="a", next_token="b"),
            pb.ScoreTokenRequest(accumulated_text="ab", next_token="c"),
        ]
        out = list(svc.ScoreStream(iter(reqs), _ServicerContext()))
        assert len(out) == 2, "stream must close after the halt verdict"
        assert out[0].verdict.halted is False
        assert out[1].verdict.halted is True

    def test_stream_skips_empty_accumulated_text(self):
        scorer = FakeScorer()
        svc = CoherenceScoringService(scorer=scorer)
        reqs = [
            pb.ScoreTokenRequest(accumulated_text="", next_token="  "),
            pb.ScoreTokenRequest(accumulated_text="", next_token=""),
            pb.ScoreTokenRequest(accumulated_text="hello", next_token=""),
        ]
        out = list(svc.ScoreStream(iter(reqs), _ServicerContext()))
        assert len(out) == 1
        assert scorer.calls[0][1] == "hello"

    def test_stream_pushes_new_documents_only_once(self):
        store_pushes: list[str] = []

        class RecordingStore:
            def add(self, key, value):
                store_pushes.append(value)

            def retrieve_context(self, *_a, **_kw):  # pragma: no cover
                return ""

        svc = CoherenceScoringService(
            scorer=FakeScorer(score=0.9, approved=True),
            store=RecordingStore(),  # type: ignore[arg-type]
        )
        doc = "France."
        reqs = [
            pb.ScoreTokenRequest(accumulated_text="a", next_token="b", documents=[doc]),
            pb.ScoreTokenRequest(accumulated_text="ab", next_token="c", documents=[doc]),
            pb.ScoreTokenRequest(accumulated_text="abc", next_token="d", documents=[doc]),
        ]
        list(svc.ScoreStream(iter(reqs), _ServicerContext()))
        assert store_pushes.count(doc) == 1


class TestEndToEnd:
    def test_round_trip_via_grpc_channel(self):
        svc = CoherenceScoringService(scorer=FakeScorer(score=0.88, approved=True))
        with _server_with(svc) as addr, grpc.insecure_channel(addr) as channel:
            stub = rpc.CoherenceScoringStub(channel)
            resp = stub.ScoreClaim(
                pb.ScoreClaimRequest(claim="test", documents=["supporting"]),
                timeout=5,
            )
        assert resp.verdict.score == pytest.approx(0.88)

    def test_stop_event_shuts_server(self):
        svc = CoherenceScoringService(scorer=FakeScorer())
        stop = threading.Event()
        server, _port = serve(listen_addr="[::]:0", service=svc, stop_event=stop)
        stop.set()
        for _ in range(100):
            time.sleep(0.01)
            try:
                server.wait_for_termination(timeout=0.01)
                break
            except Exception:  # pragma: no cover
                continue


class TestHelpers:
    def test_prompt_from_documents_joins(self):
        s = _prompt_from_documents(["d1", "d2"], "claim")
        assert "d1" in s and "d2" in s and "Claim: claim" in s

    def test_prompt_no_documents_returns_claim(self):
        assert _prompt_from_documents([], "claim") == "claim"

    def test_score_to_verdict_maps_halt(self):
        verdict = _score_to_verdict(
            score=FakeScore(0.2), approved=False, hard_limit=0.5
        )
        assert verdict.halted is True
        assert verdict.halt_reason == pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD
        assert verdict.hard_limit == pytest.approx(0.5)


# --- test doubles -----------------------------------------------------


class _AbortError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _ServicerContext:
    """Minimal ``grpc.ServicerContext`` stand-in. Only implements the
    surface our service actually touches."""

    def abort(self, code, message):
        raise _AbortError(code, message)

    def is_active(self) -> bool:
        return True

    def invocation_metadata(self):
        return []
