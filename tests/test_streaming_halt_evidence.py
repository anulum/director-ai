# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Streaming Halt Evidence Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from director_ai.core import (
    CoherenceScorer,
    GroundTruthStore,
    StreamingKernel,
)
from director_ai.core.types import HaltEvidence


class TestHaltWithScorer:
    def _make_scorer(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue")
        return CoherenceScorer(threshold=0.5, ground_truth_store=store, use_nli=False)

    def test_halt_with_scorer_builds_evidence(self):
        kernel = StreamingKernel(hard_limit=0.99)
        scorer = self._make_scorer()

        def always_low(_token):
            return 0.3

        session = kernel.stream_tokens(
            iter(["bad", "output"]), always_low, scorer=scorer
        )
        assert session.halted
        assert session.halt_evidence_structured is not None
        assert isinstance(session.halt_evidence_structured, HaltEvidence)
        assert session.halt_evidence_structured.reason
        assert session.halt_evidence_structured.suggested_action

    def test_halt_without_scorer_no_structured(self):
        kernel = StreamingKernel(hard_limit=0.99)

        def always_low(_token):
            return 0.3

        session = kernel.stream_tokens(iter(["bad"]), always_low)
        assert session.halted
        assert session.halt_evidence_structured is None


class TestHaltEventField:
    def test_halt_event_has_evidence_field(self):
        kernel = StreamingKernel(hard_limit=0.99)
        store = GroundTruthStore()
        store.add("test", "test fact")
        scorer = CoherenceScorer(threshold=0.5, ground_truth_store=store, use_nli=False)

        session = kernel.stream_tokens(iter(["x"]), lambda _t: 0.1, scorer=scorer)
        assert session.halted
        halt_event = session.events[-1]
        assert hasattr(halt_event, "halt_evidence")
        assert halt_event.halt_evidence is not None


class TestSuggestedAction:
    def test_hard_limit_action(self):
        action = StreamingKernel._suggested_action("hard_limit (0.3 < 0.5)")
        assert "temperature" in action.lower() or "KB" in action

    def test_window_avg_action(self):
        action = StreamingKernel._suggested_action("window_avg (0.4 < 0.55)")
        assert "drifting" in action.lower() or "context" in action.lower()

    def test_downward_trend_action(self):
        action = StreamingKernel._suggested_action("downward_trend (0.2 > 0.15)")
        assert "degrading" in action.lower() or "rephrase" in action.lower()

    def test_unknown_reason_gives_generic(self):
        action = StreamingKernel._suggested_action("something_else")
        assert len(action) > 0
