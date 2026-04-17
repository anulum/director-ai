# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Halt Evidence Tests
"""Multi-angle tests for streaming kernel halt evidence pipeline.

Covers: halt evidence with scorer, halt without scorer, halt event fields,
suggested action per halt reason, parametrised halt scores, pipeline
integration, and performance documentation.
"""

import pytest

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
            iter(["bad", "output"]),
            always_low,
            scorer=scorer,
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

    @pytest.mark.parametrize(
        "reason",
        [
            "hard_limit (0.1 < 0.5)",
            "window_avg (0.3 < 0.6)",
            "downward_trend (0.2 > 0.1)",
            "custom_reason",
        ],
    )
    def test_all_reasons_produce_action(self, reason):
        action = StreamingKernel._suggested_action(reason)
        assert isinstance(action, str)
        assert len(action) > 0


class TestHaltEvidenceParametrised:
    """Parametrised halt evidence tests."""

    @pytest.mark.parametrize("score", [0.01, 0.1, 0.3, 0.49])
    def test_various_halt_scores(self, score):
        kernel = StreamingKernel(hard_limit=0.5)
        session = kernel.stream_tokens(iter(["bad"]), lambda _: score)
        assert session.halted
        assert session.halt_reason is not None

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
    def test_various_thresholds_with_scorer(self, threshold):
        store = GroundTruthStore()
        store.add("test", "test fact")
        scorer = CoherenceScorer(
            threshold=threshold, ground_truth_store=store, use_nli=False
        )
        kernel = StreamingKernel(hard_limit=0.99)
        session = kernel.stream_tokens(iter(["x"]), lambda _: 0.1, scorer=scorer)
        assert session.halted


class TestHaltEvidencePerformanceDoc:
    """Document halt evidence pipeline performance."""

    def test_halt_evidence_has_all_fields(self):
        store = GroundTruthStore()
        store.add("test", "test fact")
        scorer = CoherenceScorer(threshold=0.5, ground_truth_store=store, use_nli=False)
        kernel = StreamingKernel(hard_limit=0.99)
        session = kernel.stream_tokens(iter(["x"]), lambda _: 0.1, scorer=scorer)

        ev = session.halt_evidence_structured
        assert ev is not None
        assert hasattr(ev, "reason")
        assert hasattr(ev, "last_score")
        assert hasattr(ev, "evidence_chunks")
        assert hasattr(ev, "suggested_action")

    def test_halt_evidence_chain_complete(self):
        """Full pipeline: StreamingKernel → halt → scorer → evidence → action."""
        store = GroundTruthStore()
        store.add("sky", "The sky is blue")
        scorer = CoherenceScorer(threshold=0.5, ground_truth_store=store, use_nli=False)
        kernel = StreamingKernel(hard_limit=0.99)
        session = kernel.stream_tokens(
            iter(["wrong", "output"]), lambda _: 0.2, scorer=scorer
        )

        assert session.halted
        ev = session.halt_evidence_structured
        assert ev is not None
        assert ev.reason != ""
        assert ev.suggested_action != ""
        assert isinstance(ev.evidence_chunks, list)
