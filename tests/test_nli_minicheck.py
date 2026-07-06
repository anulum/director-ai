# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MiniCheck NLI Backend Tests
"""Typed guard tests for the MiniCheck NLI backend.

The companion real-surface coverage lives in
``tests/test_nli_scorer_real_surface.py``. This file keeps direct edge-case
guards for fallback, cache, and claim-coverage behavior without loading a remote
MiniCheck checkpoint during local verification.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

import pytest

from director_ai.core import CoherenceScorer
from director_ai.core.nli import NLIScorer
from director_ai.core.scoring import scorer as scorer_module


@dataclass(frozen=True)
class _MiniCheckCall:
    """Captured MiniCheck package call."""

    docs: list[str]
    claims: list[str]


class _MiniCheckPackage:
    """Small MiniCheck package protocol used by the public scorer."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[_MiniCheckCall] = []

    def score(self, *, docs: list[str], claims: list[str]) -> list[float]:
        """Return the configured support probabilities for one call."""
        assert len(docs) == len(claims)
        assert len(self._scores) == len(claims)
        self.calls.append(_MiniCheckCall(docs=docs, claims=claims))
        return list(self._scores)


class _MiniCheckState(Protocol):
    """Mutable MiniCheck state exposed by NLIScorer instances."""

    _minicheck: _MiniCheckPackage
    _minicheck_loaded: bool


class _CoherenceMiniCheckState(Protocol):
    """Mutable cached MiniCheck state exposed by CoherenceScorer."""

    _minicheck_nli: NLIScorer | None


class _UnavailableMiniCheckScorer(NLIScorer):
    """NLIScorer subclass that records unavailable MiniCheck loading."""

    def __init__(
        self,
        *,
        use_model: bool = True,
        backend: str = "minicheck",
        minicheck_variant: str = "deberta-v3-large",
    ) -> None:
        super().__init__(
            use_model=use_model,
            backend=backend,
            minicheck_variant=minicheck_variant,
        )
        self.ensure_calls = 0

    def _ensure_minicheck(self) -> bool:
        """Record the load attempt and report MiniCheck unavailable."""
        self.ensure_calls += 1
        return False


class _ConstructedUnavailableMiniCheckScorer(_UnavailableMiniCheckScorer):
    """Unavailable scorer that tracks construction by CoherenceScorer."""

    instances: ClassVar[list[_ConstructedUnavailableMiniCheckScorer]] = []

    def __init__(
        self,
        *,
        use_model: bool = True,
        backend: str = "minicheck",
        minicheck_variant: str = "deberta-v3-large",
    ) -> None:
        super().__init__(
            use_model=use_model,
            backend=backend,
            minicheck_variant=minicheck_variant,
        )
        self.instances.append(self)


class _ReadyMiniCheckScorer(NLIScorer):
    """NLIScorer subclass that reports a ready MiniCheck backend."""

    instances: ClassVar[list[_ReadyMiniCheckScorer]] = []

    def __init__(
        self,
        *,
        use_model: bool = True,
        backend: str = "minicheck",
        minicheck_variant: str = "deberta-v3-large",
    ) -> None:
        super().__init__(
            use_model=use_model,
            backend=backend,
            minicheck_variant=minicheck_variant,
        )
        self.instances.append(self)

    def _ensure_minicheck(self) -> bool:
        """Report MiniCheck readiness without loading a checkpoint."""
        return True


class _ClaimCoverageScorer:
    """Sentence-level scorer used by MiniCheck claim coverage guards."""

    def __init__(self, divergences: list[float]) -> None:
        self._divergences = divergences
        self.calls: list[tuple[str, str]] = []

    def score(self, premise: str, hypothesis: str) -> float:
        """Return configured divergences while recording each claim."""
        self.calls.append((premise, hypothesis))
        if len(self._divergences) == 1:
            return self._divergences[0]
        return self._divergences[len(self.calls) - 1]


def _install_minicheck_package(
    scorer: NLIScorer,
    minicheck: _MiniCheckPackage,
) -> None:
    """Install a local MiniCheck protocol object on an NLI scorer."""
    state = cast(_MiniCheckState, scorer)
    state._minicheck = minicheck
    state._minicheck_loaded = True


class TestMiniCheckBackend:
    """Guard direct NLIScorer MiniCheck backend behavior."""

    def test_invalid_backend_raises(self) -> None:
        """Unknown NLI backends should fail fast."""
        with pytest.raises(ValueError, match="backend must be one of"):
            NLIScorer(backend="nonexistent")

    def test_deberta_backend_default(self) -> None:
        """The default NLI backend should remain DeBERTa."""
        scorer = NLIScorer(use_model=False)
        assert scorer.backend == "deberta"

    def test_minicheck_fallback_to_heuristic(self) -> None:
        """MiniCheck should fall back when the package is unavailable."""
        scorer = NLIScorer(backend="minicheck")
        result = scorer.score("The sky is blue.", "The sky is blue.")
        assert 0.0 <= result <= 1.0

    def test_minicheck_dispatches_correctly(self) -> None:
        """MiniCheck backend scoring should attempt MiniCheck loading once."""
        scorer = _UnavailableMiniCheckScorer()
        result = scorer.score("A", "B")
        assert scorer.ensure_calls == 1
        assert 0.0 <= result <= 1.0

    def test_minicheck_with_protocol_package(self) -> None:
        """Package probabilities should convert to divergence scores."""
        minicheck = _MiniCheckPackage([0.8])
        scorer = NLIScorer(backend="minicheck")
        _install_minicheck_package(scorer, minicheck)

        result = scorer.score("The sky is blue.", "The sky is blue.")

        assert result == pytest.approx(0.2)
        assert minicheck.calls == [
            _MiniCheckCall(
                docs=["The sky is blue."],
                claims=["The sky is blue."],
            )
        ]

    def test_minicheck_high_contradiction(self) -> None:
        """Low MiniCheck support probability should mean high divergence."""
        minicheck = _MiniCheckPackage([0.1])
        scorer = NLIScorer(backend="minicheck")
        _install_minicheck_package(scorer, minicheck)

        result = scorer.score("Earth orbits Sun.", "Sun orbits Earth.")

        assert result == pytest.approx(0.9)

    def test_score_batch_uses_backend(self) -> None:
        """Batch MiniCheck scoring should preserve docs and claims order."""
        minicheck = _MiniCheckPackage([0.5, 0.7])
        scorer = NLIScorer(backend="minicheck")
        _install_minicheck_package(scorer, minicheck)

        results = scorer.score_batch([("A", "B"), ("C", "D")])

        assert results == pytest.approx([0.5, 0.3])
        assert minicheck.calls == [_MiniCheckCall(docs=["A", "C"], claims=["B", "D"])]

    @pytest.mark.parametrize("backend", ["deberta", "onnx", "minicheck"])
    def test_valid_backends_accepted(self, backend: str) -> None:
        """All built-in backend names should construct successfully."""
        scorer = NLIScorer(backend=backend, use_model=False)
        assert scorer.backend == backend

    @pytest.mark.parametrize(
        ("mc_score", "expected_divergence"),
        [
            (1.0, 0.0),
            (0.5, 0.5),
            (0.0, 1.0),
        ],
    )
    def test_minicheck_score_to_divergence(
        self,
        mc_score: float,
        expected_divergence: float,
    ) -> None:
        """MiniCheck support probabilities should invert to divergence."""
        minicheck = _MiniCheckPackage([mc_score])
        scorer = NLIScorer(backend="minicheck")
        _install_minicheck_package(scorer, minicheck)

        result = scorer.score("A", "B")

        assert result == pytest.approx(expected_divergence)


class TestMiniCheckPipelineIntegration:
    """Verify MiniCheck wiring through the public CoherenceScorer pipeline."""

    def test_scorer_with_minicheck_backend(self) -> None:
        """The public scorer should accept MiniCheck as a backend name."""
        scorer = CoherenceScorer(use_nli=False, scorer_backend="minicheck")
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


class TestMiniCheckPerformanceDoc:
    """Document deterministic MiniCheck fallback latency."""

    def test_heuristic_fallback_fast(self) -> None:
        """Heuristic fallback should stay below the local smoke threshold."""
        import time

        scorer = NLIScorer(backend="minicheck", use_model=False)
        t0 = time.perf_counter()
        for _ in range(100):
            scorer.score("test", "test")
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 1.0


class TestGetMinicheckScorer:
    """Guard CoherenceScorer MiniCheck scorer caching behavior."""

    def test_returns_none_when_minicheck_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unavailable MiniCheck construction should cache ``None``."""
        _ConstructedUnavailableMiniCheckScorer.instances = []
        monkeypatch.setattr(
            scorer_module,
            "NLIScorer",
            _ConstructedUnavailableMiniCheckScorer,
        )
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)

        result = scorer._get_minicheck_scorer()

        assert result is None
        assert scorer._minicheck_nli is None
        assert len(_ConstructedUnavailableMiniCheckScorer.instances) == 1

    def test_caches_none_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cached unavailable MiniCheck result should not retry construction."""
        _ConstructedUnavailableMiniCheckScorer.instances = []
        monkeypatch.setattr(
            scorer_module,
            "NLIScorer",
            _ConstructedUnavailableMiniCheckScorer,
        )
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)

        scorer._get_minicheck_scorer()
        first_count = len(_ConstructedUnavailableMiniCheckScorer.instances)
        result = scorer._get_minicheck_scorer()

        assert result is None
        assert scorer._minicheck_nli is None
        assert first_count == 1
        assert len(_ConstructedUnavailableMiniCheckScorer.instances) == first_count

    def test_returns_cached_value_on_subsequent_calls(self) -> None:
        """An existing MiniCheck scorer should be returned directly."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        sentinel = _ReadyMiniCheckScorer()
        state = cast(_CoherenceMiniCheckState, scorer)
        state._minicheck_nli = sentinel

        assert scorer._get_minicheck_scorer() is sentinel

    def test_caches_successful_scorer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ready MiniCheck scorer should be cached after construction."""
        _ReadyMiniCheckScorer.instances = []
        monkeypatch.setattr(scorer_module, "NLIScorer", _ReadyMiniCheckScorer)
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)

        result = scorer._get_minicheck_scorer()

        assert len(_ReadyMiniCheckScorer.instances) == 1
        assert result is _ReadyMiniCheckScorer.instances[0]
        assert scorer._minicheck_nli is _ReadyMiniCheckScorer.instances[0]


class TestMinicheckClaimCoverage:
    """Guard MiniCheck sentence-level claim coverage behavior."""

    def test_all_supported(self) -> None:
        """All low-divergence claims should produce full coverage."""
        coverage_scorer = _ClaimCoverageScorer([0.1])

        coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
            coverage_scorer, "Source text.", "First sentence. Second sentence."
        )

        assert len(sents) >= 2
        assert coverage == 1.0
        assert all(divergence == 0.1 for divergence in divs)

    def test_none_supported(self) -> None:
        """All high-divergence claims should produce zero coverage."""
        coverage_scorer = _ClaimCoverageScorer([0.9])

        coverage, divs, _sents = CoherenceScorer._minicheck_claim_coverage(
            coverage_scorer, "Source text.", "First sentence. Second sentence."
        )

        assert coverage == 0.0
        assert all(divergence == 0.9 for divergence in divs)

    def test_partial_support(self) -> None:
        """Mixed claim divergences should report fractional coverage."""
        coverage_scorer = _ClaimCoverageScorer([0.1, 0.9])

        coverage, divs, _sents = CoherenceScorer._minicheck_claim_coverage(
            coverage_scorer, "Source text.", "Good claim. Bad claim."
        )

        assert coverage == pytest.approx(0.5)
        assert divs == [0.1, 0.9]

    def test_empty_summary(self) -> None:
        """An empty summary has no unsupported claims."""
        coverage_scorer = _ClaimCoverageScorer([0.9])

        coverage, divs, sents = CoherenceScorer._minicheck_claim_coverage(
            coverage_scorer, "Source text.", ""
        )

        assert coverage == 1.0
        assert divs == []
        assert sents == []

    def test_nltk_import_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When nltk is unavailable, period splitting should still work."""
        monkeypatch.setitem(sys.modules, "nltk", None)
        monkeypatch.setitem(sys.modules, "nltk.tokenize", None)
        coverage_scorer = _ClaimCoverageScorer([0.2])

        coverage, _divs, sents = CoherenceScorer._minicheck_claim_coverage(
            coverage_scorer, "Source.", "First. Second."
        )

        assert len(sents) == 2
        assert coverage == 1.0
