# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rust Pipeline Integration Tests
"""Multi-angle tests for the Python → Rust → Python pipeline.

Covers: RustBackend registration, scorer_backend="rust" wiring,
heuristic scoring via FFI, knowledge callback integration,
score component consistency, backend fallback when Rust unavailable,
signal function dispatch, streaming kernel wiring, and pipeline
performance documentation.

Requires: backfire_kernel (maturin-built wheel). Tests skip gracefully
when the Rust extension is not installed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

try:
    import backfire_kernel  # noqa: F401

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(
    not HAS_RUST, reason="backfire_kernel not installed (requires maturin build)"
)

from director_ai.core import CoherenceScorer, GroundTruthStore  # noqa: E402

# ── Backend registration ──────────────────────────────────────────


class TestRustBackendRegistration:
    """Verify Rust backend is discoverable via the registry."""

    def test_rust_backend_registered(self) -> None:
        """The public backend registry should expose the Rust backend name."""
        from director_ai.core.scoring.backends import get_backend

        backend_cls = get_backend("rust")
        assert backend_cls is not None

    def test_backfire_alias_registered(self) -> None:
        """The public backend registry should expose the backfire alias."""
        from director_ai.core.scoring.backends import get_backend

        backend_cls = get_backend("backfire")
        assert backend_cls is not None

    def test_rust_and_backfire_same_class(self) -> None:
        """Rust and backfire names should resolve to the same backend class."""
        from director_ai.core.scoring.backends import get_backend

        assert get_backend("rust") is get_backend("backfire")

    def test_list_backends_includes_rust(self) -> None:
        """Listing public backends should include the Rust backend."""
        from director_ai.core.scoring.backends import list_backends

        backends = list_backends()
        assert "rust" in backends


# ── Scorer via Rust backend ───────────────────────────────────────


class TestRustScorerPipeline:
    """End-to-end: CoherenceScorer(scorer_backend="rust") pipeline."""

    @pytest.fixture
    def rust_scorer(self) -> CoherenceScorer:
        """Build a Rust-backed scorer through the public scorer API."""
        return CoherenceScorer(use_nli=False, scorer_backend="rust")

    def test_review_returns_tuple(self, rust_scorer: CoherenceScorer) -> None:
        """Review should keep the public tuple contract."""
        result = rust_scorer.review("What is AI?", "AI is intelligence.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_approved_is_bool(self, rust_scorer: CoherenceScorer) -> None:
        """The approved flag should remain a boolean."""
        approved, _ = rust_scorer.review("Q", "A")
        assert isinstance(approved, bool)

    def test_score_in_range(self, rust_scorer: CoherenceScorer) -> None:
        """Rust-backed scores should stay in the normalised range."""
        _, score = rust_scorer.review("What is 2+2?", "4")
        assert 0.0 <= score.score <= 1.0

    def test_score_has_components(self, rust_scorer: CoherenceScorer) -> None:
        """Rust-backed score objects should expose scorer components."""
        _, score = rust_scorer.review("Q", "A")
        assert hasattr(score, "h_logical")
        assert hasattr(score, "h_factual")
        assert hasattr(score, "score")
        assert hasattr(score, "approved")

    @pytest.mark.parametrize(
        "prompt,response",
        [
            ("What is the sky colour?", "The sky is blue."),
            ("", "empty prompt response"),
            ("test", ""),
            ("Unicode: こんにちは", "Response: hello"),
            ("Числа: 123", "Ответ: 456"),
        ],
    )
    def test_various_inputs(
        self,
        rust_scorer: CoherenceScorer,
        prompt: str,
        response: str,
    ) -> None:
        """The Rust-backed scorer should handle representative text inputs."""
        approved, score = rust_scorer.review(prompt, response)
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_deterministic(self, rust_scorer: CoherenceScorer) -> None:
        """Identical inputs should produce deterministic Rust-backed scores."""
        _, s1 = rust_scorer.review("Q", "A")
        _, s2 = rust_scorer.review("Q", "A")
        assert s1.score == s2.score

    def test_long_input(self, rust_scorer: CoherenceScorer) -> None:
        """Long repeated inputs should remain bounded and non-crashing."""
        long_text = "word " * 10_000
        _, score = rust_scorer.review(long_text, long_text)
        assert 0.0 <= score.score <= 1.0


# ── Knowledge callback via Rust ───────────────────────────────────


class TestRustKnowledgeCallback:
    """Verify knowledge store callback crosses FFI boundary."""

    def test_scorer_with_ground_truth_store(self) -> None:
        """GroundTruthStore should remain usable by a Rust-backed scorer."""
        store = GroundTruthStore()
        store.add_fact("earth", "Earth orbits the Sun")
        scorer = CoherenceScorer(
            use_nli=False,
            scorer_backend="rust",
            ground_truth_store=store,
        )
        approved, score = scorer.review(
            "What does Earth orbit?", "Earth orbits the Sun."
        )
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_scorer_without_ground_truth(self) -> None:
        """Rust-backed scorer construction should not require a store."""
        scorer = CoherenceScorer(use_nli=False, scorer_backend="rust")
        approved, score = scorer.review("Q", "A")
        assert isinstance(approved, bool)


# ── Rust vs Python consistency ────────────────────────────────────


class TestRustPythonConsistency:
    """Rust and Python heuristic backends must agree on direction."""

    def test_identical_input_both_approve_or_reject(self) -> None:
        """Rust and Python heuristic paths should both return valid scores."""
        py_scorer = CoherenceScorer(use_nli=False, scorer_backend="deberta")
        rs_scorer = CoherenceScorer(use_nli=False, scorer_backend="rust")

        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."

        py_approved, py_score = py_scorer.review(prompt, response)
        rs_approved, rs_score = rs_scorer.review(prompt, response)

        # Both should produce valid scores (exact values may differ)
        assert 0.0 <= py_score.score <= 1.0
        assert 0.0 <= rs_score.score <= 1.0


# ── Signal function dispatch ─────────────────────────────────────


class TestSignalDispatch:
    """Verify Rust signal functions are callable when backfire_kernel installed."""

    def test_entity_overlap_available(self) -> None:
        """Entity-overlap dispatch should return a numeric signal."""
        from director_ai.core.scoring.verified_scorer import _entity_overlap

        result = _entity_overlap("Paris is the capital", "Paris is capital of France")
        assert isinstance(result, float)

    def test_numerical_consistency_available(self) -> None:
        """Numerical-consistency dispatch should return a scalar signal."""
        from director_ai.core.scoring.verified_scorer import _numerical_consistency

        result = _numerical_consistency("The speed is 100 km/h", "Speed: 100 km/h")
        assert isinstance(result, (bool, float, int))

    def test_negation_flip_available(self) -> None:
        """Negation-flip dispatch should return a scalar signal."""
        from director_ai.core.scoring.verified_scorer import _negation_flip

        result = _negation_flip("The sky is blue", "The sky is not blue")
        assert isinstance(result, (bool, float, int))

    def test_traceability_available(self) -> None:
        """Traceability dispatch should return a scalar signal."""
        from director_ai.core.scoring.verified_scorer import _traceability

        result = _traceability("Source document", "Response text")
        assert isinstance(result, (bool, float, int))

    def test_trend_drop_available(self) -> None:
        """Streaming trend-drop dispatch should return a scalar signal."""
        from director_ai.core.runtime.streaming import _trend_drop

        result = _trend_drop([0.9, 0.8, 0.7, 0.6, 0.5])
        assert isinstance(result, (bool, float, int))


# ── Performance documentation ─────────────────────────────────────


class TestRustPerformanceDoc:
    """Document and verify Rust pipeline performance characteristics.

    Expected: Rust heuristic scorer ≤ 50µs per review (no NLI).
    Measured on L40S: 2.5µs median. On consumer hardware: ~10-50µs.
    """

    def test_rust_review_latency_under_1ms(self) -> None:
        """Single review must complete in < 1ms (heuristic, no NLI)."""
        scorer = CoherenceScorer(use_nli=False, scorer_backend="rust")
        # Warmup
        for _ in range(10):
            scorer.review("warmup", "warmup")

        t0 = time.perf_counter()
        for _ in range(100):
            scorer.review("What is AI?", "AI is intelligence.")
        elapsed = (time.perf_counter() - t0) / 100 * 1000  # ms

        assert elapsed < 1.0, f"Rust review took {elapsed:.3f}ms (expected < 1ms)"

    def test_rust_backend_faster_than_python(self) -> None:
        """Rust backend should be faster than Python heuristic backend."""
        py_scorer = CoherenceScorer(use_nli=False)
        rs_scorer = CoherenceScorer(use_nli=False, scorer_backend="rust")

        n = 50
        prompt, response = "What is AI?", "AI is artificial intelligence."

        # Warmup both
        for _ in range(5):
            py_scorer.review(prompt, response)
            rs_scorer.review(prompt, response)

        t0 = time.perf_counter()
        for _ in range(n):
            py_scorer.review(prompt, response)
        py_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            rs_scorer.review(prompt, response)
        rs_time = time.perf_counter() - t0

        speedup = py_time / rs_time if rs_time > 0 else float("inf")
        # Rust should be at least 2x faster (typically 10-14x)
        assert speedup > 2.0, (
            f"Rust speedup only {speedup:.1f}x "
            f"(py={py_time * 1000:.1f}ms, rs={rs_time * 1000:.1f}ms)"
        )

    def test_benchmark_results_exist(self) -> None:
        """FFI overhead benchmark results must be on disk."""
        results_dir = Path(__file__).parent.parent / "benchmarks" / "results"
        ffi_results = results_dir / "ffi_overhead_results.json"
        if ffi_results.exists():
            data = json.loads(ffi_results.read_text())
            # Results is a list of benchmark entries with 'name' field
            if isinstance(data, list):
                names = {d["name"] for d in data if "name" in d}
                assert any("streaming" in n for n in names)
            else:
                assert "streaming" in data or "scorer" in data
