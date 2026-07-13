# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — claim-decomposition benchmark tests
"""Multi-angle tests for benchmarks/run_claim_decomp_benchmark.py.

Exercises the pure scoring and metric logic with real collaborators (the
production sentence splitter and a real :class:`AtomicClaimDecomposer`
driven by a plain-function transport) and plain protocol fakes — no
mock library, so the tests run against the real surface. The GPU/dataset
paths (model construction, ``_load_wice``, ``build_local_transport``,
``main``) are out of scope here; they are exercised on the metered run.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_claim_decomp_benchmark import (
    CONFIGS,
    ConfigScores,
    _balanced_accuracy,
    _prf1,
    _subclaims,
    _support_score,
    best_threshold,
    metrics_at_threshold,
    run_claim_decomp_benchmark,
    score_configs,
    summarise,
)
from director_ai.core.scoring.claim_decomposition import AtomicClaimDecomposer
from director_ai.core.text_segmentation import split_sentences

# ── Collaborators ──────────────────────────────────────────────────


class OverlapPredictor:
    """Fake NLI: entailment prob = fraction of hypothesis words in premise."""

    def score(self, premise: str, hypothesis: str) -> float:
        words = {w.lower().strip(".,") for w in hypothesis.split() if len(w) > 3}
        if not words:
            return 1.0
        prem = {w.lower().strip(".,") for w in premise.split()}
        return len(words & prem) / len(words)


def _and_split_transport(
    model: str, messages: list[dict[str, str]], max_tokens: int
) -> str | None:
    """Split the passage on ' and ' into a JSON claims list."""
    passage = json.loads(messages[1]["content"])["passage"]
    parts = [p.strip() for p in passage.split(" and ") if p.strip()]
    return json.dumps({"claims": parts if len(parts) > 1 else [passage]})


def _failing_transport(
    model: str, messages: list[dict[str, str]], max_tokens: int
) -> str | None:
    """Always fail so the decomposer takes its sentence fallback."""
    return None


@pytest.fixture
def llm_decomposer() -> AtomicClaimDecomposer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return AtomicClaimDecomposer(
            provider="openai", model="fake", transport=_and_split_transport
        )


@pytest.fixture
def fallback_decomposer() -> AtomicClaimDecomposer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return AtomicClaimDecomposer(
            provider="openai", model="fake", transport=_failing_transport
        )


ROWS = [
    {
        "doc": "Alice was born in Paris. Alice directed the film Nova.",
        "claim": "Alice was born in Paris and Alice directed Nova",
        "label": 1,
        "dataset": "Wice",
    },
    {
        "doc": "Bob was born in Rome.",
        "claim": "Bob was born in Rome and Bob won an Oscar",
        "label": 0,
        "dataset": "Wice",
    },
    {
        "doc": "Carla plays tennis.",
        "claim": "Carla plays tennis",
        "label": 1,
        "dataset": "Wice",
    },
    {
        "doc": "Dan lives in Berlin.",
        "claim": "Dan lives in Tokyo",
        "label": 0,
        "dataset": "Wice",
    },
]


# ── _subclaims ─────────────────────────────────────────────────────


class TestSubclaims:
    """Strategy dispatch and its non-empty / backend guarantees."""

    def test_no_decomp_keeps_whole_claim(self) -> None:
        subs, backend = _subclaims(
            "no-decomp", "X and Y", decomposer=None, splitter=split_sentences
        )
        assert subs == ["X and Y"]
        assert backend is None

    def test_regex_splits_on_sentences(self) -> None:
        subs, backend = _subclaims(
            "regex-decomp",
            "First fact. Second fact.",
            decomposer=None,
            splitter=split_sentences,
        )
        assert len(subs) == 2
        assert backend is None

    def test_regex_unsentenced_claim_stays_whole(self) -> None:
        subs, _ = _subclaims(
            "regex-decomp",
            "one clause no period",
            decomposer=None,
            splitter=split_sentences,
        )
        assert subs == ["one clause no period"]

    def test_regex_empty_split_degrades_to_claim(self) -> None:
        # A splitter that returns nothing must not yield an empty sub-claim set.
        subs, _ = _subclaims(
            "regex-decomp", "text", decomposer=None, splitter=lambda _t: []
        )
        assert subs == ["text"]

    def test_llm_decomp_splits_and_labels_backend(
        self, llm_decomposer: AtomicClaimDecomposer
    ) -> None:
        subs, backend = _subclaims(
            "llm-decomp",
            "A happened and B happened",
            decomposer=llm_decomposer,
            splitter=split_sentences,
        )
        assert subs == ["A happened", "B happened"]
        assert backend == "llm"

    def test_llm_decomp_fallback_backend(
        self, fallback_decomposer: AtomicClaimDecomposer
    ) -> None:
        subs, backend = _subclaims(
            "llm-decomp",
            "A happened and B happened",
            decomposer=fallback_decomposer,
            splitter=split_sentences,
        )
        assert backend == "sentence-fallback"
        assert subs  # non-empty via the splitter

    def test_llm_decomp_requires_decomposer(self) -> None:
        with pytest.raises(ValueError, match="requires a decomposer"):
            _subclaims("llm-decomp", "x", decomposer=None, splitter=split_sentences)

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown strategy"):
            _subclaims("bogus", "x", decomposer=None, splitter=split_sentences)


# ── _support_score ─────────────────────────────────────────────────


class TestSupportScore:
    """Weakest-link (min) aggregation over sub-claims."""

    def test_min_over_subclaims(self) -> None:
        pred = OverlapPredictor()
        doc = "Bob was born in Rome."
        # "Bob won an Oscar" is unsupported → drives the min to a low value.
        subs = ["Bob was born in Rome", "Bob won an Oscar"]
        assert _support_score(doc, subs, pred) == pytest.approx(0.0)

    def test_single_subclaim_is_its_own_score(self) -> None:
        pred = OverlapPredictor()
        assert _support_score(
            "Carla plays tennis.", ["Carla plays tennis"], pred
        ) == pytest.approx(1.0)


# ── score_configs ──────────────────────────────────────────────────


class TestScoreConfigs:
    """Row scoring across strategies stays paired and skips bad rows."""

    def test_skips_incomplete_rows(self, llm_decomposer: AtomicClaimDecomposer) -> None:
        rows = [*ROWS, {"doc": "", "claim": "x", "label": 1}]
        scores = score_configs(
            rows,
            OverlapPredictor(),
            decomposer=llm_decomposer,
            splitter=split_sentences,
        )
        for strat in CONFIGS:
            assert len(scores[strat].labels) == len(ROWS)

    def test_llm_decomp_granularity_exceeds_regex_on_compounds(
        self, llm_decomposer: AtomicClaimDecomposer
    ) -> None:
        scores = score_configs(
            ROWS,
            OverlapPredictor(),
            decomposer=llm_decomposer,
            splitter=split_sentences,
        )
        # regex keeps the un-punctuated compound whole; the LLM splits it.
        assert scores["regex-decomp"].n_subclaims == [1, 1, 1, 1]
        assert scores["llm-decomp"].n_subclaims == [2, 2, 1, 1]
        assert scores["llm-decomp"].backends == {"llm": 4}

    def test_decomposition_never_raises_unsupported_score(
        self, llm_decomposer: AtomicClaimDecomposer
    ) -> None:
        scores = score_configs(
            ROWS,
            OverlapPredictor(),
            decomposer=llm_decomposer,
            splitter=split_sentences,
        )
        # Bob row (idx 1): weakest-link min must be <= the whole-claim score.
        assert scores["llm-decomp"].scores[1] <= scores["no-decomp"].scores[1]

    def test_strategy_subset_only_scores_requested(self) -> None:
        scores = score_configs(
            ROWS,
            OverlapPredictor(),
            decomposer=None,
            splitter=split_sentences,
            strategies=("no-decomp", "regex-decomp"),
        )
        assert set(scores) == {"no-decomp", "regex-decomp"}


# ── metrics ────────────────────────────────────────────────────────


class TestMetrics:
    """Balanced accuracy, PRF1, thresholding, sweep."""

    def test_balanced_accuracy_perfect(self) -> None:
        assert _balanced_accuracy([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_balanced_accuracy_degenerate_single_class(self) -> None:
        assert _balanced_accuracy([1, 1, 1], [1, 1, 0]) == 0.0

    def test_prf1_perfect(self) -> None:
        prec, rec, f1 = _prf1([1, 0, 1, 0], [1, 0, 1, 0], target=1)
        assert (prec, rec, f1) == (1.0, 1.0, 1.0)

    def test_prf1_zero_division_guard(self) -> None:
        # No predictions of the target class → precision/recall/f1 all 0.
        prec, rec, f1 = _prf1([0, 0], [0, 0], target=1)
        assert (prec, rec, f1) == (0.0, 0.0, 0.0)

    def test_metrics_at_threshold_separates(self) -> None:
        m = metrics_at_threshold([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
        assert m["balanced_accuracy"] == 1.0
        assert m["hallucination_f1"] == 1.0
        assert m["supported_f1"] == 1.0
        assert m["threshold"] == 0.5

    def test_threshold_direction(self) -> None:
        # A threshold above every score labels all as hallucination (0).
        m = metrics_at_threshold([1, 1], [0.4, 0.4], 0.95)
        assert m["supported_recall"] == 0.0

    def test_best_threshold_maximises_ba(self) -> None:
        b = best_threshold([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
        assert b["balanced_accuracy"] == 1.0
        assert 0.2 < b["threshold"] <= 0.8


# ── summarise ──────────────────────────────────────────────────────


class TestSummarise:
    """Report assembly and the llm−regex delta block."""

    def test_full_report_has_delta(self, llm_decomposer: AtomicClaimDecomposer) -> None:
        scores = score_configs(
            ROWS,
            OverlapPredictor(),
            decomposer=llm_decomposer,
            splitter=split_sentences,
        )
        report = summarise(scores)
        assert set(report["per_config"]) == set(CONFIGS)
        delta = report["delta_llm_minus_regex"]
        assert set(delta) == {
            "hallucination_f1_fixed",
            "balanced_accuracy_fixed",
            "hallucination_f1_oracle",
            "balanced_accuracy_oracle",
        }

    def test_per_config_block_shape(
        self, llm_decomposer: AtomicClaimDecomposer
    ) -> None:
        scores = score_configs(
            ROWS,
            OverlapPredictor(),
            decomposer=llm_decomposer,
            splitter=split_sentences,
        )
        block = summarise(scores)["per_config"]["llm-decomp"]
        assert block["samples"] == 4
        assert block["positives"] == 2
        assert block["avg_subclaims"] == pytest.approx(1.5)
        assert "fixed_0.5" in block and "oracle" in block

    def test_delta_absent_without_both_configs(self) -> None:
        scores = score_configs(
            ROWS,
            OverlapPredictor(),
            decomposer=None,
            splitter=split_sentences,
            strategies=("no-decomp",),
        )
        report = summarise(scores)
        assert "delta_llm_minus_regex" not in report

    def test_empty_config_scores_avg_subclaims_zero(self) -> None:
        report = summarise({"no-decomp": ConfigScores(strategy="no-decomp")})
        assert report["per_config"]["no-decomp"]["avg_subclaims"] == 0.0


# ── run_claim_decomp_benchmark (injected) ──────────────────────────


class TestRunBenchmark:
    """Orchestration with everything injected (no models, no dataset)."""

    def test_injected_run_returns_report(
        self, llm_decomposer: AtomicClaimDecomposer
    ) -> None:
        report = run_claim_decomp_benchmark(
            rows=ROWS,
            predictor=OverlapPredictor(),
            decomposer=llm_decomposer,
            splitter=split_sentences,
            decomposer_model="fake",
        )
        assert report["meta"]["samples"] == 4
        assert report["meta"]["decomposer_model"] == "fake"
        assert "elapsed_s" in report["meta"]
        assert "delta_llm_minus_regex" in report
