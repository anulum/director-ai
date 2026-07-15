# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — long-context sweep harness tests
"""Multi-angle tests for benchmarks/run_longcontext_bench.py.

Exercises the pure matrix/aggregation/operating-point logic with real
collaborators (the production sentence splitter and the benchmark passage
anchorer) and plain protocol fakes — no mock library. The heavy paths
(model construction, dataset download, ``main``) run only on the metered
GPU pass and are out of scope here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_longcontext_bench import (
    AGGREGATIONS,
    BASELINE_CATCH,
    MATCHED_FPR,
    TASK_VARIANTS,
    MatrixEntry,
    Row,
    aggregate,
    build_premise,
    matrix_from_json,
    matrix_to_json,
    operating_points,
    rows_from_halueval,
    run_longcontext_bench,
    score_matrix,
    summarise,
    threshold_at_matched_fpr,
)
from director_ai.core.text_segmentation import split_sentences

# ── Collaborators ──────────────────────────────────────────────────


class OverlapPredictor:
    """Fake checker: support = fraction of claim words present in premise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, premise: str, hypothesis: str) -> float:
        self.calls.append((premise, hypothesis))
        words = {w.lower().strip(".,") for w in hypothesis.split() if len(w) > 3}
        if not words:
            return 1.0
        prem = {w.lower().strip(".,") for w in premise.split()}
        return len(words & prem) / len(words)


def _summ_sample(doc: str, right: str, bad: str) -> dict:
    return {"document": doc, "right_summary": right, "hallucinated_summary": bad}


def _dlg_sample(knowledge: str, history: str, right: str, bad: str) -> dict:
    return {
        "knowledge": knowledge,
        "dialogue_history": history,
        "right_response": right,
        "hallucinated_response": bad,
    }


# ── rows_from_halueval ─────────────────────────────────────────────


class TestRowExtraction:
    def test_summarization_yields_paired_rows(self):
        rows = rows_from_halueval(
            "summarization", [_summ_sample("The sky is blue.", "Sky blue.", "Sky red.")]
        )
        assert [r.hallucinated for r in rows] == [False, True]
        assert all(r.task == "summarization" for r in rows)
        assert all(r.doc == "The sky is blue." for r in rows)
        assert all(r.history == "" for r in rows)

    def test_dialogue_carries_knowledge_and_history_separately(self):
        rows = rows_from_halueval(
            "dialogue",
            [
                _dlg_sample(
                    "Zodiac stars Jake Gyllenhaal.",
                    "A: seen Zodiac?",
                    "Yes.",
                    "Tom Hanks stars.",
                )
            ],
        )
        assert all(r.doc == "Zodiac stars Jake Gyllenhaal." for r in rows)
        assert all(r.history == "A: seen Zodiac?" for r in rows)

    def test_dialogue_with_only_history_still_yields_rows(self):
        rows = rows_from_halueval(
            "dialogue", [_dlg_sample("", "A: hello", "Hi.", "Bye planet.")]
        )
        assert len(rows) == 2

    def test_missing_fields_are_skipped(self):
        assert rows_from_halueval("summarization", [_summ_sample("", "a", "b")]) == []
        assert rows_from_halueval("summarization", [_summ_sample("doc", "", "")]) == []

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="unsupported task"):
            rows_from_halueval("qa", [])


# ── build_premise ──────────────────────────────────────────────────


class TestEvidenceComposition:
    def test_prefix3000_truncates(self):
        row = Row("summarization", "x" * 4000, "", "resp", False)
        assert build_premise(row, "prefix3000", "claim") == "x" * 3000

    def test_fulldoc_keeps_everything(self):
        row = Row("summarization", "x" * 4000, "", "resp", False)
        assert build_premise(row, "fulldoc", "claim") == "x" * 4000

    def test_anchored_selects_claim_relevant_sentences(self):
        # 8 sentences; only one mentions the claim's entities and it sits
        # PAST a 3000-char prefix in spirit — anchoring must find it anyway.
        filler = " ".join(
            f"Filler sentence number {i} about weather patterns." for i in range(7)
        )
        doc = filler + " The Zodiac film stars Jake Gyllenhaal as Robert Graysmith."
        row = Row("summarization", doc, "", "resp", False)
        premise = build_premise(row, "anchored@5", "Zodiac stars Jake Gyllenhaal")
        assert "Jake Gyllenhaal" in premise
        assert len(premise) < len(doc)

    def test_anchored_empty_selection_degrades_to_prefix(self):
        row = Row("summarization", "Tiny.", "", "resp", False)
        # one <3-word sentence -> no passages -> prefix fallback
        assert build_premise(row, "anchored@5", "claim") == "Tiny."

    def test_dialogue_history_variant_is_current_production_shape(self):
        row = Row("dialogue", "K fact.", "A: hi", "resp", False)
        assert build_premise(row, "history", "claim") == "A: hi"

    def test_dialogue_knowledge_history_prepends_knowledge(self):
        row = Row("dialogue", "K fact.", "A: hi", "resp", False)
        assert build_premise(row, "knowledge+history", "claim") == "K fact.\nA: hi"

    def test_dialogue_knowledge_history_without_knowledge(self):
        row = Row("dialogue", "", "A: hi", "resp", False)
        assert build_premise(row, "knowledge+history", "claim") == "A: hi"

    def test_unknown_variant_raises(self):
        row = Row("summarization", "doc", "", "resp", False)
        with pytest.raises(ValueError, match="unknown evidence variant"):
            build_premise(row, "prefix9000", "claim")


# ── aggregate ──────────────────────────────────────────────────────


class TestAggregations:
    def test_min_is_weakest_link(self):
        assert aggregate([0.9, 0.2, 0.8], "min") == 0.2

    def test_mean(self):
        assert aggregate([0.5, 1.0], "mean") == 0.75

    def test_low2mean_averages_two_weakest(self):
        assert aggregate([0.9, 0.1, 0.3], "low2mean") == pytest.approx(0.2)

    def test_low2mean_single_claim(self):
        assert aggregate([0.4], "low2mean") == 0.4

    def test_coverage_fraction_at_cut(self):
        assert aggregate([0.9, 0.4, 0.6, 0.1], "coverage") == 0.5

    def test_single_fact_swap_dilution_reproduced(self):
        # D3: a 4-claim summary with ONE swapped fact — coverage stays high
        # (looks supported) while weakest-link exposes the bad claim.
        scores = [0.9, 0.85, 0.8, 0.05]
        assert aggregate(scores, "coverage") == 0.75
        assert aggregate(scores, "min") == 0.05

    def test_empty_scores_raise(self):
        with pytest.raises(ValueError, match="at least one claim"):
            aggregate([], "min")

    def test_unknown_aggregation_raises(self):
        with pytest.raises(ValueError, match="unknown aggregation"):
            aggregate([0.5], "median")

    def test_all_declared_aggregations_run(self):
        for agg in AGGREGATIONS:
            assert 0.0 <= aggregate([0.2, 0.7], agg) <= 1.0


# ── operating points ───────────────────────────────────────────────


class TestOperatingPoints:
    def test_threshold_respects_fpr_budget(self):
        good = [i / 100 for i in range(1, 101)]  # 0.01..1.00
        t = threshold_at_matched_fpr(good, 0.05)
        flagged = sum(1 for s in good if s < t)
        assert flagged <= 5
        assert t == pytest.approx(0.06)

    def test_zero_fpr_budget_flags_no_good(self):
        good = [0.3, 0.5, 0.7]
        t = threshold_at_matched_fpr(good, 0.0)
        assert sum(1 for s in good if s < t) == 0

    def test_empty_good_raises(self):
        with pytest.raises(ValueError, match="needs good scores"):
            threshold_at_matched_fpr([], 0.05)

    def test_separable_scores_get_full_catch(self):
        pts = operating_points([0.8, 0.9, 0.95], [0.1, 0.2, 0.3], 0.05)
        assert pts["catch_at_matched_fpr"] == 1.0
        assert pts["actual_fpr"] == 0.0
        assert pts["oracle_balanced_accuracy"] == 1.0

    def test_inseparable_scores_get_no_catch_at_matched_fpr(self):
        same = [0.5] * 10
        pts = operating_points(same, same, 0.045)
        assert pts["catch_at_matched_fpr"] == 0.0
        assert pts["actual_fpr"] == 0.0

    def test_actual_fpr_never_exceeds_target(self):
        good = [0.11, 0.42, 0.55, 0.62, 0.68, 0.71, 0.79, 0.83, 0.9, 0.97]
        bad = [0.05, 0.3, 0.45, 0.5, 0.66, 0.72, 0.8, 0.88, 0.93, 0.99]
        for target in (0.0, 0.025, 0.045, 0.1, 0.3):
            pts = operating_points(good, bad, target)
            assert pts["actual_fpr"] <= target + 1e-12


# ── score_matrix ───────────────────────────────────────────────────


class TestScoreMatrix:
    def test_matrix_covers_every_task_variant(self):
        rows = rows_from_halueval(
            "summarization",
            [_summ_sample("The sky is blue today.", "Sky is blue.", "Sky is green.")],
        ) + rows_from_halueval(
            "dialogue",
            [
                _dlg_sample(
                    "Zodiac stars Jake Gyllenhaal.",
                    "A: seen Zodiac?",
                    "Gyllenhaal stars.",
                    "Tom Hanks stars.",
                )
            ],
        )
        matrix = score_matrix(rows, OverlapPredictor(), splitter=split_sentences)
        assert len(matrix) == 4
        for entry in matrix:
            assert set(entry.scores) == set(TASK_VARIANTS[entry.task])
            for scores in entry.scores.values():
                assert scores and all(0.0 <= s <= 1.0 for s in scores)

    def test_predictor_calls_are_memoised(self):
        # right + hallucinated share the SAME premise for prefix/fulldoc and
        # a repeated claim, so unique calls < naive claims×variants product.
        doc = "Alpha beta gamma delta epsilon zeta."
        rows = rows_from_halueval(
            "summarization",
            [_summ_sample(doc, "Alpha beta gamma.", "Alpha beta gamma.")],
        )
        predictor = OverlapPredictor()
        score_matrix(rows, predictor, splitter=split_sentences)
        # identical (premise, claim) pairs must hit the checker exactly once
        assert len(predictor.calls) == len(set(predictor.calls))

    def test_unsplittable_response_degrades_to_whole_response(self):
        rows = [Row("dialogue", "K.", "H.", "ok", False)]
        matrix = score_matrix(rows, OverlapPredictor(), splitter=lambda _s: [])
        assert all(len(v) == 1 for v in matrix[0].scores.values())

    def test_d1_knowledge_variant_scores_higher_for_grounded_reply(self):
        # The D1 mechanism end-to-end: a reply refutable only from knowledge
        # scores low vs history alone but high once knowledge is composed in.
        rows = [
            Row(
                "dialogue",
                "The Zodiac film stars Jake Gyllenhaal as Robert Graysmith.",
                "Speaker A: Have you seen the Zodiac movie?",
                "Zodiac stars Jake Gyllenhaal.",
                False,
            )
        ]
        matrix = score_matrix(rows, OverlapPredictor(), splitter=split_sentences)
        entry = matrix[0]
        assert max(entry.scores["knowledge+history"]) > max(entry.scores["history"])


# ── data-loader import surface ─────────────────────────────────────


class TestHaluEvalDataSurface:
    def test_loader_module_is_pytest_free(self):
        # The 2026-07-15 metered run failed because halueval_eval imports
        # pytest at module level; the loader must stay importable lean.
        import ast

        src = (
            Path(__file__).resolve().parent.parent / "benchmarks" / "_halueval_data.py"
        ).read_text(encoding="utf-8")
        names = {
            alias.name.split(".")[0]
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "pytest" not in names

    def test_halueval_eval_re_exports_loader(self):
        from benchmarks import _halueval_data, halueval_eval

        assert halueval_eval._download_task_data.__module__ == _halueval_data.__name__
        assert halueval_eval._DATASET_URLS is _halueval_data._DATASET_URLS


# ── matrix serialisation round-trip ────────────────────────────────


class TestMatrixSerialisation:
    def test_round_trip_preserves_everything(self):
        matrix = [
            MatrixEntry("summarization", True, {"fulldoc": [0.1, 0.9]}),
            MatrixEntry("dialogue", False, {"history": [0.5]}),
        ]
        recovered = matrix_from_json(matrix_to_json(matrix))
        assert [e.task for e in recovered] == ["summarization", "dialogue"]
        assert [e.hallucinated for e in recovered] == [True, False]
        assert recovered[0].scores == {"fulldoc": [0.1, 0.9]}


# ── summarise + end-to-end offline run ─────────────────────────────


def _synthetic_matrix() -> list[MatrixEntry]:
    """Deterministic matrix where anchored/knowledge variants separate the
    classes and the production baselines (prefix/history) do not."""
    entries: list[MatrixEntry] = []
    for i in range(20):
        good_anchor = 0.8 + (i % 5) * 0.02
        bad_anchor = 0.1 + (i % 5) * 0.02
        entries.append(
            MatrixEntry(
                "summarization",
                False,
                {
                    "prefix3000": [0.5],
                    "fulldoc": [0.55 + (i % 3) * 0.01],
                    "anchored@5": [good_anchor, 0.9],
                },
            )
        )
        entries.append(
            MatrixEntry(
                "summarization",
                True,
                {
                    "prefix3000": [0.5],
                    "fulldoc": [0.54 + (i % 3) * 0.01],
                    "anchored@5": [bad_anchor, 0.9],
                },
            )
        )
        entries.append(
            MatrixEntry(
                "dialogue",
                False,
                {"history": [0.5], "knowledge+history": [0.85 + (i % 4) * 0.01]},
            )
        )
        entries.append(
            MatrixEntry(
                "dialogue",
                True,
                {"history": [0.5], "knowledge+history": [0.15 + (i % 4) * 0.01]},
            )
        )
    return entries


class TestSummarise:
    def test_report_shape_and_grid_coverage(self):
        report = summarise(_synthetic_matrix())
        for task, variants in TASK_VARIANTS.items():
            block = report[task]
            assert block["n_good"] == block["n_bad"] == 20
            assert block["matched_fpr_target"] == MATCHED_FPR[task]
            assert block["baseline_catch_tracked_200"] == BASELINE_CATCH[task]
            assert set(block["grid"]) == set(variants)
            for cell in block["grid"].values():
                assert set(cell) == set(AGGREGATIONS)

    def test_separating_variants_win_over_flat_baselines(self):
        report = summarise(_synthetic_matrix())
        assert (
            report["summarization"]["best_by_matched_catch"]["variant"] == "anchored@5"
        )
        assert (
            report["dialogue"]["best_by_matched_catch"]["variant"]
            == "knowledge+history"
        )
        # flat baseline variants cannot catch anything at the matched FPR
        for task, baseline_variant in (
            ("summarization", "prefix3000"),
            ("dialogue", "history"),
        ):
            cell = report[task]["grid"][baseline_variant]["min"]
            assert cell["catch_at_matched_fpr"] == 0.0

    def test_missing_task_omitted(self):
        only_summ = [e for e in _synthetic_matrix() if e.task == "summarization"]
        report = summarise(only_summ)
        assert "dialogue" not in report


class TestRunOffline:
    def test_run_with_injected_rows_and_predictor(self):
        rows = rows_from_halueval(
            "summarization",
            [
                _summ_sample(
                    "The sky is blue today over Paris.",
                    "Sky is blue over Paris.",
                    "Sky is green over Cairo.",
                )
            ],
        ) + rows_from_halueval(
            "dialogue",
            [
                _dlg_sample(
                    "Zodiac stars Jake Gyllenhaal.",
                    "A: seen Zodiac?",
                    "It stars Jake Gyllenhaal.",
                    "It stars Tom Hanks.",
                )
            ],
        )
        report, matrix = run_longcontext_bench(
            rows=rows, predictor=OverlapPredictor(), splitter=split_sentences
        )
        assert report["meta"]["rows"] == 4
        assert set(report["per_task"]) == {"summarization", "dialogue"}
        assert len(matrix) == 4

    def test_run_from_prescored_matrix_needs_no_predictor(self):
        report, matrix = run_longcontext_bench(matrix=_synthetic_matrix())
        assert report["meta"]["rows"] == 80
        assert len(matrix) == 80
        best = report["per_task"]["dialogue"]["best_by_matched_catch"]
        assert best["catch_at_matched_fpr"] == 1.0

    def test_meta_records_matched_fpr_provenance(self):
        report, _ = run_longcontext_bench(matrix=_synthetic_matrix())
        assert (
            report["meta"]["matched_fpr_source"]
            == "benchmarks/results/judge_bench_nli_only_200.json"
        )
