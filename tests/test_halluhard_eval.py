# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HalluHard benchmark harness tests
"""Offline tests for the HalluHard grounding harness.

Exercises the JSONL loader (research_question field, blank-line and empty-field
skipping, max_samples, missing file), the sample and metric aggregates
(grounded_fraction / citation_coverage, micro-averaged grounded_rate /
hallucination_rate, the vacuous empty case), and a full ``run_halluhard`` pass
driving a stub generator's well-formed citing answer through the real resolve →
fetch → judge chain with stubs. No model, network, or dataset download.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.halluhard_eval import (
    HalluHardMetrics,
    HalluHardSample,
    _load_halluhard,
    run_halluhard,
)
from director_ai.core.citation_grounding import SourceFetcher

_ANSWER = (
    "Mass and radius constrain the equation of state [1]. "
    "The sky is green cheese [2].\n\n"
    "References:\n"
    "[1] Bogdanov 10.3847/2041-8213/ab50c5\n"
    "[2] Nonsense arXiv:2411.04368\n"
)


class _FakeGen:
    def generate_candidates(self, prompt, n):
        return [{"text": _ANSWER}]


class _StubScorer:
    def score(self, premise, hypothesis):
        return 0.1 if "equation of state" in premise else 0.95


class _StubHttp:
    def get(self, url, *, headers=None):
        if "crossref" in url:
            return (
                200,
                b'{"message":{"title":["T"],'
                b'"abstract":"Mass and radius constrain the equation of state."}}',
                "application/json",
            )
        if "arxiv" in url:
            return (
                200,
                b"<feed><entry><title>X</title>"
                b"<summary>A benchmark, nothing about cheese.</summary></entry></feed>",
                "application/atom+xml",
            )
        return 404, b"", ""


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


class TestLoader:
    def test_loads_research_questions(self, tmp_path):
        src = _write_jsonl(
            tmp_path / "hh.jsonl",
            [{"research_question": "Q1?", "doi": "x"}, {"research_question": "Q2?"}],
        )
        records = _load_halluhard(source=src)
        assert [r["research_question"] for r in records] == ["Q1?", "Q2?"]

    def test_skips_blank_and_empty_question(self, tmp_path):
        path = tmp_path / "hh.jsonl"
        path.write_text(
            json.dumps({"research_question": "Q1?"})
            + "\n\n"
            + json.dumps({"research_question": ""})
            + "\n"
            + json.dumps({"doi": "no-question-field"})
            + "\n",
            encoding="utf-8",
        )
        assert len(_load_halluhard(source=str(path))) == 1

    def test_max_samples(self, tmp_path):
        src = _write_jsonl(
            tmp_path / "hh.jsonl",
            [{"research_question": f"Q{i}?"} for i in range(5)],
        )
        assert len(_load_halluhard(2, source=src)) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_halluhard(source=str(tmp_path / "absent.jsonl"))


class TestSampleAndMetrics:
    def test_sample_fractions(self):
        s = HalluHardSample("q", n_claims=4, n_cited=3, n_grounded=2)
        assert s.grounded_fraction == pytest.approx(0.5)
        assert s.citation_coverage == pytest.approx(0.75)

    def test_zero_claims_is_vacuous(self):
        s = HalluHardSample("q", n_claims=0, n_cited=0, n_grounded=0)
        assert s.grounded_fraction == 1.0
        assert s.citation_coverage == 1.0

    def test_metrics_micro_average(self):
        metrics = HalluHardMetrics(
            [
                HalluHardSample("a", n_claims=4, n_cited=4, n_grounded=1),
                HalluHardSample("b", n_claims=6, n_cited=3, n_grounded=4),
            ]
        )
        assert metrics.total_claims == 10
        assert metrics.grounded_rate == pytest.approx(0.5)  # (1+4)/10
        assert metrics.citation_coverage == pytest.approx(0.7)  # (4+3)/10
        assert metrics.hallucination_rate == pytest.approx(0.5)

    def test_empty_metrics_vacuous(self):
        metrics = HalluHardMetrics()
        assert metrics.grounded_rate == 1.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.to_dict()["total_questions"] == 0

    def test_to_dict_shape(self):
        d = HalluHardMetrics([HalluHardSample("q", 2, 2, 1)]).to_dict()
        assert set(d) == {
            "total_questions",
            "total_claims",
            "grounded_rate",
            "citation_coverage",
            "hallucination_rate",
        }


class TestRunHalluHard:
    def _metrics(self, records, **kw):
        return run_halluhard(
            records,
            generator=_FakeGen(),
            scorer=_StubScorer(),
            fetcher=SourceFetcher(http=_StubHttp()),
            followups=(),
            **kw,
        )

    def test_single_turn_chain(self):
        records = [{"research_question": "How is the EOS constrained?"}]
        metrics = self._metrics(records)
        # Two assertions: the EOS claim is grounded, the cheese claim is not.
        assert metrics.total == 1
        assert metrics.total_claims == 2
        assert metrics.grounded_rate == pytest.approx(0.5)
        assert metrics.citation_coverage == pytest.approx(1.0)

    def test_multi_turn_aggregates_across_turns(self):
        records = [{"research_question": "Q?"}]
        metrics = run_halluhard(
            records,
            generator=_FakeGen(),
            scorer=_StubScorer(),
            fetcher=SourceFetcher(http=_StubHttp()),
            followups=("Tell me more.", "Are you sure?"),
        )
        # Three turns × two assertions each.
        assert metrics.total_claims == 6
        assert metrics.grounded_rate == pytest.approx(0.5)

    def test_unfetchable_citation_is_ungrounded(self):
        # Every fetch 404s → no evidence → cited claims are not grounded.
        records = [{"research_question": "Q?"}]
        metrics = run_halluhard(
            records,
            generator=_FakeGen(),
            scorer=_StubScorer(),
            fetcher=SourceFetcher(
                http=type("_404", (), {"get": lambda *a, **k: (404, b"", "")})()
            ),
            followups=(),
        )
        assert metrics.grounded_rate == 0.0
        assert metrics.citation_coverage == pytest.approx(1.0)
