# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Temporal Consistency Graph Tests
"""Multi-angle tests for structured-claim temporal contradiction detection."""

from __future__ import annotations

import math

import pytest

from director_ai.core.temporal_consistency import (
    FUNCTIONAL_VALUE,
    POLARITY,
    TemporalClaim,
    TemporalConsistencyGraph,
    TemporalContradiction,
)


def _claim(**kw) -> TemporalClaim:
    base = {"subject": "patient:1", "predicate": "has_condition", "timestamp": 1.0}
    base.update(kw)
    return TemporalClaim(**base)


class TestTemporalClaim:
    def test_requires_subject(self):
        with pytest.raises(ValueError, match="subject"):
            _claim(subject="  ")

    def test_requires_predicate(self):
        with pytest.raises(ValueError, match="predicate"):
            _claim(predicate="")

    def test_rejects_bad_tenant(self):
        with pytest.raises(ValueError, match="tenant"):
            _claim(tenant_id="bad tenant!")

    def test_allows_empty_tenant(self):
        assert _claim(tenant_id="").tenant_id == ""

    def test_rejects_non_finite_timestamp(self):
        with pytest.raises(ValueError, match="finite"):
            _claim(timestamp=math.inf)

    def test_key(self):
        assert _claim(tenant_id="t1").key == ("t1", "patient:1", "has_condition")

    def test_to_dict_omits_source_text_by_default(self):
        d = _claim(source_text="raw phrasing", value="diabetes").to_dict()
        assert "source_text" not in d
        assert d["value"] == "diabetes"

    def test_to_dict_includes_source_text_when_requested(self):
        d = _claim(source_text="raw").to_dict(include_text=True)
        assert d["source_text"] == "raw"

    def test_subject_and_predicate_trimmed(self):
        claim = _claim(subject="  s ", predicate=" p ")
        assert claim.subject == "s"
        assert claim.predicate == "p"


class TestPolarityContradiction:
    def test_assert_then_negate_contradicts(self):
        g = TemporalConsistencyGraph()
        g.record(
            _claim(timestamp=1.0, value="diabetes", polarity=True, session_id="mon")
        )
        found = g.record(
            _claim(timestamp=2.0, value="diabetes", polarity=False, session_id="tue")
        )
        assert len(found) == 1
        assert found[0].kind == POLARITY
        assert found[0].earlier.session_id == "mon"
        assert found[0].later.session_id == "tue"

    def test_same_polarity_same_value_no_contradiction(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="diabetes"))
        assert g.record(_claim(timestamp=2.0, value="diabetes")) == ()

    def test_different_value_no_polarity_contradiction(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="diabetes", polarity=True))
        # Negating a *different* value is not a contradiction of the first.
        assert g.record(_claim(timestamp=2.0, value="asthma", polarity=False)) == ()

    def test_out_of_order_timestamps_ordered(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=5.0, value="x", polarity=True, session_id="late"))
        found = g.record(
            _claim(timestamp=2.0, value="x", polarity=False, session_id="early")
        )
        assert found[0].earlier.session_id == "early"
        assert found[0].later.session_id == "late"


class TestFunctionalContradiction:
    def test_functional_value_change_contradicts(self):
        g = TemporalConsistencyGraph(functional_predicates={"diagnosis"})
        g.record(_claim(predicate="diagnosis", timestamp=1.0, value="diabetes"))
        found = g.record(_claim(predicate="diagnosis", timestamp=2.0, value="healthy"))
        assert [c.kind for c in found] == [FUNCTIONAL_VALUE]

    def test_non_functional_value_change_is_allowed(self):
        g = TemporalConsistencyGraph()  # has_condition is multi-valued
        g.record(_claim(timestamp=1.0, value="diabetes"))
        assert g.record(_claim(timestamp=2.0, value="asthma")) == ()

    def test_functional_requires_both_positive_polarity(self):
        g = TemporalConsistencyGraph(functional_predicates={"diagnosis"})
        g.record(_claim(predicate="diagnosis", timestamp=1.0, value="diabetes"))
        # A negation of a different value is not a functional-value conflict.
        assert (
            g.record(
                _claim(
                    predicate="diagnosis",
                    timestamp=2.0,
                    value="healthy",
                    polarity=False,
                )
            )
            == ()
        )

    def test_functional_requires_both_values_present(self):
        g = TemporalConsistencyGraph(functional_predicates={"diagnosis"})
        g.record(_claim(predicate="diagnosis", timestamp=1.0, value=""))
        assert g.record(_claim(predicate="diagnosis", timestamp=2.0, value="x")) == ()

    def test_functional_predicates_property(self):
        g = TemporalConsistencyGraph(functional_predicates={"diagnosis", ""})
        assert g.functional_predicates == frozenset({"diagnosis"})


class TestMultiplePriors:
    def test_contradicts_each_conflicting_prior(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True))
        g.record(_claim(timestamp=2.0, value="x", polarity=True))
        # A negation now conflicts with BOTH prior assertions.
        found = g.record(_claim(timestamp=3.0, value="x", polarity=False))
        assert len(found) == 2
        assert {c.kind for c in found} == {POLARITY}


class TestQueriesAndTenancy:
    def test_history_is_chronological(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=3.0, session_id="c"))
        g.record(_claim(timestamp=1.0, session_id="a"))
        g.record(_claim(timestamp=2.0, session_id="b"))
        assert [c.session_id for c in g.history("patient:1", "has_condition")] == [
            "a",
            "b",
            "c",
        ]

    def test_tenant_isolation_no_cross_contradiction(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True, tenant_id="t1"))
        # Same subject/value/opposite polarity but a different tenant: no conflict.
        assert (
            g.record(_claim(timestamp=2.0, value="x", polarity=False, tenant_id="t2"))
            == ()
        )

    def test_contradictions_filtered_by_tenant(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True, tenant_id="t1"))
        g.record(_claim(timestamp=2.0, value="x", polarity=False, tenant_id="t1"))
        assert len(g.contradictions(tenant_id="t1")) == 1
        assert g.contradictions(tenant_id="t2") == ()
        assert len(g.contradictions()) == 1

    def test_subjects_and_counts(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(subject="a", timestamp=1.0, tenant_id="t1"))
        g.record(_claim(subject="b", timestamp=1.0, tenant_id="t1"))
        g.record(_claim(subject="c", timestamp=1.0, tenant_id="t2"))
        assert g.subjects(tenant_id="t1") == ("a", "b")
        assert g.claim_count(tenant_id="t1") == 2
        assert g.claim_count() == 3

    def test_delete_tenant_removes_claims_and_contradictions(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True, tenant_id="t1"))
        g.record(_claim(timestamp=2.0, value="x", polarity=False, tenant_id="t1"))
        assert len(g.contradictions(tenant_id="t1")) == 1
        removed = g.delete_tenant("t1")
        assert removed == 2
        assert g.claim_count(tenant_id="t1") == 0
        assert g.contradictions(tenant_id="t1") == ()


class TestReport:
    def test_clean_report_is_consistent(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="diabetes"))
        report = g.report()
        assert report["consistent"] is True
        assert report["contradiction_count"] == 0
        assert report["claim_count"] == 1

    def test_report_lists_contradictions_tenant_safe(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True, source_text="secret"))
        g.record(_claim(timestamp=2.0, value="x", polarity=False, source_text="secret"))
        report = g.report()
        assert report["consistent"] is False
        assert report["contradiction_count"] == 1
        record = report["contradictions"][0]
        assert "source_text" not in record["earlier"]
        assert "source_text" not in record["later"]

    def test_report_can_include_text(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True, source_text="raw"))
        g.record(_claim(timestamp=2.0, value="x", polarity=False, source_text="raw"))
        report = g.report(include_text=True)
        assert report["contradictions"][0]["earlier"]["source_text"] == "raw"


class TestContradictionToDict:
    def test_to_dict_structure(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True))
        found = g.record(_claim(timestamp=2.0, value="x", polarity=False))
        d = found[0].to_dict()
        assert d["kind"] == POLARITY
        assert d["subject"] == "patient:1"
        assert d["predicate"] == "has_condition"
        assert d["earlier"]["polarity"] is True
        assert d["later"]["polarity"] is False

    def test_is_dataclass_instance(self):
        g = TemporalConsistencyGraph()
        g.record(_claim(timestamp=1.0, value="x", polarity=True))
        found = g.record(_claim(timestamp=2.0, value="x", polarity=False))
        assert isinstance(found[0], TemporalContradiction)


class TestProductionGuardWiring:
    def test_guard_temporal_consistency_persists_and_detects(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        graph = guard.temporal_consistency
        # Same instance across calls -> cross-session tracking on one guard.
        assert guard.temporal_consistency is graph
        graph.record(_claim(timestamp=1.0, value="x", polarity=True, session_id="s1"))
        found = graph.record(
            _claim(timestamp=2.0, value="x", polarity=False, session_id="s2")
        )
        assert found[0].kind == POLARITY
