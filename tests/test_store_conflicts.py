# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector store conflict-ledger mixin contracts

"""Contract tests for the vector-store conflict-ledger module.

``director_ai.core.retrieval.vector_store._conflicts`` owns the
knowledge-base conflict detection of
:class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`
(retraction overlaps, protected-claim divergence, explicit contradiction
targets). These tests pin where the methods live and the write-path
wiring; the classification and dedupe behaviour matrix stays in
``tests/test_vector_store.py``.
"""

from __future__ import annotations

import pytest

from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
from director_ai.core.retrieval.vector_store._conflicts import ConflictLedgerMixin

_CONFLICT_METHODS = (
    "conflict_reports",
    "_build_conflict_reports",
    "_retraction_conflicts",
    "_protected_claim_conflicts",
    "_explicit_conflicts",
    "_protected_conflict_type",
    "_conflict_record",
    "_record_refs",
    "_first_ref",
    "_metadata_list",
    "_dedupe_conflicts",
)


class TestMixinComposition:
    def test_store_composes_the_mixin(self):
        assert issubclass(VectorGroundTruthStore, ConflictLedgerMixin)

    def test_conflict_methods_live_on_the_mixin_only(self):
        for name in _CONFLICT_METHODS:
            assert name in vars(ConflictLedgerMixin)
            assert name not in vars(VectorGroundTruthStore)

    def test_module_exports_only_the_mixin(self):
        import director_ai.core.retrieval.vector_store._conflicts as module

        assert module.__all__ == ["ConflictLedgerMixin"]


class TestWritePathWiring:
    def test_add_over_retracted_reference_records_a_conflict(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily")
        store.retract_fact("dose", reason="withdrawn")

        store.add_fact("dose", "20 mg daily")
        (report,) = store.conflict_reports(key="dose")
        assert report["conflict_type"] == "retraction_record"
        assert report["reason"] == "new fact overlaps a retracted ledger entry"

    def test_signed_fact_divergence_is_classified_as_protected(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily", metadata={"signed_fact_id": "sig-1"})
        store.add_fact("dose", "20 mg daily", metadata={"signed_fact_id": "sig-1"})
        types = {r["conflict_type"] for r in store.conflict_reports(key="dose")}
        assert "signed_fact" in types

    def test_conflict_reports_validates_blank_key(self):
        store = VectorGroundTruthStore()
        with pytest.raises(ValueError, match="key"):
            store.conflict_reports(key="   ")

    def test_reports_are_tenant_scoped_copies(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily", metadata={"signed_fact_id": "sig-1"})
        store.add_fact("dose", "20 mg daily", metadata={"signed_fact_id": "sig-1"})
        reports = store.conflict_reports()
        assert reports and store.conflict_reports(tenant_id="other") == []
        reports[0]["conflict_type"] = "mutated"
        assert store.conflict_reports()[0]["conflict_type"] != "mutated"
