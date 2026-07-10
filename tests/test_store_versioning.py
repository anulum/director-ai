# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector store version-ledger mixin contracts

"""Contract tests for the vector-store version-ledger module.

``director_ai.core.retrieval.vector_store._versioning`` owns the
semantic-version bookkeeping of
:class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`
(version records, retraction/replacement ledgers, freshness signals).
These tests pin where the methods live and the ledger's write/read
round-trip; the full behaviour matrix (tenant isolation, conflict
interplay, retrieval filtering) stays in ``tests/test_vector_store.py``.
"""

from __future__ import annotations

import pytest

from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
from director_ai.core.retrieval.vector_store._versioning import VersionLedgerMixin

_LEDGER_METHODS = (
    "fact_version",
    "fact_version_record",
    "version_manifest",
    "retraction_records",
    "replacement_records",
    "freshness_status_signals",
    "retract_fact",
    "replace_fact",
    "_build_version_metadata",
    "_commit_version_metadata",
    "_normalised_claim_metadata",
    "_version_key",
    "_content_hash",
    "_next_semver",
)


class TestMixinComposition:
    def test_store_composes_the_mixin(self):
        assert issubclass(VectorGroundTruthStore, VersionLedgerMixin)

    def test_ledger_methods_live_on_the_mixin_only(self):
        for name in _LEDGER_METHODS:
            assert name in vars(VersionLedgerMixin)
            assert name not in vars(VectorGroundTruthStore)

    def test_module_exports_only_the_mixin(self):
        import director_ai.core.retrieval.vector_store._versioning as module

        assert module.__all__ == ["VersionLedgerMixin"]


class TestLedgerRoundTrip:
    def test_replace_bumps_version_and_records_superseded_hash(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily")
        first = store.fact_version_record("dose")
        assert first is not None and first["version"] == "1.0.0"

        event = store.replace_fact("dose", "20 mg daily", reason="new guidance")
        assert event["from_version"] == "1.0.0"
        assert event["to_version"] == "1.0.1"
        assert event["from_hash"] == first["content_hash"]
        assert store.replacement_records()[0]["reason"] == "new guidance"

    def test_retract_removes_fact_and_appends_ledger_event(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily")
        event = store.retract_fact("dose", reason="withdrawn")
        assert event["event"] == "retracted"
        assert store.retraction_records()[0]["reason"] == "withdrawn"
        assert store.facts == {}

    def test_retract_unknown_fact_raises_key_error(self):
        store = VectorGroundTruthStore()
        with pytest.raises(KeyError, match="cannot retract unknown fact"):
            store.retract_fact("ghost")

    def test_non_string_reason_is_rejected(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily")
        with pytest.raises(ValueError, match="reason must be a string"):
            store.retract_fact("dose", reason=None)  # type: ignore[arg-type]


class TestFreshnessSignals:
    def test_status_metadata_surfaces_as_numeric_signal(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact(
            "dose",
            "10 mg daily",
            metadata={"kb_citation_status": "active", "kb_source_timestamp": "1700"},
        )
        (signal,) = store.freshness_status_signals()
        assert signal["status"] == "active"
        assert signal["published_at"] == 1700.0

    def test_non_numeric_timestamp_is_rejected(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact(
            "dose",
            "10 mg daily",
            metadata={"kb_source_timestamp": "yesterday"},
        )
        with pytest.raises(ValueError, match="numeric timestamp"):
            store.freshness_status_signals()
