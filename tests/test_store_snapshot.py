# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector store snapshot-audit mixin contracts

"""Contract tests for the vector-store snapshot-audit module.

``director_ai.core.retrieval.vector_store._snapshot`` owns the
deterministic KB snapshot view of
:class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`
(sorted snapshot records, the domain-separated Merkle root, the compact
audit payload). These tests pin where the methods live and the
root/payload determinism; the ledger behaviour matrix stays in
``tests/test_vector_store.py``.
"""

from __future__ import annotations

import hashlib

from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
from director_ai.core.retrieval.vector_store._snapshot import SnapshotAuditMixin

_SNAPSHOT_METHODS = (
    "kb_snapshot_records",
    "kb_snapshot_root",
    "kb_snapshot_audit_record",
    "_snapshot_leaf",
    "_merkle_root_hex",
)


class TestMixinComposition:
    def test_store_composes_the_mixin(self):
        assert issubclass(VectorGroundTruthStore, SnapshotAuditMixin)

    def test_snapshot_methods_live_on_the_mixin_only(self):
        for name in _SNAPSHOT_METHODS:
            assert name in vars(SnapshotAuditMixin)
            assert name not in vars(VectorGroundTruthStore)

    def test_module_exports_only_the_mixin(self):
        import director_ai.core.retrieval.vector_store._snapshot as module

        assert module.__all__ == ["SnapshotAuditMixin"]


class TestDeterminism:
    def test_empty_snapshot_root_is_the_domain_separated_empty_digest(self):
        store = VectorGroundTruthStore()
        expected = hashlib.sha256(b"director-ai/kb-snapshot/v1/empty").hexdigest()
        assert store.kb_snapshot_root() == expected

    def test_same_content_yields_same_root_across_stores(self):
        def build() -> VectorGroundTruthStore:
            store = VectorGroundTruthStore(tenant_id="t1")
            store.add_fact("b", "second fact")
            store.add_fact("a", "first fact")
            return store

        assert build().kb_snapshot_root() == build().kb_snapshot_root()

    def test_records_are_sorted_by_tenant_key_kind_index(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("b", "second fact")
        store.add_fact("a", "first fact")
        assert [r["key"] for r in store.kb_snapshot_records()] == ["a", "b"]

    def test_audit_record_counts_ledgers_and_matches_root(self):
        store = VectorGroundTruthStore(tenant_id="t1")
        store.add_fact("dose", "10 mg daily")
        store.retract_fact("dose", reason="withdrawn")
        payload = store.kb_snapshot_audit_record()
        assert payload["event"] == "kb_snapshot"
        assert payload["record_count"] == 1
        assert payload["retraction_count"] == 1
        assert payload["merkle_root"] == store.kb_snapshot_root()
