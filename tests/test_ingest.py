# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” VectorGroundTruthStore Ingest Tests

from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore


class TestIngest:
    def test_ingest_text_list(self):
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        count = store.ingest(["fact one", "fact two", "fact three"])
        assert count == 3
        assert store.backend.count() >= 3

    def test_ingest_empty_list(self):
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        initial_count = store.backend.count()
        count = store.ingest([])
        assert count == 0
        assert store.backend.count() == initial_count

    def test_ingest_retrieves(self):
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        store.ingest(["The refund policy is 30 days from purchase."])
        result = store.retrieve_context("refund policy")
        assert result is not None
        assert "refund" in result.lower()

    def test_add_fact_retrieves(self):
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        store.add_fact("capital", "Paris is the capital of France")
        result = store.retrieve_context("capital of France")
        assert result is not None
        assert "Paris" in result

    def test_retrieve_from_empty_backend(self):
        backend = InMemoryBackend()
        store = VectorGroundTruthStore(backend=backend)
        store.facts.clear()
        result = store.retrieve_context("anything")
        assert result is None
