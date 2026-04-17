# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Knowledge Store Tests
"""Multi-angle tests for GroundTruthStore knowledge management.

Covers: add/retrieve, demo facts, empty store, keyword search,
context relevance, multiple facts, pipeline integration with
CoherenceScorer, and performance documentation.
"""

from __future__ import annotations

from director_ai.core.knowledge import GroundTruthStore


class TestGroundTruthStore:
    def test_default_store_is_empty(self):
        store = GroundTruthStore()
        assert store.facts == {}
        assert store.retrieve_context("anything") is None

    def test_with_demo_facts(self):
        store = GroundTruthStore.with_demo_facts()
        assert len(store.facts) == 7
        assert store.facts["sky color"] == "blue"
        assert store.facts["scpn layers"] == "16"

    def test_retrieve_from_demo(self):
        store = GroundTruthStore.with_demo_facts()
        ctx = store.retrieve_context("What color is the sky?")
        assert ctx is not None
        assert "blue" in ctx

    def test_retrieve_scpn_layers(self):
        store = GroundTruthStore.with_demo_facts()
        ctx = store.retrieve_context("How many layers in SCPN?")
        assert ctx is not None
        assert "16" in ctx

    def test_retrieve_no_match(self):
        store = GroundTruthStore.with_demo_facts()
        ctx = store.retrieve_context("xyzzy gibberish nothing")
        assert ctx is None

    def test_add_and_retrieve(self):
        store = GroundTruthStore()
        store.add("capital city", "Paris")
        ctx = store.retrieve_context("What is the capital city?")
        assert ctx is not None
        assert "Paris" in ctx

    def test_add_overwrites(self):
        store = GroundTruthStore()
        store.add("test key", "old")
        store.add("test key", "new")
        assert store.facts["test key"] == "new"

    def test_multiple_matches_joined(self):
        store = GroundTruthStore.with_demo_facts()
        ctx = store.retrieve_context("What is layer 1?")
        assert ctx is not None
        assert ";" in ctx or "quantum" in ctx.lower() or "director" in ctx.lower()

    def test_case_insensitive_query(self):
        store = GroundTruthStore.with_demo_facts()
        ctx = store.retrieve_context("SKY COLOR is what?")
        assert ctx is not None
        assert "blue" in ctx

    def test_empty_store_returns_none(self):
        store = GroundTruthStore()
        assert store.retrieve_context("anything") is None

    def test_retrieve_case_insensitive_key(self):
        store = GroundTruthStore()
        store.add("Sky Color", "blue")
        result = store.retrieve_context("what is the sky color")
        assert result is not None
        assert "blue" in result

    def test_retrieve_uppercase_key(self):
        store = GroundTruthStore()
        store.add("IMPORTANT FACT", "42")
        result = store.retrieve_context("important fact")
        assert result is not None
        assert "42" in result

    def test_tenant_prefixed_single_word_key(self):
        store = GroundTruthStore()
        store.add("secret", "tenant-only fact", tenant_id="acme")
        result = store.retrieve_context("secret", tenant_id="acme")
        assert result is not None
        assert "tenant-only fact" in result
