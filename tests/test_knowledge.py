# SPDX-License-Identifier: Apache-2.0
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

import pytest

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

    def test_retrieve_matches_ingested_chunk_by_value(self):
        # Regression (LOW-5): API-ingested chunks are keyed by an opaque cid with
        # no query words, so key-only matching never found them by content. The
        # value-aware fallback must retrieve the chunk text.
        store = GroundTruthStore()
        store.facts["mydoc:chunk:0"] = "the sky is blue and clear today"
        result = store.retrieve_context("what colour is the sky")
        assert result is not None
        assert "blue" in result

    def test_value_match_respects_tenant(self):
        store = GroundTruthStore()
        store.facts["acme:mydoc:chunk:0"] = "the sky is blue and clear today"
        assert "blue" in (
            store.retrieve_context("what colour is the sky", tenant_id="acme") or ""
        )
        assert (
            store.retrieve_context("what colour is the sky", tenant_id="other") is None
        )

    def test_value_match_stopwords_do_not_overmatch(self):
        # A query of only function words must not match a value on stop words.
        store = GroundTruthStore()
        store.facts["doc:chunk:0"] = "the meeting is at noon"
        assert store.retrieve_context("what is the") is None

    def test_value_match_ranks_below_curated_key_hit(self):
        # A curated, semantically-keyed fact outranks a value-only content match.
        store = GroundTruthStore()
        store.add("refund policy", "curated answer")
        store.facts["doc:chunk:0"] = "our refund policy is generous"
        result = store.retrieve_context("refund policy", top_k=1)
        assert result == "curated answer"

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("", "value", "key"),
            ("   ", "value", "key"),
            ("key", "", "value"),
            ("key", "   ", "value"),
        ],
    )
    def test_add_rejects_empty_fact_fields(self, key, value, message):
        store = GroundTruthStore()
        with pytest.raises(ValueError, match=message):
            store.add(key, value)

    @pytest.mark.parametrize("query", ["", "   "])
    def test_retrieve_rejects_empty_query(self, query):
        store = GroundTruthStore.with_demo_facts()
        with pytest.raises(ValueError, match="query"):
            store.retrieve_context(query)

    def test_retrieve_rejects_negative_top_k(self):
        store = GroundTruthStore.with_demo_facts()
        with pytest.raises(ValueError, match="top_k"):
            store.retrieve_context("sky", top_k=-1)

    def test_retrieve_ranks_by_overlap(self):
        store = GroundTruthStore()
        store.add("refund policy", "policy hit")
        store.add("refund process details", "process hit")
        result = store.retrieve_context("refund policy details", top_k=1)
        assert result == "policy hit"

    def test_retrieval_ranks_by_word_overlap(self):
        # _word_overlap now delegates to the shared director_ai.core.text_overlap
        # helper (dispatch + fallback covered by test_text_overlap); this pins the
        # retrieval ranking that relies on the overlap value.
        store = GroundTruthStore()
        store.add("refund policy", "policy first")
        store.add("weather report", "weather second")
        result = store.retrieve_context("refund policy", top_k=1)
        assert result == "policy first"
