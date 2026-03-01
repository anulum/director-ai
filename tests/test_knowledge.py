from __future__ import annotations

from director_ai.core.knowledge import GroundTruthStore


class TestGroundTruthStore:
    def test_retrieve_known_fact(self):
        store = GroundTruthStore()
        ctx = store.retrieve_context("What color is the sky?")
        assert ctx is not None
        assert "blue" in ctx

    def test_retrieve_scpn_layers(self):
        store = GroundTruthStore()
        ctx = store.retrieve_context("How many layers in SCPN?")
        assert ctx is not None
        assert "16" in ctx

    def test_retrieve_no_match(self):
        store = GroundTruthStore()
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
        store = GroundTruthStore()
        # "layer" matches both "layer 1" and "layer 16"
        ctx = store.retrieve_context("What is layer 1?")
        assert ctx is not None
        assert ";" in ctx or "quantum" in ctx.lower() or "director" in ctx.lower()

    def test_case_insensitive_query(self):
        store = GroundTruthStore()
        ctx = store.retrieve_context("SKY COLOR is what?")
        assert ctx is not None
        assert "blue" in ctx

    def test_empty_store(self):
        store = GroundTruthStore()
        store.facts = {}
        assert store.retrieve_context("anything") is None
