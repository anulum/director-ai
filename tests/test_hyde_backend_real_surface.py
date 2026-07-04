# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for HyDE retrieval wiring."""

from __future__ import annotations

import director_ai
from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.hyde import HyDEBackend
from director_ai.core.retrieval.vector_store import (
    InMemoryBackend,
    VectorGroundTruthStore,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _QuestionAnswerGenerator:
    """Deterministic pseudo-document generator preserving the HyDE protocol."""

    def __init__(self, pseudo_document: str) -> None:
        """Store the pseudo-document emitted for every production query."""
        self.pseudo_document = pseudo_document
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        """Record the HyDE prompt and return the configured pseudo-document."""
        self.prompts.append(prompt)
        return self.pseudo_document


def _hyde_store() -> VectorGroundTruthStore:
    """Build the dependency-free production store with HyDE enabled."""
    config = DirectorConfig(
        hybrid_retrieval=False,
        reranker_enabled=False,
        hyde_enabled=True,
        hyde_prompt_template="Answer this with indexed facts: {query}",
    )
    store = config.build_store()
    assert isinstance(store, VectorGroundTruthStore)
    assert isinstance(store.backend, HyDEBackend)
    return store


def test_hyde_backend_unit_guard_declares_this_real_surface_companion() -> None:
    """The legacy HyDE unit guard should declare this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_hyde_backend.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_hyde_backend_real_surface.py" in reason


def test_configured_hyde_store_retrieves_through_raw_query_fallback() -> None:
    """Configured HyDE stores should retrieve through the production store path."""
    store = _hyde_store()
    store.add(
        "refund-policy",
        "Refund approval requires manager signoff before shipment.",
        tenant_id="tenant-alpha",
    )

    context = store.retrieve_context(
        "Which rule controls refund approval?",
        top_k=1,
        tenant_id="tenant-alpha",
    )
    missing_context = store.retrieve_context(
        "Which rule controls refund approval?",
        top_k=1,
        tenant_id="tenant-beta",
    )

    assert context == (
        "refund-policy: Refund approval requires manager signoff before shipment."
    )
    assert missing_context is None


def test_hyde_generator_pseudo_document_drives_real_backend_retrieval() -> None:
    """A real HyDE generator should steer retrieval and annotate result metadata."""
    generator = _QuestionAnswerGenerator("invoice reminders improve revenue retention")
    base = InMemoryBackend()
    backend = HyDEBackend(
        base=base,
        generator=generator,
        template="Produce a factual passage for: {query}",
    )
    backend.add(
        "billing-policy",
        "Invoice reminders improve revenue retention for overdue accounts.",
        {"tenant_id": "tenant-alpha", "source": "policy"},
    )
    backend.add(
        "shipping-policy",
        "Shipping windows are set by warehouse operations.",
        {"tenant_id": "tenant-alpha", "source": "policy"},
    )

    results = backend.query(
        "Which process helps account health?",
        n_results=1,
        tenant_id="tenant-alpha",
    )

    assert generator.prompts == [
        "Produce a factual passage for: Which process helps account health?"
    ]
    assert results[0]["id"] == "billing-policy"
    assert results[0]["text"] == (
        "Invoice reminders improve revenue retention for overdue accounts."
    )
    assert results[0]["metadata"]["source"] == "policy"
    assert results[0]["metadata"]["hyde_original_query"] == (
        "Which process helps account health?"
    )
    assert results[0]["metadata"]["hyde_pseudo_doc"] == (
        "invoice reminders improve revenue retention"
    )


def test_hyde_annotations_do_not_mutate_stored_metadata() -> None:
    """HyDE query annotations should not be written back into indexed metadata."""
    generator = _QuestionAnswerGenerator("invoice reminders retention")
    base = InMemoryBackend()
    backend = HyDEBackend(base=base, generator=generator)
    backend.add(
        "billing-policy",
        "Invoice reminders improve retention.",
        {"tenant_id": "tenant-alpha", "source": "policy"},
    )

    annotated_results = backend.query(
        "Which process helps retention?",
        n_results=1,
        tenant_id="tenant-alpha",
    )
    stored_results = base.query(
        "invoice reminders retention",
        n_results=1,
        tenant_id="tenant-alpha",
    )

    assert "hyde_pseudo_doc" in annotated_results[0]["metadata"]
    assert stored_results[0]["metadata"] == {
        "tenant_id": "tenant-alpha",
        "source": "policy",
    }


def test_public_hyde_export_matches_backend_class() -> None:
    """The package-level lazy export should expose the production backend class."""
    assert director_ai.HyDEBackend is HyDEBackend
