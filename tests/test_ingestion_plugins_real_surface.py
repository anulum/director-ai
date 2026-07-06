# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - ingestion plugin real-surface tests
"""Real-surface coverage for external ingestion plugins and KB storage."""

from __future__ import annotations

from collections.abc import Iterator

from director_ai.core.retrieval.ingestion import IngestedDocument, IngestionPlugin
from director_ai.core.retrieval.ingestion.base import chunks
from director_ai.core.retrieval.vector_store.store import VectorGroundTruthStore
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _PublicSurfacePlugin(IngestionPlugin):
    """Small plugin using only the public ingestion protocol."""

    name = "public-surface"

    def iter_documents(self) -> Iterator[IngestedDocument]:
        """Yield production-shaped records as an external plugin would."""
        yield IngestedDocument(
            key="refund-policy",
            text="Refund approvals require a signed operator receipt.",
            source=self.name,
            source_id="runbook-42",
            metadata={
                "external_id": "drive-runbook-42",
                "source_timestamp": "1710000000.0",
                "updated_timestamp": "1710000300.0",
                "citation_status": "fresh",
                "status_observed_at": "1710000600.0",
            },
        )
        yield IngestedDocument(
            key="blank-note",
            text="   ",
            source=self.name,
            source_id="blank-1",
        )


def test_ingestion_plugin_unit_guard_declares_this_real_surface_companion() -> None:
    """The stub-heavy ingestion unit guard must point at this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_ingestion_plugins.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_ingestion_plugins_real_surface.py" in reason


def test_public_ingestion_plugin_writes_vector_store_provenance() -> None:
    """A public plugin should write retrievable, tenant-scoped KB evidence."""
    store = VectorGroundTruthStore(tenant_id="tenant-alpha")
    plugin = _PublicSurfacePlugin()

    written = plugin.ingest(store, tenant_id="tenant-alpha")
    context = store.retrieve_context("signed refund operator receipt", top_k=1)
    chunks = store.retrieve_context_with_chunks(
        "signed refund operator receipt",
        top_k=1,
    )
    version_record = store.fact_version_record(
        "refund-policy",
        tenant_id="tenant-alpha",
    )
    freshness = store.freshness_status_signals(
        "tenant-alpha",
        key="refund-policy",
    )

    assert written == 1
    assert context == (
        "refund-policy: Refund approvals require a signed operator receipt."
    )
    assert len(chunks) == 1
    assert chunks[0].source == "vector:tenant-alpha::refund-policy"
    assert version_record is not None
    assert version_record["source_id"] == "runbook-42"
    assert version_record["external_id"] == "drive-runbook-42"
    assert version_record["record_kind"] == "fact"
    assert freshness == [
        {
            "source_id": "drive-runbook-42",
            "status": "fresh",
            "status_source": "",
            "published_at": 1710000000.0,
            "updated_at": 1710000300.0,
            "observed_at": 1710000600.0,
        }
    ]


def test_public_chunk_helper_handles_empty_iterables() -> None:
    """The public chunk helper should stream no batches for empty sources."""
    assert list(chunks((), 8)) == []
