# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from director_ai.core import GroundTruthStore


def test_empty_store_retrieval():
    store = GroundTruthStore()
    result = store.retrieve_context("What is the refund policy?")
    # Should return empty, not crash
    assert result == "" or result is None


def test_add_then_retrieve():
    store = GroundTruthStore()
    store.add_fact("refund", "Refunds within 30 days only")
    result = store.retrieve_context("refund policy")
    assert "30 days" in result


def test_empty_query():
    store = GroundTruthStore()
    store.add_fact("test", "test value")
    result = store.retrieve_context("")
    assert result is None or isinstance(result, str)


def test_very_long_fact():
    store = GroundTruthStore()
    long_fact = "word " * 50_000
    store.add_fact("long", long_fact)
    result = store.retrieve_context("long fact")
    assert isinstance(result, str)
