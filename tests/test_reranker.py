# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” RerankedBackend Tests

from __future__ import annotations

from director_ai.core.vector_store import InMemoryBackend


def test_reranked_backend_import_error():
    """RerankedBackend raises ImportError without sentence-transformers."""
    try:
        from director_ai.core.vector_store import RerankedBackend

        base = InMemoryBackend()
        # Will raise ImportError if sentence-transformers not installed
        RerankedBackend(base=base)
    except ImportError as e:
        assert "sentence-transformers" in str(e)


def test_reranked_backend_delegates_add():
    """add() delegates to base backend."""
    from unittest.mock import MagicMock

    from director_ai.core.vector_store import RerankedBackend

    base = InMemoryBackend()
    base.add("d1", "hello world")

    mock_ce = MagicMock()

    rb = RerankedBackend.__new__(RerankedBackend)
    rb._base = base
    rb._multiplier = 3
    rb._reranker = mock_ce

    rb.add("d2", "test doc")
    assert rb.count() == 2


def test_reranked_backend_query_reranks():
    """query() retrieves more candidates then reranks."""
    from unittest.mock import MagicMock

    from director_ai.core.vector_store import RerankedBackend

    base = InMemoryBackend()
    base.add("d1", "cat sat on mat")
    base.add("d2", "dog ran fast")
    base.add("d3", "cat likes fish")

    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs: [
        0.9 - i * 0.3 for i in range(len(pairs))
    ]

    rb = RerankedBackend.__new__(RerankedBackend)
    rb._base = base
    rb._multiplier = 3
    rb._reranker = mock_ce

    results = rb.query("cat", n_results=2)
    assert len(results) <= 2
    mock_ce.predict.assert_called_once()


def test_reranked_backend_empty_store():
    """query() on empty store returns empty list."""
    from unittest.mock import MagicMock

    from director_ai.core.vector_store import RerankedBackend

    base = InMemoryBackend()
    mock_ce = MagicMock()

    rb = RerankedBackend.__new__(RerankedBackend)
    rb._base = base
    rb._multiplier = 3
    rb._reranker = mock_ce

    assert rb.query("test") == []
    mock_ce.predict.assert_not_called()
