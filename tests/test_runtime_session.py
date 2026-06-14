# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime ConversationSession tests

from __future__ import annotations

import pytest

from director_ai.core.runtime.contradiction_tracker import ContradictionReport
from director_ai.core.runtime.session import ConversationSession


def _high_divergence(_premise: str, _hypothesis: str) -> float:
    return 0.9


def test_get_contradiction_report_empty_session():
    session = ConversationSession()
    report = session.get_contradiction_report()
    assert isinstance(report, ContradictionReport)
    assert report.pair_count == 0
    assert report.worst_pair is None
    assert report.contradiction_index == 0.0


def test_get_contradiction_report_reflects_updates_without_adding_turn():
    session = ConversationSession()
    session.update_contradictions("the sky is blue", _high_divergence)
    session.update_contradictions("the sky is not blue", _high_divergence)

    # get_contradiction_report reads the current state and must not
    # itself record a turn or mutate the tracker.
    report1 = session.get_contradiction_report()
    report2 = session.get_contradiction_report()

    assert report1.pair_count >= 1
    assert report1.contradiction_index == pytest.approx(0.9)
    assert report1.worst_pair is not None
    # Idempotent read: two consecutive reports are identical, and no
    # conversation turn was appended by reporting.
    assert report2.pair_count == report1.pair_count
    assert len(session) == 0


def test_get_contradiction_report_matches_last_update_return():
    session = ConversationSession()
    session.update_contradictions("revenue rose sharply", _high_divergence)
    returned = session.update_contradictions("revenue fell sharply", _high_divergence)
    fetched = session.get_contradiction_report()
    assert fetched.pair_count == returned.pair_count
    assert fetched.contradiction_index == pytest.approx(returned.contradiction_index)
