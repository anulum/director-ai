# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — standalone interlock tests

from __future__ import annotations

import pytest

from director_ai.interlock import InterlockKernel, InterlockPolicy


def test_interlock_allows_stream_with_bring_your_own_scorer() -> None:
    kernel = InterlockKernel(InterlockPolicy(hard_limit=0.4))

    result = kernel.run(["safe", " text"], scorer=lambda text: 0.91)

    assert result.decision == "allow"
    assert result.output == "safe text"
    assert result.halt_event is None
    assert result.scores == (0.91, 0.91)


def test_interlock_halts_before_appending_low_score_token() -> None:
    scores = iter([0.9, 0.2])
    kernel = InterlockKernel(
        InterlockPolicy(
            hard_limit=0.5,
            hook_id="standalone.test",
            policy_id="policy.interlock.regulated",
        )
    )

    result = kernel.run(["safe", " unsafe"], scorer=lambda text: next(scores))

    assert result.decision == "halt"
    assert result.output == "safe"
    assert result.halt_index == 1
    assert result.halt_reason == "hard_limit"
    assert result.halt_event is not None
    assert result.halt_event.policy_decision == "halt"
    assert result.halt_event.evidence_refs == ("interlock://token/1",)
    assert result.halt_event.attributes["policy_id"] == "policy.interlock.regulated"
    assert "unsafe" not in str(result.to_dict())


def test_interlock_window_average_can_warn_without_halting() -> None:
    scores = iter([0.9, 0.42, 0.43])
    kernel = InterlockKernel(
        InterlockPolicy(
            hard_limit=0.2,
            window_size=2,
            window_threshold=0.5,
            warn_only=True,
        )
    )

    result = kernel.run(["a", "b", "c"], scorer=lambda text: next(scores))

    assert result.decision == "warn"
    assert result.output == "abc"
    assert result.halt_reason == "window_average"
    assert result.halt_event is not None
    assert result.halt_event.policy_decision == "warn"


def test_interlock_rejects_invalid_scorer_output() -> None:
    kernel = InterlockKernel(InterlockPolicy())

    with pytest.raises(ValueError, match="score"):
        kernel.run(["x"], scorer=lambda text: 1.5)


def test_interlock_uses_score_attribute_results() -> None:
    class ScoreResult:
        score = 0.82

    kernel = InterlockKernel(InterlockPolicy(hard_limit=0.5))

    result = kernel.run(["x"], scorer=lambda text: ScoreResult())

    assert result.decision == "allow"
    assert result.scores == (0.82,)


def test_root_lazy_import_export() -> None:
    from director_ai import InterlockKernel as RootInterlockKernel

    assert RootInterlockKernel is InterlockKernel
