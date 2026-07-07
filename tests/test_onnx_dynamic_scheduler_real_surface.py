# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - ONNX dynamic scheduler real-surface tests
"""Real-surface coverage for public ONNX dynamic scheduler wiring."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from director_ai.core.nli import NLIScorer as PublicNLIScorer
from director_ai.core.nli import OnnxDynamicBatcher as PublicOnnxDynamicBatcher
from director_ai.core.scoring.nli import OnnxDynamicBatcher as RuntimeOnnxDynamicBatcher
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

Pair = tuple[str, str]
ScoreFn = Callable[[list[Pair]], list[float]]


def _public_constant_scorer(pairs: list[Pair]) -> list[float]:
    """Return one public scheduler score per input pair."""
    return [0.25] * len(pairs)


def test_onnx_dynamic_scheduler_unit_guard_declares_real_surface_companion() -> None:
    """The ONNX scheduler guard should name its public companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_onnx_dynamic_scheduler.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_onnx_dynamic_scheduler_real_surface.py" in reason


def test_public_onnx_dynamic_scheduler_is_runtime_scheduler() -> None:
    """The public compatibility import should expose the runtime scheduler."""
    assert PublicOnnxDynamicBatcher is RuntimeOnnxDynamicBatcher


def test_public_onnx_dynamic_scheduler_flushes_on_elapsed_timeout() -> None:
    """The public scheduler should flush buffered work after its timeout."""
    scorer: ScoreFn = _public_constant_scorer
    batcher = PublicOnnxDynamicBatcher(
        scorer,
        max_batch=4,
        flush_timeout_ms=1.0,
    )

    assert batcher.submit([("premise", "hypothesis")]) == []
    time.sleep(0.01)

    assert batcher.submit([]) == [0.25]


@pytest.mark.parametrize(
    ("onnx_batch_size", "onnx_flush_timeout_ms"),
    [(0, 10.0), (16, -1.0)],
)
def test_public_onnx_scorer_rejects_invalid_scheduler_config(
    onnx_batch_size: int,
    onnx_flush_timeout_ms: float,
) -> None:
    """The public ONNX scorer should reject invalid scheduler settings."""
    with pytest.raises(ValueError):
        PublicNLIScorer(
            backend="onnx",
            onnx_batch_size=onnx_batch_size,
            onnx_flush_timeout_ms=onnx_flush_timeout_ms,
        )
