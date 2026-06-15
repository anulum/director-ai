# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CLIP discrimination benchmark tests

"""Offline tests for the multimodal discrimination benchmark.

The separation metric is checked directly; the hash-bag stage runs without any
optional dependency; the CLIP stage degrades to available:false when open_clip is
absent (the common CI case) rather than fabricating a number.
"""

from __future__ import annotations

import importlib.util

from benchmarks.multimodal_clip_discrimination import (
    run_benchmark,
    separation,
)


def test_separation_empty():
    assert separation([], [0.1]) == {"n_matched": 0, "n_mismatched": 1}


def test_separation_perfect():
    s = separation([0.9, 0.8], [0.2, 0.1])
    assert s["gap"] > 0
    assert s["pairwise_auc"] == 1.0


def test_separation_no_signal():
    # Identical distributions -> all ties -> AUC 0.5, zero gap.
    s = separation([0.5, 0.5], [0.5, 0.5])
    assert s["pairwise_auc"] == 0.5
    assert s["gap"] == 0.0


def test_run_benchmark_hashbag_runs_clip_degrades_without_dep():
    result = run_benchmark()
    assert result["benchmark"] == "multimodal_clip_discrimination"
    # Hash-bag baseline is dependency-free and always produces a metric.
    assert result["hashbag_baseline"]["available"] is True
    assert "pairwise_auc" in result["hashbag_baseline"]
    # CLIP requires open_clip; without it the stage degrades honestly.
    if importlib.util.find_spec("open_clip") is None:
        assert result["clip"]["available"] is False
        assert "multimodal" in result["clip"]["reason"]
    else:
        assert result["clip"]["available"] is True
        assert result["clip"]["gap"] > 0  # CLIP must separate matched captions
