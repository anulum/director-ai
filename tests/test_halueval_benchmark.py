# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HaluEval benchmark tests

"""Slow, model-backed HaluEval benchmark smoke tests.

These moved out of ``benchmarks/halueval_eval.py`` so the benchmark module no
longer imports pytest at module scope — a plain ``python -m
benchmarks.ragtruth_eval`` (which transitively imports halueval_eval) was
breaking on minimal remote runners that lack pytest. They live here, in the
``tests`` collection path, guarded on model availability so offline CI skips
them cleanly (the same pattern as ``test_nli_integration``).
"""

from __future__ import annotations

import pytest

from benchmarks.halueval_eval import run_halueval_benchmark
from director_ai.core.nli import NLIScorer

_model_ok = NLIScorer(use_model=True, device="cpu").model_available


@pytest.mark.slow
@pytest.mark.skipif(not _model_ok, reason="DeBERTa NLI model not available")
def test_halueval_qa_sample():
    """Run the HaluEval QA benchmark on a small sample with the NLI model."""
    result = run_halueval_benchmark(tasks=["qa"], use_nli=True, max_samples_per_task=25)
    assert result.overall.total > 0


@pytest.mark.slow
@pytest.mark.skipif(not _model_ok, reason="DeBERTa NLI model not available")
def test_halueval_full():
    """Run the full HaluEval benchmark across all tasks with the NLI model."""
    result = run_halueval_benchmark(use_nli=True, max_samples_per_task=200)
    assert result.overall.total > 0
