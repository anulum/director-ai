# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shared Test Fixtures

import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest

from director_ai.core import (
    CoherenceAgent,
    CoherenceScorer,
    GroundTruthStore,
    MockGenerator,
    SafetyKernel,
)

# faiss-cpu 1.13.1 AVX2 .pyd hangs on DLL init (Windows);
# generic backend loads in ~2s and CI uses Linux where this is not needed.
os.environ.setdefault("FAISS_OPT_LEVEL", "generic")

# ── Stub heavy optional deps so benchmark tests can patch them in CI ────
# When llama_cpp / datasets are not installed (CI base environment),
# ``unittest.mock.patch("llama_cpp.Llama")`` fails because the module
# does not exist in sys.modules. Pre-populating with empty stubs lets
# patch() succeed; the actual implementation is always mocked in tests.
#
# IMPORTANT: only stub if the package is truly missing (importlib cannot
# find it). If the real package is installed, we must NOT overwrite it
# because other tests (e.g. test_build_judge_dataset) import real symbols.
# Only stub llama_cpp (never datasets — test_build_judge_dataset uses
# pytest.importorskip("datasets") and a stub would defeat that guard).
if "llama_cpp" not in sys.modules and importlib.util.find_spec("llama_cpp") is None:
    _stub = types.ModuleType("llama_cpp")
    _stub.__spec__ = importlib.machinery.ModuleSpec("llama_cpp", None)
    _stub.Llama = None  # type: ignore[attr-defined]
    sys.modules["llama_cpp"] = _stub


@pytest.fixture
def agent():
    """Pre-configured CoherenceAgent (mock mode, demo facts)."""
    store = GroundTruthStore.with_demo_facts()
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)
    return CoherenceAgent(_scorer=scorer, _store=store)


@pytest.fixture
def scorer():
    """CoherenceScorer with heuristic scoring and demo facts."""
    store = GroundTruthStore.with_demo_facts()
    return CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)


@pytest.fixture
def strict_scorer():
    """CoherenceScorer with a strict threshold (0.7), heuristic scoring."""
    store = GroundTruthStore.with_demo_facts()
    return CoherenceScorer(threshold=0.7, ground_truth_store=store, use_nli=False)


@pytest.fixture
def kernel():
    """SafetyKernel instance."""
    return SafetyKernel()


@pytest.fixture
def store():
    """GroundTruthStore with demo facts."""
    return GroundTruthStore.with_demo_facts()


@pytest.fixture
def generator():
    """MockGenerator instance."""
    return MockGenerator()
