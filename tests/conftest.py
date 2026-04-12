# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shared Test Fixtures

import importlib.machinery
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
for _mod_name in ("llama_cpp", "datasets"):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        _stub.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
        sys.modules[_mod_name] = _stub

# Ensure stub modules have the attributes that benchmark tests patch.
_llama_stub = sys.modules.get("llama_cpp")
if _llama_stub is not None and not hasattr(_llama_stub, "Llama"):
    _llama_stub.Llama = None  # type: ignore[attr-defined]
_ds_stub = sys.modules.get("datasets")
if _ds_stub is not None and not hasattr(_ds_stub, "load_dataset"):
    _ds_stub.load_dataset = None  # type: ignore[attr-defined]


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
