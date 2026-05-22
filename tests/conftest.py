# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shared Test Fixtures

import importlib.machinery
import importlib.util
import logging
import os
import sys
import types

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
for import_path in (SRC, ROOT):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

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
# Stub llama_cpp when not installed so benchmark tests can patch it.
if "llama_cpp" not in sys.modules and importlib.util.find_spec("llama_cpp") is None:
    _stub = types.ModuleType("llama_cpp")
    _stub.__spec__ = importlib.machinery.ModuleSpec("llama_cpp", None)
    _stub.Llama = None  # type: ignore[attr-defined]
    sys.modules["llama_cpp"] = _stub

# NOTE: we intentionally do NOT stub ``datasets`` here because
# test_build_judge_dataset.py and test_data_pipeline.py use
# pytest.importorskip("datasets") and a global stub defeats that
# guard. Benchmark tests that need to patch datasets.load_dataset
# must use the ``_ensure_datasets_stub`` fixture below.

# PyTorch is not safely re-importable after its package modules are removed
# from ``sys.modules`` because its C extension keeps process-global state.
# Tests may temporarily patch ``sys.modules["torch"] = None``; loading the
# real module once here makes patch.dict restore the real package afterwards.
try:  # pragma: no cover - exercised indirectly by dependency isolation tests
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover - only on dependency bootstrap faults
    logging.getLogger(__name__).warning("Torch preload unavailable: %s", exc)

_DEPENDENCY_SENTINEL_PREFIXES = (
    "torch",
    "transformers",
    "sentence_transformers",
)


def _remove_missing_dependency_sentinels() -> None:
    """Remove ``sys.modules`` sentinels left by missing-dependency tests."""
    for name, module in list(sys.modules.items()):
        root = name.split(".", 1)[0]
        if (
            root in _DEPENDENCY_SENTINEL_PREFIXES
            and module is None
            and root not in sys.modules
        ):
            del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_missing_dependency_sentinels():
    """Keep missing-dependency simulations from poisoning later tests.

    Several tests intentionally set ``sys.modules["torch"] = None`` or the
    equivalent for transformer stacks to exercise import-error paths. PyTorch
    cannot be re-imported cleanly while those sentinels remain, so each test
    gets a clean boundary.
    """
    _remove_missing_dependency_sentinels()
    yield
    _remove_missing_dependency_sentinels()


@pytest.fixture(autouse=False)
def _ensure_datasets_stub():
    """Temporarily ensure a ``datasets`` module exists in sys.modules.

    Benchmark tests that call ``patch("datasets.load_dataset")`` need
    the module to exist.  This fixture inserts a minimal stub before
    the test and removes it afterwards — so it never leaks into tests
    that rely on ``importorskip("datasets")``.
    """
    _inserted = False
    if "datasets" not in sys.modules:
        _ds = types.ModuleType("datasets")
        _ds.__spec__ = importlib.machinery.ModuleSpec("datasets", None)
        _ds.load_dataset = None  # type: ignore[attr-defined]
        sys.modules["datasets"] = _ds
        _inserted = True
    yield
    if _inserted and "datasets" in sys.modules:
        del sys.modules["datasets"]


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
