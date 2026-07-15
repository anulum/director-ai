# SPDX-License-Identifier: Apache-2.0
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

_MANDATORY_ACCELERATOR_FAILURE_NODEIDS = frozenset(
    line.strip()
    for line in """
tests/test_continual_adversarial.py::TestContinualRustSums::test_rust_sum_type_error_falls_back
tests/test_cyber_physical.py::TestAABB::test_contains_rust_exception_falls_back_to_python
tests/test_cyber_physical.py::TestAABB::test_contains_rust_non_runtime_exception_falls_back_to_python
tests/test_cyber_physical.py::TestSimpleKinematicModel::test_inverse_rust_exception_falls_back_to_python
tests/test_cyber_physical.py::TestSimpleKinematicModel::test_inverse_rust_non_runtime_exception_falls_back_to_python
tests/test_cyber_physical.py::TestSphere::test_sphere_rust_exceptions_fall_back_to_python
tests/test_cyber_physical.py::TestSphere::test_sphere_rust_non_runtime_exceptions_fall_back_to_python
tests/test_distilled_scorer.py::TestSoftmax::test_rust_softmax_non_runtime_fallback
tests/test_emergence_oracle.py::TestEmergenceRustSums::test_rust_sum_type_error_falls_back_to_python
tests/test_emergence_oracle.py::TestGraphRustSums::test_graph_rust_sum_type_error_falls_back
tests/test_emergence_oracle.py::TestSpectrumRustSums::test_spectrum_rust_sum_type_error_falls_back
tests/test_injection_detector.py::TestFallbackSplit::test_rust_sentence_splitter_non_runtime_fallback
tests/test_injection_detector.py::TestFallbackSplit::test_rust_sentence_splitter_runtime_fallback
tests/test_injection_detector.py::TestFallbackSplit::test_rust_sentence_splitter_type_error_fallback
tests/test_injection_detector.py::TestGracefulDegradation::test_python_fallback_scores_claims_when_rust_bidir_raises_non_runtime
tests/test_injection_detector.py::TestGracefulDegradation::test_python_fallback_scores_claims_when_rust_verdict_raises
tests/test_irreversibility.py::TestStandardNormalQuantile::test_rust_quantile_type_error_falls_back_to_python
tests/test_irreversibility.py::TestWilsonScore::test_rust_wilson_type_error_falls_back_to_python
tests/test_lite_scorer.py::TestLiteScorerRustFallback::test_batch_falls_back_when_rust_ffi_raises
tests/test_lite_scorer.py::TestLiteScorerRustFallback::test_batch_falls_back_when_rust_ffi_raises_non_runtime
tests/test_lite_scorer.py::TestLiteScorerRustFallback::test_batch_falls_back_when_rust_ffi_raises_type_error
tests/test_lite_scorer.py::TestLiteScorerRustFallback::test_score_falls_back_when_rust_ffi_raises
tests/test_lite_scorer.py::TestLiteScorerRustFallback::test_score_falls_back_when_rust_ffi_raises_non_runtime
tests/test_lite_scorer.py::TestLiteScorerRustFallback::test_score_falls_back_when_rust_ffi_raises_type_error
tests/test_meta_guard.py::TestMetaAnalyzerRustKernels::test_rust_mean_type_error_falls_back
tests/test_moderation.py::TestKeywordToxicity::test_rust_scanner_non_runtime_exception_falls_back_to_python
tests/test_moderation.py::TestKeywordToxicity::test_rust_scanner_type_error_falls_back_to_python
tests/test_moderation.py::TestRegexPII::test_rust_scanner_non_runtime_exception_falls_back_to_python
tests/test_moderation.py::TestRegexPII::test_rust_scanner_type_error_falls_back_to_python
tests/test_multi_scale_alignment.py::TestAlignerRustSums::test_rust_sum_type_error_falls_back_to_python
tests/test_numeric_verifier.py::TestRustNumericAdapter::test_rust_numeric_exception_falls_back_to_python
tests/test_numeric_verifier.py::TestRustNumericAdapter::test_rust_numeric_non_runtime_exception_falls_back_to_python
tests/test_online_calibrator.py::TestCalibrationReport::test_rust_wilson_type_error_falls_back_to_python
tests/test_policy_compiler.py::TestSplitConformal::test_rust_conformal_type_error_falls_back_to_python
tests/test_prompt_risk_routing.py::TestRoutingRustSums::test_rust_sum_type_error_falls_back_to_python
tests/test_reasoning_verifier.py::TestExtractSteps::test_extract_steps_reverts_to_python_when_rust_extractor_raises
tests/test_reasoning_verifier.py::TestExtractSteps::test_extract_steps_reverts_to_python_when_rust_extractor_raises_non_runtime
tests/test_reasoning_verifier.py::TestExtractSteps::test_extract_steps_reverts_to_python_when_rust_extractor_raises_type_error
tests/test_rules_scorer.py::TestWordOverlapRule::test_rust_exception_falls_back_to_python
tests/test_rules_scorer.py::TestWordOverlapRule::test_rust_non_runtime_exception_falls_back_to_python
tests/test_sanitizer.py::TestRustUnicodeFallback::test_rust_unicode_exception_falls_back_to_python
tests/test_sanitizer.py::TestRustUnicodeFallback::test_rust_unicode_non_runtime_exception_falls_back_to_python
tests/test_sustainability.py::TestCarbonTracker::test_rust_mean_type_error_falls_back_to_python
tests/test_sustainability.py::TestForecaster::test_rust_ema_runtime_error_falls_back_to_python
tests/test_sustainability.py::TestTelemetryRustSums::test_rust_sum_type_error_falls_back_to_python
tests/test_swarm_economics.py::TestSwarmEconomicsRustSums::test_rust_sum_type_error_falls_back_to_python
tests/test_swarm_equilibrium.py::TestScorer::test_rust_mean_type_error_falls_back_to_python
tests/test_task_scoring_paths.py::TestMiniCheckClaimCoverage::test_python_reducer_fallback_on_non_runtime_rust_error
tests/test_task_scoring_paths.py::TestMiniCheckClaimCoverage::test_python_reducer_fallback_on_rust_runtime_error
tests/test_task_scoring_paths.py::TestMiniCheckClaimCoverage::test_python_reducer_fallback_on_rust_type_error
tests/test_task_scoring_paths.py::TestTaskDetectionFallback::test_rust_detector_exception_falls_back_to_python
tests/test_task_scoring_paths.py::TestTaskDetectionFallback::test_rust_detector_non_runtime_exception_falls_back_to_python
tests/test_temporal_freshness.py::TestCurrentReference::test_rust_exception_falls_back_to_python_detection
tests/test_temporal_freshness.py::TestCurrentReference::test_rust_non_runtime_exception_falls_back_to_python_detection
tests/test_temporal_freshness.py::TestCurrentReference::test_rust_type_error_falls_back_to_python_detection
tests/test_verified_scorer.py::TestEntityOverlap::test_rust_exception_falls_back_to_python
tests/test_verified_scorer.py::TestEntityOverlap::test_rust_non_runtime_exception_falls_back_to_python
tests/test_verified_scorer.py::TestEntityOverlap::test_rust_type_error_falls_back_to_python
tests/test_verified_scorer.py::TestNegationFlip::test_rust_negation_non_runtime_exception_falls_back_to_python
tests/test_verified_scorer.py::TestNegationFlip::test_rust_negation_type_error_falls_back_to_python
tests/test_verified_scorer.py::TestNumericalConsistency::test_rust_numerical_non_runtime_exception_falls_back_to_python
tests/test_verified_scorer.py::TestNumericalConsistency::test_rust_numerical_type_error_falls_back_to_python
tests/test_verified_scorer.py::TestSplitSentences::test_rust_sentence_splitter_non_runtime_fallback
tests/test_verified_scorer.py::TestSplitSentences::test_rust_sentence_splitter_runtime_fallback
tests/test_verified_scorer.py::TestTraceability::test_rust_traceability_exception_falls_back_to_python
tests/test_verified_scorer.py::TestTraceability::test_rust_traceability_non_runtime_exception_falls_back_to_python
tests/test_verified_scorer.py::TestTraceability::test_rust_traceability_type_error_falls_back_to_python
tests/test_zk_attestation.py::TestCommitmentBackend::test_rust_challenge_exception_falls_back_to_python
tests/test_zk_attestation.py::TestCommitmentBackend::test_rust_challenge_non_runtime_exception_falls_back_to_python
tests/test_zk_attestation.py::TestCommitmentBackend::test_rust_challenge_type_error_falls_back_to_python
tests/test_zk_attestation.py::TestMerkleCommitment::test_rust_merkle_auth_path_non_runtime_exception_falls_back_to_python
tests/test_zk_attestation.py::TestMerkleCommitment::test_rust_merkle_exception_falls_back_to_python
tests/test_zk_attestation.py::TestMerkleCommitment::test_rust_merkle_mixed_non_runtime_exceptions_fall_back_to_python
tests/test_zk_attestation.py::TestMerkleCommitment::test_rust_merkle_non_runtime_exception_falls_back_to_python
tests/test_zk_attestation.py::TestMerkleCommitment::test_rust_merkle_root_non_runtime_exception_falls_back_to_python
tests/test_zk_attestation.py::TestMerkleCommitment::test_rust_merkle_walk_path_non_runtime_exception_falls_back_to_python
""".splitlines()
    if line.strip()
)


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Assert stale Rust fallback tests now propagate mandatory FFI failures."""
    if pyfuncitem.nodeid not in _MANDATORY_ACCELERATOR_FAILURE_NODEIDS:
        return None
    testargs = {
        arg: pyfuncitem.funcargs[arg]
        for arg in pyfuncitem._fixtureinfo.argnames
        if arg in pyfuncitem.funcargs
    }
    with pytest.raises(
        (RuntimeError, ValueError, TypeError, ArithmeticError),
        match="ffi|unavailable|mandatory",
    ):
        pyfuncitem.obj(**testargs)
    return True


def _remove_missing_dependency_sentinels() -> None:
    """Remove ``sys.modules`` sentinels left by missing-dependency tests."""
    for name, module in list(sys.modules.items()):
        root = name.split(".", 1)[0]
        if root in _DEPENDENCY_SENTINEL_PREFIXES and module is None:
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
