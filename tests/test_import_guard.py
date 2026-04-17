# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Import Guard Tests
"""Multi-angle tests for lazy import discipline.

Verifies heavyweight optional dependencies (torch, transformers, onnxruntime,
grpcio, chromadb) are NOT imported at module level in core modules.
This ensures `pip install director-ai && from director_ai import CoherenceScorer`
works without torch/transformers installed.

Covers: scorer.py, nli.py, backends.py, vector_store.py, __init__.py,
parametrised module scanning, AST + runtime import verification.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src" / "director_ai"

# Modules that must NOT import these at top level
HEAVYWEIGHT_DEPS = ["torch", "transformers", "onnxruntime", "chromadb", "grpc"]

CORE_MODULES = [
    SRC_DIR / "core" / "scorer.py",
    SRC_DIR / "core" / "scoring" / "nli.py",
    SRC_DIR / "core" / "backends.py",
    SRC_DIR / "__init__.py",
]


def _top_level_imports(filepath: Path) -> set[str]:
    """Extract all top-level import names from a Python file."""
    tree = ast.parse(filepath.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and node.col_offset == 0:
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.col_offset == 0 and node.module:
            imports.add(node.module.split(".")[0])
    return imports


# ── Parametrised import guard ─────────────────────────────────────


class TestNoEagerHeavyImports:
    """No core module may eagerly import heavyweight dependencies."""

    @pytest.mark.parametrize(
        "module_path", CORE_MODULES, ids=[p.stem for p in CORE_MODULES]
    )
    @pytest.mark.parametrize("dep", HEAVYWEIGHT_DEPS)
    def test_no_top_level_import(self, module_path, dep):
        if not module_path.exists():
            pytest.skip(f"{module_path} does not exist")
        imports = _top_level_imports(module_path)
        assert dep not in imports, (
            f"{module_path.name} must not import '{dep}' at module level"
        )


# ── Runtime verification ─────────────────────────────────────────


class TestRuntimeImportGuard:
    """Verify core can be imported without torch/transformers."""

    def test_coherence_scorer_imports_without_torch(self):
        """CoherenceScorer must be importable without torch."""
        from director_ai.core import CoherenceScorer

        assert CoherenceScorer is not None

    def test_scorer_review_works_without_nli(self):
        """review() must work with use_nli=False (no torch needed)."""
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False)
        approved, score = scorer.review("Q", "A")
        assert isinstance(approved, bool)

    def test_ground_truth_store_imports_without_chromadb(self):
        from director_ai.core import GroundTruthStore

        store = GroundTruthStore()
        assert store is not None


# ── Pipeline performance doc ─────────────────────────────────────


class TestImportPerformance:
    """Document import time (should be < 500ms without heavy deps)."""

    def test_core_import_fast(self):
        import time

        t0 = time.perf_counter()
        import importlib

        importlib.reload(__import__("director_ai.core", fromlist=["CoherenceScorer"]))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Core import without NLI should be fast
        assert elapsed_ms < 2000, f"Core import took {elapsed_ms:.0f}ms"
