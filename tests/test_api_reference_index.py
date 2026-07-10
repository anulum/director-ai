# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - API reference consistency gate tests
"""Regression tests for public API reference documentation consistency."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

from director_ai.core.agent import CoherenceAgent
from director_ai.core.scoring.nli import NLIScorer
from director_ai.core.scoring.scorer import CoherenceScorer

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_api_reference.py"
SPEC = importlib.util.spec_from_file_location("validate_api_reference", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_api_reference = MODULE.validate_api_reference


def _api_page(name: str) -> str:
    """Return a docs-site API page as UTF-8 text."""
    return (ROOT / "docs-site" / "api" / name).read_text(encoding="utf-8")


def test_api_reference_index_matches_current_docs_and_imports() -> None:
    """The API reference index should match docs and importable symbols."""
    assert validate_api_reference(ROOT) == []


def test_signature_docs_match_current_api_surfaces() -> None:
    """Canonical API docs should describe the current public signatures."""
    scorer_init = inspect.signature(CoherenceScorer.__init__)
    agent_process = inspect.signature(CoherenceAgent.process)
    agent_aprocess = inspect.signature(CoherenceAgent.aprocess)
    nli_init = inspect.signature(NLIScorer.__init__)

    assert scorer_init.parameters["w_logic"].default is None
    assert scorer_init.parameters["w_fact"].default is None
    assert scorer_init.parameters["nli_devices"].default is None
    # Resolved via the class (ChunkingMixin provides it on NLIScorer);
    # __dict__ membership would miss mixin-provided methods.
    assert not hasattr(CoherenceScorer, "score_chunked")
    assert hasattr(NLIScorer, "score_chunked")
    assert agent_process.parameters["prompt"].annotation in {str, "str"}
    assert "query" not in agent_process.parameters
    assert agent_aprocess.parameters["prompt"].annotation in {str, "str"}
    assert "query" not in agent_aprocess.parameters
    assert nli_init.parameters["model_name"].default is None

    scorer_doc = _api_page("scorer.md")
    assert "| `w_logic` | `float \\| None` | `None` |" in scorer_doc
    assert "| `w_fact` | `float \\| None` | `None` |" in scorer_doc
    assert "| `nli_devices` | `list[str] \\| None` | `None` |" in scorer_doc
    assert "### score_chunked()" not in scorer_doc
    assert "NLIScorer.score_chunked()" in scorer_doc

    agent_doc = _api_page("agent.md")
    assert "result = agent.process(prompt: str" in agent_doc
    assert "result = await agent.aprocess(prompt: str" in agent_doc
    assert "agent.process(query: str" not in agent_doc
    assert "agent.aprocess(query: str" not in agent_doc

    nli_doc = _api_page("nli.md")
    assert "| `model_name` | `str \\| None` | `None` |" in nli_doc
    assert "resolves to FactCG-DeBERTa-v3-Large" in nli_doc

    streaming_guide = (ROOT / "docs-site" / "guide" / "streaming.md").read_text(
        encoding="utf-8"
    )
    assert "scorer.score_chunked()" not in streaming_guide
    assert "NLIScorer.score_chunked()" in streaming_guide

    scoring_guide = (ROOT / "docs-site" / "guide" / "scoring.md").read_text(
        encoding="utf-8"
    )
    assert "scorer._nli.score_chunked" not in scoring_guide
    assert "nli.score_chunked" in scoring_guide


def test_api_reference_gate_rejects_missing_markdown_target(tmp_path: Path) -> None:
    """The validator should reject links to absent Markdown files."""
    docs = tmp_path / "docs-site" / "api"
    docs.mkdir(parents=True)
    (docs / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n"
        "| [`guard()`](missing.md) | `director_ai` | stale link |\n",
        encoding="utf-8",
    )

    errors = validate_api_reference(tmp_path)

    assert errors == ["docs-site/api/index.md:5: missing markdown target missing.md"]


def test_api_reference_gate_rejects_missing_importable_symbol(tmp_path: Path) -> None:
    """The validator should reject rows for absent public symbols."""
    docs = tmp_path / "docs-site" / "api"
    docs.mkdir(parents=True)
    (docs / "guard.md").write_text("# Guard\n\n", encoding="utf-8")
    (docs / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n"
        "| [`definitely_missing()`](guard.md) | `director_ai` | stale symbol |\n",
        encoding="utf-8",
    )

    errors = validate_api_reference(tmp_path)

    assert errors == [
        "docs-site/api/index.md:5: director_ai does not expose definitely_missing"
    ]
