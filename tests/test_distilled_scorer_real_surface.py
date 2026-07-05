# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for distilled NLI local artefact validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from director_ai.core.scoring.distilled_scorer import (
    DistilledNLIBackend,
    validate_local_distilled_onnx_artifact,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _write_model_file(model_dir: Path) -> Path:
    """Create a minimal local ONNX placeholder file for validation tests."""
    model_dir.mkdir(parents=True)
    model_file = model_dir / "model.onnx"
    model_file.write_bytes(b"placeholder-onnx")
    return model_file


def test_distilled_scorer_unit_guard_declares_this_companion() -> None:
    """The mocked distilled scorer unit guard should point at this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_distilled_scorer.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_distilled_scorer_real_surface.py" in reason


def test_validate_local_distilled_onnx_artifact_resolves_allowed_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local distilled artefact validation should return resolved safe paths."""
    allowed_root = tmp_path / "allowed"
    model_dir = allowed_root / "nli-lite"
    model_file = _write_model_file(model_dir)
    tokeniser_file = model_dir / "tokenizer.json"
    tokeniser_file.write_text('{"version": "1.0"}', encoding="utf-8")
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed_root))

    artifact = validate_local_distilled_onnx_artifact(model_dir)

    assert artifact.model_dir == model_dir.resolve()
    assert artifact.model_file == model_file.resolve()
    assert artifact.tokeniser_files == (tokeniser_file.resolve(),)


def test_validate_local_distilled_onnx_artifact_rejects_missing_tokeniser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A local ONNX model without tokeniser assets is not loadable."""
    allowed_root = tmp_path / "allowed"
    model_dir = allowed_root / "missing-tokeniser"
    _write_model_file(model_dir)
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed_root))

    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        validate_local_distilled_onnx_artifact(model_dir)


def test_validate_local_distilled_onnx_artifact_rejects_missing_model_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A local artefact directory without an ONNX model is invalid."""
    allowed_root = tmp_path / "allowed"
    model_dir = allowed_root / "missing-model"
    model_dir.mkdir(parents=True)
    (model_dir / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed_root))

    with pytest.raises(FileNotFoundError, match="model file not found"):
        validate_local_distilled_onnx_artifact(model_dir)


def test_validate_local_distilled_onnx_artifact_rejects_tokeniser_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Tokeniser assets must not escape the selected artefact directory."""
    allowed_root = tmp_path / "allowed"
    model_dir = allowed_root / "nli-lite"
    _write_model_file(model_dir)
    external = tmp_path / "external"
    external.mkdir()
    external_tokeniser = external / "tokenizer.json"
    external_tokeniser.write_text('{"version": "1.0"}', encoding="utf-8")
    (model_dir / "tokenizer.json").symlink_to(external_tokeniser)
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed_root))

    with pytest.raises(PermissionError, match="tokeniser file escapes"):
        validate_local_distilled_onnx_artifact(model_dir)


def test_local_backend_fails_closed_for_incomplete_onnx_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A broken local ONNX artefact should not fall back to a hub/PyTorch load."""
    allowed_root = tmp_path / "allowed"
    model_dir = allowed_root / "incomplete"
    _write_model_file(model_dir)
    monkeypatch.setenv("DIRECTOR_ONNX_ALLOWED_DIRS", str(allowed_root))

    backend = DistilledNLIBackend(model_path=str(model_dir))

    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        backend.score("Policy requires signed approval.", "Approval was signed.")
