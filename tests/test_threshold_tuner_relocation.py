# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — threshold tuner relocation tests
"""Real-surface checks for the calibration threshold-tuner module location."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from director_ai import _cli_bench


@dataclass(frozen=True)
class _TuneResult:
    """Small public-shape result returned by the canonical tuner test double."""

    threshold: float = 0.42
    w_logic: float = 0.7
    w_fact: float = 0.3
    balanced_accuracy: float = 0.91
    precision: float = 0.81
    recall: float = 0.76
    f1: float = 0.78
    samples: int = 1


class _CanonicalTunerModule(ModuleType):
    """Typed module double for the canonical calibration tuner import path."""

    observed_samples: list[dict[str, object]]
    observed_overlay: tuple[_TuneResult, str, str] | None

    def __init__(self) -> None:
        """Initialise an empty observation buffer."""
        super().__init__("director_ai.core.calibration.tuner")
        self.observed_samples = []
        self.observed_overlay = None

    def tune(self, samples: list[dict[str, object]]) -> _TuneResult:
        """Record the CLI-supplied samples and return deterministic metrics."""
        self.observed_samples = samples
        return _TuneResult()

    def format_confidence_report(self, tune_result: _TuneResult) -> str:
        """Render a deterministic confidence report for CLI output."""
        return f"confidence: {tune_result.threshold:.2f}"

    def format_profile_overlay(
        self,
        tune_result: _TuneResult,
        *,
        profile: str,
        base_profile: str,
    ) -> str:
        """Record the overlay request and return deterministic YAML text."""
        self.observed_overlay = (tune_result, profile, base_profile)
        return "overlay: calibration\n"


def test_threshold_tuner_canonical_module_is_calibration() -> None:
    """The threshold tuner should live under the calibration package."""
    canonical = importlib.import_module("director_ai.core.calibration.tuner")
    training_compat = importlib.import_module("director_ai.core.training.tuner")
    root_compat = importlib.import_module("director_ai.core.tuner")

    assert canonical.TuneResult is training_compat.TuneResult
    assert canonical.TuneResult is root_compat.TuneResult
    assert canonical.tune is training_compat.tune
    assert canonical.tune is root_compat.tune
    assert canonical.__file__ is not None
    assert canonical.__file__.endswith("/core/calibration/tuner.py")


def test_bench_tune_prefers_canonical_tuner_when_legacy_alias_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The tune CLI should not let a loaded legacy alias shadow canonical wiring."""
    importlib.import_module("director_ai.core.training.tuner")
    dataset = tmp_path / "labels.jsonl"
    dataset.write_text(
        '{"prompt":"p","response":"r","label":true}\n',
        encoding="utf-8",
    )
    fake_tuner = _CanonicalTunerModule()
    monkeypatch.setitem(sys.modules, "director_ai.core.calibration.tuner", fake_tuner)

    _cli_bench._cmd_tune([str(dataset)])

    assert fake_tuner.observed_samples == [
        {"prompt": "p", "response": "r", "label": True}
    ]
    assert "Best threshold: 0.42" in capsys.readouterr().out


def test_bench_tune_uses_canonical_calibration_tuner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public tune CLI should import the calibration tuner, not training."""
    dataset = tmp_path / "labels.jsonl"
    dataset.write_text(
        '{"prompt":"p","response":"r","label":true}\n',
        encoding="utf-8",
    )
    output = tmp_path / "overlay.yaml"
    fake_tuner = _CanonicalTunerModule()
    monkeypatch.delitem(sys.modules, "director_ai.core.training.tuner", raising=False)
    monkeypatch.setitem(sys.modules, "director_ai.core.calibration.tuner", fake_tuner)

    _cli_bench._cmd_tune(
        [
            "--dataset",
            str(dataset),
            "--profile",
            "strict",
            "--output",
            str(output),
        ],
    )

    assert fake_tuner.observed_samples == [
        {"prompt": "p", "response": "r", "label": True}
    ]
    assert fake_tuner.observed_overlay == (_TuneResult(), "strict_tuned", "strict")
    assert output.read_text(encoding="utf-8") == "overlay: calibration\n"
    assert "Best threshold: 0.42" in capsys.readouterr().out
