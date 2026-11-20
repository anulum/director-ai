# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — finetune CLI replay and benchmark options

"""Replay and anti-regression options of the ``director-ai finetune`` CLI.

The commands run through the public CLI entry point. Only the training call
itself is replaced by a recorder, so option parsing, validation and the
printed summary are the real code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import director_ai.core.training.finetune as finetune_module
from director_ai.cli import main
from director_ai.core.training.finetune import FinetuneConfig


def _train_file(tmp_path: Path) -> Path:
    """Write a two-row training file and return its path."""
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"premise": f"P{i}", "hypothesis": f"H{i}", "label": i % 2})
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_mix_general_without_replay_data_exits_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replay mixing without a replay file stops before any training."""

    def refuse_training(*_args: object, **_kwargs: object) -> SimpleNamespace:
        raise AssertionError("training must not start")

    monkeypatch.setattr(finetune_module, "finetune_nli", refuse_training)

    with pytest.raises(SystemExit) as stopped:
        main(["finetune", str(_train_file(tmp_path)), "--mix-general"])

    assert stopped.value.code == 1
    assert "--mix-general requires --general-data FILE" in capsys.readouterr().out


def test_general_benchmark_reaches_training_and_reason_is_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The replay and benchmark files reach the config and a reason prints."""
    seen: list[FinetuneConfig] = []

    def record_training(
        path: str, *, eval_path: str | None, config: FinetuneConfig
    ) -> SimpleNamespace:
        seen.append(config)
        return SimpleNamespace(
            output_dir=config.output_dir,
            train_samples=2,
            mixed_general_samples=1,
            epochs_completed=1,
            final_loss=0.25,
            eval_samples=0,
            best_balanced_accuracy=0.0,
            regression_report={
                "recommendation": "reject",
                "regression_pp": 0.0,
                "reason": "evaluation rows overlap the training rows",
                "domain_accuracy": 0.875,
                "domain_set_selected_checkpoint": True,
            },
            onnx_path="",
        )

    monkeypatch.setattr(finetune_module, "finetune_nli", record_training)
    replay = tmp_path / "replay.jsonl"
    heldout = tmp_path / "heldout.jsonl"

    main(
        [
            "finetune",
            str(_train_file(tmp_path)),
            "--general-data",
            str(replay),
            "--auto-benchmark",
            "--general-benchmark",
            str(heldout),
        ]
    )

    config = seen[0]
    assert config.mix_general_data is True
    assert config.general_data_path == str(replay)
    assert config.auto_benchmark is True
    assert config.general_benchmark_path == str(heldout)
    out = capsys.readouterr().out
    assert "Regression:      +0.0pp → reject" in out
    assert "Reason:          evaluation rows overlap the training rows" in out
    assert "Domain bal. acc: 87.5% (selection set)" in out


def test_finetune_help_documents_the_general_benchmark(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The help text names the held-out benchmark and its reject default."""
    main(["finetune", "--help"])
    out = capsys.readouterr().out
    assert "--general-benchmark FILE" in out
    assert "without it the check reports reject" in out
    assert "--mix-general           Mix replay NLI data (needs --general-data)" in out
