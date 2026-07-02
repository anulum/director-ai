# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — threshold data feeder real-surface tests
"""Real CLI-surface coverage for the threshold-tuner data feeder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

_FEEDER_MODULE: ModuleType | None = None


class _FeederModule(Protocol):
    """Typed surface used from the dynamically loaded feeder module."""

    def normalise(
        self,
        records: Iterable[dict[str, object]],
        *,
        score_key: str,
        label_key: str,
        source_key: str | None,
    ) -> Iterator[dict[str, object]]:
        """Coerce, filter, and reshape records for the Julia tuner."""

    def write_jsonl(self, records: Iterable[dict[str, object]], path: Path) -> int:
        """Write tuner records atomically as UTF-8 JSONL."""


class _TempfileModule(Protocol):
    """Temporary-file function surface used by the feeder module."""

    NamedTemporaryFile: Callable[..., object]


class _FeederRuntimeModule(_FeederModule, Protocol):
    """Feeder module surface including its imported dependencies."""

    tempfile: _TempfileModule


def _feeder_module_object() -> ModuleType:
    """Return the loaded threshold feeder module object."""
    global _FEEDER_MODULE
    if _FEEDER_MODULE is None:
        path = (
            Path(__file__).resolve().parents[1] / "tools" / "prepare_threshold_data.py"
        )
        spec = importlib.util.spec_from_file_location("prepare_threshold_data", path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _FEEDER_MODULE = module
    return _FEEDER_MODULE


def _feeder_module() -> _FeederModule:
    """Return the loaded threshold feeder module public surface."""
    module = _feeder_module_object()
    return cast(_FeederModule, module)


def _feeder_runtime_module() -> _FeederRuntimeModule:
    """Return the loaded threshold feeder module runtime surface."""
    module = _feeder_module_object()
    return cast(_FeederRuntimeModule, module)


def _run_feeder(
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the production data-feeder script in a subprocess."""
    repo_root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(repo_root / "tools" / "prepare_threshold_data.py")]
    command.extend(args)
    return subprocess.run(
        command,
        cwd=cwd or repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file as a list of object rows."""
    return [
        cast(dict[str, object], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_prepare_threshold_data_unit_guard_has_real_surface_companion() -> None:
    """The threshold feeder guard should be backed by real CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_prepare_threshold_data.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_prepare_threshold_data_real_surface.py" in category


def test_csv_input_converts_to_julia_tuner_jsonl_over_real_cli(
    tmp_path: Path,
) -> None:
    """The CLI should convert labelled CSV rows into tuner JSONL records."""
    source = tmp_path / "scores.csv"
    source.write_text(
        "\n".join(
            [
                "score,label,dataset",
                "0.91,supported,aggrefact:summ",
                ",true,drop-missing-score",
                "0.18,hallucinated,aggrefact:diag",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "nested" / "threshold.jsonl"

    result = _run_feeder("--input", str(source), "--output", str(output))

    assert result.returncode == 0, result.stderr
    assert _jsonl_rows(output) == [
        {"score": 0.91, "label": True, "source": "aggrefact:summ"},
        {"score": 0.18, "label": False, "source": "aggrefact:diag"},
    ]
    assert "1 record(s) dropped" in result.stderr
    assert "wrote 2 records" in result.stderr


def test_json_records_accept_custom_keys_without_source_over_real_cli(
    tmp_path: Path,
) -> None:
    """The CLI should honour custom score, label, and disabled source keys."""
    source = tmp_path / "scores.json"
    source.write_text(
        json.dumps(
            {
                "records": [
                    {"probability": "0.77", "gold": "yes", "dataset": "kept-out"},
                    {"probability": 0.22, "gold": "no", "dataset": "kept-out"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "threshold.jsonl"

    result = _run_feeder(
        "--input",
        str(source),
        "--output",
        str(output),
        "--score-key",
        "probability",
        "--label-key",
        "gold",
        "--source-key",
        "",
    )

    assert result.returncode == 0, result.stderr
    assert _jsonl_rows(output) == [
        {"score": 0.77, "label": True},
        {"score": 0.22, "label": False},
    ]
    assert "source" not in output.read_text(encoding="utf-8")


def test_normalise_omits_missing_requested_source() -> None:
    """Normalisation should keep records whose optional source is absent."""
    normalise = _feeder_module().normalise

    rows = list(
        normalise(
            [{"score": 0.44, "label": "supported"}],
            score_key="score",
            label_key="label",
            source_key="dataset",
        )
    )

    assert rows == [{"score": 0.44, "label": True}]


def test_unsupported_input_extension_fails_before_output_over_real_cli(
    tmp_path: Path,
) -> None:
    """The CLI should reject unsupported input files without writing output."""
    source = tmp_path / "scores.txt"
    source.write_text("score,label\n0.1,true\n", encoding="utf-8")
    output = tmp_path / "threshold.jsonl"

    result = _run_feeder("--input", str(source), "--output", str(output))

    assert result.returncode != 0
    assert "Unsupported extension" in result.stderr
    assert not output.exists()


def test_atomic_writer_removes_temporary_output_when_iteration_fails(
    tmp_path: Path,
) -> None:
    """A failed record stream should leave neither output nor temp files."""
    write_jsonl = _feeder_module().write_jsonl
    output = tmp_path / "threshold.jsonl"

    def _broken_records() -> Iterator[dict[str, object]]:
        yield {"score": 0.5, "label": True}
        raise RuntimeError("record stream failed")

    with pytest.raises(RuntimeError, match="record stream failed"):
        write_jsonl(_broken_records(), output)

    assert not output.exists()
    assert list(tmp_path.glob(".threshold.jsonl.*.tmp")) == []


def test_atomic_writer_leaves_no_output_when_tempfile_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temp-file creation failure should not leave a target output file."""
    module = _feeder_runtime_module()
    output = tmp_path / "threshold.jsonl"

    def _raise_named_temporary_file(*_args: object, **_kwargs: object) -> object:
        raise OSError("tmpdir unavailable")

    monkeypatch.setattr(
        module.tempfile,
        "NamedTemporaryFile",
        _raise_named_temporary_file,
    )

    with pytest.raises(OSError, match="tmpdir unavailable"):
        module.write_jsonl([{"score": 0.5, "label": True}], output)

    assert not output.exists()
