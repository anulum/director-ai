# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 held-out builder real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 held-out builder CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_lite_scorer_v2_heldout.py"


def _source_rows(
    *, per_label: int = 8, distinct_sources: int = 4
) -> list[dict[str, object]]:
    """Return NLI-style rows that exercise label mapping and source balancing."""
    sources = [f"source_{index}" for index in range(distinct_sources)]
    rows: list[dict[str, object]] = []
    for index in range(per_label):
        rows.append(
            {
                "premise": f"Supported premise {index}",
                "hypothesis": f"Supported hypothesis {index}",
                "label": 0,
                "source": sources[index % len(sources)],
            }
        )
        rows.append(
            {
                "premise": f"Unsupported premise {index}",
                "hypothesis": f"Unsupported hypothesis {index}",
                "label": 2 if index % 2 else 1,
                "source": sources[(index + 1) % len(sources)],
            }
        )
    return rows


def _run_command(
    argv: list[str],
    *,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 15,
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` as a text subprocess and return the completed process."""
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        env=command_env,
        input=input_text,
        text=True,
        timeout=timeout,
    )


def _write_hf_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    """Write ``rows`` as a Hugging Face-compatible dataset directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")


def _write_dataset_provider(path: Path) -> dict[str, str]:
    """Write a protocol fixture that provides ``datasets.load_from_disk``."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "datasets.py").write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "",
                "def load_from_disk(path: str):",
                "    return json.loads((Path(path) / 'rows.json').read_text())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"PYTHONPATH": path.as_posix()}


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest for ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lite_scorer_v2_heldout_builder_cli_writes_dataset_and_manifest(
    tmp_path: Path,
) -> None:
    """The production CLI should build balanced JSONL and TOML artefacts."""
    source = tmp_path / "source-dataset"
    output = tmp_path / "benchmarks" / "heldout" / "lite_scorer_v2.jsonl"
    manifest = output.with_suffix(".manifest.toml")
    _write_hf_dataset(source, _source_rows(per_label=10, distinct_sources=5))
    env = _write_dataset_provider(tmp_path / "hf-provider")

    result = _run_command(
        [
            sys.executable,
            str(BUILDER),
            "--source",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--target-rows",
            "10",
            "--seed",
            "20260627",
            "--min-sources",
            "4",
        ],
        env=env,
    )

    assert result.returncode == 0
    assert result.stdout == f"lite_scorer_v2_heldout_built rows=10 output={output}\n"
    assert result.stderr == ""
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    label_counts = Counter(record["label"] for record in records)
    source_counts = Counter(str(record["source"]) for record in records)
    assert len(records) == 10
    assert label_counts == {True: 5, False: 5}
    assert {type(record["label"]) for record in records} == {bool}
    assert {record["source_label"] for record in records} == {0, 1, 2}
    assert len(source_counts) >= 4

    packet = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert packet["schema_version"] == "1.0.0"
    assert packet["dataset_id"] == "lite-scorer-v2-heldout"
    assert packet["source_dataset"] == source.as_posix()
    assert packet["output"] == output.as_posix()
    assert packet["rows"] == 10
    assert packet["supported_rows"] == 5
    assert packet["unsupported_rows"] == 5
    assert packet["seed"] == 20260627
    assert packet["min_sources"] == 4
    assert packet["sha256"] == _sha256(output)
    assert sum(packet["source_counts"].values()) == 10


def test_lite_scorer_v2_heldout_builder_cli_rejects_insufficient_sources(
    tmp_path: Path,
) -> None:
    """The production CLI should fail closed when source diversity is too low."""
    source = tmp_path / "source-dataset"
    output = tmp_path / "benchmarks" / "heldout" / "lite_scorer_v2.jsonl"
    manifest = output.with_suffix(".manifest.toml")
    _write_hf_dataset(source, _source_rows(per_label=4, distinct_sources=1))
    env = _write_dataset_provider(tmp_path / "hf-provider")

    result = _run_command(
        [
            sys.executable,
            str(BUILDER),
            "--source",
            str(source),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--target-rows",
            "4",
            "--seed",
            "20260627",
            "--min-sources",
            "2",
        ],
        env=env,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "source rows must contain at least 2 distinct sources\n"
    assert not output.exists()
    assert not manifest.exists()
