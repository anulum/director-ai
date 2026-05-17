# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 held-out builder tests

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools" / "build_lite_scorer_v2_heldout.py"
SPEC = importlib.util.spec_from_file_location("build_lite_scorer_v2_heldout", BUILDER)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

HeldoutBuildConfig = MODULE.HeldoutBuildConfig
build_lite_scorer_v2_heldout_from_rows = MODULE.build_lite_scorer_v2_heldout_from_rows
select_lite_scorer_v2_heldout_rows = MODULE.select_lite_scorer_v2_heldout_rows


def _rows(per_label: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(per_label):
        rows.append(
            {
                "premise": f"Supported premise {index}",
                "hypothesis": f"Supported hypothesis {index}",
                "label": 0,
                "source": "fever" if index % 2 else "vitaminc",
            }
        )
        rows.append(
            {
                "premise": f"Unsupported premise {index}",
                "hypothesis": f"Unsupported hypothesis {index}",
                "label": 2 if index % 2 else 1,
                "source": "halueval_qa" if index % 2 else "aggrefact_RAGTruth",
            }
        )
    return rows


def test_select_lite_scorer_v2_heldout_rows_maps_nli_labels_and_balances() -> None:
    selected, errors = select_lite_scorer_v2_heldout_rows(
        _rows(),
        target_rows=10,
        seed=17,
        min_sources=3,
    )

    assert errors == []
    assert len(selected) == 10
    assert sum(row["label"] is True for row in selected) == 5
    assert sum(row["label"] is False for row in selected) == 5
    assert {row["source_label"] for row in selected} == {0, 1, 2}
    assert len({row["source"] for row in selected}) >= 3


def test_select_lite_scorer_v2_heldout_rows_is_seed_deterministic() -> None:
    first, first_errors = select_lite_scorer_v2_heldout_rows(
        _rows(12),
        target_rows=12,
        seed=99,
        min_sources=3,
    )
    second, second_errors = select_lite_scorer_v2_heldout_rows(
        _rows(12),
        target_rows=12,
        seed=99,
        min_sources=3,
    )
    third, third_errors = select_lite_scorer_v2_heldout_rows(
        _rows(12),
        target_rows=12,
        seed=100,
        min_sources=3,
    )

    assert first_errors == []
    assert second_errors == []
    assert third_errors == []
    assert first == second
    assert first != third


def test_select_lite_scorer_v2_heldout_rows_rejects_unsupported_labels() -> None:
    selected, errors = select_lite_scorer_v2_heldout_rows(
        [
            {
                "premise": "Premise",
                "hypothesis": "Hypothesis",
                "label": 7,
                "source": "bad_source",
            }
        ],
        target_rows=2,
        seed=1,
        min_sources=1,
    )

    assert selected == []
    assert errors == ["row 1: label must be one of 0, 1, or 2"]


def test_build_lite_scorer_v2_heldout_writes_jsonl_and_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "benchmarks" / "heldout" / "lite_scorer_v2.jsonl"
    manifest = output.with_suffix(".manifest.toml")
    config = HeldoutBuildConfig(
        target_rows=12,
        seed=123,
        min_sources=3,
        output=output,
        manifest=manifest,
        source_dataset="training/data/eval",
    )

    errors = build_lite_scorer_v2_heldout_from_rows(_rows(10), config)

    assert errors == []
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 12
    records = [json.loads(line) for line in lines]
    assert set(records[0]) == {
        "premise",
        "hypothesis",
        "label",
        "source",
        "source_label",
    }
    assert {type(record["label"]) for record in records} == {bool}
    packet = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert packet["schema_version"] == "1.0.0"
    assert packet["rows"] == 12
    assert packet["supported_rows"] == 6
    assert packet["unsupported_rows"] == 6
    assert packet["sha256"]
    assert packet["source_dataset"] == "training/data/eval"
