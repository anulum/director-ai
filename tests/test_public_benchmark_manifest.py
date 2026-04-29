# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - public benchmark manifest tests

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "public_accuracy_manifest.toml"


def _manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def test_public_manifest_has_required_tables():
    data = _manifest()
    table_ids = {table["id"] for table in data["public_accuracy_tables"]}

    assert {
        "readme_scoring_pyramid",
        "readme_aggrefact_leaderboard",
        "readme_routed_local_judge",
        "benchmark_report_e2e_halueval",
        "benchmark_report_local_judge",
        "benchmark_report_streaming_false_halt",
    } <= table_ids


def test_public_manifest_paths_exist_or_are_declared_optional():
    data = _manifest()
    optional_results = {"benchmarks/results/streaming_false_halt_nli.json"}

    assert (ROOT / data["cache_schema"]["path"]).exists()
    assert (ROOT / data["cache_schema"]["result_schema"]).exists()

    for table in data["public_accuracy_tables"]:
        assert (ROOT / table["public_file"]).exists()
        for runner in table["runner_files"]:
            assert (ROOT / runner).exists()
        for result_path in table["result_files"]:
            if result_path in optional_results:
                continue
            assert (ROOT / result_path).exists(), result_path
        assert table["commands"]
        assert table["metrics"]


def test_public_manifest_dataset_links_are_valid():
    data = _manifest()
    datasets = {dataset["id"]: dataset for dataset in data["datasets"]}

    for table in data["public_accuracy_tables"]:
        for dataset_id in table["datasets"]:
            assert dataset_id in datasets

    for dataset in datasets.values():
        for cache_path in dataset.get("local_cache", []):
            assert (ROOT / cache_path).exists(), cache_path


def test_public_manifest_mode_cards_keep_backend_claims_separate():
    data = _manifest()
    table_ids = {table["id"] for table in data["public_accuracy_tables"]}
    dataset_ids = {dataset["id"] for dataset in data["datasets"]}
    mode_cards = data["benchmark_mode_cards"]

    assert {card["mode_family"] for card in mode_cards} == {
        "heuristic",
        "pure_nli",
        "tuned_threshold_nli",
        "hybrid_judge",
        "local_judge",
    }

    for card in mode_cards:
        assert card["claim_boundary"]
        assert card["public_metric"]
        for table_id in card["public_tables"]:
            assert table_id in table_ids
        for dataset_id in card["datasets"]:
            assert dataset_id in dataset_ids
        for runner in card["runner_files"]:
            assert (ROOT / runner).exists()
        for result_path in card["result_files"]:
            assert (ROOT / result_path).exists(), result_path


def test_reproduction_docs_reference_manifest_and_cache_schema():
    doc = (ROOT / "benchmarks" / "PUBLIC_BENCHMARKS.md").read_text(encoding="utf-8")

    assert "benchmarks/public_accuracy_manifest.toml" in doc
    assert "benchmarks/CACHE_SCHEMA.md" in doc
    assert "readme_aggrefact_leaderboard" in doc
    assert "Benchmark Mode Cards" in doc
    assert "hybrid_remote_judge_halueval" in doc
