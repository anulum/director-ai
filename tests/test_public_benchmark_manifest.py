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
EXTERNAL_PACKET = ROOT / "benchmarks" / "external_validation_packet.toml"
LOCAL_CACHE_ACCESS_MARKERS = ("downloads", "gated", "mirrored")


def _manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def _external_packet() -> dict:
    return tomllib.loads(EXTERNAL_PACKET.read_text(encoding="utf-8"))


def _assert_result_path(path: str) -> None:
    if path.startswith(("benchmarks/results/", "benchmarks/.cache/")) and path.endswith(
        ".json"
    ):
        return
    assert (ROOT / path).exists(), path


def _requires_local_cache_in_checkout(dataset: dict) -> bool:
    access = str(dataset.get("access", "")).lower()
    return not any(marker in access for marker in LOCAL_CACHE_ACCESS_MARKERS)


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

    assert (ROOT / data["cache_schema"]["path"]).exists()
    assert (ROOT / data["cache_schema"]["result_schema"]).exists()

    for table in data["public_accuracy_tables"]:
        assert (ROOT / table["public_file"]).exists()
        for runner in table["runner_files"]:
            assert (ROOT / runner).exists()
        for result_path in table["result_files"]:
            _assert_result_path(result_path)
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
            if _requires_local_cache_in_checkout(dataset):
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
            _assert_result_path(result_path)


def test_reproduction_docs_reference_manifest_and_cache_schema():
    doc = (ROOT / "benchmarks" / "PUBLIC_BENCHMARKS.md").read_text(encoding="utf-8")

    assert "benchmarks/public_accuracy_manifest.toml" in doc
    assert "benchmarks/CACHE_SCHEMA.md" in doc
    assert "readme_aggrefact_leaderboard" in doc
    assert "Benchmark Mode Cards" in doc
    assert "hybrid_remote_judge_halueval" in doc
    assert "benchmarks/EXTERNAL_VALIDATION_PACKET.md" in doc


def test_external_validation_packet_is_linked_and_complete():
    manifest = _manifest()
    packet = _external_packet()

    assert (ROOT / manifest["external_validation"]["packet"]).exists()
    assert (ROOT / manifest["external_validation"]["manifest"]).exists()
    assert (ROOT / packet["packet_doc"]).exists()
    assert packet["public_manifest"] == "benchmarks/public_accuracy_manifest.toml"
    assert packet["acceptance"]["minimum_reproducible_tasks"] >= 3

    mode_cards = {card["id"] for card in manifest["benchmark_mode_cards"]}
    dataset_ids = {dataset["id"] for dataset in manifest["datasets"]}
    required_outputs = {output["path"] for output in packet["required_outputs"]}

    assert {
        "validation/environment.json",
        "validation/raw_results/",
        "validation/metric_recalculation.md",
        "validation/failure_cases.jsonl",
        "validation/summary.md",
    } <= required_outputs

    for task in packet["benchmark_tasks"]:
        assert task["mode_card"] in mode_cards
        assert task["dataset"] in dataset_ids
        assert (ROOT / task["runner"]).exists()
        _assert_result_path(task["expected_result"])
        assert task["command"]
        assert task["primary_metrics"]
        assert task["claim_boundary"]


def test_external_validation_doc_contains_required_sections():
    doc = (ROOT / "benchmarks" / "EXTERNAL_VALIDATION_PACKET.md").read_text(
        encoding="utf-8"
    )

    for heading in [
        "## Validation Scope",
        "## Required Environment Record",
        "## Commands",
        "## Required Report Outputs",
        "## Claim Boundary Rules",
        "## Auditor Questions",
    ]:
        assert heading in doc
