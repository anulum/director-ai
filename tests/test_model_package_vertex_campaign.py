# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Vertex model package campaign tests

from __future__ import annotations

from pathlib import Path

from benchmarks.model_package_vertex_campaign import build_run_items


def test_campaign_expands_all_public_vertex_package_stages(tmp_path: Path):
    items = build_run_items(output_root=tmp_path)

    assert len(items) == 21
    assert {item.model_alias for item in items} == {
        "balanced-default",
        "deberta-small",
        "deberta-large-nli",
    }
    assert {item.stage_id for item in items} == {
        "aggrefact_anchor_vertex",
        "ragtruth_vertex",
        "halueval_vertex",
        "financebench_vertex",
        "legal_contractnli_vertex",
        "medical_mednli_pubmedqa_vertex",
        "patronus_halubench_wire",
    }


def test_campaign_commands_pass_resolved_model_to_model_sensitive_runners(
    tmp_path: Path,
):
    items = build_run_items(output_root=tmp_path)
    commands = {item.stage_id: item.command for item in items if item.model_alias == "balanced-default"}

    assert "--model" in commands["aggrefact_anchor_vertex"]
    assert "--nli-model" in commands["halueval_vertex"]
    assert "--model" in commands["financebench_vertex"]
    assert "--model" in commands["legal_contractnli_vertex"]
    assert "--model" in commands["medical_mednli_pubmedqa_vertex"]
    assert "--model" in commands["patronus_halubench_wire"]
    assert commands["ragtruth_vertex"] == (
        "/usr/bin/python3",
        "benchmarks/run_ragtruth_freshqa.py",
    ) or commands["ragtruth_vertex"][:2] == (
        commands["ragtruth_vertex"][0],
        "benchmarks/run_ragtruth_freshqa.py",
    )


def test_campaign_filters_are_applied(tmp_path: Path):
    items = build_run_items(
        output_root=tmp_path,
        model_aliases={"deberta-small"},
        stage_ids={"financebench_vertex", "patronus_halubench_wire"},
    )

    assert [(item.model_alias, item.stage_id) for item in items] == [
        ("deberta-small", "financebench_vertex"),
        ("deberta-small", "patronus_halubench_wire"),
    ]
