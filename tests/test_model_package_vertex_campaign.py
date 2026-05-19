# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Vertex model package campaign tests

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from benchmarks import model_package_vertex_campaign as campaign
from benchmarks.model_package_vertex_campaign import (
    _result_filename,
    _validate_stage_output,
    build_run_items,
)


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
    commands = {
        item.stage_id: item.command
        for item in items
        if item.model_alias == "balanced-default"
    }

    assert "--model" in commands["aggrefact_anchor_vertex"]
    assert "--scorer-template" in commands["aggrefact_anchor_vertex"]
    template_index = commands["aggrefact_anchor_vertex"].index("--scorer-template")
    assert commands["aggrefact_anchor_vertex"][template_index + 1] == "factcg"
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


def test_campaign_item_carries_registry_scorer_template(tmp_path: Path):
    items = build_run_items(
        output_root=tmp_path,
        model_aliases={"balanced-default", "deberta-small"},
        stage_ids={"aggrefact_anchor_vertex"},
    )

    assert [(item.model_alias, item.scorer_template) for item in items] == [
        ("balanced-default", "factcg"),
        ("deberta-small", "sequence-pair"),
    ]


def test_aggrefact_quality_gate_rejects_collapsed_default_output(tmp_path: Path):
    item = build_run_items(
        output_root=tmp_path,
        model_aliases={"balanced-default"},
        stage_ids={"aggrefact_anchor_vertex"},
    )[0]
    item.output_dir.mkdir(parents=True)
    (item.output_dir / _result_filename(item.stage_id)).write_text(
        json.dumps(
            {
                "global_balanced_accuracy": 0.52,
                "predictions": [1] * 96 + [0] * 4,
            }
        ),
        encoding="utf-8",
    )

    assert "below quality gate" in _validate_stage_output(item)


def test_run_one_exports_scorer_template_to_subprocess(
    tmp_path: Path,
    monkeypatch,
):
    item = build_run_items(
        output_root=tmp_path,
        model_aliases={"balanced-default"},
        stage_ids={"aggrefact_anchor_vertex"},
    )[0]
    recorded = {}

    def fake_run(command, cwd, env, text, check):
        recorded["command"] = command
        recorded["env"] = env
        item.output_dir.mkdir(parents=True, exist_ok=True)
        (item.output_dir / _result_filename(item.stage_id)).write_text(
            json.dumps(
                {
                    "global_balanced_accuracy": 0.76,
                    "predictions": [0, 1] * 10,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(campaign, "_resolve_model_source", lambda value: "/resolved")
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setattr(
        campaign, "_move_stage_outputs", lambda stage_id, output_dir: None
    )

    result = campaign._run_one(item, output_root=tmp_path)

    assert result.returncode == 0
    assert recorded["env"]["DIRECTOR_SCORER_TEMPLATE"] == "factcg"
    assert "--scorer-template" in recorded["command"]
    assert recorded["command"][recorded["command"].index("--scorer-template") + 1] == (
        "factcg"
    )
