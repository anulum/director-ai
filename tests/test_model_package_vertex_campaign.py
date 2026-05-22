# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - model package campaign tests

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks import model_package_vertex_campaign as campaign
from benchmarks.model_package_vertex_campaign import (
    _result_filename,
    _upload_tree,
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


def test_campaign_can_run_local_output_only_without_cloud_upload(
    tmp_path: Path,
    monkeypatch,
):
    item = campaign.CampaignRunItem(
        model_alias="balanced-default",
        runtime_model="/models/factcg",
        scorer_template="factcg",
        stage_id="aggrefact_anchor_vertex",
        evidence_id="aggrefact_anchor",
        command=("python", "-m", "benchmarks.aggrefact_eval"),
        output_dir=tmp_path / "balanced-default" / "aggrefact_anchor_vertex",
    )

    monkeypatch.setattr(campaign, "build_run_items", lambda **kwargs: (item,))
    monkeypatch.setattr(campaign, "_require_free_disk", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        campaign,
        "_run_one",
        lambda item, output_root: campaign.StageResult(
            model_alias=item.model_alias,
            stage_id=item.stage_id,
            evidence_id=item.evidence_id,
            returncode=0,
            elapsed_seconds=1.0,
            output_dir=str(item.output_dir),
            uploaded_files=(),
        ),
    )
    monkeypatch.setattr(
        campaign,
        "_upload_tree",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected upload")),
    )

    results = campaign.run_campaign(
        output_root=tmp_path,
        bucket_uri="",
        prefix="",
        min_free_gb=0.0,
        upload=False,
    )

    assert len(results) == 1
    assert results[0].uploaded_files == ()
    assert (tmp_path / "campaign_summary.json").exists()


def test_upload_tree_supports_file_destination(tmp_path: Path):
    root = tmp_path / "root"
    destination = tmp_path / "provider-mounted-artifacts"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "result.json").write_text('{"ok": true}', encoding="utf-8")

    uploaded = _upload_tree(
        bucket_uri=f"file://{destination}",
        prefix="campaign/run",
        root=root,
    )

    copied = destination / "campaign" / "run" / "nested" / "result.json"
    assert copied.read_text(encoding="utf-8") == '{"ok": true}'
    assert uploaded == (copied.as_uri(),)


def test_provider_neutral_campaign_entrypoint_exposes_help():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (
            repo_root / "src",
            repo_root,
            env.get("PYTHONPATH", ""),
        )
        if str(path)
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.model_package_campaign",
            "--help",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "--upload-uri" in completed.stdout
    assert "--no-upload" in completed.stdout
