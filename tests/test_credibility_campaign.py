# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - credibility campaign tests

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.credibility_campaign import (
    CAMPAIGN_PATH,
    campaign_to_dict,
    load_campaign,
    next_runnable_stages,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_campaign_manifest_is_valid_and_vertex_scoped():
    campaign = load_campaign(CAMPAIGN_PATH)
    findings = validate_campaign(campaign, root=ROOT)

    assert findings == []
    assert campaign.vertex.project == "gotm-director-ai"
    assert campaign.vertex.region == "europe-west4"
    assert campaign.vertex.accelerator == "NVIDIA_TESLA_T4"
    assert campaign.vertex.bucket == "gs://gotm-director-ai-training"


def test_campaign_orders_anchor_before_domain_and_gated_stages():
    campaign = load_campaign(CAMPAIGN_PATH)
    stage_ids = [stage.stage_id for stage in campaign.stages]

    assert stage_ids.index("aggrefact_anchor_vertex") < stage_ids.index(
        "ragtruth_vertex"
    )
    assert stage_ids.index("halueval_vertex") < stage_ids.index(
        "patronus_halubench_wire"
    )
    assert stage_ids.index("legal_contractnli_vertex") < stage_ids.index(
        "validation_packet_freeze"
    )
    assert stage_ids.index("medical_mednli_pubmedqa_vertex") < stage_ids.index(
        "validation_packet_freeze"
    )


def test_next_runnable_stages_respect_dependencies_and_gated_access():
    campaign = load_campaign(CAMPAIGN_PATH)

    first = next_runnable_stages(campaign, completed_stage_ids=set())
    assert [stage.stage_id for stage in first] == ["aggrefact_anchor_vertex"]

    after_anchor = next_runnable_stages(
        campaign,
        completed_stage_ids={"aggrefact_anchor_vertex"},
    )
    assert {stage.stage_id for stage in after_anchor} == {
        "ragtruth_vertex",
        "halueval_vertex",
        "financebench_vertex",
    }

    after_text = next_runnable_stages(
        campaign,
        completed_stage_ids={
            "aggrefact_anchor_vertex",
            "ragtruth_vertex",
            "halueval_vertex",
            "financebench_vertex",
            "legal_contractnli_vertex",
            "medical_mednli_pubmedqa_vertex",
            "patronus_halubench_wire",
        },
    )
    assert "auw_halubench_geospatial_vertex" not in {
        stage.stage_id for stage in after_text
    }

    gated = next_runnable_stages(
        campaign,
        completed_stage_ids={
            "aggrefact_anchor_vertex",
            "ragtruth_vertex",
            "halueval_vertex",
            "financebench_vertex",
            "legal_contractnli_vertex",
            "medical_mednli_pubmedqa_vertex",
            "patronus_halubench_wire",
        },
        include_gated=True,
    )
    assert [stage.stage_id for stage in gated] == ["auw_halubench_geospatial_vertex"]


def test_campaign_dict_is_json_serialisable_and_redacts_human_tokens():
    campaign = load_campaign(CAMPAIGN_PATH)
    payload = campaign_to_dict(campaign)
    encoded = json.dumps(payload, sort_keys=True)

    assert "HF_TOKEN" in encoded
    assert "<accepted-access-token>" not in encoded
    assert "secret" not in encoded.lower()
