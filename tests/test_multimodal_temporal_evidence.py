# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multimodal temporal evidence tests

from __future__ import annotations

import json

from benchmarks import multimodal_temporal_evidence as evidence


def test_image_claim_probe_reports_allow_halt_and_caption_conflict() -> None:
    packet = evidence.run_image_claim_probe()

    assert packet["passed"] is True
    assert packet["raw_payload_leaked"] is False
    assert [(record["case"], record["decision"]) for record in packet["records"]] == [
        ("supported", "allow"),
        ("hallucinated", "halt"),
        ("caption_conflict", "halt"),
    ]
    assert all(record["matched"] for record in packet["records"])


def test_video_temporal_probe_reports_frame_level_halt() -> None:
    packet = evidence.run_video_temporal_probe()

    assert packet["passed"] is True
    assert packet["decision"] == "halt"
    assert packet["verdict"] == "temporal_inconsistent"
    assert "media://video-temporal#frame:3" in packet["evidence_refs"]
    assert packet["claim_text_leaked"] is False


def test_hashbag_fallback_probe_uses_dependency_free_guard() -> None:
    packet = evidence.run_hashbag_fallback_probe()

    assert packet["passed"] is True
    assert packet["label"] in {"consistent", "uncertain"}
    assert packet["similarity"] >= 0.0


def test_multimodal_temporal_evidence_payload_has_acceptance_summary(
    monkeypatch,
) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")

    packet = evidence.run_multimodal_temporal_evidence()

    assert packet["schema_version"] == "director-ai.multimodal-temporal-evidence.v1"
    assert packet["benchmark"] == "multimodal_temporal_evidence"
    assert packet["git_commit"] == "abc123"
    assert packet["acceptance"] == {
        "passed": True,
        "checks": {
            "image_claim_paths": True,
            "video_temporal_consistency": True,
            "hashbag_dependency_free": True,
        },
        "limits": {
            "local_only": True,
            "external_vision_nli_benchmark_included": False,
            "real_video_model_included": False,
        },
    }
    assert set(packet["probes"]) == {
        "image_claim_paths",
        "video_temporal_consistency",
        "hashbag_dependency_free",
    }


def test_main_writes_requested_output_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(evidence, "_git_commit", lambda: "abc123")
    output = tmp_path / "multimodal-temporal.json"

    exit_code = evidence.main(["--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["acceptance"]["passed"] is True
