# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — scorer-miss forensics tests
"""Behavioural tests for tenant-safe scorer-miss forensics."""

from __future__ import annotations

import pytest

from director_ai.core.observability import (
    build_forensics_report,
    render_forensics_markdown,
    render_forensics_text,
)


def _records() -> list[dict[str, object]]:
    return [
        {
            "director.eval.answer_id": "fn-1",
            "director.eval.approved": True,
            "director.eval.score": 0.82,
            "director.eval.threshold": 0.6,
            "director.eval.scorer": "nli",
            "director.eval.model": "model-a",
            "director.eval.model_revision": "rev-a",
            "director.eval.domain": "legal",
            "director.eval.evidence_count": 0,
            "label": "hallucination",
        },
        {
            "answer_id": "fp-1",
            "approved": False,
            "score": 0.41,
            "threshold": 0.6,
            "scorer": "heuristic",
            "model": "model-b",
            "domain": "medical",
            "evidence_count": 2,
            "unsupported_claims": 1,
            "expected_label": "grounded",
        },
        {
            "case_id": "ok-1",
            "decision": "halt",
            "score": 0.2,
            "threshold": 0.6,
            "scorer": "nli",
            "label": "hallucinated",
        },
    ]


def test_build_forensics_report_classifies_misses_and_counts() -> None:
    report = build_forensics_report(_records())

    assert report.total_records == 3
    assert report.labelled_records == 3
    assert report.misses_total == 2
    assert report.false_negatives == 1
    assert report.false_positives == 1
    assert report.missed_by_scorer == {"heuristic": 1, "nli": 1}
    assert report.missed_by_domain == {"legal": 1, "medical": 1}


def test_false_negative_action_points_to_kb_refresh_when_no_evidence() -> None:
    report = build_forensics_report(_records())
    case = report.cases[0]

    assert case.outcome == "false_negative"
    assert case.knowledge_state == "kb:unversioned:no_evidence"
    assert case.recommended_action == "refresh_or_add_governed_facts"
    assert "no evidence" in case.reason


def test_false_positive_action_points_to_retrieval_mapping() -> None:
    report = build_forensics_report(_records())
    case = report.cases[1]

    assert case.outcome == "false_positive"
    assert case.knowledge_state == "kb:unversioned:unsupported_claims"
    assert case.recommended_action == "review_retrieval_source_mapping"


def test_report_payload_is_tenant_safe() -> None:
    payload = build_forensics_report(_records()).to_dict()

    assert payload["privacy"]["raw_prompt_included"] is False
    assert payload["privacy"]["raw_response_included"] is False
    assert payload["privacy"]["raw_evidence_text_included"] is False
    assert all("raw_text" not in case for case in payload["cases"])


def test_renderers_include_summary_and_actions() -> None:
    report = build_forensics_report(_records())

    text = render_forensics_text(report)
    markdown = render_forensics_markdown(report)

    assert "misses_total: 2" in text
    assert "fn-1: false_negative" in text
    assert "# Guardrail Forensics" in markdown
    assert "refresh_or_add_governed_facts" in markdown


def test_markdown_empty_report_explains_no_records_or_misses() -> None:
    markdown = render_forensics_markdown(build_forensics_report([]))

    assert "No scorer misses in the labelled window." in markdown
    assert "No records supplied." in markdown


def test_unlabelled_records_are_retained_but_not_counted_as_misses() -> None:
    report = build_forensics_report(
        [
            {
                "director.eval.decision": "allow",
                "director.eval.score": 0.9,
                "director.eval.threshold": 0.6,
            }
        ]
    )

    assert report.labelled_records == 0
    assert report.misses_total == 0
    assert report.cases[0].outcome == "unlabelled_allow"


def test_decision_aliases_and_kb_version_are_normalised() -> None:
    report = build_forensics_report(
        [
            {
                "director.eval.decision": "approved",
                "director.eval.score": 0.7,
                "director.eval.threshold": 0.6,
                "director.eval.evidence_count": 1,
                "director.eval.kb_version": "2026-06-19",
                "review_label": "correct",
            },
            {
                "decision": "rejected",
                "score": 0.4,
                "threshold": 0.6,
                "knowledge_version": "2026-06-18",
                "label": "false",
            },
        ]
    )

    assert report.cases[0].outcome == "correct_allow"
    assert report.cases[0].knowledge_state == "kb:2026-06-19:evidence_present"
    assert report.cases[0].recommended_action == "no_operator_action"
    assert report.cases[1].outcome == "correct_halt"
    assert report.cases[1].knowledge_state == "kb:2026-06-18:no_evidence"


def test_false_negative_actions_cover_unsupported_claim_variants() -> None:
    report = build_forensics_report(
        [
            {
                "approved": True,
                "score": 0.8,
                "threshold": 0.6,
                "evidence_count": 2,
                "unsupported_claims": 0,
                "label": "hallucination",
            },
            {
                "approved": True,
                "score": 0.8,
                "threshold": 0.6,
                "evidence_count": 2,
                "unsupported_claims": 1,
                "label": "hallucination",
            },
        ]
    )

    assert report.cases[0].reason.endswith("without unsupported claims")
    assert (
        report.cases[0].recommended_action
        == "add_counterexample_and_recalibrate_scorer"
    )
    assert report.cases[1].reason.endswith("unsupported-claim metadata")
    assert report.cases[1].recommended_action == "inspect_claim_attribution_thresholds"


def test_rejects_record_without_decision_or_approval() -> None:
    with pytest.raises(ValueError, match="approved"):
        build_forensics_report([{"score": 0.5, "threshold": 0.6}])


@pytest.mark.parametrize(
    "record",
    [
        {"approved": True, "score": True, "threshold": 0.6},
        {"approved": True, "score": 0.5, "threshold": False},
        {"approved": True, "score": 0.5, "threshold": 0.6, "evidence_count": True},
        {"approved": True, "score": 0.5, "threshold": 0.6, "unsupported_claims": 1.2},
    ],
)
def test_rejects_non_numeric_forensics_fields(record: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_forensics_report([record])
