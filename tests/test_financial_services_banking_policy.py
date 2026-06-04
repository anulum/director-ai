# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Financial services banking policy tests

from __future__ import annotations

import json

import director_ai.core.customer_model_factory as cmf
from director_ai.core.financial_services import (
    __all__ as financial_services_exports,
)
from director_ai.core.financial_services import (
    assess_banking_response,
)
from director_ai.core.financial_services.banking_policy import (
    DEFAULT_DEPOSIT_INSURANCE_LIMIT_USD,
    BankingPolicyReport,
)


def _codes(report: BankingPolicyReport) -> set[str]:
    return {finding.code for finding in report.findings}


def test_standard_deposit_insurance_claim_with_evidence_is_approved() -> None:
    report = assess_banking_response(
        prompt="What is the standard FDIC deposit coverage limit?",
        response=(
            "FDIC insurance covers up to $250,000 per depositor, per insured "
            "bank, for each ownership category."
        ),
        evidence_refs=("policy://fdic/deposit-insurance/current",),
        numeric_evidence_refs=("policy://fdic/deposit-insurance/current#limit",),
        policy_refs=("policy://financial-services/deposit-disclosures",),
    )

    assert report.approved is True
    assert report.requires_human_review is False
    assert report.findings == ()
    assert DEFAULT_DEPOSIT_INSURANCE_LIMIT_USD == 250_000


def test_inflated_deposit_insurance_claim_blocks_response() -> None:
    report = assess_banking_response(
        prompt="What is the standard FDIC deposit coverage limit?",
        response="FDIC insurance covers up to $500,000 per depositor.",
        evidence_refs=("policy://fdic/deposit-insurance/current",),
        numeric_evidence_refs=("policy://fdic/deposit-insurance/current#limit",),
        policy_refs=("policy://financial-services/deposit-disclosures",),
    )

    assert report.approved is False
    assert report.requires_human_review is True
    assert "deposit_insurance_limit_mismatch" in _codes(report)
    assert report.highest_severity == "critical"


def test_product_rate_claim_requires_numeric_evidence_reference() -> None:
    report = assess_banking_response(
        prompt="What is the current savings account APY?",
        response="The current savings account APY is 4.25%.",
        evidence_refs=("policy://bank-products/savings-disclosure",),
        policy_refs=("policy://financial-services/rate-disclosures",),
    )

    assert report.approved is False
    assert report.requires_human_review is True
    assert "numeric_evidence_required" in _codes(report)


def test_investment_recommendation_escalates_to_human_review() -> None:
    report = assess_banking_response(
        prompt="What should I do with this volatile stock?",
        response="You should buy shares now because the return is guaranteed.",
        evidence_refs=("policy://financial-services/advice-controls",),
        numeric_evidence_refs=("policy://financial-services/advice-controls",),
        policy_refs=("policy://financial-services/advice-controls",),
    )

    assert report.approved is False
    assert report.requires_human_review is True
    assert "investment_advice_escalation_required" in _codes(report)


def test_customer_complaint_response_requires_human_review_acknowledgement() -> None:
    report = assess_banking_response(
        prompt="I need to dispute an unauthorized wire transfer.",
        response="We opened a dispute case and will contact you with next steps.",
        evidence_refs=("policy://financial-services/complaint-handling",),
        policy_refs=("policy://financial-services/complaint-handling",),
    )

    assert report.approved is False
    assert report.requires_human_review is True
    assert "complaint_escalation_required" in _codes(report)

    reviewed = assess_banking_response(
        prompt="I need to dispute an unauthorized wire transfer.",
        response="We opened a dispute case and will contact you with next steps.",
        evidence_refs=("policy://financial-services/complaint-handling",),
        policy_refs=("policy://financial-services/complaint-handling",),
        human_review_acknowledged=True,
    )

    assert reviewed.approved is True
    assert "complaint_escalation_required" not in _codes(reviewed)


def test_report_serialisation_omits_raw_customer_text() -> None:
    prompt = "Customer secret phrase: disputed payroll deposit."
    response = "FDIC insurance covers up to $500,000 per depositor."

    report = assess_banking_response(
        prompt=prompt,
        response=response,
        evidence_refs=("policy://fdic/deposit-insurance/current",),
        numeric_evidence_refs=("policy://fdic/deposit-insurance/current#limit",),
    )
    payload = report.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["approved"] is False
    assert prompt not in encoded
    assert response not in encoded
    assert "deposit_insurance_limit_mismatch" in encoded


def test_financial_services_public_surface_stays_outside_private_factory() -> None:
    assert "assess_banking_response" in financial_services_exports
    assert not hasattr(cmf, "assess_banking_response")
