# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Banking response policy assessment

"""Deterministic banking-response policy assessment.

This module is a runtime guardrail, not legal or regulatory advice. Operators
must configure policy references, numeric evidence references, and human-review
workflow ownership for their jurisdiction and product catalogue.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_DEPOSIT_INSURANCE_LIMIT_USD = 250_000

Severity = Literal["low", "medium", "high", "critical"]
Action = Literal["audit", "halt", "escalate", "block"]

_SEVERITY_RANK: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

_MONEY_RE = re.compile(
    r"\$\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>thousand|million|billion|k|m|bn)?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*"
    r"(?:percent|percentage points?)\b)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_INVESTMENT_ADVICE_RE = re.compile(
    r"\b(?:buy|sell|short|overweight|underweight)\b.{0,80}"
    r"\b(?:stock|stocks|share|shares|security|securities|bond|fund|crypto)\b",
    re.IGNORECASE | re.DOTALL,
)
_GUARANTEED_RETURN_RE = re.compile(
    r"\b(?:guaranteed return|risk-free investment|cannot lose money)\b",
    re.IGNORECASE,
)
_FINANCIAL_CRIME_CONTROL_BYPASS_RE = re.compile(
    r"\b(?:avoid(?:s|ing)?|bypass(?:es|ing)?|evad(?:e|es|ing)|skip(?:s|ping)?|"
    r"circumvent(?:s|ing)?)\b.{0,120}"
    r"\b(?:kyc|aml|sanctions?|screening|identity verification|customer due diligence)\b"
    r"|"
    r"\b(?:split|structure|break up)\b.{0,120}"
    r"\b(?:transfer|payment|deposit|withdrawal|transaction)s?\b.{0,120}"
    r"\b(?:avoid(?:s|ing)?|bypass(?:es|ing)?|evad(?:e|es|ing)|skip(?:s|ping)?|"
    r"circumvent(?:s|ing)?)\b.{0,120}"
    r"\b(?:kyc|aml|sanctions?|screening|reporting|review)\b",
    re.IGNORECASE | re.DOTALL,
)
_CREDIT_APPROVAL_GUARANTEE_RE = re.compile(
    r"\b(?:guaranteed|automatic|certain)\b.{0,80}"
    r"\b(?:approval|approved|qualify|acceptance)\b.{0,120}"
    r"\b(?:loan|mortgage|credit|line of credit|card|overdraft)\b"
    r"|"
    r"\b(?:loan|mortgage|credit|line of credit|card|overdraft)\b.{0,120}"
    r"\b(?:approved|approval|qualify|acceptance)\b.{0,80}"
    r"\b(?:regardless of|without)\b.{0,80}"
    r"\b(?:credit|income|affordability|underwriting|employment)\b",
    re.IGNORECASE | re.DOTALL,
)

_CITATION_TERMS = (
    "account",
    "annual percentage yield",
    "apy",
    "coverage limit",
    "deposit insurance",
    "fdic",
    "fee",
    "interest rate",
    "loan",
    "minimum balance",
    "mortgage",
    "wire transfer",
)

_NUMERIC_CONTEXT_TERMS = (
    "annual percentage yield",
    "apy",
    "balance",
    "coverage",
    "deposit insurance",
    "fee",
    "interest rate",
    "limit",
    "minimum",
    "payment",
    "rate",
    "transfer",
)

_COMPLAINT_TERMS = (
    "chargeback",
    "complaint",
    "dispute",
    "error resolution",
    "fraud",
    "unauthorised",
    "unauthorized",
)


@dataclass(frozen=True)
class BankingPolicyFinding:
    """One banking policy finding safe to persist in audit logs."""

    code: str
    severity: Severity
    action: Action
    detail: str
    policy_refs: tuple[str, ...] = ()
    evidence_required: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the finding without raw customer prompt or response text."""

        return {
            "code": self.code,
            "severity": self.severity,
            "action": self.action,
            "detail": self.detail,
            "policy_refs": list(self.policy_refs),
            "evidence_required": list(self.evidence_required),
        }


@dataclass(frozen=True)
class BankingPolicyReport:
    """Result of a banking response policy assessment."""

    approved: bool
    requires_human_review: bool
    jurisdiction: str
    product_line: str
    policy_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    numeric_evidence_refs: tuple[str, ...]
    findings: tuple[BankingPolicyFinding, ...]

    @property
    def highest_severity(self) -> str:
        """Return the highest finding severity, or ``none`` when clean."""

        if not self.findings:
            return "none"
        return max(
            (finding.severity for finding in self.findings),
            key=lambda severity: _SEVERITY_RANK[severity],
        )

    @property
    def blocked_codes(self) -> tuple[str, ...]:
        """Return finding codes that prevent automatic approval."""

        return tuple(
            finding.code
            for finding in self.findings
            if finding.action in {"block", "halt", "escalate"}
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a deterministic JSON-safe shape."""

        return {
            "approved": self.approved,
            "requires_human_review": self.requires_human_review,
            "jurisdiction": self.jurisdiction,
            "product_line": self.product_line,
            "policy_refs": list(self.policy_refs),
            "evidence_refs": list(self.evidence_refs),
            "numeric_evidence_refs": list(self.numeric_evidence_refs),
            "highest_severity": self.highest_severity,
            "blocked_codes": list(self.blocked_codes),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def assess_banking_response(
    prompt: str,
    response: str,
    *,
    evidence_refs: Iterable[str] = (),
    numeric_evidence_refs: Iterable[str] = (),
    policy_refs: Iterable[str] = (),
    jurisdiction: str = "US",
    product_line: str = "deposit",
    deposit_insurance_limit_usd: int = DEFAULT_DEPOSIT_INSURANCE_LIMIT_USD,
    human_review_acknowledged: bool = False,
) -> BankingPolicyReport:
    """Assess a banking response for evidence and escalation requirements."""

    clean_evidence_refs = _normalise_refs(evidence_refs)
    clean_numeric_refs = _normalise_refs(numeric_evidence_refs)
    clean_policy_refs = _normalise_refs(policy_refs)
    findings: list[BankingPolicyFinding] = []

    if _requires_citation(response) and not clean_evidence_refs:
        findings.append(
            BankingPolicyFinding(
                code="citation_required",
                severity="medium",
                action="halt",
                detail="Banking product or regulatory claim lacks an evidence reference.",
                policy_refs=clean_policy_refs,
                evidence_required=("evidence_refs",),
            )
        )

    if _requires_numeric_evidence(response) and not clean_numeric_refs:
        findings.append(
            BankingPolicyFinding(
                code="numeric_evidence_required",
                severity="high",
                action="escalate",
                detail="Numeric banking claim lacks a numeric evidence reference.",
                policy_refs=clean_policy_refs,
                evidence_required=("numeric_evidence_refs",),
            )
        )

    if _deposit_limit_mismatch(response, deposit_insurance_limit_usd):
        findings.append(
            BankingPolicyFinding(
                code="deposit_insurance_limit_mismatch",
                severity="critical",
                action="block",
                detail="Deposit insurance coverage claim conflicts with the configured limit.",
                policy_refs=clean_policy_refs,
                evidence_required=("deposit_insurance_limit_usd",),
            )
        )

    if _contains_investment_recommendation(response):
        findings.append(
            BankingPolicyFinding(
                code="investment_advice_escalation_required",
                severity="high",
                action="escalate",
                detail="Investment recommendation requires licensed human review.",
                policy_refs=clean_policy_refs,
                evidence_required=("human_review_acknowledged",),
            )
        )

    if (
        _contains_complaint_or_dispute(prompt, response)
        and not human_review_acknowledged
    ):
        findings.append(
            BankingPolicyFinding(
                code="complaint_escalation_required",
                severity="high",
                action="escalate",
                detail="Complaint or dispute handling requires human review.",
                policy_refs=clean_policy_refs,
                evidence_required=("human_review_acknowledged",),
            )
        )

    if _contains_financial_crime_control_bypass(response):
        findings.append(
            BankingPolicyFinding(
                code="financial_crime_control_bypass_blocked",
                severity="critical",
                action="block",
                detail="Response advises bypassing financial-crime controls.",
                policy_refs=clean_policy_refs,
                evidence_required=("financial_crime_control_review",),
            )
        )

    if _contains_credit_approval_guarantee(response):
        findings.append(
            BankingPolicyFinding(
                code="credit_approval_guarantee_blocked",
                severity="critical",
                action="block",
                detail="Credit or lending approval guarantee bypasses underwriting review.",
                policy_refs=clean_policy_refs,
                evidence_required=("lending_underwriting_review",),
            )
        )

    requires_human_review = any(
        finding.action in {"block", "escalate"} for finding in findings
    )
    approved = not any(
        finding.action in {"block", "halt", "escalate"} for finding in findings
    )
    return BankingPolicyReport(
        approved=approved,
        requires_human_review=requires_human_review,
        jurisdiction=jurisdiction,
        product_line=product_line,
        policy_refs=clean_policy_refs,
        evidence_refs=clean_evidence_refs,
        numeric_evidence_refs=clean_numeric_refs,
        findings=tuple(findings),
    )


def _normalise_refs(refs: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    clean: list[str] = []
    for ref in refs:
        candidate = ref.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            clean.append(candidate)
    return tuple(clean)


def _requires_citation(response: str) -> bool:
    text = response.casefold()
    return any(term in text for term in _CITATION_TERMS) or bool(
        _MONEY_RE.search(response)
    )


def _requires_numeric_evidence(response: str) -> bool:
    text = response.casefold()
    has_numeric_claim = bool(_MONEY_RE.search(response) or _PERCENT_RE.search(response))
    return has_numeric_claim and any(term in text for term in _NUMERIC_CONTEXT_TERMS)


def _deposit_limit_mismatch(response: str, configured_limit: int) -> bool:
    for sentence in _deposit_insurance_sentences(response):
        for amount in _money_amounts(sentence):
            if amount != configured_limit:
                return True
    return False


def _deposit_insurance_sentences(response: str) -> tuple[str, ...]:
    sentences = _SENTENCE_SPLIT_RE.split(response.strip())
    return tuple(
        sentence
        for sentence in sentences
        if _mentions_deposit_insurance_limit(sentence)
    )


def _mentions_deposit_insurance_limit(text: str) -> bool:
    lowered = text.casefold()
    return (
        "fdic" in lowered
        or "deposit insurance" in lowered
        or ("insurance covers" in lowered and "deposit" in lowered)
    )


def _money_amounts(text: str) -> tuple[int, ...]:
    amounts: list[int] = []
    for match in _MONEY_RE.finditer(text):
        raw_amount = float(match.group("amount").replace(",", ""))
        scale = (match.group("scale") or "").casefold()
        multiplier = {
            "thousand": 1_000,
            "k": 1_000,
            "million": 1_000_000,
            "m": 1_000_000,
            "billion": 1_000_000_000,
            "bn": 1_000_000_000,
        }.get(scale, 1)
        amounts.append(int(raw_amount * multiplier))
    return tuple(amounts)


def _contains_investment_recommendation(response: str) -> bool:
    return bool(
        _INVESTMENT_ADVICE_RE.search(response) or _GUARANTEED_RETURN_RE.search(response)
    )


def _contains_complaint_or_dispute(prompt: str, response: str) -> bool:
    text = f"{prompt}\n{response}".casefold()
    return any(term in text for term in _COMPLAINT_TERMS)


def _contains_financial_crime_control_bypass(response: str) -> bool:
    return bool(_FINANCIAL_CRIME_CONTROL_BYPASS_RE.search(response))


def _contains_credit_approval_guarantee(response: str) -> bool:
    return bool(_CREDIT_APPROVAL_GUARANTEE_RE.search(response))
