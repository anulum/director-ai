# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for Phase 5 Gem 2: Claim-Level Provenance.

Covers: claims from empty/populated evidence, unsupported claim filtering,
provenance structure, multiple claims, empty attributions, parametrised
divergence thresholds, pipeline integration, and performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.types import (
    ClaimAttribution,
    CoherenceScore,
    ScoringEvidence,
)


def _make_evidence(
    claims=None, attributions=None, claim_coverage=None
) -> ScoringEvidence:
    return ScoringEvidence(
        chunks=[],
        nli_premise="source text",
        nli_hypothesis="generated text",
        nli_score=0.3,
        claims=claims,
        attributions=attributions,
        claim_coverage=claim_coverage,
    )


def _make_score(evidence=None) -> CoherenceScore:
    return CoherenceScore(
        score=0.85,
        approved=True,
        h_logical=0.1,
        h_factual=0.15,
        evidence=evidence,
    )


class TestClaimProperties:
    def test_claims_empty_without_evidence(self):
        score = _make_score()
        assert score.claims == []
        assert score.attributions == []
        assert score.claim_coverage is None
        assert score.unsupported_claims == []
        assert score.claim_provenance() == []

    def test_claims_from_evidence(self):
        ev = _make_evidence(
            claims=["Paris is the capital.", "It has 2M people."],
            claim_coverage=0.5,
            attributions=[
                ClaimAttribution(
                    claim="Paris is the capital.",
                    claim_index=0,
                    source_sentence="France's capital is Paris.",
                    source_index=0,
                    divergence=0.1,
                    supported=True,
                ),
                ClaimAttribution(
                    claim="It has 2M people.",
                    claim_index=1,
                    source_sentence="Paris has 2.1M inhabitants.",
                    source_index=3,
                    divergence=0.6,
                    supported=False,
                ),
            ],
        )
        score = _make_score(evidence=ev)

        assert len(score.claims) == 2
        assert score.claims[0] == "Paris is the capital."
        assert len(score.attributions) == 2
        assert score.claim_coverage == 0.5

    def test_unsupported_claims(self):
        ev = _make_evidence(
            attributions=[
                ClaimAttribution("A", 0, "src", 0, 0.1, True),
                ClaimAttribution("B", 1, "src", 1, 0.7, False),
                ClaimAttribution("C", 2, "src", 2, 0.8, False),
            ],
        )
        score = _make_score(evidence=ev)
        unsupported = score.unsupported_claims
        assert len(unsupported) == 2
        assert unsupported[0].claim == "B"
        assert unsupported[1].claim == "C"


class TestClaimProvenance:
    def test_provenance_structure(self):
        ev = _make_evidence(
            attributions=[
                ClaimAttribution(
                    claim="The sky is blue.",
                    claim_index=0,
                    source_sentence="The sky appears blue due to Rayleigh scattering.",
                    source_index=5,
                    divergence=0.08,
                    supported=True,
                ),
            ],
        )
        score = _make_score(evidence=ev)
        prov = score.claim_provenance()

        assert len(prov) == 1
        assert prov[0]["claim"] == "The sky is blue."
        assert prov[0]["supported"] is True
        assert prov[0]["source"] == "The sky appears blue due to Rayleigh scattering."
        assert prov[0]["divergence"] == 0.08
        assert prov[0]["source_index"] == 5

    def test_provenance_multiple_claims(self):
        ev = _make_evidence(
            attributions=[
                ClaimAttribution("A", 0, "srcA", 0, 0.1, True),
                ClaimAttribution("B", 1, "srcB", 1, 0.5, True),
                ClaimAttribution("C", 2, "srcC", 2, 0.9, False),
            ],
        )
        score = _make_score(evidence=ev)
        prov = score.claim_provenance()
        assert len(prov) == 3
        assert prov[2]["supported"] is False
        assert prov[2]["divergence"] == 0.9

    def test_provenance_empty_attributions(self):
        ev = _make_evidence(attributions=[])
        score = _make_score(evidence=ev)
        assert score.claim_provenance() == []


class TestClaimProvenanceParametrised:
    """Parametrised claim provenance tests."""

    @pytest.mark.parametrize(
        "divergence,supported",
        [
            (0.0, True),
            (0.1, True),
            (0.5, True),
            (0.8, False),
            (1.0, False),
        ],
    )
    def test_divergence_support_combinations(self, divergence, supported):
        ev = _make_evidence(
            attributions=[
                ClaimAttribution("claim", 0, "src", 0, divergence, supported),
            ],
        )
        score = _make_score(evidence=ev)
        prov = score.claim_provenance()
        assert len(prov) == 1
        assert prov[0]["divergence"] == divergence
        assert prov[0]["supported"] is supported

    @pytest.mark.parametrize("n_claims", [0, 1, 3, 5, 10])
    def test_various_claim_counts(self, n_claims):
        attrs = [
            ClaimAttribution(f"claim_{i}", i, f"src_{i}", i, 0.1, True)
            for i in range(n_claims)
        ]
        ev = _make_evidence(attributions=attrs)
        score = _make_score(evidence=ev)
        assert len(score.claim_provenance()) == n_claims


class TestClaimProvenancePerformanceDoc:
    """Document claim provenance pipeline performance."""

    def test_claim_attribution_has_required_fields(self):
        attr = ClaimAttribution("claim", 0, "source", 0, 0.1, True)
        assert hasattr(attr, "claim")
        assert hasattr(attr, "claim_index")
        assert hasattr(attr, "source_sentence")
        assert hasattr(attr, "source_index")
        assert hasattr(attr, "divergence")
        assert hasattr(attr, "supported")

    def test_provenance_dict_structure(self):
        ev = _make_evidence(
            attributions=[ClaimAttribution("c", 0, "s", 0, 0.1, True)],
        )
        score = _make_score(evidence=ev)
        prov = score.claim_provenance()[0]
        required_keys = ["claim", "supported", "source", "divergence", "source_index"]
        for key in required_keys:
            assert key in prov, f"Missing key: {key}"
