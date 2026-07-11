# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for temporal freshness scoring pipeline."""

from __future__ import annotations

import time

import director_ai.core.scoring.temporal_freshness as temporal_mod
from director_ai.core.scoring.temporal_freshness import (
    CitationStatusSignal,
    score_temporal_freshness,
)


class TestPositionDetection:
    def test_ceo_reference(self):
        result = score_temporal_freshness("The CEO of Apple is Tim Cook.")
        assert result.has_temporal_claims
        assert any(c.claim_type == "position" for c in result.claims)

    def test_president_reference(self):
        result = score_temporal_freshness("The president of France is Emmanuel Macron.")
        assert any(c.claim_type == "position" for c in result.claims)

    def test_no_position(self):
        result = score_temporal_freshness("Water boils at 100 degrees Celsius.")
        pos_claims = [c for c in result.claims if c.claim_type == "position"]
        assert len(pos_claims) == 0


class TestStatisticDetection:
    def test_population(self):
        result = score_temporal_freshness("The population of Japan is 125 million.")
        assert any(c.claim_type == "statistic" for c in result.claims)

    def test_gdp(self):
        result = score_temporal_freshness("GDP of Germany was 4.2 trillion.")
        assert any(c.claim_type == "statistic" for c in result.claims)


class TestCurrentReference:
    def test_currently_flag(self):
        result = score_temporal_freshness(
            "The company currently employs 50,000 people worldwide."
        )
        assert any(c.claim_type == "current_reference" for c in result.claims)

    def test_as_of_flag(self):
        result = score_temporal_freshness("As of 2024, the market share was 15%.")
        assert any(c.claim_type == "current_reference" for c in result.claims)

    def test_python_path_detects_current_reference_with_domain_hint(self):
        result = score_temporal_freshness(
            "As of 2024, the market share was 15%.",
            domain="finance",
        )

        claim = next(c for c in result.claims if c.claim_type == "current_reference")
        assert "As of 2024" in claim.text
        assert claim.reason == "Temporal claim may not reflect current state"

    def test_rust_exception_falls_back_to_python_detection(self, monkeypatch):
        monkeypatch.setattr(temporal_mod, "_RUST_TEMPORAL", True)
        monkeypatch.setattr(
            temporal_mod,
            "rust_score_temporal_freshness",
            lambda _text: (_ for _ in ()).throw(RuntimeError("ffi fail")),
            raising=False,
        )
        result = score_temporal_freshness(
            "As of 2024, the market share was 15%.",
            source_timestamp=None,
            citation_statuses=None,
            domain="",
        )
        assert any(c.claim_type == "current_reference" for c in result.claims)

    def test_rust_non_runtime_exception_falls_back_to_python_detection(
        self, monkeypatch
    ):
        monkeypatch.setattr(temporal_mod, "_RUST_TEMPORAL", True)
        monkeypatch.setattr(
            temporal_mod,
            "rust_score_temporal_freshness",
            lambda _text: (_ for _ in ()).throw(ValueError("ffi fail")),
            raising=False,
        )
        result = score_temporal_freshness(
            "As of 2024, the market share was 15%.",
            source_timestamp=None,
            citation_statuses=None,
            domain="",
        )
        assert any(c.claim_type == "current_reference" for c in result.claims)

    def test_rust_type_error_falls_back_to_python_detection(self, monkeypatch):
        monkeypatch.setattr(temporal_mod, "_RUST_TEMPORAL", True)
        monkeypatch.setattr(
            temporal_mod,
            "rust_score_temporal_freshness",
            lambda _text: (_ for _ in ()).throw(TypeError("ffi fail")),
            raising=False,
        )
        result = score_temporal_freshness(
            "As of 2024, the market share was 15%.",
            source_timestamp=None,
            citation_statuses=None,
            domain="",
        )
        assert any(c.claim_type == "current_reference" for c in result.claims)


class TestRecordDetection:
    def test_world_record(self):
        result = score_temporal_freshness(
            "The world record for 100m sprint is 9.58 seconds."
        )
        assert any(c.claim_type == "record" for c in result.claims)

    def test_superlative(self):
        result = score_temporal_freshness("The tallest building in the world is.")
        assert any(c.claim_type == "record" for c in result.claims)

    def test_python_path_detects_record_with_source_timestamp(self):
        result = score_temporal_freshness(
            "The world record for 100m sprint is 9.58 seconds.",
            source_timestamp=time.time(),
        )

        claim = next(c for c in result.claims if c.claim_type == "record")
        assert "world record" in claim.text
        assert claim.reason == "Records and rankings change over time"


class TestStalenessRisk:
    def test_fresh_source_lower_risk(self):
        now = time.time()
        result_fresh = score_temporal_freshness(
            "The CEO of Apple is Tim Cook.",
            source_timestamp=now - 86400,  # 1 day old
        )
        result_stale = score_temporal_freshness(
            "The CEO of Apple is Tim Cook.",
            source_timestamp=now - 365 * 86400,  # 1 year old
        )
        fresh_risk = result_fresh.overall_staleness_risk
        stale_risk = result_stale.overall_staleness_risk
        assert stale_risk > fresh_risk

    def test_no_source_timestamp(self):
        result = score_temporal_freshness("The CEO of Apple is Tim Cook.")
        assert result.overall_staleness_risk > 0

    def test_stale_claims_property(self):
        result = score_temporal_freshness(
            "The CEO of Apple is someone. The population of Earth is 8 billion.",
            source_timestamp=time.time() - 400 * 86400,
        )
        assert len(result.stale_claims) >= 1


class TestCleanText:
    def test_no_temporal_claims(self):
        result = score_temporal_freshness(
            "The mathematical constant pi is approximately 3.14159."
        )
        assert not result.has_temporal_claims
        assert result.overall_staleness_risk == 0.0


class TestExternalCitationStatus:
    def test_retracted_source_sets_external_risk(self):
        result = score_temporal_freshness(
            "Trial X reported a 12 percent response rate.",
            citation_statuses=[
                CitationStatusSignal(
                    source_id="doi:10.example/withdrawn",
                    status="retracted",
                    status_source="publisher-feed",
                )
            ],
        )

        assert result.external_status_risk == 1.0
        assert result.overall_staleness_risk == 1.0
        assert result.risky_statuses[0].status == "retracted"

    def test_mapping_status_signal_is_supported(self):
        old_source = time.time() - 400 * 86400

        result = score_temporal_freshness(
            "The population estimate is 12 million.",
            citation_statuses=[
                {
                    "source_id": "dataset:population",
                    "status": "active",
                    "published_at": old_source,
                }
            ],
            max_age_days=180,
        )

        assert result.citation_status_verdicts[0].source_id == "dataset:population"
        assert result.external_status_risk == 0.5
        assert result.overall_staleness_risk >= result.external_status_risk

    def test_high_stakes_domain_shortens_age_window(self):
        old_source = time.time() - 60 * 86400
        neutral = score_temporal_freshness(
            "The current guideline recommends follow-up testing.",
            citation_statuses=[
                {
                    "source_id": "guideline:one",
                    "status": "active",
                    "published_at": old_source,
                }
            ],
        )
        medical = score_temporal_freshness(
            "The current guideline recommends follow-up testing.",
            citation_statuses=[
                {
                    "source_id": "guideline:one",
                    "status": "active",
                    "published_at": old_source,
                }
            ],
            domain="medical",
        )

        assert medical.external_status_risk > neutral.external_status_risk

    def test_status_reason_branches_are_operator_specific(self):
        result = score_temporal_freshness(
            "The source status is externally supplied.",
            citation_statuses=[
                {"source_id": "paper:stale", "status": "stale"},
                {"source_id": "paper:updated", "status": "updated"},
                {"source_id": "paper:active", "status": "active", "observed_at": ""},
            ],
        )

        reasons = {
            verdict.source_id: verdict.reason
            for verdict in result.citation_status_verdicts
        }
        assert reasons["paper:stale"] == "Source has a newer external status"
        assert reasons["paper:updated"] == "Source changed after first publication"
        assert (
            reasons["paper:active"] == "Source status does not increase freshness risk"
        )

    def test_invalid_age_window_is_rejected(self):
        try:
            score_temporal_freshness("current claim", max_age_days=0)
        except ValueError as exc:
            assert "max_age_days" in str(exc)
        else:
            raise AssertionError("expected ValueError")


def test_status_feed_produces_citation_status_verdicts():
    from director_ai.core.scoring.temporal_freshness import CitationStatusVerdict

    result = score_temporal_freshness(
        "Trial X reported a 12 percent response rate.",
        citation_statuses=[
            CitationStatusSignal(
                source_id="doi:10.example/withdrawn",
                status="retracted",
                status_source="publisher-feed",
            )
        ],
    )

    risky = result.risky_statuses
    assert risky and all(isinstance(v, CitationStatusVerdict) for v in risky)
    verdict = risky[0]
    assert verdict.source_id == "doi:10.example/withdrawn"
    assert verdict.status == "retracted"
    assert verdict.status_source == "publisher-feed"
    assert verdict.risk == 1.0
    assert verdict.reason
