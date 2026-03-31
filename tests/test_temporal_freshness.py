# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Phase 5 Gem 8: Temporal Freshness Scoring."""

from __future__ import annotations

import time

from director_ai.core.scoring.temporal_freshness import score_temporal_freshness


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


class TestRecordDetection:
    def test_world_record(self):
        result = score_temporal_freshness(
            "The world record for 100m sprint is 9.58 seconds."
        )
        assert any(c.claim_type == "record" for c in result.claims)

    def test_superlative(self):
        result = score_temporal_freshness("The tallest building in the world is.")
        assert any(c.claim_type == "record" for c in result.claims)


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
