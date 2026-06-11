# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — threat-intelligence tests

"""Multi-angle tests for STIX-aligned threat-intelligence matching.

Covers indicator validation and matching (substring/regex/sha256), the matcher's
severity-ordered results / attribution / duplicate rejection, and the STIX 2.1
bundle importer (pattern subset, compound decomposition, attribution resolution,
severity, and the skip-don't-guess handling of unparsable input), plus
ProductionGuard wiring.
"""

from __future__ import annotations

import hashlib

import pytest

from director_ai.core.threat_intel import (
    IndicatorType,
    Severity,
    ThreatIndicator,
    ThreatIntelligenceMatcher,
    ThreatMatch,
    from_stix_bundle,
)

_HASH = hashlib.sha256(b"malware").hexdigest()


def _ind(**kw) -> ThreatIndicator:
    base = dict(
        id="i1", name="n", indicator_type=IndicatorType.SUBSTRING, pattern="evil"
    )
    base.update(kw)
    return ThreatIndicator(**base)


class TestIndicator:
    def test_substring_match(self):
        ind = _ind(pattern="Verify Your Account")
        assert ind.matches("please verify your account now") is True
        assert ind.matches("nothing here") is False

    def test_regex_match(self):
        ind = _ind(indicator_type=IndicatorType.REGEX, pattern=r"api[_-]?key")
        assert ind.matches("send api_key=1") is True
        assert ind.matches("no secret") is False

    def test_sha256_match(self):
        ind = _ind(indicator_type=IndicatorType.SHA256, pattern=_HASH)
        assert ind.matches("malware") is True
        assert ind.matches("benign") is False

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"id": "  "}, "id is required"),
            ({"pattern": ""}, "pattern is required"),
            ({"indicator_type": IndicatorType.REGEX, "pattern": "(unclosed"}, "regex"),
            ({"indicator_type": IndicatorType.SHA256, "pattern": "nothex"}, "sha256"),
        ],
    )
    def test_validation(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _ind(**kw)

    def test_to_dict(self):
        d = _ind(attribution="APT29", severity=Severity.HIGH, labels=("a",)).to_dict()
        assert d["attribution"] == "APT29"
        assert d["severity"] == "high"
        assert d["labels"] == ["a"]


class TestMatcher:
    def _matcher(self) -> ThreatIntelligenceMatcher:
        return ThreatIntelligenceMatcher(
            [
                _ind(id="low", pattern="lure", severity=Severity.LOW, attribution="A"),
                _ind(
                    id="crit",
                    indicator_type=IndicatorType.REGEX,
                    pattern=r"key=\w+",
                    severity=Severity.CRITICAL,
                    attribution="B",
                ),
            ]
        )

    def test_match_sorted_by_severity(self):
        hits = self._matcher().match("a lure with key=abc")
        assert [h.indicator_id for h in hits] == ["crit", "low"]

    def test_no_match(self):
        assert self._matcher().match("clean text") == []

    def test_is_threat(self):
        m = self._matcher()
        assert m.is_threat("lure") is True
        assert m.is_threat("clean") is False

    def test_attributions_deduped_sorted(self):
        m = self._matcher()
        m.add(_ind(id="dup", pattern="lure", attribution="A"))  # same attribution A
        assert m.attributions("a lure with key=x") == ("A", "B")

    def test_attributions_skip_empty(self):
        m = ThreatIntelligenceMatcher([_ind(id="x", pattern="lure", attribution="")])
        assert m.attributions("lure") == ()

    def test_duplicate_id_rejected(self):
        m = self._matcher()
        with pytest.raises(ValueError, match="duplicate indicator id"):
            m.add(_ind(id="low", pattern="z"))

    def test_indicator_count(self):
        assert self._matcher().indicator_count == 2

    def test_match_to_dict_tenant_safe(self):
        hit = self._matcher().match("key=abc")[0]
        assert set(hit.to_dict()) == {
            "indicator_id",
            "name",
            "indicator_type",
            "attribution",
            "severity",
        }
        assert isinstance(hit, ThreatMatch)


class TestStixImport:
    def test_substring_with_attribution(self):
        bundle = {
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--a",
                    "name": "Evil domain",
                    "pattern": "[domain-name:value = 'evil.example.com']",
                    "labels": ["malicious-activity"],
                    "x_severity": "high",
                },
                {"type": "intrusion-set", "id": "intrusion-set--x", "name": "APT29"},
                {
                    "type": "relationship",
                    "relationship_type": "indicates",
                    "source_ref": "indicator--a",
                    "target_ref": "intrusion-set--x",
                },
            ]
        }
        [ind] = from_stix_bundle(bundle)
        assert ind.indicator_type is IndicatorType.SUBSTRING
        assert ind.pattern == "evil.example.com"
        assert ind.attribution == "APT29"
        assert ind.severity is Severity.HIGH
        assert ind.labels == ("malicious-activity",)

    def test_sha256_and_regex_patterns(self):
        bundle = {
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--h",
                    "pattern": f"[file:hashes.'SHA-256' = '{_HASH}']",
                },
                {
                    "type": "indicator",
                    "id": "indicator--r",
                    "pattern": "[url:value MATCHES 'http://bad[0-9]+']",
                },
            ]
        }
        kinds = {i.id: i.indicator_type for i in from_stix_bundle(bundle)}
        assert kinds["indicator--h"] is IndicatorType.SHA256
        assert kinds["indicator--r"] is IndicatorType.REGEX

    def test_compound_pattern_decomposed(self):
        bundle = {
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--c",
                    "pattern": "[a:b = 'one' AND c:d = 'two']",
                }
            ]
        }
        ids = sorted(i.id for i in from_stix_bundle(bundle))
        assert ids == ["indicator--c#0", "indicator--c#1"]

    def test_attributed_to_threat_actor(self):
        bundle = {
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--a",
                    "pattern": "[x:y = 'z']",
                },
                {"type": "threat-actor", "id": "threat-actor--t", "name": "Lazarus"},
                {
                    "type": "relationship",
                    "relationship_type": "attributed-to",
                    "source_ref": "indicator--a",
                    "target_ref": "threat-actor--t",
                },
            ]
        }
        assert from_stix_bundle(bundle)[0].attribution == "Lazarus"

    def test_unparsable_and_invalid_objects_skipped(self):
        bundle = {
            "objects": [
                "not-a-mapping",
                {
                    "type": "indicator",
                    "id": "indicator--noid-pattern",
                    "pattern": "junk",
                },
                {"type": "indicator", "id": "", "pattern": "[x:y = 'v']"},
                {
                    "type": "indicator",
                    "id": "indicator--empty",
                    "pattern": "[x:y = '']",
                },
                {
                    "type": "indicator",
                    "id": "indicator--ok",
                    "pattern": "[x:y = 'good']",
                },
            ]
        }
        out = from_stix_bundle(bundle)
        assert [i.id for i in out] == ["indicator--ok"]

    def test_invalid_severity_defaults_medium(self):
        bundle = {
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--s",
                    "pattern": "[x:y = 'v']",
                    "x_severity": "apocalyptic",
                }
            ]
        }
        assert from_stix_bundle(bundle)[0].severity is Severity.MEDIUM

    def test_irrelevant_relationship_ignored(self):
        bundle = {
            "objects": [
                {"type": "indicator", "id": "indicator--a", "pattern": "[x:y = 'v']"},
                {"type": "intrusion-set", "id": "intrusion-set--x", "name": "APT"},
                {
                    "type": "relationship",
                    "relationship_type": "uses",  # not indicates/attributed-to
                    "source_ref": "indicator--a",
                    "target_ref": "intrusion-set--x",
                },
            ]
        }
        assert from_stix_bundle(bundle)[0].attribution == ""

    def test_unresolved_attribution_target(self):
        # An 'indicates' relationship whose target is not a known actor leaves
        # the indicator unattributed rather than inventing one.
        bundle = {
            "objects": [
                {"type": "indicator", "id": "indicator--a", "pattern": "[x:y = 'v']"},
                {
                    "type": "relationship",
                    "relationship_type": "indicates",
                    "source_ref": "indicator--a",
                    "target_ref": "intrusion-set--missing",
                },
            ]
        }
        assert from_stix_bundle(bundle)[0].attribution == ""

    def test_escaped_quote_in_value(self):
        bundle = {
            "objects": [
                {
                    "type": "indicator",
                    "id": "indicator--q",
                    "pattern": r"[x:y = 'o\'brien']",
                }
            ]
        }
        assert from_stix_bundle(bundle)[0].pattern == "o'brien"

    def test_empty_bundle(self):
        assert from_stix_bundle({}) == []


class TestGuardWiring:
    def test_production_guard_exposes_threat_intel(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        ti = guard.threat_intel
        assert isinstance(ti, ThreatIntelligenceMatcher)
        assert guard.threat_intel is ti  # cached
        ti.add(_ind(id="w", pattern="lure", attribution="APT29"))
        assert ti.attributions("a lure") == ("APT29",)
