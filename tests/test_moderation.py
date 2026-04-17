# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — moderation tests (PII + toxicity + policy wiring)

"""Multi-angle coverage of moderation detectors: regex PII
(email/SSN/phone/IBAN/IPv4/passport), Presidio adapter against a
stub analyser, keyword toxicity seed list + custom patterns,
Detoxify adapter against a stub classifier, and Policy.with_moderation
end-to-end with mixed detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from director_ai.core.safety.moderation import (
    DetoxifyDetector,
    KeywordToxicityDetector,
    ModerationMatch,
    ModerationResult,
    PresidioPIIDetector,
    RegexPIIDetector,
)
from director_ai.core.safety.policy import Policy

# --- Regex PII -------------------------------------------------------


class TestRegexPII:
    def test_email_detected(self):
        res = RegexPIIDetector().analyse("contact me at a.b+c@example.com soon")
        cats = {m.category for m in res.matches}
        assert "email" in cats

    def test_ssn_detected(self):
        res = RegexPIIDetector().analyse("my ssn is 123-45-6789")
        assert any(m.category == "ssn" for m in res.matches)

    def test_iban_detected(self):
        res = RegexPIIDetector().analyse(
            "wire to CH9300762011623852957 by friday"
        )
        assert any(m.category == "iban" for m in res.matches)

    def test_ipv4_detected(self):
        res = RegexPIIDetector().analyse("server 10.0.0.5 responded")
        assert any(m.category == "ipv4" for m in res.matches)

    def test_credit_card_detected(self):
        res = RegexPIIDetector().analyse("card 4111-1111-1111-1111 declined")
        assert any(m.category == "credit_card" for m in res.matches)

    def test_empty_input_returns_no_matches(self):
        res = RegexPIIDetector().analyse("")
        assert res.matches == []
        assert res.flagged is False

    def test_extra_patterns_added(self):
        det = RegexPIIDetector(extra_patterns=[("license", r"\bDL-\d{6}\b")])
        res = det.analyse("license DL-123456 expires tomorrow")
        assert any(m.category == "license" for m in res.matches)

    def test_extra_patterns_reject_invalid_regex(self):
        with pytest.raises(ValueError, match="invalid regex"):
            RegexPIIDetector(extra_patterns=[("broken", "[")])

    def test_match_snippet_trims_window(self):
        text = "prefix " * 20 + "4111-1111-1111-1111" + " suffix" * 20
        det = RegexPIIDetector()
        res = det.analyse(text)
        m = res.matches[0]
        snippet = m.snippet(window=10)
        assert "4111-1111-1111-1111" in snippet
        assert len(snippet) <= len("4111-1111-1111-1111") + 20 + 10

    def test_prefer_rust_false_forces_python_backend(self):
        det = RegexPIIDetector(prefer_rust=False)
        assert det.backend == "python"
        # Python backend still finds the same hits.
        res = det.analyse("card 4111-1111-1111-1111 email a@b.co")
        cats = {m.category for m in res.matches}
        assert "credit_card" in cats
        assert "email" in cats

    def test_default_uses_rust_when_available(self):
        det = RegexPIIDetector()
        try:
            import backfire_kernel  # noqa: F401 — probe only
        except ImportError:
            pytest.skip("backfire_kernel not installed — nothing to assert")
        assert det.backend == "rust"

    def test_rust_and_python_agree_on_categories(self):
        try:
            import backfire_kernel  # noqa: F401 — probe only
        except ImportError:
            pytest.skip("backfire_kernel not installed — cannot compare")
        text = (
            "email a@b.co, card 4111-1111-1111-1111, "
            "ssn 123-45-6789, ip 10.0.0.1"
        )
        rust = RegexPIIDetector(prefer_rust=True)
        py = RegexPIIDetector(prefer_rust=False)
        rust_cats = sorted(m.category for m in rust.analyse(text).matches)
        py_cats = sorted(m.category for m in py.analyse(text).matches)
        assert rust_cats == py_cats


# --- Presidio adapter ------------------------------------------------


@dataclass
class _StubPresidioResult:
    entity_type: str
    start: int
    end: int
    score: float = 0.9


class _StubPresidio:
    def __init__(self, results: list[_StubPresidioResult]) -> None:
        self.results = results
        self.calls: list[dict[str, Any]] = []

    def analyze(self, **kwargs: Any) -> list[_StubPresidioResult]:
        self.calls.append(kwargs)
        return list(self.results)


class TestPresidio:
    def test_basic_match(self):
        stub = _StubPresidio(
            [_StubPresidioResult("PERSON", 5, 12, score=0.85)]
        )
        det = PresidioPIIDetector(stub)
        res = det.analyse("hi   Sotek visits ANULUM")
        assert len(res.matches) == 1
        assert res.matches[0].category == "person"
        assert res.matches[0].score == pytest.approx(0.85)

    def test_below_threshold_dropped(self):
        stub = _StubPresidio(
            [_StubPresidioResult("PERSON", 0, 5, score=0.3)]
        )
        det = PresidioPIIDetector(stub, score_threshold=0.5)
        assert det.analyse("hello").matches == []

    def test_entities_filter_forwarded(self):
        stub = _StubPresidio([])
        det = PresidioPIIDetector(
            stub, entities=["PERSON", "LOCATION"], language="de"
        )
        det.analyse("Text")
        assert stub.calls[0]["entities"] == ["PERSON", "LOCATION"]
        assert stub.calls[0]["language"] == "de"

    def test_fallback_when_entities_not_accepted(self):
        class _EntityUnawareStub:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def analyze(self, **kwargs: Any) -> list[_StubPresidioResult]:
                if "entities" in kwargs:
                    raise TypeError("unexpected kw 'entities'")
                self.calls.append(kwargs)
                return [_StubPresidioResult("EMAIL_ADDRESS", 0, 5)]

        stub = _EntityUnawareStub()
        det = PresidioPIIDetector(stub, entities=["EMAIL_ADDRESS"])
        res = det.analyse("test@x")
        assert len(res.matches) == 1
        assert "entities" not in stub.calls[0]

    def test_rejects_none_analyzer(self):
        with pytest.raises(ValueError, match="analyzer is required"):
            PresidioPIIDetector(None)  # type: ignore[arg-type]

    def test_empty_input_short_circuits(self):
        stub = _StubPresidio(
            [_StubPresidioResult("PERSON", 0, 5, score=0.9)]
        )
        det = PresidioPIIDetector(stub)
        assert det.analyse("").matches == []
        assert stub.calls == []


# --- Keyword toxicity ------------------------------------------------


class TestKeywordToxicity:
    def test_seed_list_matches(self):
        det = KeywordToxicityDetector()
        res = det.analyse("go kill yourself loser")
        cats = {m.category for m in res.matches}
        assert "keyword" in cats or "self_harm_encouragement" in cats

    def test_threat_category_matches(self):
        det = KeywordToxicityDetector()
        res = det.analyse("i will kill you next time")
        assert any(m.category == "threat" for m in res.matches)

    def test_empty_input_returns_empty(self):
        det = KeywordToxicityDetector()
        assert det.analyse("").matches == []

    def test_extra_keywords_respected(self):
        det = KeywordToxicityDetector(extra_keywords=["bad_token_xyz"])
        res = det.analyse("this contains bad_token_xyz and nothing else")
        assert any(m.category == "keyword" for m in res.matches)

    def test_case_insensitive_default(self):
        det = KeywordToxicityDetector()
        assert det.analyse("I WILL KILL YOU").flagged

    def test_case_sensitive_flag(self):
        det = KeywordToxicityDetector(case_sensitive=True)
        assert det.analyse("I WILL KILL YOU").flagged is False
        assert det.analyse("i will kill you").flagged is True

    def test_invalid_extra_pattern_rejected(self):
        with pytest.raises(ValueError, match="invalid regex"):
            KeywordToxicityDetector(extra_patterns=[("cat", "[")])

    def test_benign_text_passes(self):
        det = KeywordToxicityDetector()
        assert det.analyse("The weather is pleasant today.").matches == []


# --- Detoxify adapter ------------------------------------------------


class _StubDetoxify:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[str] = []

    def predict(self, text: str) -> dict[str, float]:
        self.calls.append(text)
        return dict(self.scores)


class TestDetoxify:
    def test_high_score_triggers_match(self):
        det = DetoxifyDetector(
            _StubDetoxify({"toxicity": 0.8, "insult": 0.2}),
            score_threshold=0.5,
        )
        res = det.analyse("offensive line here")
        cats = {m.category for m in res.matches}
        assert "toxicity" in cats
        assert "insult" not in cats

    def test_threshold_respected(self):
        det = DetoxifyDetector(
            _StubDetoxify({"toxicity": 0.55}), score_threshold=0.6
        )
        assert det.analyse("x").matches == []

    def test_categories_filter(self):
        det = DetoxifyDetector(
            _StubDetoxify({"toxicity": 0.9, "threat": 0.9}),
            categories={"threat"},
        )
        res = det.analyse("x")
        assert [m.category for m in res.matches] == ["threat"]

    def test_rejects_none_classifier(self):
        with pytest.raises(ValueError, match="classifier is required"):
            DetoxifyDetector(None)  # type: ignore[arg-type]

    def test_empty_input_short_circuits(self):
        stub = _StubDetoxify({"toxicity": 0.9})
        det = DetoxifyDetector(stub)
        assert det.analyse("").matches == []
        assert stub.calls == []

    def test_match_spans_full_text(self):
        det = DetoxifyDetector(
            _StubDetoxify({"toxicity": 0.9}), score_threshold=0.5
        )
        res = det.analyse("abc def")
        assert res.matches[0].start == 0
        assert res.matches[0].end == len("abc def")


# --- Policy integration ----------------------------------------------


class _FixedDetector:
    name = "fixed"

    def __init__(self, categories: list[str]) -> None:
        self._categories = categories

    def analyse(self, text: str) -> ModerationResult:
        return ModerationResult(
            detector=self.name,
            matches=[
                ModerationMatch(
                    detector=self.name,
                    category=c,
                    start=0,
                    end=len(text),
                    text=text,
                )
                for c in self._categories
            ],
        )


class TestPolicyIntegration:
    def test_with_moderation_appends_detectors(self):
        base = Policy(forbidden=["nope"])
        policy = base.with_moderation([_FixedDetector(["pii_email"])])
        assert len(policy.moderation_detectors) == 1
        # original policy untouched
        assert base.moderation_detectors == []

    def test_moderation_match_produces_violation(self):
        policy = Policy().with_moderation(
            [_FixedDetector(["pii_email", "toxicity:insult"])]
        )
        violations = policy.check("some text")
        rules = {v.rule for v in violations}
        assert "moderation:fixed:pii_email" in rules
        assert "moderation:fixed:toxicity:insult" in rules

    def test_no_matches_means_no_violations(self):
        policy = Policy().with_moderation([_FixedDetector([])])
        assert policy.check("clean text") == []

    def test_policy_without_moderation_still_works(self):
        policy = Policy(forbidden=["banned"])
        assert policy.check("banned phrase here")
        assert policy.check("innocent text") == []
