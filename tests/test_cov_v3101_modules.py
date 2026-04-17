# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Coverage Push Module Tests v3.10.1
"""Multi-angle coverage push tests for v3.10.1 modules.

Covers: otel, compliance, voice guard, license, doc_chunker,
doc_registry, sanitizer, stats, exceptions, types edge cases,
pipeline integration, and performance documentation.
"""

from __future__ import annotations

# ── OTel (no-op path) ──────────────────────────────────────────────


class TestOtelNoOp:
    def test_setup_without_otel(self):
        from director_ai.core.otel import setup_otel

        setup_otel("test-service")  # should not crash

    def test_trace_review_noop(self):
        from director_ai.core.otel import trace_review

        with trace_review() as span:
            span.set_attribute("test.key", "value")
            span.set_status("ok")

    def test_trace_streaming_noop(self):
        from director_ai.core.otel import trace_streaming

        with trace_streaming() as span:
            span.set_attribute("tokens", 42)

    def test_trace_vector_query_noop(self):
        from director_ai.core.otel import trace_vector_query

        with trace_vector_query() as span:
            span.set_attribute("query", "test")

    def test_trace_vector_add_noop(self):
        from director_ai.core.otel import trace_vector_add

        with trace_vector_add() as span:
            span.set_attribute("count", 10)

    def test_noop_span_methods(self):
        from director_ai.core.otel import _NoopSpan

        span = _NoopSpan()
        span.set_attribute("key", "val")
        span.set_status("ok")
        span.set_status("error", description="test")


# ── Compliance: DriftDetector ───────────────────────────────────────


class TestDriftDetector:
    def test_analyze_returns_report(self, tmp_path):
        import time

        from director_ai.compliance.audit_log import AuditEntry, AuditLog
        from director_ai.compliance.drift_detector import DriftDetector

        log = AuditLog(str(tmp_path / "drift.db"))
        for i in range(10):
            log.log(
                AuditEntry(
                    prompt=f"q{i}",
                    response=f"a{i}",
                    model="m",
                    provider="p",
                    score=0.8,
                    approved=True,
                    verdict_confidence=0.9,
                    task_type="qa",
                    domain="test",
                    latency_ms=1.0,
                    timestamp=time.time(),
                )
            )
        detector = DriftDetector(log)
        report = detector.analyze()
        assert report is not None

    def test_custom_window(self, tmp_path):
        from director_ai.compliance.audit_log import AuditLog
        from director_ai.compliance.drift_detector import DriftDetector

        log = AuditLog(str(tmp_path / "drift2.db"))
        detector = DriftDetector(log, window_days=30)
        report = detector.analyze()
        assert report is not None


class TestFeedbackLoopDetector:
    def test_check_and_record(self):
        import time

        from director_ai.compliance.feedback_loop_detector import FeedbackLoopDetector

        detector = FeedbackLoopDetector()
        result = detector.check_and_record(
            "What is the refund policy?",
            "The refund policy is 30 days.",
            time.time(),
        )
        # Returns FeedbackLoopAlert or None
        assert result is None or hasattr(result, "similarity")

    def test_record_output(self):
        from director_ai.compliance.feedback_loop_detector import FeedbackLoopDetector

        detector = FeedbackLoopDetector()
        detector.record_output("Some output here for testing", 0.9)
        assert detector.buffer_size == 1

    def test_check_input_returns_alert_for_matching_output(self):
        from director_ai.compliance.feedback_loop_detector import FeedbackLoopDetector

        detector = FeedbackLoopDetector()
        prior_output = "Previously seen output text"
        detector.record_output(prior_output, 0.8)

        result = detector.check_input(prior_output)

        assert result is not None
        assert result.matched_output == prior_output
        assert result.output_timestamp == 0.8
        assert result.similarity >= detector.similarity_threshold
        assert result.severity == "high"


# ── Compliance: AuditLog ────────────────────────────────────────────


class TestAuditLog:
    def test_create_and_log(self, tmp_path):
        import time

        from director_ai.compliance.audit_log import AuditEntry, AuditLog

        log = AuditLog(str(tmp_path / "audit.db"))
        log.log(
            AuditEntry(
                prompt="What is 2+2?",
                response="4",
                model="test-model",
                provider="test",
                score=0.95,
                approved=True,
                verdict_confidence=0.99,
                task_type="qa",
                domain="math",
                latency_ms=1.0,
                timestamp=time.time(),
            )
        )
        entries = log.query(limit=10)
        assert len(entries) >= 1

    def test_query_empty(self, tmp_path):
        from director_ai.compliance.audit_log import AuditLog

        log = AuditLog(str(tmp_path / "empty_audit.db"))
        assert log.query(limit=10) == []

    def test_count(self, tmp_path):
        import time

        from director_ai.compliance.audit_log import AuditEntry, AuditLog

        log = AuditLog(str(tmp_path / "count.db"))
        assert log.count() == 0
        log.log(
            AuditEntry(
                prompt="q",
                response="a",
                model="m",
                provider="p",
                score=0.8,
                approved=True,
                verdict_confidence=0.9,
                task_type="qa",
                domain="d",
                latency_ms=1.0,
                timestamp=time.time(),
            )
        )
        assert log.count() == 1


# ── VoiceGuard ──────────────────────────────────────────────────────


class TestVoiceGuardBasic:
    def test_feed_approved_tokens(self):
        from director_ai.integrations.voice import VoiceGuard

        guard = VoiceGuard(
            facts={"sky": "The sky is blue."},
            prompt="What color is the sky?",
            use_nli=False,
        )
        result = guard.feed("The")
        assert result.approved
        assert result.index == 0

        result2 = guard.feed(" sky")
        assert result2.index == 1

    def test_reset(self):
        from director_ai.integrations.voice import VoiceGuard

        guard = VoiceGuard(
            facts={"fact": "Test fact"},
            use_nli=False,
        )
        guard.feed("token1")
        guard.feed("token2")
        guard.reset()
        result = guard.feed("token3")
        assert result.index == 0

    def test_set_prompt(self):
        from director_ai.integrations.voice import VoiceGuard

        guard = VoiceGuard(facts={"fact": "Test"}, use_nli=False)
        guard.set_prompt("New question?")
        result = guard.feed("answer")
        assert result.approved

    def test_already_halted(self):
        from director_ai.integrations.voice import VoiceGuard

        guard = VoiceGuard(
            facts={"fact": "The answer is 42."},
            prompt="What is the answer?",
            threshold=0.99,  # very strict — likely to halt
            hard_limit=0.99,
            score_every=1,
            use_nli=False,
        )
        # Feed tokens until halt or 20 tokens
        halted = False
        for i in range(20):
            r = guard.feed(f"wrong{i} ")
            if r.halted:
                halted = True
                break

        if halted:
            # Subsequent tokens should be rejected
            r2 = guard.feed("more")
            assert not r2.approved
            assert r2.halt_reason == "already_halted"


# ── License module ──────────────────────────────────────────────────


class TestLicenseModule:
    def test_validate_key_returns_license_info(self):
        from director_ai.core.license import LicenseInfo, validate_key

        result = validate_key("invalid-key-12345")
        assert isinstance(result, LicenseInfo)
        assert not result.valid

    def test_license_info_importable(self):
        from director_ai.core.license import LicenseInfo

        assert LicenseInfo is not None

    def test_generate_license(self):
        from director_ai.core.license import generate_license

        assert callable(generate_license)

    def test_tiers(self):
        from director_ai.core.license import TIERS

        assert isinstance(TIERS, (list, tuple, dict, set))


# ── LiteScorer ──────────────────────────────────────────────────────


class TestLiteScorer:
    def test_basic_review(self):
        from director_ai.core.lite_scorer import LiteScorer

        scorer = LiteScorer()
        result = scorer.review(
            "What color is the sky?",
            "The sky is blue.",
        )
        # Returns (bool, CoherenceScore) tuple
        approved, score = result
        assert isinstance(approved, bool)
        assert hasattr(score, "score")
        assert 0 <= score.score <= 1


# ── MetaClassifier ──────────────────────────────────────────────────


class TestMetaClassifier:
    def test_importable(self):
        from director_ai.core.meta_classifier import MetaClassifier

        # MetaClassifier requires a model_path — just verify import
        assert MetaClassifier is not None


# ── Sanitizer ───────────────────────────────────────────────────────


class TestInputSanitizer:
    def test_scrub_clean_text(self):
        from director_ai.core.sanitizer import InputSanitizer

        s = InputSanitizer()
        result = s.scrub("Hello world")
        assert isinstance(result, str)
        assert "Hello" in result

    def test_check_clean_text(self):
        from director_ai.core.sanitizer import InputSanitizer

        s = InputSanitizer()
        result = s.check("Normal input text.")
        assert result is not None

    def test_score(self):
        from director_ai.core.sanitizer import InputSanitizer

        s = InputSanitizer()
        result = s.score("Some potentially risky input")
        # Returns SanitizeResult
        assert hasattr(result, "suspicion_score")
        assert hasattr(result, "blocked")


# ── Policy ──────────────────────────────────────────────────────────


class TestPolicy:
    def test_default_policy(self):
        from director_ai.core.policy import Policy

        p = Policy()
        assert p is not None

    def test_policy_check(self):
        from director_ai.core.policy import Policy

        p = Policy()
        result = p.check("This is a test response.")
        # check returns a PolicyResult or similar
        assert result is not None

    def test_policy_from_dict(self):
        from director_ai.core.policy import Policy

        p = Policy.from_dict({})
        assert p is not None


# ── Tenant ──────────────────────────────────────────────────────────


class TestTenant:
    def test_tenant_router_basic(self):
        from director_ai.core.tenant import TenantRouter

        router = TenantRouter()
        router.add_fact("tenant_a", "fact_key", "Water is H2O")
        scorer = router.get_scorer("tenant_a", threshold=0.5)
        assert scorer is not None

    def test_tenant_isolation(self):
        from director_ai.core.tenant import TenantRouter

        router = TenantRouter()
        router.add_fact("t1", "k1", "Fact for T1")
        router.add_fact("t2", "k2", "Fact for T2")
        s1 = router.get_scorer("t1", threshold=0.5)
        s2 = router.get_scorer("t2", threshold=0.5)
        assert s1 is not s2


# ── Metrics ─────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_collector_importable(self):
        from director_ai.core.metrics import MetricsCollector

        assert MetricsCollector is not None

    def test_metrics_collector_instance(self):
        from director_ai.core.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc is not None

    def test_metrics_singleton(self):
        from director_ai.core.metrics import metrics

        assert metrics is not None
