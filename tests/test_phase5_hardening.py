# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — test_phase5_hardening.py

from __future__ import annotations

import time

from director_ai.core.cache import ScoreCache
from director_ai.core.kernel import SafetyKernel
from director_ai.core.sanitizer import InputSanitizer

# â”€â”€ Sanitizer: new injection patterns â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestSanitizerNewPatterns:
    def setup_method(self):
        self.san = InputSanitizer()

    def test_base64_payload_with_padding(self):
        payload = "A" * 80 + "=="
        r = self.san.score(payload)
        assert r.suspicion_score > 0
        assert "base64_payload" in r.matches

    def test_short_base64_not_blocked(self):
        r = self.san.check("SGVsbG8gV29ybGQ=")
        assert r.blocked is False

    def test_unicode_escape_injection(self):
        r = self.san.score(r"\u0069\u0067\u006e\u006f\u0072\u0065")
        assert r.suspicion_score > 0
        assert "unicode_escape_injection" in r.matches

    def test_control_char_vt(self):
        r = self.san.score("normal text\x0binjection")
        assert r.suspicion_score > 0
        assert "control_char_injection" in r.matches

    def test_control_char_ff(self):
        r = self.san.score("page\x0cbreak")
        assert r.suspicion_score > 0
        assert "control_char_injection" in r.matches

    def test_control_char_escape(self):
        r = self.san.score("some\x1b[31m colored")
        assert r.suspicion_score > 0
        assert "control_char_injection" in r.matches

    def test_bidi_override(self):
        r = self.san.score("normal \u202e reversed")
        assert r.suspicion_score > 0
        assert "bidi_override" in r.matches

    def test_bidi_lre(self):
        r = self.san.score("text \u202a embedded")
        assert r.suspicion_score > 0
        assert "bidi_override" in r.matches

    def test_clean_text_not_blocked(self):
        r = self.san.check("This is a perfectly normal question about AI safety.")
        assert r.blocked is False

    def test_normal_alphanumeric_not_base64(self):
        r = self.san.check("a" * 100)
        assert r.blocked is False


# â”€â”€ Kernel: timeout enforcement â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestKernelTimeout:
    def test_total_timeout_interrupts(self):
        k = SafetyKernel(total_timeout=0.05)

        def slow_tokens():
            for tok in ["a", "b", "c", "d"]:
                time.sleep(0.02)
                yield tok

        result = k.stream_output(slow_tokens(), lambda t: 0.9)
        assert "TOTAL TIMEOUT" in result

    def test_token_timeout_interrupts(self):
        k = SafetyKernel(token_timeout=0.01)

        def slow_callback(tok):
            time.sleep(0.05)
            return 0.9

        result = k.stream_output(["a", "b", "c"], slow_callback)
        assert "TOKEN TIMEOUT" in result

    def test_no_timeout_passes_normally(self):
        k = SafetyKernel()
        result = k.stream_output(["a", "b", "c"], lambda t: 0.9)
        assert result == "abc"

    def test_timeout_deactivates_kernel(self):
        k = SafetyKernel(total_timeout=0.05)

        def slow():
            time.sleep(0.2)
            yield "a"

        k.stream_output(slow(), lambda t: 0.9)
        assert k.is_active is False


# â”€â”€ Cache: generation versioning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestCacheGeneration:
    def test_initial_generation_zero(self):
        c = ScoreCache(max_size=16)
        assert c.generation == 0

    def test_invalidate_bumps_generation(self):
        c = ScoreCache(max_size=16)
        c.invalidate()
        assert c.generation == 1

    def test_old_generation_entry_evicted(self):
        c = ScoreCache(max_size=16)
        c.put("q", "p", 0.8, 0.1, 0.2)
        assert c.get("q", "p") is not None
        c.invalidate()
        assert c.get("q", "p") is None

    def test_new_generation_entry_survives(self):
        c = ScoreCache(max_size=16)
        c.invalidate()
        c.put("q", "p", 0.8, 0.1, 0.2)
        assert c.get("q", "p") is not None

    def test_multiple_invalidations(self):
        c = ScoreCache(max_size=16)
        c.put("q", "p", 0.8, 0.1, 0.2)
        c.invalidate()
        c.invalidate()
        c.invalidate()
        assert c.generation == 3
        assert c.get("q", "p") is None
        c.put("q", "p", 0.9, 0.0, 0.1)
        assert c.get("q", "p") is not None


# â”€â”€ Rust scorer opt-in â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestRustScorerIntegration:
    def test_default_agent_prefers_rust_if_available(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        # When backfire_kernel is installed, agent uses RustCoherenceScorer
        # directly; otherwise it falls back to CoherenceScorer.
        assert agent.scorer.__class__.__name__ in (
            "RustCoherenceScorer",
            "CoherenceScorer",
        )

    def test_rust_registry_probe(self):
        from director_ai.core.backends import list_backends

        backends = list_backends()
        assert isinstance(backends, dict)
