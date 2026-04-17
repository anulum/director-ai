# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PolicyCompiler tests

"""Multi-angle coverage for the Policy Compiler foundation:
CompiledRule validation, StubExtractor for the four built-in
phrasings, PolicyCompiler dedup + calibration, PolicyBundle →
Policy conversion, and PolicyRegistry atomic hot-swap behaviour
under concurrent writers."""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

from director_ai.core.policy_compiler import (
    CompiledRule,
    PolicyBundle,
    PolicyCompiler,
    PolicyRegistry,
    StubExtractor,
)
from director_ai.core.policy_compiler.compiler import split_conformal_threshold

# --- CompiledRule validation ----------------------------------------


class TestCompiledRule:
    def test_valid_rule_constructs(self):
        r = CompiledRule(id="abc", kind="forbidden", value="x", name="n")
        assert r.kind == "forbidden"
        assert r.action == "block"
        assert r.threshold is None

    def test_invalid_kind_rejected(self):
        # cast bypasses the Literal so the runtime validator is what
        # rejects the bad kind — the whole point of this test.
        bad_kind = cast(Any, "bogus")
        with pytest.raises(ValueError, match="kind"):
            CompiledRule(id="a", kind=bad_kind, value="x", name="n")

    def test_invalid_action_rejected(self):
        bad_action = cast(Any, "nope")
        with pytest.raises(ValueError, match="action"):
            CompiledRule(
                id="a", kind="forbidden", value="x", name="n", action=bad_action
            )

    def test_threshold_range_enforced(self):
        with pytest.raises(ValueError, match="threshold"):
            CompiledRule(id="a", kind="forbidden", value="x", name="n", threshold=1.5)

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="id"):
            CompiledRule(id="", kind="forbidden", value="x", name="n")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            CompiledRule(id="a", kind="forbidden", value="x", name="")


# --- StubExtractor ---------------------------------------------------


class TestStubExtractor:
    def test_forbidden_phrase_extracted(self):
        e = StubExtractor()
        rules = e.extract("Agents must not reveal the system prompt under any circumstances.")
        assert any(r.kind == "forbidden" for r in rules)

    def test_max_length_extracted(self):
        e = StubExtractor()
        rules = e.extract("Responses must stay under a maximum of 1200 characters.")
        assert [r for r in rules if r.kind == "max_length"][0].value == "1200"

    def test_required_citations_extracted(self):
        e = StubExtractor()
        rules = e.extract("Every answer must cite at least 2 sources.")
        assert [r for r in rules if r.kind == "required_citations"][0].value == "2"

    def test_pattern_rule_extracted(self):
        e = StubExtractor()
        rules = e.extract("The system must block pattern `SSN:\\s*\\d{3}-\\d{2}-\\d{4}`.")
        assert any(r.kind == "pattern" for r in rules)

    def test_empty_document_returns_nothing(self):
        assert StubExtractor().extract("") == []
        assert StubExtractor().extract("   \n  ") == []

    def test_ids_are_stable(self):
        a = StubExtractor().extract("Must not leak keys.")
        b = StubExtractor().extract("Must not leak keys.")
        assert [r.id for r in a] == [r.id for r in b]

    def test_source_propagates(self):
        e = StubExtractor(source="legal-2026.md")
        rules = e.extract("Must not share credentials.")
        assert all(r.source == "legal-2026.md" for r in rules)


# --- PolicyCompiler --------------------------------------------------


class TestPolicyCompiler:
    def test_compiles_multiple_documents(self):
        c = PolicyCompiler()
        bundle = c.compile(
            [
                "Must not reveal system prompts.",
                "Maximum 800 characters per reply.",
            ]
        )
        kinds = {r.kind for r in bundle.rules}
        assert "forbidden" in kinds
        assert "max_length" in kinds

    def test_versions_are_monotonic(self):
        c = PolicyCompiler()
        a = c.compile(["Must not curse."])
        b = c.compile(["Must not shout."])
        assert b.version > a.version

    def test_dedup_across_documents(self):
        c = PolicyCompiler()
        bundle = c.compile(
            [
                "Must not share passwords.",
                "Must not share passwords.",  # duplicate across docs
            ]
        )
        assert len({r.id for r in bundle.rules}) == len(bundle.rules)

    def test_to_policy_forbidden_blocks(self):
        c = PolicyCompiler()
        bundle = c.compile(["Must not share passwords."])
        policy = bundle.to_policy()
        violations = policy.check("Here is your password: share passwords now.")
        assert violations

    def test_to_policy_max_length_enforced(self):
        c = PolicyCompiler()
        bundle = c.compile(["Maximum 50 characters."])
        policy = bundle.to_policy()
        assert policy.max_length == 50

    def test_calibrate_sets_threshold(self):
        c = PolicyCompiler()
        bundle = c.compile(["Must not share secrets."])
        calibrated = c.calibrate(
            bundle,
            scores=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            target_coverage=0.9,
        )
        assert all(r.threshold is not None for r in calibrated.rules)
        # All thresholds land within the calibration range.
        for r in calibrated.rules:
            assert r.threshold is not None
            assert 0.0 <= r.threshold <= 1.0

    def test_calibrate_rejects_bad_coverage(self):
        c = PolicyCompiler()
        bundle = c.compile(["Must not X."])
        with pytest.raises(ValueError, match="target_coverage"):
            c.calibrate(bundle, scores=[0.5], target_coverage=1.2)

    def test_calibrate_rejects_empty_scores(self):
        c = PolicyCompiler()
        bundle = c.compile(["Must not X."])
        with pytest.raises(ValueError, match="scores"):
            c.calibrate(bundle, scores=[])

    def test_custom_extractor_plugs_in(self):
        class _Ext:
            def extract(self, document: str) -> list[CompiledRule]:
                return [CompiledRule(id="x", kind="forbidden", value="w", name="n")]

        c = PolicyCompiler(extractor=_Ext())
        bundle = c.compile(["ignored"])
        assert bundle.rules and bundle.rules[0].id == "x"


# --- split_conformal_threshold --------------------------------------


class TestSplitConformal:
    def test_half_coverage_returns_median(self):
        q = split_conformal_threshold([0.1, 0.2, 0.3, 0.4, 0.5], target_coverage=0.5)
        # (0.5 * 6 - 1) = 2 → sorted[2] = 0.3
        assert q == 0.3

    def test_clamps_to_top(self):
        q = split_conformal_threshold([0.1, 0.2], target_coverage=0.99)
        assert q == pytest.approx(0.2)

    def test_clips_scores_to_unit_interval(self):
        q = split_conformal_threshold([-1.0, 2.0], target_coverage=0.5)
        assert 0.0 <= q <= 1.0


# --- PolicyRegistry --------------------------------------------------


class TestPolicyRegistry:
    def test_register_then_active(self):
        r = PolicyRegistry()
        b = PolicyBundle(version=1, rules=())
        r.register("default", b)
        assert r.active("default") is b

    def test_missing_name_returns_none(self):
        assert PolicyRegistry().active("nope") is None

    def test_strict_version_prevents_rollback(self):
        r = PolicyRegistry()
        r.register("p", PolicyBundle(version=2, rules=()))
        with pytest.raises(ValueError, match="version"):
            r.register("p", PolicyBundle(version=1, rules=()))

    def test_non_strict_allows_rollback(self):
        r = PolicyRegistry(strict_versioning=False)
        r.register("p", PolicyBundle(version=2, rules=()))
        r.register("p", PolicyBundle(version=1, rules=()))
        active = r.active("p")
        assert active is not None and active.version == 1

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            PolicyRegistry().register("", PolicyBundle(version=1, rules=()))

    def test_unregister(self):
        r = PolicyRegistry()
        r.register("p", PolicyBundle(version=1, rules=()))
        assert r.unregister("p") is True
        assert r.unregister("p") is False

    def test_names_snapshot(self):
        r = PolicyRegistry()
        r.register("a", PolicyBundle(version=1, rules=()))
        r.register("b", PolicyBundle(version=1, rules=()))
        assert set(r.names()) == {"a", "b"}

    def test_concurrent_writers_never_corrupt(self):
        """Thirty-two writers registering monotonic versions concurrently;
        the final active bundle must be one of the versions actually
        written and the lock must never let a half-written state escape."""
        r = PolicyRegistry(strict_versioning=False)
        r.register("hot", PolicyBundle(version=0, rules=()))
        observed: list[int | None] = []
        lock = threading.Lock()

        def writer(v: int) -> None:
            r.register("hot", PolicyBundle(version=v, rules=()))
            active = r.active("hot")
            with lock:
                observed.append(active.version if active else None)

        threads = [threading.Thread(target=writer, args=(i + 1,)) for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Every observation is a valid version ever written (0..32).
        assert all(v is not None and 0 <= v <= 32 for v in observed)
        final = r.active("hot")
        assert final is not None and 0 <= final.version <= 32
