# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — E2E feature-matrix invariants

"""End-to-end invariants for the three opt-in safety hooks.

For every hook, assert three independent facts:

* **default-off preserves**: an agent constructed without the hook
  produces the same output shape as a fresh ``CoherenceAgent()``
  — no silent engagement.
* **on-changes**: the hook, once attached, visibly changes
  behaviour on the specific branch it is meant to guard.
* **failure-surfaces**: an attached hook that rejects the call
  surfaces the rejection through the public API (halted
  ReviewResult for containment, RuntimeError for grounding /
  passport when the module is not attached, typed verdict
  returned when attached and unhappy).

Plus a handful of cross-hook invariants: composition does not
interfere, the orchestrator itself round-trips its schema, and a
synthetic regression is detected by the default rule set.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.containment import (
    BreakoutDetector,
    BreakoutFinding,
    ContainmentAttestor,
    ContainmentGuard,
    RealityAnchor,
)
from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    JointChain,
    PhysicalAction,
    SimpleKinematicModel,
    Vec3,
    WorkspaceConstraint,
)
from director_ai.core.zk_attestation import (
    CommitmentBackend,
    CrossOrgPassport,
    MinimumCoherence,
    PassportIssuer,
    PassportVerifier,
)

_HMAC_KEY = b"e2e-feature-matrix-test-key-32by"
_CANONICAL_PROMPT = "Paris is the capital of France."


# ─── helpers ──────────────────────────────────────────────────────


def _fresh_agent(**kwargs) -> CoherenceAgent:
    return CoherenceAgent(**kwargs)


def _make_guard(
    detector: BreakoutDetector | None = None,
) -> tuple[ContainmentGuard, RealityAnchor]:
    attestor = ContainmentAttestor(key=_HMAC_KEY, issuer="host://e2e")
    guard = ContainmentGuard(
        attestor=attestor,
        detector=detector or BreakoutDetector(),
    )
    anchor = attestor.mint(session_id=f"sess-{uuid.uuid4().hex[:8]}", scope="sandbox")
    return guard, anchor


def _make_hook(workspace_bounds: float = 5.0) -> GroundingHook:
    chain = JointChain(base=Vec3(0, 0, 0), link_lengths=(1.0, 1.0))
    model = SimpleKinematicModel(chain=chain)
    envelope = AABB(
        min_corner=Vec3(-workspace_bounds, -workspace_bounds, -workspace_bounds),
        max_corner=Vec3(workspace_bounds, workspace_bounds, workspace_bounds),
    )
    return GroundingHook(
        model=model,
        constraints=(WorkspaceConstraint(name="cell", envelope=envelope),),
    )


def _make_passport_pair() -> tuple[PassportIssuer, PassportVerifier]:
    issuer = PassportIssuer(key=_HMAC_KEY, issuing_org="org://source")
    verifier = PassportVerifier(
        issuer_keys={"org://source": _HMAC_KEY},
        backends={"commitment": CommitmentBackend(key=_HMAC_KEY)},
    )
    return issuer, verifier


# ─── containment feature matrix ──────────────────────────────────


class TestContainmentFeatureMatrix:
    def test_default_off_preserves_behaviour(self):
        """A fresh agent with no containment hook produces an
        output that does NOT start with the block marker."""
        agent = _fresh_agent()
        result = agent.process(_CANONICAL_PROMPT)
        assert not result.output.startswith("[CONTAINMENT-BLOCK]")
        assert result.halted is False
        assert agent.containment_guard is None
        assert agent.containment_anchor is None

    def test_on_changes_behaviour_when_blocking(self):
        """An attached always-block detector converts the output
        into a halted ReviewResult carrying the finding."""

        class _AlwaysBlock(BreakoutDetector):
            def scan(self, event, anchored_scope, claimed_scope=None):
                del event, anchored_scope, claimed_scope
                return [
                    BreakoutFinding(
                        category="policy",
                        severity="high",
                        detail="e2e",
                    ),
                ]

        guard, anchor = _make_guard(detector=_AlwaysBlock())
        agent = _fresh_agent(containment_guard=guard, containment_anchor=anchor)
        result = agent.process(_CANONICAL_PROMPT)
        assert result.output.startswith("[CONTAINMENT-BLOCK]")
        assert result.halted is True

    def test_failure_surfaces_through_halt_evidence(self):
        """The guard's block reason must appear in
        ``halt_evidence.suggested_action`` — otherwise a caller
        has no audit trail for why the output was suppressed."""
        guard, anchor = _make_guard()
        # Tamper the MAC so the attestor rejects the anchor.
        tampered = RealityAnchor(
            session_id=anchor.session_id,
            scope=anchor.scope,
            issuer=anchor.issuer,
            created_at=anchor.created_at,
            nonce=anchor.nonce,
            mac="0" * 64,
        )
        agent = _fresh_agent(containment_guard=guard, containment_anchor=tampered)
        result = agent.process(_CANONICAL_PROMPT)
        assert result.halted is True
        assert result.halt_evidence is not None
        assert result.halt_evidence.reason == "containment_block"
        assert "mac_mismatch" in result.halt_evidence.suggested_action

    def test_configuration_mismatch_rejected_at_construction(self):
        guard, _ = _make_guard()
        with pytest.raises(ValueError, match="together"):
            _fresh_agent(containment_guard=guard, containment_anchor=None)
        with pytest.raises(ValueError, match="together"):
            _fresh_agent(containment_guard=None, containment_anchor=None)  # noop, ok
            _fresh_agent(containment_guard=None, containment_anchor=_make_guard()[1])


# ─── grounding feature matrix ─────────────────────────────────────


class TestGroundingFeatureMatrix:
    def test_default_off_raises_on_verify(self):
        """No hook attached → verify_physical_action raises
        RuntimeError. Silent no-op would be dangerous."""
        agent = _fresh_agent()
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1.0, 0.0, 0.0),
            velocity_magnitude=0.1,
            torque_magnitude=0.5,
        )
        with pytest.raises(RuntimeError, match="grounding_hook"):
            agent.verify_physical_action(action)

    def test_on_allows_in_workspace_action(self):
        agent = _fresh_agent(grounding_hook=_make_hook())
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1.0, 0.0, 0.0),
            velocity_magnitude=0.1,
            torque_magnitude=0.5,
        )
        verdict = agent.verify_physical_action(action)
        assert verdict.allowed is True
        assert verdict.violations == ()

    def test_failure_surfaces_violation_details(self):
        agent = _fresh_agent(grounding_hook=_make_hook(workspace_bounds=2.0))
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1000.0, 0.0, 0.0),
            velocity_magnitude=0.1,
            torque_magnitude=0.5,
        )
        verdict = agent.verify_physical_action(action)
        assert verdict.allowed is False
        assert any(v.constraint == "cell" for v in verdict.violations)


# ─── passport feature matrix ──────────────────────────────────────


class TestPassportFeatureMatrix:
    def _samples(self) -> list[dict[str, object]]:
        return [
            {"coherence": 0.95, "halted": False, "breakout": False} for _ in range(32)
        ]

    def test_default_off_raises_on_verify(self):
        agent = _fresh_agent()
        issuer, _ = _make_passport_pair()
        passport = issuer.issue(
            agent_id="a",
            samples=self._samples(),
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        with pytest.raises(RuntimeError, match="passport_verifier"):
            agent.verify_passport(passport)

    def test_on_accepts_valid_passport(self):
        issuer, verifier = _make_passport_pair()
        agent = _fresh_agent(passport_verifier=verifier)
        passport = issuer.issue(
            agent_id="a",
            samples=self._samples(),
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        assert agent.verify_passport(passport).accepted is True

    def test_failure_surfaces_on_tampered_mac(self):
        issuer, verifier = _make_passport_pair()
        agent = _fresh_agent(passport_verifier=verifier)
        passport = issuer.issue(
            agent_id="a",
            samples=self._samples(),
            statements=[MinimumCoherence(name="c", threshold=0.9, samples_min=1)],
        )
        tampered = CrossOrgPassport(
            agent_id=passport.agent_id,
            issuing_org=passport.issuing_org,
            created_at=passport.created_at,
            entries=passport.entries,
            mac="0" * 64,
        )
        verdict = agent.verify_passport(tampered)
        assert verdict.accepted is False
        assert verdict.signature_ok is False


# ─── cross-hook composition ──────────────────────────────────────


class TestHookComposition:
    def test_all_three_hooks_coexist(self):
        """Every hook attached simultaneously — no interference,
        each invariant still holds independently."""
        guard, anchor = _make_guard()
        hook = _make_hook()
        _, verifier = _make_passport_pair()
        agent = _fresh_agent(
            containment_guard=guard,
            containment_anchor=anchor,
            grounding_hook=hook,
            passport_verifier=verifier,
        )
        # Each accessor returns the exact object we passed in.
        assert agent.containment_guard is guard
        assert agent.containment_anchor is anchor
        assert agent.grounding_hook is hook
        assert agent.passport_verifier is verifier
        # Core pipeline still runs cleanly under full composition.
        result = agent.process(_CANONICAL_PROMPT)
        assert not result.output.startswith("[CONTAINMENT-BLOCK]")

    def test_hooks_do_not_leak_into_subsequent_construction(self):
        """Constructing an agent with hooks and then constructing
        another agent without hooks must not share state."""
        guard, anchor = _make_guard()
        first = _fresh_agent(containment_guard=guard, containment_anchor=anchor)
        assert first.containment_guard is guard
        second = _fresh_agent()
        assert second.containment_guard is None


# ─── orchestrator self-invariants ─────────────────────────────────


class TestOrchestratorInvariants:
    def test_schema_round_trip(self):
        """Every RunReport the orchestrator emits must parse back
        through :meth:`RunReport.from_json` without drift."""
        from benchmarks.orchestrator.environment import capture_environment
        from benchmarks.orchestrator.schema import (
            MetricResult,
            RunReport,
            SuiteEntry,
        )

        env = capture_environment(runner="ci")
        report = RunReport(
            run_id=str(uuid.uuid4()),
            timestamp_utc="2026-04-18T12:00:00Z",
            environment=env,
            entries=(
                SuiteEntry(
                    name="synthetic",
                    kind="smoke",
                    status="passed",
                    metrics=(MetricResult(name="value", value=1.0, unit="bool"),),
                    wall_clock_seconds=0.1,
                ),
            ),
        )
        round_tripped = RunReport.from_json(report.to_json())
        assert round_tripped.run_id == report.run_id
        assert round_tripped.timestamp_utc == report.timestamp_utc
        assert round_tripped.environment.git_commit == env.git_commit
        assert round_tripped.entries[0].name == "synthetic"
        assert round_tripped.entries[0].metrics[0].value == 1.0

    def test_regression_detects_synthetic_accuracy_drop(self, tmp_path):
        """A 5 pp accuracy drop must fire the
        ``balanced_accuracy`` rule on the AggreFact macro case."""
        from benchmarks.orchestrator.environment import capture_environment
        from benchmarks.orchestrator.regression import (
            RegressionRule,
            detect_regressions,
        )
        from benchmarks.orchestrator.schema import (
            MetricResult,
            RunReport,
            SuiteEntry,
        )

        env = capture_environment(runner="ci")

        def _report(value: float) -> RunReport:
            return RunReport(
                run_id=str(uuid.uuid4()),
                timestamp_utc="2026-04-18T12:00:00Z",
                environment=env,
                entries=(
                    SuiteEntry(
                        name="nli_tier5_aggrefact_macro",
                        kind="accuracy",
                        status="passed",
                        metrics=(
                            MetricResult(
                                name="balanced_accuracy",
                                value=value,
                                unit="ratio",
                            ),
                        ),
                        wall_clock_seconds=10.0,
                    ),
                ),
            )

        baseline = _report(0.758)
        current = _report(0.708)
        rules = (
            RegressionRule(
                case_name="nli_tier5_aggrefact_macro",
                metric="balanced_accuracy",
                absolute_tolerance=0.02,
                severity="high",
            ),
        )
        result = detect_regressions(current, baseline, rules)
        assert not result.clean
        assert len(result.findings) == 1
        finding = result.findings[0]
        assert finding.rule.metric == "balanced_accuracy"
        assert finding.severity == "high"
        assert finding.absolute_delta == pytest.approx(-0.05)

    def test_regression_stays_clean_on_self_comparison(self, tmp_path):
        """Comparing a report against itself never produces a
        finding — the orchestrator is self-consistent."""
        from benchmarks.orchestrator.environment import capture_environment
        from benchmarks.orchestrator.regression import (
            default_rules,
            detect_regressions,
        )
        from benchmarks.orchestrator.schema import (
            MetricResult,
            RunReport,
            SuiteEntry,
        )

        env = capture_environment(runner="ci")
        report = RunReport(
            run_id="a",
            timestamp_utc="2026-04-18T12:00:00Z",
            environment=env,
            entries=(
                SuiteEntry(
                    name="rust_parity_safety",
                    kind="smoke",
                    status="passed",
                    metrics=(
                        MetricResult(name="pass_count", value=80.0, unit="count"),
                    ),
                    wall_clock_seconds=5.0,
                ),
            ),
        )
        result = detect_regressions(report, report, default_rules())
        assert result.clean

    def test_schema_file_written_and_parseable(self, tmp_path):
        """``to_file`` / ``from_file`` round-trip on disk."""
        from benchmarks.orchestrator.environment import capture_environment
        from benchmarks.orchestrator.schema import RunReport, SuiteEntry

        env = capture_environment(runner="ci")
        report = RunReport(
            run_id="b",
            timestamp_utc="2026-04-18T12:00:00Z",
            environment=env,
            entries=(
                SuiteEntry(
                    name="synthetic",
                    kind="smoke",
                    status="passed",
                    metrics=(),
                    wall_clock_seconds=0.0,
                ),
            ),
        )
        target = Path(tmp_path) / "report.json"
        report.to_file(target)
        assert target.exists()
        parsed_raw = json.loads(target.read_text())
        assert parsed_raw["run_id"] == "b"
        round_tripped = RunReport.from_file(target)
        assert round_tripped.entries[0].name == "synthetic"
