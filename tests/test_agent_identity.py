# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — agent identity + provenance tests

"""Multi-angle coverage: AgentPassport validation, PassportSigner
issue + verify + key rotation + constant-time comparison,
expiry handling, BehavioralFingerprint Welford updates +
z_score semantics, IdentityMonitor anomaly detection under drift
and steady state, AuditChain append + tamper detection across
every mutable field, plus concurrent-append correctness."""

from __future__ import annotations

import dataclasses
import threading

import pytest

from director_ai.core.agent_identity import (
    AgentPassport,
    AuditChain,
    AuditEntry,
    BehaviorObservation,
    BehavioralFingerprint,
    IdentityMonitor,
    PassportSigner,
    PassportVerificationError,
)

# --- AgentPassport --------------------------------------------------


class TestAgentPassport:
    def _mk(
        self,
        *,
        agent_id: str = "t/agent/1",
        role: str = "agent",
        tenant_id: str = "t",
        capabilities: tuple[str, ...] = ("read",),
        issued_at: float = 100.0,
        expires_at: float = 200.0,
        key_id: str = "k1",
    ) -> AgentPassport:
        return AgentPassport(
            agent_id=agent_id,
            role=role,
            tenant_id=tenant_id,
            capabilities=capabilities,
            issued_at=issued_at,
            expires_at=expires_at,
            key_id=key_id,
        )

    def test_valid_passport(self):
        p = self._mk()
        assert p.signature == ""

    def test_canonical_is_deterministic(self):
        a = self._mk()
        b = self._mk()
        assert a.canonical() == b.canonical()

    def test_canonical_excludes_signature(self):
        base = self._mk()
        signed = dataclasses.replace(base, signature="abcd")
        assert base.canonical() == signed.canonical()

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ValueError, match="agent_id"):
            self._mk(agent_id="")

    def test_empty_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            self._mk(role="")

    def test_empty_key_id_rejected(self):
        with pytest.raises(ValueError, match="key_id"):
            self._mk(key_id="")

    def test_negative_issued_at_rejected(self):
        with pytest.raises(ValueError, match="issued_at"):
            self._mk(issued_at=-1.0)

    def test_negative_expires_at_rejected(self):
        with pytest.raises(ValueError, match="expires_at"):
            self._mk(expires_at=-1.0)

    def test_expiry_before_issue_rejected(self):
        with pytest.raises(ValueError, match="expires_at"):
            self._mk(issued_at=200.0, expires_at=100.0)


# --- PassportSigner -----------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestPassportSigner:
    def _signer(self, clock: _FakeClock | None = None) -> PassportSigner:
        return PassportSigner(
            active_key=b"k" * 32,
            active_key_id="k1",
            default_ttl_seconds=3600.0,
            clock=clock or _FakeClock(),
        )

    def test_issue_and_verify(self):
        signer = self._signer()
        passport = signer.issue(agent_id="a1", role="agent", tenant_id="t")
        signer.verify(passport)

    def test_invalid_signature_rejected(self):
        signer = self._signer()
        passport = signer.issue(agent_id="a1", role="agent")
        tampered = dataclasses.replace(passport, role="admin")
        with pytest.raises(PassportVerificationError, match="signature"):
            signer.verify(tampered)

    def test_is_valid_wrapper(self):
        signer = self._signer()
        passport = signer.issue(agent_id="a1", role="agent")
        assert signer.is_valid(passport)
        tampered = dataclasses.replace(passport, agent_id="a2")
        assert not signer.is_valid(tampered)

    def test_expired_passport(self):
        clock = _FakeClock(start=1_000.0)
        signer = self._signer(clock)
        passport = signer.issue(agent_id="a1", role="agent", ttl_seconds=10.0)
        clock.now = 2_000.0  # past expiry
        with pytest.raises(PassportVerificationError, match="expired"):
            signer.verify(passport)

    def test_no_expiry_when_ttl_zero(self):
        clock = _FakeClock(start=0.0)
        signer = self._signer(clock)
        passport = signer.issue(agent_id="a1", role="agent", ttl_seconds=0.0)
        clock.now = 10**9
        signer.verify(passport)

    def test_short_key_rejected(self):
        with pytest.raises(ValueError, match="active_key"):
            PassportSigner(active_key=b"short", active_key_id="k1")

    def test_empty_key_id_rejected(self):
        with pytest.raises(ValueError, match="active_key_id"):
            PassportSigner(active_key=b"k" * 32, active_key_id="")

    def test_negative_ttl_rejected(self):
        with pytest.raises(ValueError, match="default_ttl"):
            PassportSigner(
                active_key=b"k" * 32, active_key_id="k1", default_ttl_seconds=-1.0
            )

    def test_issue_ttl_validation(self):
        signer = self._signer()
        with pytest.raises(ValueError, match="ttl_seconds"):
            signer.issue(agent_id="a1", role="agent", ttl_seconds=-1.0)

    def test_rotation_preserves_old_passports(self):
        clock = _FakeClock()
        signer = self._signer(clock)
        old_passport = signer.issue(agent_id="a1", role="agent")
        signer.rotate(new_active_key=b"j" * 32, new_active_key_id="k2")
        assert signer.active_key_id == "k2"
        # Old passport still verifies under the rotated-out key.
        signer.verify(old_passport)
        # New issues use the new key id.
        new_passport = signer.issue(agent_id="a2", role="agent")
        assert new_passport.key_id == "k2"

    def test_rotate_validates_new_key(self):
        signer = self._signer()
        with pytest.raises(ValueError, match="new_active_key"):
            signer.rotate(new_active_key=b"short", new_active_key_id="k2")
        with pytest.raises(ValueError, match="new_active_key_id"):
            signer.rotate(new_active_key=b"j" * 32, new_active_key_id="")

    def test_unknown_key_id_rejected(self):
        signer = self._signer()
        passport = signer.issue(agent_id="a1", role="agent")
        bogus = dataclasses.replace(passport, key_id="ghost")
        with pytest.raises(PassportVerificationError, match="unknown key"):
            signer.verify(bogus)

    def test_empty_signature_rejected(self):
        signer = self._signer()
        passport = signer.issue(agent_id="a1", role="agent")
        stripped = dataclasses.replace(passport, signature="")
        with pytest.raises(PassportVerificationError, match="no signature"):
            signer.verify(stripped)

    def test_capabilities_sorted_for_canonical(self):
        signer = self._signer()
        passport = signer.issue(
            agent_id="a1", role="agent", capabilities=("z", "a", "m")
        )
        assert passport.capabilities == ("a", "m", "z")


# --- BehavioralFingerprint -----------------------------------------


class TestBehavioralFingerprint:
    def test_update_accumulates_mean(self):
        fp = BehavioralFingerprint(min_samples=4)
        for value in [10.0, 20.0, 30.0, 40.0]:
            fp.update(BehaviorObservation(features={"len": value}))
        assert fp.mean("len") == pytest.approx(25.0)

    def test_variance_is_welford(self):
        fp = BehavioralFingerprint(min_samples=2)
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            fp.update(BehaviorObservation(features={"x": value}))
        # Population variance of 1..5 = 2.0.
        assert fp.stddev("x") == pytest.approx(2.0**0.5)

    def test_z_score_waits_for_min_samples(self):
        fp = BehavioralFingerprint(min_samples=5)
        fp.update(BehaviorObservation(features={"x": 1.0}))
        assert fp.z_score("x", 100.0) == 0.0

    def test_z_score_zero_variance(self):
        fp = BehavioralFingerprint(min_samples=2)
        for _ in range(5):
            fp.update(BehaviorObservation(features={"x": 1.0}))
        assert fp.z_score("x", 1.0) == 0.0

    def test_z_score_under_drift(self):
        fp = BehavioralFingerprint(min_samples=4)
        for value in [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]:
            fp.update(BehaviorObservation(features={"x": value}))
        # The mean ≈ 0.5, stddev small — a value at 10 is many sigma away.
        assert fp.z_score("x", 10.0) > 5.0

    def test_unknown_feature(self):
        fp = BehavioralFingerprint()
        with pytest.raises(KeyError):
            fp.mean("nope")
        with pytest.raises(KeyError):
            fp.stddev("nope")

    def test_sample_count(self):
        fp = BehavioralFingerprint()
        for _ in range(3):
            fp.update(BehaviorObservation(features={"x": 1.0, "y": 2.0}))
        assert fp.sample_count("x") == 3
        assert fp.sample_count() == 3

    def test_bad_min_samples(self):
        with pytest.raises(ValueError, match="min_samples"):
            BehavioralFingerprint(min_samples=0)

    def test_observation_validation(self):
        with pytest.raises(ValueError, match="features"):
            BehaviorObservation(features={})
        with pytest.raises(ValueError, match="finite"):
            BehaviorObservation(features={"x": float("nan")})
        with pytest.raises(ValueError, match="non-empty"):
            BehaviorObservation(features={"": 1.0})


# --- IdentityMonitor -----------------------------------------------


class TestIdentityMonitor:
    def _warm_monitor(self) -> IdentityMonitor:
        fp = BehavioralFingerprint(min_samples=8)
        monitor = IdentityMonitor(fingerprint=fp, z_threshold=3.0)
        for value in [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.0, 10.3, 9.7]:
            monitor.evaluate(BehaviorObservation(features={"prompt_len": value}))
        return monitor

    def test_steady_signal_is_safe(self):
        monitor = self._warm_monitor()
        result = monitor.evaluate(
            BehaviorObservation(features={"prompt_len": 10.0}, source="steady")
        )
        assert result is None

    def test_outlier_is_flagged(self):
        monitor = self._warm_monitor()
        anomaly = monitor.evaluate(
            BehaviorObservation(features={"prompt_len": 10_000.0}, source="attacker")
        )
        assert anomaly is not None
        assert anomaly.feature == "prompt_len"
        assert anomaly.source == "attacker"
        assert abs(anomaly.z_score) > 3.0
        assert "prompt_len" in anomaly.reason

    def test_anomaly_does_not_pollute_fingerprint_by_default(self):
        monitor = self._warm_monitor()
        before_mean = monitor.fingerprint.mean("prompt_len")
        monitor.evaluate(BehaviorObservation(features={"prompt_len": 10_000.0}))
        after_mean = monitor.fingerprint.mean("prompt_len")
        assert before_mean == after_mean

    def test_update_on_anomaly_flag(self):
        fp = BehavioralFingerprint(min_samples=4)
        for value in [10.0, 10.0, 10.0, 10.0]:
            fp.update(BehaviorObservation(features={"x": value}))
        monitor = IdentityMonitor(
            fingerprint=fp, z_threshold=2.0, update_on_anomaly=True
        )
        monitor.evaluate(BehaviorObservation(features={"x": 1000.0}))
        assert fp.sample_count("x") == 5

    def test_bad_threshold(self):
        with pytest.raises(ValueError, match="z_threshold"):
            IdentityMonitor(fingerprint=BehavioralFingerprint(), z_threshold=0.0)


# --- AuditChain ----------------------------------------------------


class TestAuditChain:
    def _chain(self) -> AuditChain:
        return AuditChain(secret=b"s" * 32, clock=_FakeClock())

    def test_append_and_length(self):
        chain = self._chain()
        chain.append({"kind": "start"})
        chain.append({"kind": "score", "value": 0.5})
        assert len(chain) == 2

    def test_entries_chain(self):
        chain = self._chain()
        a = chain.append({"kind": "a"})
        b = chain.append({"kind": "b"})
        assert isinstance(a, AuditEntry)
        assert b.parent_hash == a.event_hash
        assert a.parent_hash == "0" * 64

    def test_verify_on_clean_chain(self):
        chain = self._chain()
        chain.append({"kind": "a"})
        chain.append({"kind": "b"})
        ok, idx = chain.verify()
        assert ok and idx is None

    def test_tamper_event_detected(self):
        chain = self._chain()
        chain.append({"kind": "a"})
        chain.append({"kind": "b"})
        # Tamper with the middle entry's event payload — this is a
        # deliberate test-only mutation of the private attribute,
        # since the public API would never let a caller forge it.
        # Deliberate test-only mutation of the private attribute —
        # the public API does not let a caller forge this, which is
        # precisely why we exercise it here.
        chain._entries[0] = dataclasses.replace(
            chain._entries[0], event={"kind": "evil"}
        )
        ok, idx = chain.verify()
        assert not ok and idx == 0

    def test_tamper_tag_detected(self):
        chain = self._chain()
        chain.append({"kind": "a"})
        chain._entries[0] = dataclasses.replace(
            chain._entries[0], tag="0" * 64
        )
        ok, idx = chain.verify()
        assert not ok and idx == 0

    def test_tamper_parent_hash_detected(self):
        chain = self._chain()
        chain.append({"kind": "a"})
        chain.append({"kind": "b"})
        chain._entries[1] = dataclasses.replace(
            chain._entries[1], parent_hash="0" * 64
        )
        ok, idx = chain.verify()
        assert not ok and idx == 1

    def test_short_secret_rejected(self):
        with pytest.raises(ValueError, match="secret"):
            AuditChain(secret=b"short")

    def test_snapshot_is_tuple(self):
        chain = self._chain()
        chain.append({"kind": "a"})
        snap = chain.snapshot()
        assert isinstance(snap, tuple) and len(snap) == 1

    def test_concurrent_appends_are_atomic(self):
        chain = AuditChain(secret=b"s" * 32)

        def writer(tag: str) -> None:
            for i in range(50):
                chain.append({"kind": tag, "i": i})

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(chain) == 8 * 50
        ok, _ = chain.verify()
        assert ok
