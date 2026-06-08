# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Federated-DP evidence packet

"""Formal privacy and poisoning-resilience evidence for federated calibration.

:class:`~director_ai.core.federated_dp.calibration.FederatedCalibrationRound`
aggregates clipped per-tenant updates under Gaussian noise. This module turns the
round's parameters into the two pieces of evidence a regulated deployment needs
before trusting cross-tenant calibration:

* **Formal ``(ε, δ)`` bound** — the per-round Gaussian mechanism composed over the
  applied rounds with the tight Rényi-DP accountant. The clipped mean has L2
  sensitivity ``2·C / n`` to one tenant (``C`` the clip norm, ``n`` the cohort),
  and the noise added to the mean is ``N(0, (m·C / n)²)`` for noise multiplier
  ``m``, so the effective noise multiplier is ``z = m / 2`` (cohort-independent).
  Composing ``R`` rounds at ``z`` and converting at ``δ`` gives the formal bound.

* **Poisoning bound** — clipping is what makes the aggregate poisoning-resilient.
  A coalition of ``f`` malicious tenants out of ``n`` can move the clipped mean by
  at most ``2·f·C / n`` per round (each can swing its clipped contribution across
  the full ``[-C, C]`` range), so the certified worst-case parameter shift the
  coalition can induce is ``lr · 2·f·C / n`` per round and ``R`` times that over
  ``R`` rounds. Without clipping a single tenant could move it without bound; the
  bound is the resilience guarantee.

:meth:`FederatedDPEvidence.simulate_poisoning` runs an attacked round trajectory
against an all-honest baseline (shared noise seed, so the noise cancels and the
residual is purely the poisoning effect) and checks the observed shift stays
within the certified bound — empirical evidence to accompany the certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

from director_ai.core.federated_privacy.rdp_accountant import (
    DPGuarantee,
    RenyiAccountant,
)

from .calibration import FederatedCalibrationRound


@dataclass(frozen=True)
class PoisoningBound:
    """The certified worst-case parameter shift a malicious coalition can induce."""

    num_malicious: int
    cohort_size: int
    clip_norm: float
    learning_rate: float
    rounds: int
    per_round_shift: float
    total_shift: float

    @property
    def fraction_malicious(self) -> float:
        """Fraction of the cohort controlled by the coalition."""
        return self.num_malicious / self.cohort_size

    def to_dict(self) -> dict[str, float | int]:
        """Tenant-safe view of the certified bound."""
        return {
            "num_malicious": self.num_malicious,
            "cohort_size": self.cohort_size,
            "fraction_malicious": self.fraction_malicious,
            "clip_norm": self.clip_norm,
            "learning_rate": self.learning_rate,
            "rounds": self.rounds,
            "per_round_shift": self.per_round_shift,
            "total_shift": self.total_shift,
        }


@dataclass(frozen=True)
class PoisoningSimulation:
    """The observed vs certified poisoning shift from a simulated attack."""

    bound: PoisoningBound
    observed_shift: float
    baseline_value: float
    attacked_value: float

    @property
    def within_bound(self) -> bool:
        """Whether the observed shift stayed within the certified bound."""
        return abs(self.observed_shift) <= self.bound.total_shift + 1e-12

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe view of the simulation outcome."""
        return {
            "bound": self.bound.to_dict(),
            "observed_shift": self.observed_shift,
            "baseline_value": self.baseline_value,
            "attacked_value": self.attacked_value,
            "within_bound": self.within_bound,
        }


@dataclass(frozen=True)
class FederatedDPEvidencePacket:
    """Combined formal-privacy and poisoning-resilience evidence for a round."""

    rounds: int
    noise_multiplier: float
    effective_noise_multiplier: float
    epsilon: float
    delta: float
    rdp_order: float
    poisoning: PoisoningBound

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe view of the whole evidence packet."""
        return {
            "rounds": self.rounds,
            "noise_multiplier": self.noise_multiplier,
            "effective_noise_multiplier": self.effective_noise_multiplier,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "rdp_order": self.rdp_order,
            "poisoning": self.poisoning.to_dict(),
        }


class FederatedDPEvidence:
    """Produce formal-privacy and poisoning-resilience evidence for a round.

    Parameters
    ----------
    calibration_round:
        The :class:`FederatedCalibrationRound` whose parameters (clip norm, noise
        multiplier, learning rate) drive the evidence.
    """

    def __init__(self, calibration_round: FederatedCalibrationRound) -> None:
        self._round = calibration_round

    @property
    def effective_noise_multiplier(self) -> float:
        """The cohort-independent Gaussian noise multiplier ``z = m / 2``."""
        return self._round.noise_multiplier / 2.0

    def _resolved_rounds(self, rounds: int | None) -> int:
        resolved = self._round.rounds_applied if rounds is None else rounds
        if resolved < 0:
            raise ValueError("rounds must be non-negative")
        return resolved

    def epsilon_bound(self, *, delta: float, rounds: int | None = None) -> DPGuarantee:
        """Formal ``(ε, δ)``-DP bound for the applied (or requested) rounds.

        Composes the per-round Gaussian mechanism at the effective noise
        multiplier with the Rényi-DP accountant and converts at ``δ``. ``rounds``
        defaults to the round's :attr:`rounds_applied`.
        """
        steps = self._resolved_rounds(rounds)
        accountant = RenyiAccountant()
        accountant.compose_gaussian(
            noise_multiplier=self.effective_noise_multiplier, steps=steps
        )
        return accountant.epsilon(delta=delta)

    def poisoning_bound(
        self,
        *,
        num_malicious: int,
        cohort_size: int,
        rounds: int | None = None,
    ) -> PoisoningBound:
        """Certified worst-case parameter shift for ``num_malicious`` attackers."""
        if cohort_size < 1:
            raise ValueError("cohort_size must be >= 1")
        if num_malicious < 0:
            raise ValueError("num_malicious must be non-negative")
        if num_malicious > cohort_size:
            raise ValueError("num_malicious cannot exceed cohort_size")
        steps = self._resolved_rounds(rounds)
        clip_norm = self._round.clip_norm
        learning_rate = self._round.learning_rate
        per_round_shift = learning_rate * 2.0 * num_malicious * clip_norm / cohort_size
        return PoisoningBound(
            num_malicious=num_malicious,
            cohort_size=cohort_size,
            clip_norm=clip_norm,
            learning_rate=learning_rate,
            rounds=steps,
            per_round_shift=per_round_shift,
            total_shift=steps * per_round_shift,
        )

    def evidence_packet(
        self,
        *,
        delta: float,
        num_malicious: int,
        cohort_size: int,
        rounds: int | None = None,
    ) -> FederatedDPEvidencePacket:
        """Combine the formal ``(ε, δ)`` bound and the poisoning bound."""
        steps = self._resolved_rounds(rounds)
        guarantee = self.epsilon_bound(delta=delta, rounds=steps)
        poisoning = self.poisoning_bound(
            num_malicious=num_malicious, cohort_size=cohort_size, rounds=steps
        )
        return FederatedDPEvidencePacket(
            rounds=steps,
            noise_multiplier=self._round.noise_multiplier,
            effective_noise_multiplier=self.effective_noise_multiplier,
            epsilon=guarantee.epsilon,
            delta=guarantee.delta,
            rdp_order=guarantee.order,
            poisoning=poisoning,
        )

    def simulate_poisoning(
        self,
        *,
        num_malicious: int,
        cohort_size: int,
        honest_update: float,
        rounds: int,
        attacker_update: float | None = None,
        seed: int = 0,
    ) -> PoisoningSimulation:
        """Run an attacked trajectory against an all-honest baseline.

        ``num_malicious`` attackers submit ``attacker_update`` (default ``+clip_norm``,
        the maximally adversarial value); the remaining tenants and the whole
        baseline cohort submit ``honest_update``. Both trajectories share the noise
        seed, so the Gaussian noise cancels in the difference and the observed
        shift is purely the poisoning effect. The result is checked against the
        certified bound.
        """
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        bound = self.poisoning_bound(
            num_malicious=num_malicious, cohort_size=cohort_size, rounds=rounds
        )
        clip_norm = self._round.clip_norm
        noise_multiplier = self._round.noise_multiplier
        learning_rate = self._round.learning_rate
        attack_value = clip_norm if attacker_update is None else attacker_update
        start = self._round.value

        def _run(malicious: int) -> float:
            arena = FederatedCalibrationRound(
                start,
                clip_norm=clip_norm,
                noise_multiplier=noise_multiplier,
                min_cohort=1,
                learning_rate=learning_rate,
                value_bounds=(-1e18, 1e18),
                seed=seed,
            )
            for _ in range(rounds):
                for i in range(cohort_size):
                    is_attacker = i < malicious
                    arena.submit_update(
                        tenant_id=f"t{i}",
                        update=attack_value if is_attacker else honest_update,
                    )
                arena.aggregate()
            return arena.value

        baseline_value = _run(0)
        attacked_value = _run(num_malicious)
        return PoisoningSimulation(
            bound=bound,
            observed_shift=attacked_value - baseline_value,
            baseline_value=baseline_value,
            attacked_value=attacked_value,
        )
