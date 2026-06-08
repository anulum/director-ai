# Federated-DP Evidence

::: director_ai.core.federated_dp.evidence.FederatedDPEvidence

::: director_ai.core.federated_dp.evidence.FederatedDPEvidencePacket

::: director_ai.core.federated_dp.evidence.PoisoningBound

::: director_ai.core.federated_dp.evidence.PoisoningSimulation

## Boundary

[`FederatedCalibrationRound`](federated-dp-calibration.md) aggregates clipped
per-tenant updates under Gaussian noise. Before a regulated deployment trusts
that cross-tenant calibration, it needs two pieces of evidence — this module
produces both from the round's parameters.

### Formal `(ε, δ)` bound

The clipped mean has L2 sensitivity `2·C / n` to one tenant (`C` the clip norm,
`n` the cohort); the noise added to the mean is `N(0, (m·C / n)²)` for noise
multiplier `m`, so the **effective noise multiplier is `z = m / 2`**, independent
of cohort size. Composing `R` rounds at `z` with the
[Rényi-DP accountant](rdp-accountant.md) and converting at `δ` gives the formal
bound — the tight composition, not the loose basic/advanced `(ε, δ)` sum.

### Poisoning bound

Clipping is what makes the aggregate poisoning-resilient. A coalition of `f`
malicious tenants out of `n` can move the clipped mean by at most `2·f·C / n` per
round (each can swing its clipped contribution across the full `[-C, C]` range),
so the certified worst-case parameter shift is `lr · 2·f·C / n` per round and `R`
times that over `R` rounds. Without clipping a single tenant could move the
parameter without bound; the certificate is the resilience guarantee.

`simulate_poisoning` runs an attacked trajectory against an all-honest baseline
with a shared noise seed (so the noise cancels and the residual is purely the
poisoning effect) and confirms the observed shift stays within the certificate.

```python
from director_ai.core.federated_dp import FederatedCalibrationRound, FederatedDPEvidence

round_ = FederatedCalibrationRound(0.6, clip_norm=0.1, noise_multiplier=1.0)
evidence = FederatedDPEvidence(round_)

packet = evidence.evidence_packet(delta=1e-5, num_malicious=2, cohort_size=10, rounds=10)
print(packet.epsilon, packet.poisoning.total_shift)

sim = evidence.simulate_poisoning(
    num_malicious=2, cohort_size=10, honest_update=0.0, rounds=10
)
assert sim.within_bound
```

`ProductionGuard.federated_dp_evidence()` builds one over a calibration round
(the guard's default round, or one passed in).
