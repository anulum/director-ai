# Rényi-DP Accountant

::: director_ai.core.federated_privacy.rdp_accountant.RenyiAccountant

::: director_ai.core.federated_privacy.rdp_accountant.gaussian_rdp

::: director_ai.core.federated_privacy.rdp_accountant.DPGuarantee

## Boundary

`RenyiAccountant` tracks Rényi differential privacy (RDP) at a grid of orders
and converts to an `(ε, δ)`-DP guarantee at release time. It is the tight route
for composing many applications of the Gaussian mechanism — DP decoding and DP
score release — where the `(ε, δ)` `PrivacyAccountant`'s basic (linear) and
advanced (Dwork-Rothblum-Vadhan) composition are loose.

Use `PrivacyAccountant` for pure-`ε` Laplace queries (such as DP retrieval
ranking) and `RenyiAccountant` for Gaussian-mechanism pipelines that compose
over many rounds.

```python
from director_ai.core.federated_privacy import RenyiAccountant

accountant = RenyiAccountant()
# 50 rounds of the Gaussian mechanism at noise multiplier z = σ/Δ = 4.0.
accountant.compose_gaussian(noise_multiplier=4.0, steps=50)

guarantee = accountant.epsilon(delta=1e-5)
print(guarantee.epsilon, guarantee.order)
```

## Mathematics

All formulae are the published RDP results (Mironov, *Rényi Differential
Privacy*, CSF 2017); no constants are fabricated.

| Step | Result |
|------|--------|
| Gaussian RDP | A query with L2-sensitivity `Δ` perturbed by `N(0, σ²)` is `(α, α·Δ²/(2σ²))`-RDP for every `α > 1`. With noise multiplier `z = σ/Δ` this is `α/(2z²)`. |
| Composition | RDP is additive at a fixed order: `(α, ε₁) ∘ (α, ε₂) = (α, ε₁ + ε₂)`-RDP. |
| Conversion | An `(α, ε_RDP)`-RDP mechanism is `(ε_RDP + ln(1/δ)/(α − 1), δ)`-DP. The reported `ε` is the minimum of that bound over the order grid. |

The default order grid is the de-facto standard: fractional orders `1.1 … 10.9`
plus integer orders `11 … 63`. `total_rdp_mass()` exposes a monotone scalar for
budget displays; it is not itself a privacy quantity (RDP is per-order) and
leaks no curve detail.
