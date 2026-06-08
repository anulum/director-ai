# Secure Multi-Party Aggregation (SMPC)

Score and aggregate across confidential parties — hospital A and hospital B, or
two tenants — without any party revealing its input to another. The privacy comes
from **secret sharing**: a value is split into shares that individually reveal
nothing, and only the aggregate is reconstructed.

> **Scope.** This is the information-theoretic secret-sharing layer: additive
> (`n`-of-`n`) and Shamir (`t`-of-`n`) sharing plus secure summation — the SMPC
> primitive for *linear* federated aggregation (sum, mean, count). Multiplicative
> MPC (SPDZ/MASCOT with Beaver triples and an interactive online phase) is a
> separate networked protocol and is out of scope here.

## Additive secure aggregation

Each party splits its value into shares; the aggregator sums the shares
component-wise and reconstructs only the total. No single party's value is ever
materialised.

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.federated_privacy.secret_sharing import split

agg = ProductionGuard(DirectorConfig()).secure_aggregator(party_count=3)
for value in (10, 20, 12):
    agg.submit(split(value, party_count=3))
print(agg.reconstruct())          # 42 — the sum, never any single value
```

Additive sharing is `n`-of-`n`: every party must contribute, and one dropout loses
the result.

## Shamir threshold sharing

Shamir `t`-of-`n` sharing tolerates dropouts and collusion: *any* `t` of the `n`
parties reconstruct the secret, while fewer than `t` learn nothing — so up to
`n - t` parties can drop out and up to `t - 1` can collude without breaking
privacy.

```python
from director_ai.core.federated_privacy import (
    shamir_split, shamir_reconstruct, shamir_sum_shares,
)

shares = shamir_split(secret, party_count=5, threshold=3)   # 3-of-5
secret_back = shamir_reconstruct(shares[:3])                 # any 3 suffice

# Secure summation across parties (additively homomorphic), dropout-tolerant:
groups = [shamir_split(v, party_count=5, threshold=3) for v in private_values]
total_shares = shamir_sum_shares(groups)
total = shamir_reconstruct(total_shares[:3])                 # = sum(private_values)
```

The field modulus is the Mersenne prime `2**127 - 1`, so the Lagrange
interpolation that reconstructs the secret always has the modular inverses it
needs. Shares carry no plaintext, so the protocol metadata is safe to route over
any transport. Production code must leave `seed` unset so the system CSPRNG is
used; a seed is for reproducible tests only and requires `allow_insecure_seed=True`.
