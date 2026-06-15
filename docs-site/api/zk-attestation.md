# Cross-Org Passport + ZK Attestation

`director_ai.core.zk_attestation` lets organisation **A** hand an
agent over to organisation **B** with cryptographic evidence of the
agent's past behaviour, without releasing raw interaction logs.

The subpackage ships:

- **Typed statements** — `AttestationStatement` Protocol plus four
  concrete claims (`MinimumCoherence`, `MaximumHaltRate`,
  `DomainExperience`, `NoBreakoutEvents`).
- **`CommitmentBackend`** — the default shipped backend. Commits all
  samples under an HMAC-SHA256 Merkle tree, then opens a
  root-derived random subset for the verifier to spot-check.
- **`ZkSnarkBackend`** — Protocol for plug-in groth16 / plonk
  adapters (arkworks, gnark, snarkjs). Shipping such a backend is
  deliberately out-of-scope — the Protocol and verifier wiring live
  here so operators can slot one in without touching the passport
  format.
- **`CrossOrgPassport`** + `PassportIssuer` + `PassportVerifier` —
  signed bundle with an HMAC-SHA256 MAC over the canonical header,
  verified end-to-end by the receiving organisation.

## Honest naming

`CommitmentBackend` is a commitment + challenge-response scheme under
the random-oracle assumption on HMAC-SHA256. It is **not** a full
zero-knowledge proof — opened samples are revealed in the clear. The
class docstring says so:

> Caveat (honest naming): this is a commitment + spot-check scheme,
> *not* a zero-knowledge proof. It reveals opened samples in the
> clear. For a full ZK proof use a `ZkSnarkBackend` adapter.

## Quick start

```python
from director_ai.core.zk_attestation import (
    CommitmentBackend, MinimumCoherence, NoBreakoutEvents,
    PassportIssuer, PassportVerifier,
)

# Source org — issue a passport.
issuer = PassportIssuer(key=SOURCE_HMAC_KEY, issuing_org="org://source")
passport = issuer.issue(
    agent_id="agent-001",
    samples=history_samples,                  # list of dicts
    statements=[
        MinimumCoherence(name="coherence", threshold=0.9, samples_min=10_000),
        NoBreakoutEvents(name="no_break", samples_min=10_000),
    ],
)

# Receiving org — verify.
verifier = PassportVerifier(
    issuer_keys={"org://source": SOURCE_HMAC_KEY},   # PKI / out-of-band
    backends={"commitment": CommitmentBackend(key=SOURCE_HMAC_KEY)},
)
verdict = verifier.verify(passport)
assert verdict.accepted, verdict.failures
```

## Typed statements

Each claim is a frozen dataclass with a unique `name`, threshold
parameters, a cheap `evaluate_sample(sample) -> float` method used
by the prover, and an `accepts(aggregate, total_samples) -> bool`
predicate used by the verifier.

| Statement | Claim |
|---|---|
| `MinimumCoherence(threshold, samples_min)` | Mean coherence ≥ `threshold` over ≥ `samples_min` samples. |
| `MaximumHaltRate(max_rate, samples_min)` | Halts / total ≤ `max_rate` over ≥ `samples_min` samples. |
| `DomainExperience(domain, hours_min)` | Sum of `duration_seconds` over samples with matching `domain` ≥ `hours_min * 3600`. |
| `NoBreakoutEvents(samples_min)` | Zero samples tagged `breakout=True` across ≥ `samples_min` samples. |

Operators extend this with their own claim by implementing the
Protocol — no changes to the issuer / verifier are required.

## `CommitmentBackend` protocol

1. **Prover** holds the private sample list `s_0 … s_{n-1}` and a
   128-bit blinding factor per sample. For each sample it computes
   `leaf_i = HMAC-SHA256(key, r_i || s_i)` and publishes the Merkle
   root along with `n`.
2. The challenge indices are derived deterministically from the
   commitment root via HMAC-SHA256 PRF counter expansion — the prover
   cannot cherry-pick a favourable subset.
3. **Prover** opens each challenge index by revealing `(r_i, s_i)`
   and the Merkle authentication path; the proof carries the prover's
   reported aggregate.
4. **Verifier** recomputes every opened leaf, walks each path back to
   the published root, cross-checks the aggregate against the opened
   subset, and re-derives the expected challenge indices from the
   root. Any mismatch is rejected.

Per-proof parameters:

- `challenge_size=32` is the default opened-subset size. Larger
  values tighten the bound on cheating geometrically, at the cost of
  revealing more raw samples.

## `CrossOrgPassport`

Frozen bundle with `agent_id`, `issuing_org`, `created_at`, a tuple
of statement entries and an HMAC-SHA256 `mac` over the canonical
header. The header escapes `|` and `\` in the free-text fields so a
statement whose name contains the delimiter cannot collide with a
different passport layout.

## Proof backend dependency boundary

`CommitmentBackend` is the supported in-package backend and has no heavy proof
runtime. Real groth16 / plonk adapters must live behind `ZkSnarkBackend` and be
installed in a selected adapter runtime, not the default API process.

Recommended boundary:

- Pin the prover, verifier, circuit artefacts, and proving key by immutable
  release or digest.
- Include a circuit id in each adapter proof and reject passports with unknown
  ids.
- Run prover work in a separate process or service with CPU, memory, and wall
  clock limits.
- Keep `CommitmentBackend` enabled as the fallback acceptance path during
  adapter rollout.
- Do not let adapter errors skip passport verification; return a failed
  `PassportVerdict` with the backend reason.

## `PassportIssuer`

- `issue(agent_id, samples, statements, backends=None)` proves every
  statement with the default backend (or an override from the
  `backends` name map), collects the proofs, and signs the whole
  bundle.
- Rejects short keys (< 32 bytes), empty issuer / agent identifiers,
  empty sample / statement lists, and non-callable clocks.

## `PassportVerifier`

- `issuer_keys` maps issuing-org identifier → the shared HMAC secret
  for that org (populated out-of-band through the same PKI flow you
  use for any cross-org trust).
- `backends` maps `kind` → `AttestationBackend` instance; the
  commitment backend's `kind` is `"commitment"`. Plug-in zk-SNARK
  backends register themselves under their own `kind`.
- `verify(passport)` returns a `PassportVerdict` with `accepted`,
  `signature_ok`, and `failures: tuple[(statement_name, reason), ...]`
  so the receiving org can distinguish cryptographic rejection from
  behavioural non-conformance. The verdict also includes
  `safety_event`, using the `zk_attestation.passport` hook id and
  tenant-safe failure references.

## CoherenceAgent wiring

```python
from director_ai.core.agent import CoherenceAgent

agent = CoherenceAgent(passport_verifier=verifier)
verdict = agent.verify_passport(incoming_passport)
```

`verify_passport` raises `RuntimeError` when no verifier is attached
— the check is opt-in.

## `SchnorrAttestationBackend` (zero-knowledge of sample values)

The `CommitmentBackend` proves a statement by opening a random subset of
samples — sound, but it reveals those raw samples. `SchnorrAttestationBackend`
hides them. Each sample's contribution `v_i` is sealed in an additively
homomorphic Pedersen commitment `C_i = g^{v_i}·h^{r_i} (mod p)` over a
prime-order group with `log_g h` unknown, so a commitment is perfectly hiding.
By homomorphism the product of the per-sample commitments is a commitment to the
aggregate `A = Σ v_i`; the prover reveals only `A` and a non-interactive Schnorr
proof of knowledge of the blinding `R` in `(∏ C_i)·g^{-A} = h^R` (Fiat-Shamir
over SHA-256).

```python
from director_ai.core.zk_attestation import (
    SchnorrAttestationBackend, MinimumCoherence,
)

backend = SchnorrAttestationBackend()
statement = MinimumCoherence(name="coherence", threshold=0.8, samples_min=8)
proof = backend.prove(statement, samples)        # private samples in, ZK proof out
accepted, reason = backend.verify(statement, proof)
```

What it hides and what it does not, stated plainly:

- **hidden:** every individual sample value and blinding (perfect hiding);
- **revealed:** the aggregate `A` and the public statement/threshold;
- **not proven here:** that the committed `v_i` are honest evaluations of real
  samples — compose with the spot-checking `CommitmentBackend`, or a real SNARK,
  for that. Hiding the aggregate itself (revealing only "threshold met") needs a
  zero-knowledge range proof (Bulletproofs / SNARK), which is the
  `ZkSnarkBackend` plug-in's job.

The default group is a generated 2048-bit safe prime; `PedersenParameters` are
re-verified (primality of `p` and `q = (p-1)/2`, subgroup membership of the
generators) at construction, so a corrupted constant fails fast. Measured with
`python -m benchmarks.zk_schnorr`: soundness/completeness checks pass, prove and
verify are single modular-exponentiation-bound operations over the 2048-bit
group (see `benchmarks/results/zk_schnorr.json`).

## `BulletproofRangeBackend` (zero-knowledge of the aggregate too)

The Schnorr backend hides the individual values but still reveals the aggregate.
`BulletproofRangeBackend` hides the aggregate as well: it proves `Σ vᵢ ≥
threshold` while disclosing neither the values nor their sum — only the public
threshold and the pass/fail decision.

Each sample value is sealed in a Ristretto Pedersen commitment `Cᵢ`. By
homomorphism the published commitments sum to `C_agg = Σ Cᵢ`, a commitment to the
aggregate. The prover forms `C_d = C_agg − threshold·B` (a commitment to
`d = aggregate − threshold`) and a dalek **Bulletproof** proves `d ∈ [0, 2^bits)`,
i.e. `aggregate ≥ threshold`. The verifier recomputes `C_d` from the published
per-sample commitments, so the range proof is bound to the real committed data —
a prover cannot prove the bound against fabricated values.

```python
from director_ai.core.zk_attestation import (
    BulletproofRangeBackend, MinimumCoherence,
)

backend = BulletproofRangeBackend()                       # 32-bit range default
statement = MinimumCoherence(name="coherence", threshold=0.8, samples_min=8)
proof = backend.prove(statement, samples)                 # raises if the bar is not met
accepted, reason = backend.verify(statement, proof)       # neither values nor aggregate leak
```

- **hidden:** every sample value *and* the aggregate;
- **revealed:** the public threshold and the accept/reject decision;
- **bound to data:** the range proof verifies against the recomputed aggregate
  commitment, not a free-floating value, so a false "threshold met" is unprovable.

This backend is **Rust-only** — the Bulletproof and Ristretto arithmetic live in
the `backfire_kernel` extension (dalek `bulletproofs`); there is no pure-Python
fallback, because a correct, constant-time Bulletproof in Python would be neither
safe nor practical, and construction raises if the kernel is absent. A proof is
~600 bytes regardless of sample count; measured with
`python -m benchmarks.zk_bulletproof_range` (see
`benchmarks/results/zk_bulletproof_range.json`).
