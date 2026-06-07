# Cryptographic Output Integrity

Non-repudiation and a tamper-evident audit trail for model outputs (OWASP ML09) —
what finance and healthcare deployments require and what no accuracy-focused
guardrail provides. Two composable controls:

- **Tamper-evident ledger** — an append-only hash chain over output digests.
  Stdlib only, always available, tenant-safe (digests, never raw payloads).
- **Ed25519 signing** — detached signatures a third party can verify with only
  the public key. Needs the optional `cryptography` backend
  (`pip install director-ai[crypto]`).

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig

integrity = ProductionGuard(DirectorConfig()).output_integrity()

# Sign an output for non-repudiation.
signed = integrity.sign("The diagnosis is X.", {"tenant": "t1", "model": "factcg"})
integrity.verify(signed)          # True (the public key travels inside `signed`)

# Record its digest in the tamper-evident ledger (no raw output retained).
integrity.record("The diagnosis is X.", {"tenant": "t1"})
integrity.verify_ledger()         # True — chain intact
```

## Signing (non-repudiation)

`OutputSigner` holds an Ed25519 key (generated, or reproducible from a 32-byte
`seed` loaded from a secret manager). `sign()` returns a `SignedOutput` carrying
the output, metadata, the detached signature, and the public key:

```python
from director_ai.core.output_integrity import OutputSigner, verify_signed_output

signer = OutputSigner(seed=my_32_byte_seed)
signed = signer.sign("answer", {"request_id": "r-1"})
verify_signed_output(signed)      # anyone with `signed` can verify
```

Verification fails if the output, the metadata, the signature, the algorithm, or
the public key has been altered — the signature binds all of them. Without the
`cryptography` backend, signing raises `MissingCryptoBackendError` with the
install hint; the ledger below still works.

## Tamper-evident ledger

`TamperEvidentLedger` chains each entry to the previous one:
`entry_hash = SHA-256(prev_hash ‖ payload_digest)`. Altering, removing, or
reordering any past entry breaks every subsequent hash, and `verify()` recomputes
the chain to detect it.

```python
from director_ai.core.output_integrity import TamperEvidentLedger

ledger = TamperEvidentLedger()
ledger.append({"output_id": "o-1", "digest": "..."})
ledger.append({"output_id": "o-2", "digest": "..."})
ledger.verify()                                  # True
TamperEvidentLedger.verify_entries(exported)     # verify an exported copy
```

Only digests are stored, never raw payloads, so the ledger can be exported for an
external auditor without leaking interaction content. `LedgerEntry.to_dict()` and
`SignedOutput.to_dict()` are JSON-safe.
