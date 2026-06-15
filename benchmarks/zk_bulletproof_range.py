# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Bulletproof range-proof attestation benchmark

"""Measure the aggregate-hiding range-proof backend.

The proof is a dalek Bulletproof over Ristretto, so the measurements are
correctness and the prove/verify cost:

* **Correctness** — a satisfying statement proves and verifies; a tampered
  commitment is rejected; an unsatisfying statement is unprovable (raises).
  Reported as the fraction of labelled checks that behave correctly.
* **Latency** — mean prove and verify time per attestation over a fixed sample
  count.
* **Proof size** — Bulletproof bytes and the number of per-sample commitments.

Output: ``benchmarks/results/zk_bulletproof_range.json``. Reproduce with
``python -m benchmarks.zk_bulletproof_range``.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

from benchmarks._common import RESULTS_DIR
from director_ai.core.zk_attestation.bulletproof_range import BulletproofRangeBackend
from director_ai.core.zk_attestation.statements import MinimumCoherence

_STMT = MinimumCoherence(name="coherence", threshold=0.8, samples_min=8)


def _samples(coherence: float, n: int) -> list[dict]:
    return [{"coherence": coherence} for _ in range(n)]


def correctness() -> dict:
    backend = BulletproofRangeBackend()
    checks = 0
    passed = 0
    proof = backend.prove(_STMT, _samples(0.9, 16))
    checks += 1
    passed += int(backend.verify(_STMT, proof)[0] is True)
    forged = replace(proof, commitments=(bytes(32),) + proof.commitments[1:])
    checks += 1
    passed += int(backend.verify(_STMT, forged)[0] is False)
    checks += 1
    try:
        backend.prove(_STMT, _samples(0.5, 16))
        unprovable = False
    except ValueError:
        unprovable = True
    passed += int(unprovable)
    return {"checks": checks, "correct": round(passed / checks, 4)}


def latency(samples: int, repeats: int) -> dict:
    backend = BulletproofRangeBackend()
    data = _samples(0.9, samples)
    t0 = time.perf_counter()
    proofs = [backend.prove(_STMT, data) for _ in range(repeats)]
    prove_ms = (time.perf_counter() - t0) / repeats * 1000.0
    t0 = time.perf_counter()
    for proof in proofs:
        backend.verify(_STMT, proof)
    verify_ms = (time.perf_counter() - t0) / repeats * 1000.0
    return {
        "samples": samples,
        "prove_ms": round(prove_ms, 3),
        "verify_ms": round(verify_ms, 3),
    }


def proof_size(samples: int) -> dict:
    backend = BulletproofRangeBackend()
    proof = backend.prove(_STMT, _samples(0.9, samples))
    return {
        "samples": samples,
        "proof_bytes": len(proof.proof),
        "commitments": len(proof.commitments),
        "bits": proof.bits,
    }


def run(*, samples: int = 16, repeats: int = 20) -> dict:
    return {
        "benchmark": "zk_bulletproof_range",
        "correctness": correctness(),
        "latency": latency(samples, repeats),
        "proof_size": proof_size(samples),
        "backend": "dalek bulletproofs range proof over Ristretto (aggregate hidden)",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "zk_bulletproof_range.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    c, lat, sz = result["correctness"], result["latency"], result["proof_size"]
    print("\nBulletproof range-proof attestation:")
    print(f"  correctness={c['correct']:.2f} ({c['checks']} checks)")
    print(f"  prove={lat['prove_ms']:.1f} ms  verify={lat['verify_ms']:.1f} ms")
    print(f"  proof={sz['proof_bytes']} bytes, {sz['commitments']} commitments")


if __name__ == "__main__":
    main()
