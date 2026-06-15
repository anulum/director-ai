# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Pedersen + Schnorr ZK attestation benchmark

"""Measure the zero-knowledge attestation backend.

The proof is modular exponentiation over a 2048-bit group, so the measurements
are correctness and the prove/verify cost:

* **Soundness/completeness** — an honest proof over a satisfying statement is
  accepted; a proof with a tampered aggregate is rejected. Reported as the
  fraction of the labelled checks that behave correctly.
* **Latency** — mean prove and verify time per attestation over a fixed sample
  count.
* **Proof size** — the number of group elements published (one commitment per
  sample plus the aggregate and the Schnorr pair).

Output: ``benchmarks/results/zk_schnorr.json``. Reproduce with
``python -m benchmarks.zk_schnorr``.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

from benchmarks._common import RESULTS_DIR
from director_ai.core.zk_attestation.schnorr import (
    DEFAULT_PARAMETERS,
    SchnorrAttestationBackend,
)
from director_ai.core.zk_attestation.statements import MinimumCoherence

_STMT = MinimumCoherence(name="coherence", threshold=0.8, samples_min=8)


def _samples(coherence: float, n: int) -> list[dict]:
    return [{"coherence": coherence} for _ in range(n)]


def correctness() -> dict:
    backend = SchnorrAttestationBackend()
    checks = 0
    passed = 0
    # honest + satisfying -> accept
    proof = backend.prove(_STMT, _samples(0.9, 16))
    checks += 1
    passed += int(backend.verify(_STMT, proof)[0] is True)
    # tampered aggregate -> reject
    forged = replace(proof, aggregate_scaled=proof.aggregate_scaled + 1)
    checks += 1
    passed += int(backend.verify(_STMT, forged)[0] is False)
    # unsatisfying statement -> reject with the threshold reason
    weak = backend.prove(_STMT, _samples(0.5, 16))
    checks += 1
    ok, reason = backend.verify(_STMT, weak)
    passed += int(ok is False and reason == "statement_threshold_not_met")
    return {"checks": checks, "correct": round(passed / checks, 4)}


def latency(samples: int, repeats: int) -> dict:
    backend = SchnorrAttestationBackend()
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
    backend = SchnorrAttestationBackend()
    proof = backend.prove(_STMT, _samples(0.9, samples))
    return {
        "samples": samples,
        "group_elements": len(proof.commitments) + 2,  # commitments + Y-residual + t
        "modulus_bits": DEFAULT_PARAMETERS.p.bit_length(),
    }


def run(*, samples: int = 16, repeats: int = 20) -> dict:
    return {
        "benchmark": "zk_schnorr",
        "correctness": correctness(),
        "latency": latency(samples, repeats),
        "proof_size": proof_size(samples),
        "backend": "pedersen-schnorr NIZK over a 2048-bit safe-prime group",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "zk_schnorr.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    c, lat = result["correctness"], result["latency"]
    print("\nPedersen + Schnorr ZK attestation:")
    print(f"  correctness={c['correct']:.2f} ({c['checks']} checks)")
    print(
        f"  prove={lat['prove_ms']:.1f} ms  verify={lat['verify_ms']:.1f} ms "
        f"({lat['samples']} samples)"
    )
    print(f"  modulus={result['proof_size']['modulus_bits']} bits")


if __name__ == "__main__":
    main()
