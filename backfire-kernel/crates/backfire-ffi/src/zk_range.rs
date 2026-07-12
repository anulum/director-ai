// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — Bulletproof range-proof attestation kernel

//! Zero-knowledge proof that an aggregate over hidden per-sample values meets a
//! public threshold, without revealing the aggregate.
//!
//! Each sample value `v_i` is sealed in a Ristretto Pedersen commitment
//! `C_i = v_i·B + r_i·B_blinding`. By homomorphism the sum `C_agg = Σ C_i`
//! commits to `A = Σ v_i` with blinding `R = Σ r_i`. The prover forms
//! `C_d = C_agg − threshold·B`, which commits to `d = A − threshold`, and uses a
//! Bulletproof (dalek `bulletproofs`) to prove `d ∈ [0, 2^bits)` — i.e.
//! `A ≥ threshold` — revealing neither `A` nor `d`. The verifier recomputes
//! `C_d` from the published per-sample commitments and checks the proof, so the
//! range proof is bound to the actual committed data, not a free-floating value.
//!
//! The threshold is public (the compliance bar both parties agree on); the
//! individual sample values and the aggregate stay hidden.

use bulletproofs::{BulletproofGens, PedersenGens, RangeProof};
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use curve25519_dalek::traits::Identity;
use merlin::Transcript;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::OsRng;

const TRANSCRIPT_LABEL: &[u8] = b"director-ai/zk-attestation/bulletproof/v1";

fn transcript(context: &[u8]) -> Transcript {
    let mut t = Transcript::new(TRANSCRIPT_LABEL);
    t.append_message(b"context", context);
    t
}

fn check_bits(bits: usize) -> PyResult<()> {
    if matches!(bits, 8 | 16 | 32 | 64) {
        Ok(())
    } else {
        Err(PyValueError::new_err("bits must be one of 8, 16, 32, 64"))
    }
}

/// Prove `Σ values ≥ threshold` in zero knowledge.
///
/// Returns `(proof_bytes, [per_sample_commitment_bytes, ...])`. Raises
/// `ValueError` when the threshold is not met (a false statement is unprovable),
/// when the difference does not fit in `bits`, or on bad parameters.
#[pyfunction]
pub fn rust_bulletproof_prove_threshold(
    values: Vec<u64>,
    threshold: u64,
    bits: usize,
    context: &[u8],
) -> PyResult<(Vec<u8>, Vec<Vec<u8>>)> {
    check_bits(bits)?;
    if values.is_empty() {
        return Err(PyValueError::new_err("values must be non-empty"));
    }

    let pc_gens = PedersenGens::default();
    let bp_gens = BulletproofGens::new(bits, 1);
    let mut rng = OsRng;

    let mut commitments: Vec<RistrettoPoint> = Vec::with_capacity(values.len());
    let mut blindings: Vec<Scalar> = Vec::with_capacity(values.len());
    let mut aggregate: u128 = 0;
    for &v in &values {
        let r = Scalar::random(&mut rng);
        commitments.push(pc_gens.commit(Scalar::from(v), r));
        blindings.push(r);
        aggregate = aggregate
            .checked_add(u128::from(v))
            .ok_or_else(|| PyValueError::new_err("aggregate overflow"))?;
    }

    let aggregate: u64 = aggregate
        .try_into()
        .map_err(|_| PyValueError::new_err("aggregate exceeds u64"))?;
    let difference = aggregate
        .checked_sub(threshold)
        .ok_or_else(|| PyValueError::new_err("threshold not met: aggregate < threshold"))?;
    if bits < 64 && difference >= (1u64 << bits) {
        return Err(PyValueError::new_err("difference exceeds 2^bits"));
    }

    let blinding_sum: Scalar = blindings.iter().sum();
    let mut prover_transcript = transcript(context);
    let (proof, committed) = RangeProof::prove_single(
        &bp_gens,
        &pc_gens,
        &mut prover_transcript,
        difference,
        &blinding_sum,
        bits,
    )
    .map_err(|err| PyValueError::new_err(format!("prove failed: {err:?}")))?;

    // The proof's own commitment must equal C_agg − threshold·B, binding the
    // range proof to the published per-sample commitments.
    let c_agg: RistrettoPoint = commitments.iter().sum();
    let c_d = c_agg - pc_gens.B * Scalar::from(threshold);
    if committed != c_d.compress() {
        return Err(PyValueError::new_err("internal commitment mismatch"));
    }

    let commitment_bytes: Vec<Vec<u8>> = commitments
        .iter()
        .map(|c| c.compress().as_bytes().to_vec())
        .collect();
    Ok((proof.to_bytes(), commitment_bytes))
}

/// Verify a [`rust_bulletproof_prove_threshold`] proof. Returns `false` on any
/// malformed input or failed check; never raises on cryptographic failure.
#[pyfunction]
pub fn rust_bulletproof_verify_threshold(
    proof_bytes: &[u8],
    commitments: Vec<Vec<u8>>,
    threshold: u64,
    bits: usize,
    context: &[u8],
) -> PyResult<bool> {
    check_bits(bits)?;
    if commitments.is_empty() {
        return Ok(false);
    }
    let proof = match RangeProof::from_bytes(proof_bytes) {
        Ok(proof) => proof,
        Err(_) => return Ok(false),
    };

    let pc_gens = PedersenGens::default();
    let bp_gens = BulletproofGens::new(bits, 1);

    let mut c_agg = RistrettoPoint::identity();
    for raw in &commitments {
        if raw.len() != 32 {
            return Ok(false);
        }
        let compressed = match CompressedRistretto::from_slice(raw) {
            Ok(compressed) => compressed,
            Err(_) => return Ok(false),
        };
        match compressed.decompress() {
            Some(point) => c_agg += point,
            None => return Ok(false),
        }
    }

    let c_d = (c_agg - pc_gens.B * Scalar::from(threshold)).compress();
    let mut verifier_transcript = transcript(context);
    Ok(proof
        .verify_single(&bp_gens, &pc_gens, &mut verifier_transcript, &c_d, bits)
        .is_ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CTX: &[u8] = b"test-context";

    #[test]
    fn prove_then_verify_round_trips() {
        let (proof, commitments) =
            rust_bulletproof_prove_threshold(vec![10, 20, 30], 50, 16, CTX).unwrap();
        assert!(commitments.iter().all(|c| c.len() == 32));
        assert!(rust_bulletproof_verify_threshold(&proof, commitments, 50, 16, CTX).unwrap());
    }

    #[test]
    fn verify_rejects_wrong_threshold_context_and_tampering() {
        let (proof, commitments) =
            rust_bulletproof_prove_threshold(vec![10, 20, 30], 50, 16, CTX).unwrap();

        // Different public threshold changes C_d — proof no longer binds.
        assert!(
            !rust_bulletproof_verify_threshold(&proof, commitments.clone(), 40, 16, CTX).unwrap()
        );
        // Different transcript context.
        assert!(!rust_bulletproof_verify_threshold(
            &proof,
            commitments.clone(),
            50,
            16,
            b"other-context"
        )
        .unwrap());
        // Tampered proof bytes.
        let mut bad_proof = proof.clone();
        bad_proof[0] ^= 0x01;
        assert!(
            !rust_bulletproof_verify_threshold(&bad_proof, commitments.clone(), 50, 16, CTX)
                .unwrap()
        );
        // Tampered commitment bytes (still 32 bytes, wrong point/bits).
        let mut bad_commitments = commitments;
        bad_commitments[0][0] ^= 0x01;
        assert!(!rust_bulletproof_verify_threshold(&proof, bad_commitments, 50, 16, CTX).unwrap());
    }

    #[test]
    fn verify_handles_malformed_inputs_without_raising() {
        let (proof, commitments) =
            rust_bulletproof_prove_threshold(vec![5, 6], 10, 8, CTX).unwrap();

        // Empty commitment set, garbage proof, and wrong-length commitment
        // all report false rather than raising.
        assert!(!rust_bulletproof_verify_threshold(&proof, vec![], 10, 8, CTX).unwrap());
        assert!(
            !rust_bulletproof_verify_threshold(b"garbage", commitments.clone(), 10, 8, CTX)
                .unwrap()
        );
        assert!(
            !rust_bulletproof_verify_threshold(&proof, vec![vec![0_u8; 31]], 10, 8, CTX).unwrap()
        );
    }

    #[test]
    fn prove_rejects_bad_parameters() {
        // Unsupported bit width (prover and verifier).
        assert!(rust_bulletproof_prove_threshold(vec![1], 1, 12, CTX).is_err());
        assert!(rust_bulletproof_verify_threshold(b"", vec![], 1, 12, CTX).is_err());
        // Empty values.
        assert!(rust_bulletproof_prove_threshold(vec![], 1, 8, CTX).is_err());
        // False statement: aggregate below threshold is unprovable.
        assert!(rust_bulletproof_prove_threshold(vec![1, 2], 10, 8, CTX).is_err());
        // Difference does not fit the requested range.
        assert!(rust_bulletproof_prove_threshold(vec![300], 0, 8, CTX).is_err());
        // Aggregate exceeding u64 is rejected before proving.
        assert!(rust_bulletproof_prove_threshold(vec![u64::MAX, 1], 0, 64, CTX).is_err());
    }
}
