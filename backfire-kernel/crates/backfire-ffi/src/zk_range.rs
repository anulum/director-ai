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
