# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum and Spiking Witness Validators

"""Validators of quantum acceptance specs and of the arms' execution receipts.

An exact-statevector spec uses no shots, a non-negative solver tolerance, no
mitigation and no split of its error rate. A finite-shot or QPU spec uses at
least 256 shots per Pauli group, a noise-floor multiplier of at least 3, and
splits its error rate exactly between clusters and shots (Q-3, Q-4): each
rate is read as the decimal it was written as, and the sum is taken in the
exact decimal context that traps any rounding, so ``0.1 + 0.2`` equals
``0.3``. Without mitigation the overhead cap is 1. The physical margin and
the multiplier are declared trusts. Each arm's receipt matches the spec's
mode and shot budget exactly, stays within its gate depth and mitigation
overhead, and carries the witnesses its mode needs (Q-6). The classical
baseline is named and no global optimum is claimed (Q-5). The seeds are in
:mod:`.quantum_seeds`, the preflight and parity witnesses in
:mod:`.family_witnesses`.
"""

from __future__ import annotations

from decimal import Decimal, Inexact, localcontext

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
)
from .errors import QuantumAcceptanceError
from .quantum_records import (
    OPTIMISATION_SENSES,
    QUANTUM_EXECUTION_MODES,
    QuantumCircuitExecutionReceipt,
    QuantumShotNoiseAcceptanceSpec,
)
from .spend_contracts import MONEY_CONTEXT

E = QuantumAcceptanceError
MIN_SHOTS_PER_PAULI_GROUP = 256
"""Q-6: the fewest shots per Pauli group a sampled arm may use."""
MIN_NOISE_FLOOR_MULTIPLIER = 3.0
"""A sampled arm's bound must span at least this many shot standard errors."""


def _check_sampled_alphas(spec: QuantumShotNoiseAcceptanceSpec) -> None:
    """Check that a sampled spec splits its error rate exactly (Q-3)."""
    if spec.shots_per_pauli_group < MIN_SHOTS_PER_PAULI_GROUP:
        raise E(f"a sampled arm needs at least {MIN_SHOTS_PER_PAULI_GROUP} shots")
    if spec.exact_solver_tolerance is not None:
        raise E("only an exact statevector names a solver tolerance")
    if spec.shot_noise_floor_multiplier < MIN_NOISE_FLOOR_MULTIPLIER:
        raise E("a sampled arm's noise-floor multiplier is at least 3")
    cluster = require_finite(spec.alpha_cluster, "alpha_cluster", E)
    shot = require_finite(spec.alpha_shot, "alpha_shot", E)
    if cluster <= 0.0 or shot <= 0.0:
        raise E("alpha_cluster and alpha_shot must be above zero")
    try:
        with localcontext(MONEY_CONTEXT):
            split = Decimal(str(cluster)) + Decimal(str(shot))
    except Inexact as error:
        raise E(
            "the error split needs more precision than the exact check holds"
        ) from error
    if split != Decimal(str(spec.alpha_total)):
        raise E("alpha_cluster and alpha_shot must sum exactly to alpha_total")


def validate_shot_noise_spec(
    spec: QuantumShotNoiseAcceptanceSpec,
) -> QuantumShotNoiseAcceptanceSpec:
    """Check a variational quantum acceptance spec (Q-3 to Q-5).

    Parameters
    ----------
    spec : QuantumShotNoiseAcceptanceSpec
        The spec to check.

    Returns
    -------
    QuantumShotNoiseAcceptanceSpec
        The same spec.

    Raises
    ------
    QuantumAcceptanceError
        If the mode's shot, tolerance, mitigation or error-split rules fail,
        a global optimum is claimed, or a field is malformed.
    MalformedDigestError
        If the classical baseline digest is malformed.
    """
    if not isinstance(spec, QuantumShotNoiseAcceptanceSpec):
        raise E("a QuantumShotNoiseAcceptanceSpec is required")
    require_choice(spec.schema, "schema", {"quantum-shot-noise-acceptance.v1"}, E)
    mode = require_choice(
        spec.execution_mode, "execution_mode", QUANTUM_EXECUTION_MODES, E
    )
    require_choice(
        spec.optimisation_sense, "optimisation_sense", OPTIMISATION_SENSES, E
    )
    require_int(spec.shots_per_pauli_group, "shots_per_pauli_group", E, minimum=0)
    mitigation = require_bool(spec.mitigation_enabled, "mitigation_enabled", E)
    require_bool(spec.certifies_ansatz_infimum, "certifies_ansatz_infimum", E)
    gamma = require_finite(
        spec.max_mitigation_overhead_gamma, "max_mitigation_overhead_gamma", E
    )
    if gamma < 1.0 or (not mitigation and gamma != 1.0):
        raise E("the mitigation overhead cap is at least 1, and 1 without mitigation")
    alpha = require_finite(spec.alpha_total, "alpha_total", E)
    if not 0.0 < alpha < 1.0:
        raise E("alpha_total must lie in (0, 1)")
    for name in ("min_physical_margin", "shot_noise_floor_multiplier"):
        if require_finite(getattr(spec, name), name, E) < 0.0:
            raise E(f"{name} must not be negative")
    require_int(spec.max_two_qubit_gate_depth, "max_two_qubit_gate_depth", E, minimum=0)
    require_sha256(spec.classical_baseline_digest, "classical_baseline_digest")
    if spec.global_optimum_claimed is not False:
        raise E("a variational study never claims a global optimum")
    if mode != "exact_statevector":
        _check_sampled_alphas(spec)
        return spec
    if spec.shots_per_pauli_group != 0 or mitigation:
        raise E("an exact statevector uses no shots and no mitigation")
    tolerance = require_finite(spec.exact_solver_tolerance, "exact_solver_tolerance", E)
    if tolerance < 0.0:
        raise E("exact_solver_tolerance must not be negative")
    if spec.alpha_cluster is not None or spec.alpha_shot is not None:
        raise E("an exact statevector does not split its error rate")
    return spec


OPTIONAL_RECEIPT_DIGESTS = (
    "transpilation_witness_digest",
    "mitigation_witness_digest",
    "qpu_result_pack_digest",
)


def validate_receipt_shape(
    receipt: QuantumCircuitExecutionReceipt,
) -> QuantumCircuitExecutionReceipt:
    """Check an execution receipt's shape, without its spec.

    Parameters
    ----------
    receipt : QuantumCircuitExecutionReceipt
        The receipt to check.

    Returns
    -------
    QuantumCircuitExecutionReceipt
        The same receipt.

    Raises
    ------
    QuantumAcceptanceError
        If a count is negative, the overhead is below 1, or a field is
        malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(receipt, QuantumCircuitExecutionReceipt):
        raise E("a QuantumCircuitExecutionReceipt is required")
    require_choice(
        receipt.schema, "schema", {"quantum-circuit-execution-receipt.v1"}, E
    )
    require_sha256(receipt.arm_content_digest, "arm_content_digest")
    require_choice(receipt.execution_mode, "execution_mode", QUANTUM_EXECUTION_MODES, E)
    for name in ("observed_shots_per_pauli_group", "observed_two_qubit_gate_depth"):
        require_int(getattr(receipt, name), name, E, minimum=0)
    gamma = require_finite(
        receipt.observed_max_mitigation_gamma, "observed_max_mitigation_gamma", E
    )
    if gamma < 1.0:
        raise E("the observed mitigation overhead must be at least 1")
    for name in OPTIONAL_RECEIPT_DIGESTS:
        if getattr(receipt, name) is not None:
            require_sha256(getattr(receipt, name), name)
    return receipt


def validate_execution_receipt(
    receipt: QuantumCircuitExecutionReceipt, spec: QuantumShotNoiseAcceptanceSpec
) -> QuantumCircuitExecutionReceipt:
    """Check one arm's execution receipt against the spec (Q-6).

    Parameters
    ----------
    receipt : QuantumCircuitExecutionReceipt
        The arm's receipt.
    spec : QuantumShotNoiseAcceptanceSpec
        The study's validated acceptance spec.

    Returns
    -------
    QuantumCircuitExecutionReceipt
        The same receipt.

    Raises
    ------
    QuantumAcceptanceError
        If the mode or shot budget differs from the spec, the gate depth or
        mitigation overhead exceeds its cap, or a witness is missing or
        out of place.
    MalformedDigestError
        If a digest is malformed.
    """
    validate_receipt_shape(receipt)
    if receipt.execution_mode != spec.execution_mode:
        raise E("the receipt ran in another mode than the spec")
    if receipt.observed_shots_per_pauli_group != spec.shots_per_pauli_group:
        raise E("the receipt's shots differ from the spec's matched budget")
    if receipt.observed_two_qubit_gate_depth > spec.max_two_qubit_gate_depth:
        raise E("the receipt's two-qubit gate depth exceeds the cap")
    if receipt.observed_max_mitigation_gamma > spec.max_mitigation_overhead_gamma:
        raise E("the observed mitigation overhead exceeds the cap")
    qpu = spec.execution_mode == "hardware_qpu"
    for name, required in (
        ("mitigation_witness_digest", spec.mitigation_enabled),
        ("qpu_result_pack_digest", qpu),
    ):
        if required != (getattr(receipt, name) is not None):
            raise E(f"{name} is given exactly when its mode or mitigation needs it")
    if qpu and receipt.transpilation_witness_digest is None:
        raise E("a QPU receipt names its transpilation witness")
    return receipt


def validate_paired_receipts(
    candidate: QuantumCircuitExecutionReceipt,
    baseline: QuantumCircuitExecutionReceipt,
    spec: QuantumShotNoiseAcceptanceSpec,
) -> QuantumShotNoiseAcceptanceSpec:
    """Check both arms' receipts against one spec, so their budgets match.

    Parameters
    ----------
    candidate : QuantumCircuitExecutionReceipt
        The candidate arm's receipt.
    baseline : QuantumCircuitExecutionReceipt
        The baseline arm's receipt.
    spec : QuantumShotNoiseAcceptanceSpec
        The study's acceptance spec.

    Returns
    -------
    QuantumShotNoiseAcceptanceSpec
        The same spec.

    Raises
    ------
    QuantumAcceptanceError
        If the spec or a receipt is refused, or both receipts name one arm.
    MalformedDigestError
        If a digest is malformed.
    """
    validate_shot_noise_spec(spec)
    validate_execution_receipt(candidate, spec)
    validate_execution_receipt(baseline, spec)
    if candidate.arm_content_digest == baseline.arm_content_digest:
        raise E("the candidate and the baseline must be different arms")
    return spec
