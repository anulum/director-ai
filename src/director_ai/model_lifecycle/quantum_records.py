# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum and Spiking Witness Records

"""Records of variational quantum acceptance and spiking lowering parity.

A variational quantum study fixes, in its acceptance spec, how its arms run
(an exact statevector, a finite-shot simulator or a QPU), their matched shot
budget per Pauli group, the error split between clusters and shots, the
physical margin and the sole classical baseline; it never claims a global
optimum. Each arm's execution receipt is checked against the spec, and each
run's seeds against the mode. A barren-plateau witness records the gradient
variance preflight. A spiking lowering parity report ties a checkpoint to the
native engine. The rules are in :mod:`.quantum_contracts` and
:mod:`.quantum_acceptance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QuantumExecutionMode = Literal[
    "exact_statevector",
    "finite_shot_simulator",
    "hardware_qpu",
]
QUANTUM_EXECUTION_MODES: frozenset[str] = frozenset(
    {"exact_statevector", "finite_shot_simulator", "hardware_qpu"}
)
OPTIMISATION_SENSES: frozenset[str] = frozenset(
    {"minimise_energy", "maximise_fidelity"}
)
SPIKING_RESET_SEMANTICS: frozenset[str] = frozenset({"subtractive", "hard_to_v_reset"})


@dataclass(frozen=True, slots=True)
class SpikingLoweringParityReport:
    """Parity witness between NIR-in-Safetensors checkpoint and native engine."""

    schema: Literal["spiking-lowering-parity-report.v1"]
    checkpoint_content_digest: str
    hardware_numeric_contract_digest: str
    blocking_underflow_or_overflow_count: Literal[0]
    reset_semantics: Literal["subtractive", "hard_to_v_reset"]
    surrogate_family: str
    dt_seconds: float
    simulation_steps_t: int
    victor_purpura_or_van_rossum_distance: float
    classification_agreement_rate: float
    passed: bool


@dataclass(frozen=True, slots=True)
class QuantumVariationalSeedTuple:
    """Mode-aware seed tuple for variational quantum training and evaluation."""

    schema: Literal["quantum-variational-seeds.v1"]
    parameter_init_seed: int
    problem_instance_shuffle_seed: int
    shot_sampling_seed: int | None
    transpiler_layout_seed: int | None
    mitigation_calibration_seed: int | None
    hardware_entropy_attestation_digest: str | None


@dataclass(frozen=True, slots=True)
class BarrenPlateauWitnessReceipt:
    """Empirical parameter-landscape gradient-variance preflight witness."""

    schema: Literal["barren-plateau-witness-receipt.v1"]
    companion_digest: str
    split_pool_digest: str
    sampled_parameter_points: int
    min_observed_grad_variance: float
    threshold_grad_variance: float
    passed: bool


@dataclass(frozen=True, slots=True)
class QuantumShotNoiseAcceptanceSpec:
    """Paired statistical acceptance specification for variational quantum arms.

    On the sampled modes ``shot_noise_floor_multiplier`` is at least 3; an
    exact statevector ignores it. ``min_physical_margin`` and the multiplier
    are declared trusts. ``certifies_ansatz_infimum`` is informational, since
    ``global_optimum_claimed`` is always ``False``.
    """

    schema: Literal["quantum-shot-noise-acceptance.v1"]
    execution_mode: QuantumExecutionMode
    optimisation_sense: Literal["minimise_energy", "maximise_fidelity"]
    shots_per_pauli_group: int
    exact_solver_tolerance: float | None
    mitigation_enabled: bool
    max_mitigation_overhead_gamma: float
    alpha_total: float
    alpha_cluster: float | None
    alpha_shot: float | None
    min_physical_margin: float
    shot_noise_floor_multiplier: float
    max_two_qubit_gate_depth: int
    classical_baseline_digest: str
    certifies_ansatz_infimum: bool
    global_optimum_claimed: Literal[False]


@dataclass(frozen=True, slots=True)
class QuantumCircuitExecutionReceipt:
    """Per-arm execution receipt checked against QuantumShotNoiseAcceptanceSpec."""

    schema: Literal["quantum-circuit-execution-receipt.v1"]
    arm_content_digest: str
    execution_mode: QuantumExecutionMode
    observed_shots_per_pauli_group: int
    observed_two_qubit_gate_depth: int
    observed_max_mitigation_gamma: float
    transpilation_witness_digest: str | None
    mitigation_witness_digest: str | None
    qpu_result_pack_digest: str | None
