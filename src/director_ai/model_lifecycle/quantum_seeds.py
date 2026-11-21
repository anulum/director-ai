# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum Seed Validator

"""The mode-aware seed rule of a variational quantum run (Q-7).

Every run sets its parameter-initialisation and instance-shuffle seeds. A
finite-shot simulator also sets its shot-sampling seed; a QPU sets none, but
sets its transpiler layout seed and attests its hardware entropy. A
mitigation calibration seed is set exactly when mitigation is enabled. All
active seeds are distinct non-negative 63-bit integers.
"""

from __future__ import annotations

from ._checks import require_choice, require_int, require_sha256, require_unique
from .errors import QuantumAcceptanceError
from .quantum_contracts import validate_shot_noise_spec
from .quantum_records import QuantumShotNoiseAcceptanceSpec, QuantumVariationalSeedTuple

E = QuantumAcceptanceError
SEED_LIMIT = 2**63
"""Seeds are non-negative 63-bit integers."""


def _seed(value: object, field: str) -> int:
    """Return a non-negative 63-bit seed, else refuse."""
    seed = require_int(value, field, E, minimum=0)
    if seed >= SEED_LIMIT:
        raise E(f"{field} must be below 2**63")
    return seed


REQUIRED_SEEDS = ("parameter_init_seed", "problem_instance_shuffle_seed")
OPTIONAL_SEEDS = (
    "shot_sampling_seed",
    "transpiler_layout_seed",
    "mitigation_calibration_seed",
)


def _presence(value: object, field: str, required: bool) -> None:
    """Refuse a mode seed that is missing when required or present otherwise."""
    if required and value is None:
        raise E(f"{field} must be set in this mode")
    if not required and value is not None:
        raise E(f"{field} must be None in this mode")


def validate_seed_tuple(
    seeds: QuantumVariationalSeedTuple,
) -> QuantumVariationalSeedTuple:
    """Check a seed tuple's shape: in-range seeds, all distinct.

    Parameters
    ----------
    seeds : QuantumVariationalSeedTuple
        The seeds to check.

    Returns
    -------
    QuantumVariationalSeedTuple
        The same seeds.

    Raises
    ------
    QuantumAcceptanceError
        If a seed is out of range or repeats another set seed, or a field is
        malformed.
    MalformedDigestError
        If the hardware entropy attestation is malformed.
    """
    if not isinstance(seeds, QuantumVariationalSeedTuple):
        raise E("a QuantumVariationalSeedTuple is required")
    require_choice(seeds.schema, "schema", {"quantum-variational-seeds.v1"}, E)
    active = [_seed(getattr(seeds, name), name) for name in REQUIRED_SEEDS]
    for name in OPTIONAL_SEEDS:
        if getattr(seeds, name) is not None:
            active.append(_seed(getattr(seeds, name), name))
    if seeds.hardware_entropy_attestation_digest is not None:
        require_sha256(
            seeds.hardware_entropy_attestation_digest,
            "hardware_entropy_attestation_digest",
        )
    require_unique(active, "active seeds", E)
    return seeds


def validate_quantum_seeds(
    seeds: QuantumVariationalSeedTuple, spec: QuantumShotNoiseAcceptanceSpec
) -> QuantumVariationalSeedTuple:
    """Check a run's seeds against its spec's mode (Q-7).

    Parameters
    ----------
    seeds : QuantumVariationalSeedTuple
        The run's seeds.
    spec : QuantumShotNoiseAcceptanceSpec
        The study's acceptance spec.

    Returns
    -------
    QuantumVariationalSeedTuple
        The same seeds.

    Raises
    ------
    QuantumAcceptanceError
        If a seed is present or missing against its mode, is out of range,
        or repeats another active seed, or the spec is malformed.
    MalformedDigestError
        If the hardware entropy attestation is malformed.
    """
    validate_shot_noise_spec(spec)
    validate_seed_tuple(seeds)
    mode = spec.execution_mode
    _presence(
        seeds.shot_sampling_seed, "shot_sampling_seed", mode == "finite_shot_simulator"
    )
    _presence(
        seeds.mitigation_calibration_seed,
        "mitigation_calibration_seed",
        spec.mitigation_enabled,
    )
    _presence(
        seeds.hardware_entropy_attestation_digest,
        "hardware_entropy_attestation_digest",
        mode == "hardware_qpu",
    )
    if mode == "hardware_qpu":
        _presence(seeds.transpiler_layout_seed, "transpiler_layout_seed", True)
    return seeds
