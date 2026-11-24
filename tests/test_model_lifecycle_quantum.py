# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Quantum and Spiking Contract Tests

"""Quantum acceptance specs, seeds, receipts, shot noise, the gate, and witnesses."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from fractions import Fraction
from typing import Any

import pytest

from director_ai.model_lifecycle import (
    BarrenPlateauWitnessReceipt,
    MalformedDigestError,
    ModelLifecycleError,
    PairedShotInstance,
    PauliGroupShots,
    QuantumAcceptanceError,
    QuantumCircuitExecutionReceipt,
    QuantumShotNoiseAcceptanceSpec,
    QuantumVariationalSeedTuple,
    SpikingLoweringParityReport,
    SpikingParityError,
    admits_quantum_arm,
    canonical_json_bytes,
    composite_lower_bound,
    from_canonical_json_bytes,
    hoeffding_radius,
    paired_improvement,
    paired_shot_radius,
    paired_shot_standard_error,
    require_spiking_parity_passed,
    shot_variance,
    validate_barren_plateau_receipt,
    validate_execution_receipt,
    validate_paired_receipts,
    validate_quantum_seeds,
    validate_receipt_shape,
    validate_seed_tuple,
    validate_shot_noise_spec,
    validate_spiking_lowering_parity,
)
from tests.model_lifecycle_support import digest_of

SPECS = {
    "exact_statevector": QuantumShotNoiseAcceptanceSpec(
        "quantum-shot-noise-acceptance.v1",
        "exact_statevector",
        "minimise_energy",
        0,
        0.125,
        False,
        1.0,
        0.05,
        None,
        None,
        0.01,
        3.0,
        40,
        digest_of("classical"),
        False,
        False,
    ),
    "finite_shot_simulator": QuantumShotNoiseAcceptanceSpec(
        "quantum-shot-noise-acceptance.v1",
        "finite_shot_simulator",
        "minimise_energy",
        1024,
        None,
        False,
        1.0,
        0.3,
        0.1,
        0.2,
        0.01,
        3.0,
        40,
        digest_of("classical"),
        False,
        False,
    ),
    "hardware_qpu": QuantumShotNoiseAcceptanceSpec(
        "quantum-shot-noise-acceptance.v1",
        "hardware_qpu",
        "maximise_fidelity",
        1024,
        None,
        True,
        1.5,
        0.05,
        0.045,
        0.005,
        0.01,
        3.0,
        40,
        digest_of("classical"),
        True,
        False,
    ),
}
SEEDS = {
    "exact_statevector": QuantumVariationalSeedTuple(
        "quantum-variational-seeds.v1", 1, 2, None, None, None, None
    ),
    "finite_shot_simulator": QuantumVariationalSeedTuple(
        "quantum-variational-seeds.v1", 1, 2, 3, None, None, None
    ),
    "hardware_qpu": QuantumVariationalSeedTuple(
        "quantum-variational-seeds.v1", 1, 2, None, 4, 5, digest_of("entropy")
    ),
}
MODES = tuple(SPECS)


def spec(
    mode: str = "finite_shot_simulator", /, **changes: object
) -> QuantumShotNoiseAcceptanceSpec:
    """Return a valid spec of ``mode`` with ``changes`` applied."""
    return replace(SPECS[mode], **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def receipt(
    mode: str = "finite_shot_simulator", arm: str = "candidate", /, **changes: object
) -> QuantumCircuitExecutionReceipt:
    """Return a receipt that matches the spec of ``mode``."""
    qpu, mitigated = mode == "hardware_qpu", SPECS[mode].mitigation_enabled
    base = QuantumCircuitExecutionReceipt(
        "quantum-circuit-execution-receipt.v1",
        digest_of(arm),
        mode,  # type: ignore[arg-type]  # the helper takes any string so refusal tests can pass unknown values
        SPECS[mode].shots_per_pauli_group,
        40,
        1.4 if mitigated else 1.0,
        digest_of("transpile") if qpu else None,
        digest_of("mitigation") if mitigated else None,
        digest_of("qpu") if qpu else None,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


class TestSpec:
    """Q-3 to Q-5: each mode's spec rules, the exact split and no global optimum."""

    @pytest.mark.parametrize("mode", MODES)
    def test_valid_specs_round_trip(self, mode: str) -> None:
        """Each mode's spec validates and decodes; 0.1 + 0.2 == 0.3 and 0.045 + 0.005 == 0.05."""
        item = spec(mode)
        assert validate_shot_noise_spec(item) == item
        assert (
            from_canonical_json_bytes(
                QuantumShotNoiseAcceptanceSpec, canonical_json_bytes(item)
            )
            == item
        )

    def test_tqv8_a_claimed_global_optimum_is_refused_at_decode(self) -> None:
        """The Literal[False] refuses a true claim in the JSON itself."""
        raw = canonical_json_bytes(spec())
        assert b'"global_optimum_claimed":false' in raw
        with pytest.raises(ModelLifecycleError):
            from_canonical_json_bytes(
                QuantumShotNoiseAcceptanceSpec,
                raw.replace(
                    b'"global_optimum_claimed":false', b'"global_optimum_claimed":true'
                ),
            )
        with pytest.raises(QuantumAcceptanceError, match="global optimum"):
            validate_shot_noise_spec(spec(global_optimum_claimed=True))

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"alpha_total": 0.31}, "sum exactly"),
            ({"alpha_cluster": 1e-60, "alpha_shot": 0.3}, "more precision"),
            ({"alpha_cluster": 0.0, "alpha_shot": 0.3}, "above zero"),
            ({"alpha_shot": None}, "alpha_shot"),
            ({"shots_per_pauli_group": 255}, "256"),
            ({"exact_solver_tolerance": 0.1}, "solver tolerance"),
            ({"alpha_total": 1.0}, r"\(0, 1\)"),
            ({"max_mitigation_overhead_gamma": 1.5}, "1 without mitigation"),
            (
                {"mitigation_enabled": True, "max_mitigation_overhead_gamma": 0.9},
                "at least 1",
            ),
            ({"min_physical_margin": -0.1}, "min_physical_margin"),
            ({"shot_noise_floor_multiplier": -1.0}, "multiplier"),
            ({"max_two_qubit_gate_depth": -1}, "depth"),
            ({"execution_mode": "annealer"}, "execution_mode"),
            ({"optimisation_sense": "maximise_energy"}, "optimisation_sense"),
            ({"schema": "spec.v2"}, "schema"),
            ({"mitigation_enabled": 1}, "mitigation_enabled"),
            ({"certifies_ansatz_infimum": None}, "certifies"),
        ],
        ids=[
            "TQV-4-sum",
            "inexact",
            "zero-alpha",
            "no-shot-alpha",
            "shots",
            "tolerance",
            "alpha-total",
            "gamma-cap",
            "gamma-below-1",
            "margin",
            "multiplier",
            "depth",
            "mode",
            "sense",
            "schema",
            "mitigation",
            "infimum",
        ],
    )
    def test_sampled_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """A sampled spec splits its error rate exactly and uses enough shots."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            validate_shot_noise_spec(spec(**changes))

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"exact_solver_tolerance": None}, "exact_solver_tolerance"),
            (
                {"mitigation_enabled": True, "max_mitigation_overhead_gamma": 1.5},
                "no mitigation",
            ),
            ({"shots_per_pauli_group": 256}, "no shots"),
            ({"exact_solver_tolerance": -0.1}, "negative"),
            ({"alpha_cluster": 0.01}, "does not split"),
        ],
        ids=[
            "TQV-5-tolerance",
            "TQV-5-mitigation",
            "shots",
            "negative-tolerance",
            "split",
        ],
    )
    def test_exact_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """An exact statevector uses no shots, no mitigation and a tolerance."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            validate_shot_noise_spec(spec("exact_statevector", **changes))

    def test_baseline_digest_and_type(self) -> None:
        """Q-5 names the classical baseline; other records are refused."""
        with pytest.raises(MalformedDigestError):
            validate_shot_noise_spec(spec(classical_baseline_digest="x"))
        with pytest.raises(QuantumAcceptanceError):
            validate_shot_noise_spec(SEEDS["exact_statevector"])  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestSeeds:
    """Q-7: the seeds each mode sets, all distinct 63-bit integers."""

    @pytest.mark.parametrize("mode", MODES)
    def test_valid_seeds_round_trip(self, mode: str) -> None:
        """Each mode's seeds validate and decode."""
        item = SEEDS[mode]
        assert validate_quantum_seeds(item, spec(mode)) == item
        assert (
            from_canonical_json_bytes(
                QuantumVariationalSeedTuple, canonical_json_bytes(item)
            )
            == item
        )

    def test_an_exact_run_may_set_a_layout_seed(self) -> None:
        """The transpiler layout seed is optional on an exact statevector."""
        assert validate_quantum_seeds(
            replace(SEEDS["exact_statevector"], transpiler_layout_seed=9),
            spec("exact_statevector"),
        )

    @pytest.mark.parametrize(
        ("mode", "changes", "fragment"),
        [
            (
                "exact_statevector",
                {"shot_sampling_seed": 3},
                "shot_sampling_seed must be None",
            ),
            (
                "exact_statevector",
                {"hardware_entropy_attestation_digest": digest_of("e")},
                "hardware_entropy",
            ),
            (
                "finite_shot_simulator",
                {"shot_sampling_seed": None},
                "shot_sampling_seed must be set",
            ),
            (
                "finite_shot_simulator",
                {"mitigation_calibration_seed": 7},
                "mitigation_calibration_seed must be None",
            ),
            (
                "hardware_qpu",
                {"shot_sampling_seed": 3},
                "shot_sampling_seed must be None",
            ),
            (
                "hardware_qpu",
                {"transpiler_layout_seed": None},
                "transpiler_layout_seed must be set",
            ),
            (
                "hardware_qpu",
                {"hardware_entropy_attestation_digest": None},
                "hardware_entropy_attestation_digest must be set",
            ),
            (
                "hardware_qpu",
                {"mitigation_calibration_seed": None},
                "mitigation_calibration_seed must be set",
            ),
            ("finite_shot_simulator", {"shot_sampling_seed": 1}, "active seeds"),
            ("finite_shot_simulator", {"parameter_init_seed": 2**63}, "2\\*\\*63"),
            (
                "finite_shot_simulator",
                {"problem_instance_shuffle_seed": -1},
                "problem_instance_shuffle_seed",
            ),
            (
                "finite_shot_simulator",
                {"parameter_init_seed": True},
                "parameter_init_seed",
            ),
            ("finite_shot_simulator", {"schema": "seeds.v2"}, "schema"),
        ],
    )
    def test_seed_refusals(
        self, mode: str, changes: dict[str, object], fragment: str
    ) -> None:
        """Each mode's seed pattern, range and distinctness is enforced."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            validate_quantum_seeds(replace(SEEDS[mode], **changes), spec(mode))  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields

    def test_the_shape_alone_checks_range_and_digest(self) -> None:
        """The codec's shape check refuses a bad digest or a non-tuple."""
        with pytest.raises(MalformedDigestError):
            validate_seed_tuple(
                replace(SEEDS["hardware_qpu"], hardware_entropy_attestation_digest="x")
            )
        with pytest.raises(QuantumAcceptanceError):
            validate_seed_tuple(spec())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestReceipts:
    """Q-6: each arm's receipt matches the spec; both arms share the budget."""

    @pytest.mark.parametrize("mode", MODES)
    def test_paired_receipts_pass(self, mode: str) -> None:
        """Matched receipts of two arms pass, and decode through the codec."""
        candidate, baseline = receipt(mode), receipt(mode, "baseline")
        assert validate_paired_receipts(candidate, baseline, spec(mode)) == spec(mode)
        assert (
            from_canonical_json_bytes(
                QuantumCircuitExecutionReceipt, canonical_json_bytes(candidate)
            )
            == candidate
        )

    @pytest.mark.parametrize(
        ("mode", "changes", "fragment"),
        [
            (
                "finite_shot_simulator",
                {"observed_shots_per_pauli_group": 1023},
                "matched budget",
            ),
            (
                "finite_shot_simulator",
                {"observed_two_qubit_gate_depth": 41},
                "gate depth",
            ),
            (
                "finite_shot_simulator",
                {"execution_mode": "hardware_qpu"},
                "another mode",
            ),
            (
                "finite_shot_simulator",
                {"mitigation_witness_digest": digest_of("m")},
                "mitigation_witness_digest",
            ),
            (
                "finite_shot_simulator",
                {"qpu_result_pack_digest": digest_of("q")},
                "qpu_result_pack_digest",
            ),
            ("hardware_qpu", {"observed_max_mitigation_gamma": 1.6}, "exceeds the cap"),
            (
                "hardware_qpu",
                {"mitigation_witness_digest": None},
                "mitigation_witness_digest",
            ),
            (
                "hardware_qpu",
                {"qpu_result_pack_digest": None},
                "qpu_result_pack_digest",
            ),
            (
                "hardware_qpu",
                {"transpilation_witness_digest": None},
                "transpilation witness",
            ),
        ],
        ids=[
            "TQV-6-shots",
            "TQV-6-depth",
            "mode",
            "stray-mitigation",
            "stray-qpu",
            "gamma",
            "no-mitigation",
            "no-qpu-pack",
            "no-transpilation",
        ],
    )
    def test_receipt_refusals(
        self, mode: str, changes: dict[str, object], fragment: str
    ) -> None:
        """A receipt off the spec's mode, budget or witnesses is refused."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            validate_execution_receipt(receipt(mode, **changes), spec(mode))

    def test_a_transpilation_witness_off_a_qpu_is_allowed(self) -> None:
        """A simulator may name its transpilation witness."""
        item = receipt(transpilation_witness_digest=digest_of("t"))
        assert validate_execution_receipt(item, spec()) == item

    def test_one_arm_twice_is_refused(self) -> None:
        """The candidate and the baseline are different arms."""
        with pytest.raises(QuantumAcceptanceError, match="different arms"):
            validate_paired_receipts(receipt(), receipt(), spec())

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"observed_max_mitigation_gamma": 0.99}, QuantumAcceptanceError),
            ({"observed_shots_per_pauli_group": -1}, QuantumAcceptanceError),
            ({"schema": "receipt.v2"}, QuantumAcceptanceError),
            ({"execution_mode": "annealer"}, QuantumAcceptanceError),
            ({"arm_content_digest": "x"}, MalformedDigestError),
            ({"transpilation_witness_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_shape_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """The shape check alone refuses malformed receipts."""
        with pytest.raises(error):
            validate_receipt_shape(receipt(**changes))
        with pytest.raises(QuantumAcceptanceError):
            validate_receipt_shape(spec())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


def group(
    coefficient: float = 1.0, mean: float = 0.0, shots: int = 256
) -> PauliGroupShots:
    """Return one Pauli group."""
    return PauliGroupShots(coefficient, mean, shots)


class TestShotNoise:
    """Q-1 and Q-2, pinned by hand-computed cases, with named domain refusals."""

    def test_q1_hand_computed(self) -> None:
        """0.25 x 0.96 / 256 = 0.0009375; with gamma 2 and a unit group, 4 x (0.0009375 + 1/256) = 0.019375."""
        assert shot_variance(1.0, [group(0.5, 0.2, 257)]) == pytest.approx(
            0.0009375, rel=1e-12
        )
        assert shot_variance(
            2.0, [group(0.5, 0.2, 257), group(-1.0, 0.0, 257)]
        ) == pytest.approx(0.019375, rel=1e-12)

    def test_q2_hand_computed(self) -> None:
        """sqrt(2 ln(40) / 256) = 0.169762689..."""
        assert hoeffding_radius(1.0, [group()], 0.05) == pytest.approx(
            0.16976268946757744, rel=1e-12
        )
        assert hoeffding_radius(2.0, [group()], 0.05) == pytest.approx(
            2 * 0.16976268946757744, rel=1e-12
        )
        assert hoeffding_radius(1.0, [group(coefficient=0.5)], 0.05) == pytest.approx(
            0.08488134473378872, rel=1e-12
        )

    @pytest.mark.parametrize(
        ("gamma", "groups", "alpha", "fragment"),
        [
            (1.0, [group(shots=255)], 0.05, "shots"),
            (1.0, [group(mean=1.1)], 0.05, r"\[-1, 1\]"),
            (0.9, [group()], 0.05, "at least 1"),
            (1.0, [group(coefficient=math.inf)], 0.05, "coefficient"),
            (1.0, [group()], 0.0, r"\(0, 1\)"),
            (1.0, [group()], 1.0, r"\(0, 1\)"),
            (1.0, [], 0.05, "at least one Pauli group"),
            (1.0, [(1.0, 0.0, 256)], 0.05, "PauliGroupShots"),
            (math.nan, [group()], 0.05, "gamma"),
        ],
    )
    def test_domain_refusals(
        self, gamma: float, groups: list[PauliGroupShots], alpha: float, fragment: str
    ) -> None:
        """Each domain violation is refused by name, never returned as NaN."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            hoeffding_radius(gamma, groups, alpha)
        if alpha == 0.05:
            with pytest.raises(QuantumAcceptanceError, match=fragment):
                shot_variance(gamma, groups)

    def test_paired_radii_hand_computed(self) -> None:
        """Independent: sqrt(2 ln 40 x 5 / 256); shared: 3 x sqrt(2 ln 80 / 256)."""
        pair = [PairedShotInstance(1.0, 2.0, (group(),), (group(mean=0.5),))]
        at_05 = spec(alpha_total=0.1, alpha_cluster=0.05, alpha_shot=0.05)
        assert paired_shot_radius(
            pair, at_05, shared_calibration=False
        ) == pytest.approx(0.3796009136926907, rel=1e-12)
        assert paired_shot_radius(
            pair, at_05, shared_calibration=True
        ) == pytest.approx(0.5550776952377994, rel=1e-12)
        assert paired_shot_radius(
            pair * 2, at_05, shared_calibration=False
        ) == pytest.approx(math.sqrt(2 * math.log(40) * 10 / 256) / 2, rel=1e-12)
        assert paired_shot_radius(
            pair * 2, at_05, shared_calibration=True
        ) == pytest.approx(0.5973676915297884, rel=1e-12)

    def test_c2_the_level_is_the_specs_shot_share(self) -> None:
        """The radius uses alpha_shot, never a caller's alpha or alpha_total."""
        pair = [PairedShotInstance(1.0, 2.0, (group(),), (group(mean=0.5),))]
        default = paired_shot_radius(pair, spec(), shared_calibration=False)
        assert default == pytest.approx(
            math.sqrt(2 * math.log(2 / 0.2) * 5 / 256), rel=1e-12
        )
        with pytest.raises(QuantumAcceptanceError, match="no shot radius"):
            paired_shot_radius(
                pair, spec("exact_statevector"), shared_calibration=False
            )

    def test_paired_standard_error_hand_computed(self) -> None:
        """sqrt(0.0009375 + 0.25 / 256) = 0.04375."""
        pair = [
            PairedShotInstance(
                1.0, 1.0, (group(0.5, 0.2, 257),), (group(0.5, 0.0, 257),)
            )
        ]
        assert paired_shot_standard_error(pair) == pytest.approx(0.04375, rel=1e-12)
        assert paired_shot_standard_error(pair * 2) == pytest.approx(
            0.030935921676911456, rel=1e-12
        )

    @pytest.mark.parametrize(
        ("instances", "fragment"),
        [
            ([], "at least one instance"),
            ([group()], "PairedShotInstance"),
            ([PairedShotInstance(1.0, 1.0, (group(),), (group(shots=512),))], "share"),
            ([PairedShotInstance(1.0, 0.5, (group(),), (group(),))], "at least 1"),
        ],
    )
    def test_instance_refusals(
        self, instances: list[PairedShotInstance], fragment: str
    ) -> None:
        """Paired instances are non-empty, typed, share their groups, and have overheads >= 1."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            paired_shot_radius(instances, spec(), shared_calibration=False)
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            paired_shot_standard_error(instances)


class TestAdmissionGate:
    """Case A and Case B, compared exactly, and the composite bound."""

    def test_the_composite_bound(self) -> None:
        """0.5 - 0.125 - 0.25 = 0.125; radii are finite and not negative."""
        assert composite_lower_bound(0.5, 0.125, 0.25) == 0.125
        with pytest.raises(QuantumAcceptanceError, match="negative"):
            composite_lower_bound(0.5, -0.1, 0.25)
        with pytest.raises(QuantumAcceptanceError, match="mean_improvement"):
            composite_lower_bound(math.inf, 0.1, 0.25)

    def test_case_a_spans_three_tolerances(self) -> None:
        """An exact statevector needs max(margin, 3 x 0.125) = 0.375."""
        exact = spec("exact_statevector")
        assert admits_quantum_arm(0.375, exact, shot_standard_error=None)
        assert not admits_quantum_arm(0.374, exact, shot_standard_error=None)
        with pytest.raises(QuantumAcceptanceError, match="no shot standard error"):
            admits_quantum_arm(0.5, exact, shot_standard_error=0.1)

    def test_case_b_spans_the_noise_floor_multiple(self) -> None:
        """A sampled arm needs max(margin, 3 x SE)."""
        assert admits_quantum_arm(0.375, spec(), shot_standard_error=0.125)
        assert not admits_quantum_arm(0.3749, spec(), shot_standard_error=0.125)
        assert not admits_quantum_arm(
            0.4, spec(min_physical_margin=0.5), shot_standard_error=0.125
        )
        for error, fragment in ((None, "shot_standard_error"), (-0.1, "negative")):
            with pytest.raises(QuantumAcceptanceError, match=fragment):
                admits_quantum_arm(0.5, spec(), shot_standard_error=error)

    def test_the_gate_never_rounds_in_the_arms_favour(self) -> None:
        """3.0 x 0.3 rounds down to 0.8999999999999999 in floats; exactly it is larger."""
        tight = spec(shot_noise_floor_multiplier=3.0)
        assert 3.0 * 0.3 >= 3.0 * 0.3
        assert not admits_quantum_arm(3.0 * 0.3, tight, shot_standard_error=0.3)

    def test_improvement_follows_the_sense(self) -> None:
        """Lower energy, or higher fidelity, is an improvement."""
        assert paired_improvement(1.0, 0.75, spec()) == 0.25
        assert paired_improvement(0.75, 1.0, spec("hardware_qpu")) == 0.25
        with pytest.raises(QuantumAcceptanceError, match="baseline_estimate"):
            paired_improvement(math.nan, 1.0, spec())


def barren(**changes: object) -> BarrenPlateauWitnessReceipt:
    """Return a passing barren-plateau witness with ``changes`` applied."""
    base = BarrenPlateauWitnessReceipt(
        "barren-plateau-witness-receipt.v1",
        digest_of("companion"),
        digest_of("pool"),
        64,
        0.02,
        0.01,
        True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def parity(**changes: object) -> SpikingLoweringParityReport:
    """Return a passing spiking parity report with ``changes`` applied."""
    base = SpikingLoweringParityReport(
        "spiking-lowering-parity-report.v1",
        digest_of("checkpoint"),
        digest_of("numeric"),
        0,
        "subtractive",
        "fast_sigmoid",
        0.001,
        100,
        0.5,
        0.98,
        True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


class TestWitnesses:
    """The barren-plateau preflight and TSN-3's spiking parity."""

    def test_valid_witnesses_round_trip(self) -> None:
        """Passing witnesses, and a failed preflight that says so, decode."""
        for item in (barren(), barren(min_observed_grad_variance=0.005, passed=False)):
            assert (
                from_canonical_json_bytes(
                    BarrenPlateauWitnessReceipt, canonical_json_bytes(item)
                )
                == item
            )
        failed = parity(passed=False)
        assert (
            from_canonical_json_bytes(
                SpikingLoweringParityReport, canonical_json_bytes(failed)
            )
            == failed
        )
        assert require_spiking_parity_passed(parity()) == parity()

    @pytest.mark.parametrize(
        "changes",
        [
            {"passed": False},
            {"min_observed_grad_variance": 0.005},
            {"min_observed_grad_variance": -0.1},
            {"threshold_grad_variance": 0.0},
            {"sampled_parameter_points": 0},
            {"schema": "bp.v2"},
        ],
    )
    def test_barren_refusals(self, changes: dict[str, object]) -> None:
        """The verdict follows the numbers; every field is checked."""
        with pytest.raises(QuantumAcceptanceError):
            validate_barren_plateau_receipt(barren(**changes))
        with pytest.raises(QuantumAcceptanceError):
            validate_barren_plateau_receipt(parity())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    def test_tsn3_a_failed_parity_is_refused(self) -> None:
        """A report that did not pass is a valid record but no witness."""
        assert validate_spiking_lowering_parity(parity(passed=False)).passed is False
        with pytest.raises(SpikingParityError, match="did not pass"):
            require_spiking_parity_passed(parity(passed=False))

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"blocking_underflow_or_overflow_count": 1}, "under- or overflow"),
            ({"blocking_underflow_or_overflow_count": False}, "under- or overflow"),
            ({"reset_semantics": "soft"}, "reset_semantics"),
            ({"dt_seconds": 0.0}, "dt_seconds"),
            ({"simulation_steps_t": 0}, "simulation_steps_t"),
            ({"victor_purpura_or_van_rossum_distance": -0.1}, "distance"),
            ({"classification_agreement_rate": 1.1}, "agreement"),
            ({"surrogate_family": ""}, "surrogate_family"),
            ({"passed": 1}, "passed"),
            ({"schema": "parity.v2"}, "schema"),
        ],
    )
    def test_parity_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """TSN-3: a blocking under- or overflow, or any malformed field, is refused."""
        with pytest.raises(SpikingParityError, match=fragment):
            validate_spiking_lowering_parity(parity(**changes))
        with pytest.raises(SpikingParityError):
            validate_spiking_lowering_parity(barren())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        with pytest.raises(MalformedDigestError):
            validate_spiking_lowering_parity(parity(checkpoint_content_digest="x"))


class TestDomainEdges:
    """R-1: every result is finite, or a named refusal; never NaN, inf or OverflowError."""

    @pytest.mark.parametrize(
        ("call", "fragment"),
        [
            (lambda: shot_variance(1.0, [group(coefficient=1e200)]), "shot variance"),
            (lambda: shot_variance(1e200, [group()]), "shot variance"),
            (
                lambda: shot_variance(1e150, [group(coefficient=1e154)] * 2),
                "shot variance",
            ),
            (lambda: shot_variance(1.0, [group(shots=10**400)]), "2\\*\\*53"),
            (
                lambda: hoeffding_radius(1.0, [group(coefficient=1e200)], 0.05),
                "Hoeffding radius",
            ),
            (lambda: hoeffding_radius(1.0, [group()], 5e-324), "Hoeffding radius"),
            (lambda: hoeffding_radius(1.0, [group(shots=10**400)], 0.05), "2\\*\\*53"),
            (
                lambda: hoeffding_radius(1.0, [group(coefficient=1e154)] * 500, 0.05),
                "overflows",
            ),
            (
                lambda: shot_variance(1.0, [group(coefficient=1e200, mean=1.0)]),
                "shot variance",
            ),
            (
                lambda: paired_shot_standard_error(
                    [
                        PairedShotInstance(
                            1.0,
                            1.0,
                            (group(coefficient=1e200),),
                            (group(coefficient=1e200),),
                        )
                    ]
                ),
                "shot variance",
            ),
            (lambda: composite_lower_bound(-1e308, 1e308, 1e308), "overflows"),
            (lambda: paired_improvement(1e308, -1e308, spec()), "overflows"),
        ],
        ids=[
            "V1",
            "V2",
            "V3",
            "V4",
            "H1",
            "H3",
            "H4",
            "fsum",
            "inf-times-zero",
            "S1",
            "C1",
            "improvement",
        ],
    )
    def test_overflow_is_refused_by_name(self, call: Any, fragment: str) -> None:
        """Each probe of the review is refused with a QuantumAcceptanceError."""
        with pytest.raises(QuantumAcceptanceError, match=fragment):
            call()

    def test_a_zero_coefficient_gives_a_zero_radius(self) -> None:
        """H2 and P2: a zero weight is a zero radius, not NaN, even at a tiny level."""
        assert hoeffding_radius(1.0, [group(coefficient=0.0)], 5e-324) == 0.0
        zero = [
            PairedShotInstance(
                1.0, 1.0, (group(coefficient=0.0),), (group(coefficient=0.0),)
            )
        ]
        assert paired_shot_radius(zero, spec(), shared_calibration=False) == 0.0
        assert paired_shot_radius(zero, spec(), shared_calibration=True) == 0.0

    def test_p1_an_underflowing_union_level_is_refused_by_name(self) -> None:
        """alpha_shot / 2N underflowing to zero has its own message."""
        tiny = spec(alpha_total=1e-323, alpha_cluster=5e-324, alpha_shot=5e-324)
        pair = [PairedShotInstance(1.0, 1.0, (group(),), (group(),))]
        with pytest.raises(QuantumAcceptanceError, match="underflows to zero"):
            paired_shot_radius(pair, tiny, shared_calibration=True)
        with pytest.raises(QuantumAcceptanceError, match="paired shot radius"):
            paired_shot_radius(pair, tiny, shared_calibration=False)
        zero = [
            PairedShotInstance(
                1.0, 1.0, (group(coefficient=0.0),), (group(coefficient=0.0),)
            )
        ]
        assert paired_shot_radius(zero, tiny, shared_calibration=False) == 0.0

    @pytest.mark.parametrize(
        "call",
        [
            lambda: hoeffding_radius(1.0, [group(coefficient=1e-160)], 0.05),
            lambda: hoeffding_radius(
                1e300, [group(coefficient=1.0), group(coefficient=1e-160)], 0.05
            ),
            lambda: shot_variance(1.0, [group(coefficient=1e-160)]),
            lambda: paired_shot_radius(
                [
                    PairedShotInstance(
                        1.0,
                        1.0,
                        (group(coefficient=1e-160),),
                        (group(coefficient=1e-160),),
                    )
                ],
                spec(),
                shared_calibration=False,
            ),
        ],
        ids=["radius", "radius-mixed", "variance", "paired"],
    )
    def test_n6_an_underflowing_nonzero_term_is_refused(self, call: Any) -> None:
        """A nonzero coefficient never yields a silent zero (or subnormal) term."""
        with pytest.raises(QuantumAcceptanceError, match="underflows"):
            call()

    def test_n6_exact_zeros_stay_zero(self) -> None:
        """A zero coefficient, or a group with mean exactly +-1, adds an exact zero."""
        assert shot_variance(1.0, [group(coefficient=1e-160, mean=1.0)]) == 0.0
        assert shot_variance(1.0, [group(coefficient=0.0)]) == 0.0
        normal = sys.float_info.min * 256
        assert hoeffding_radius(1.0, [group(coefficient=math.sqrt(normal))], 0.05) > 0

    def test_n7_a_non_finite_rounded_bound_is_refused(self) -> None:
        """At mean = -max a tiny radius rounds the bound down to -inf: refused."""
        with pytest.raises(QuantumAcceptanceError, match="composite lower bound"):
            composite_lower_bound(-sys.float_info.max, 1e-300, 0.0)

    def test_the_composite_is_rounded_down(self) -> None:
        """0.1 - 0.2 - 0.7 rounds up in floats; the bound is the float just below exact."""
        exact = Fraction(0.1) - Fraction(0.2) - Fraction(0.7)
        assert Fraction(0.1 - 0.2 - 0.7) > exact
        bound = composite_lower_bound(0.1, 0.2, 0.7)
        assert Fraction(bound) <= exact < Fraction(math.nextafter(bound, math.inf))
        assert composite_lower_bound(0.5, 0.125, 0.25) == 0.125


class TestNoiseFloorMinimum:
    """C-4: a sampled arm's noise-floor multiplier is at least 3."""

    @pytest.mark.parametrize("mode", ["finite_shot_simulator", "hardware_qpu"])
    def test_sampled_modes_need_three(self, mode: str) -> None:
        """3.0 passes; just below is refused."""
        assert validate_shot_noise_spec(spec(mode, shot_noise_floor_multiplier=3.0))
        with pytest.raises(QuantumAcceptanceError, match="at least 3"):
            validate_shot_noise_spec(
                spec(mode, shot_noise_floor_multiplier=math.nextafter(3.0, 0.0))
            )

    def test_an_exact_statevector_ignores_the_multiplier(self) -> None:
        """The exact mode's floor is three tolerances, whatever the multiplier."""
        assert validate_shot_noise_spec(
            spec("exact_statevector", shot_noise_floor_multiplier=0.0)
        )
