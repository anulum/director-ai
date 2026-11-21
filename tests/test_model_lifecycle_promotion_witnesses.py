# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Witnesses at Promotion Tests

"""A promoted candidate carries its family's witnesses, passed, bound and pinned."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from director_ai.model_lifecycle import (
    FAMILY_WITNESSES,
    QUANTUM_WITNESSES,
    EnvelopeError,
    FamilyWitnesses,
    PromotionDecisionError,
    QuantumAcceptanceError,
    SpikingParityError,
    canonical_json_bytes,
    from_canonical_json_bytes,
    record_digest,
    validate_family_witnesses,
    validate_promotion_or_decline,
)
from tests.model_lifecycle_support import digest_of, edge
from tests.test_model_lifecycle_promotion_gate import (
    A_BASE,
    A_CAND,
    FAMILY,
    candidate_envelope,
    inputs,
)
from tests.test_model_lifecycle_quantum import barren, parity, receipt, spec

KINDS = ("language", "dynamics", "spiking", "quantum_variational")


def promote(kind: str, **witness_changes: Any) -> dict[str, Any]:
    """Return an admissible promotion of a ``kind`` study, witnesses changed."""
    report: dict[str, object] = {}
    if kind != "dynamics":
        report["dynamics_rollout_horizon_h"] = None
    return inputs(kind=kind, report_changes=report, witness_changes=witness_changes)


def refused(kwargs: dict[str, Any], fragment: str) -> None:
    """Assert the promotion is refused by name."""
    with pytest.raises(PromotionDecisionError, match=fragment):
        validate_promotion_or_decline(**kwargs)


def unpinned(kwargs: dict[str, Any], digest: str) -> dict[str, Any]:
    """Return ``kwargs`` with ``digest`` dropped from the promotion's witness pins."""
    pins = kwargs["pins"]
    kept = tuple(d for d in pins.witness_and_parity_digests if d != digest)
    assert len(kept) == len(pins.witness_and_parity_digests) - 1
    return {**kwargs, "pins": replace(pins, witness_and_parity_digests=kept)}


class TestEveryFamily:
    """Each family promotes with its own witnesses and no others."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_bound_promotion_of_each_family_passes(self, kind: str) -> None:
        """The candidate's envelope and its family's witnesses admit it."""
        kwargs = promote(kind)
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]

    def test_the_family_table(self) -> None:
        """Spiking needs its parity report, quantum its four, the rest none."""
        assert FAMILY_WITNESSES == {
            "spiking": ("spiking_parity_report",),
            "quantum_variational": QUANTUM_WITNESSES,
        }
        assert QUANTUM_WITNESSES == (
            "barren_plateau_receipt",
            "acceptance_spec",
            "candidate_receipt",
            "baseline_receipt",
        )

    def test_a_promotion_without_witnesses_is_refused(self) -> None:
        """A promotion names its candidate's family witnesses."""
        refused({**promote("dynamics"), "family_witnesses": None}, "family witnesses")

    def test_a_decline_does_not_read_them(self) -> None:
        """A decline promotes nothing, so it needs no witnesses."""
        kwargs = {**inputs(promote=False), "family_witnesses": None}
        assert validate_promotion_or_decline(**kwargs) == kwargs["decline"]

    @pytest.mark.parametrize(
        ("kind", "name", "value"),
        [
            (
                "language",
                "spiking_parity_report",
                parity(checkpoint_content_digest=A_CAND),
            ),
            ("dynamics", "acceptance_spec", spec()),
            (
                "spiking",
                "candidate_receipt",
                receipt("finite_shot_simulator", "arm-candidate"),
            ),
            ("quantum_variational", "spiking_parity_report", parity()),
        ],
        ids=["language", "dynamics", "spiking", "quantum"],
    )
    def test_a_stray_witness_is_refused(
        self, kind: str, name: str, value: object
    ) -> None:
        """A witness the family does not use is refused, not ignored."""
        refused(promote(kind, **{name: value}), f"a {kind} candidate carries no {name}")


class TestCandidateEnvelope:
    """The envelope is the decision's candidate, of the terms' family, pinned."""

    @pytest.mark.parametrize(
        ("envelope", "fragment"),
        [
            (
                replace(
                    candidate_envelope("dynamics"),
                    content_digest=A_BASE,
                    lineage_edges=(edge(digest_of("base"), A_BASE),),
                ),
                "not the decision's candidate",
            ),
            (candidate_envelope("language"), "another family than the study's"),
            (
                replace(
                    candidate_envelope("dynamics"),
                    family_contract_digest=digest_of("x"),
                ),
                "another family contract",
            ),
        ],
        ids=["candidate", "family", "contract"],
    )
    def test_an_unbound_envelope_is_refused(
        self, envelope: object, fragment: str
    ) -> None:
        """Each binding of the envelope is refused by name."""
        kwargs = promote("dynamics")
        refused(
            {
                **kwargs,
                "family_witnesses": replace(
                    kwargs["family_witnesses"], candidate_envelope=envelope
                ),
            },
            fragment,
        )

    def test_an_unpinned_envelope_is_refused(self) -> None:
        """The envelope is itself a pinned witness."""
        kwargs = promote("language")
        digest = record_digest(kwargs["family_witnesses"].candidate_envelope)
        refused(unpinned(kwargs, digest), "candidate envelope is not a pinned witness")


class TestSpiking:
    """TSN-3: the parity report passed, is the candidate's, and is pinned."""

    def test_a_missing_report_is_refused(self) -> None:
        """A spiking candidate needs its report."""
        refused(
            promote("spiking", spiking_parity_report=None), "needs its lowering parity"
        )

    def test_a_failed_report_is_refused(self) -> None:
        """A report that did not pass refuses the promotion."""
        failed = parity(checkpoint_content_digest=A_CAND, passed=False)
        with pytest.raises(SpikingParityError, match="did not pass"):
            validate_promotion_or_decline(
                **promote("spiking", spiking_parity_report=failed)
            )

    def test_a_report_of_another_checkpoint_is_refused(self) -> None:
        """The report is of the candidate."""
        refused(
            promote("spiking", spiking_parity_report=parity()), "another checkpoint"
        )

    def test_an_unpinned_report_is_refused(self) -> None:
        """The report is a pinned witness."""
        kwargs = promote("spiking")
        digest = record_digest(kwargs["family_witnesses"].spiking_parity_report)
        refused(unpinned(kwargs, digest), "parity report is not a pinned witness")

    def test_the_numeric_contract_is_a_pinned_companion(self) -> None:
        """The hardware numeric contract is one of the pins' companion contracts."""
        kwargs = promote("spiking")
        pins = replace(kwargs["pins"], secondary_family_contract_digests=())
        refused({**kwargs, "pins": pins}, "numeric contract is not a pinned companion")


class TestQuantum:
    """Q-5, Q-6 and the preflight: bound to the terms, passed and pinned."""

    @pytest.mark.parametrize("name", QUANTUM_WITNESSES)
    def test_a_missing_witness_is_refused(self, name: str) -> None:
        """Each of the four is required."""
        refused(
            promote("quantum_variational", **{name: None}), "needs its spec, receipts"
        )

    def test_another_spec_is_refused(self) -> None:
        """The spec is the one the terms froze."""
        other = spec(shot_noise_floor_multiplier=4.0)
        refused(promote("quantum_variational", acceptance_spec=other), "not the terms'")

    def test_unpaired_receipts_are_refused(self) -> None:
        """Q-6: both receipts match the spec and name different arms."""
        same = receipt(
            "finite_shot_simulator", "arm-candidate", observed_two_qubit_gate_depth=39
        )
        with pytest.raises(QuantumAcceptanceError, match="different arms"):
            validate_promotion_or_decline(
                **promote("quantum_variational", baseline_receipt=same)
            )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            (
                {"candidate_receipt": receipt("finite_shot_simulator", "other")},
                "another arm than the candidate",
            ),
            (
                {"baseline_receipt": receipt("finite_shot_simulator", "other")},
                "another arm than the terms' baseline",
            ),
            (
                {
                    "barren_plateau_receipt": barren(
                        companion_digest=FAMILY,
                        split_pool_digest=digest_of("p"),
                        min_observed_grad_variance=0.005,
                        passed=False,
                    )
                },
                "did not pass",
            ),
            (
                {"barren_plateau_receipt": barren(split_pool_digest=digest_of("p"))},
                "another family contract",
            ),
            (
                {
                    "barren_plateau_receipt": barren(
                        companion_digest=FAMILY, split_pool_digest=digest_of("p")
                    )
                },
                "another split pool",
            ),
        ],
        ids=[
            "candidate-arm",
            "baseline-arm",
            "preflight-failed",
            "preflight-contract",
            "preflight-pool",
        ],
    )
    def test_an_unbound_witness_is_refused(
        self, changes: dict[str, Any], fragment: str
    ) -> None:
        """Each binding of the receipts and the preflight is refused by name."""
        refused(promote("quantum_variational", **changes), fragment)

    @pytest.mark.parametrize(
        ("name", "what"),
        [
            ("acceptance_spec", "acceptance spec"),
            ("candidate_receipt", "candidate receipt"),
            ("baseline_receipt", "baseline receipt"),
            ("barren_plateau_receipt", "barren-plateau witness"),
        ],
    )
    def test_an_unpinned_witness_is_refused(self, name: str, what: str) -> None:
        """Pinning is checked for each of the four."""
        kwargs = promote("quantum_variational")
        digest = record_digest(getattr(kwargs["family_witnesses"], name))
        refused(unpinned(kwargs, digest), f"the {what} is not a pinned witness")

    def test_the_classical_baseline_is_pinned(self) -> None:
        """Q-5: the spec's classical baseline is a pinned witness."""
        kwargs = promote("quantum_variational")
        digest = kwargs["family_witnesses"].acceptance_spec.classical_baseline_digest
        refused(unpinned(kwargs, digest), "classical baseline is not a pinned witness")


class TestRecord:
    """The record's shape, and its codec registration."""

    def test_a_valid_record_is_returned_and_round_trips(self) -> None:
        """Validation returns the record, and decoding validates it again."""
        for kind in KINDS:
            witnesses = promote(kind)["family_witnesses"]
            assert validate_family_witnesses(witnesses) == witnesses
            raw = canonical_json_bytes(witnesses)
            assert from_canonical_json_bytes(FamilyWitnesses, raw) == witnesses

    def test_shape_refusals(self) -> None:
        """A wrong type or schema is refused by name."""
        witnesses = promote("dynamics")["family_witnesses"]
        with pytest.raises(PromotionDecisionError, match="FamilyWitnesses is required"):
            validate_family_witnesses(object())  # type: ignore[arg-type]
        with pytest.raises(PromotionDecisionError, match="schema"):
            validate_family_witnesses(replace(witnesses, schema="v0"))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            (
                {
                    "candidate_envelope": replace(
                        candidate_envelope("dynamics"),
                        lineage_edges=(edge(digest_of("base"), A_BASE),),
                    )
                },
                EnvelopeError,
            ),
            ({"spiking_parity_report": parity(dt_seconds=0.0)}, SpikingParityError),
            (
                {"barren_plateau_receipt": barren(sampled_parameter_points=0)},
                QuantumAcceptanceError,
            ),
            ({"acceptance_spec": spec(alpha_total=0.0)}, QuantumAcceptanceError),
            (
                {
                    "candidate_receipt": replace(
                        receipt(), observed_max_mitigation_gamma=0.5
                    )
                },
                QuantumAcceptanceError,
            ),
            (
                {
                    "baseline_receipt": replace(
                        receipt(), observed_max_mitigation_gamma=0.5
                    )
                },
                QuantumAcceptanceError,
            ),
        ],
        ids=[
            "envelope",
            "parity",
            "preflight",
            "spec",
            "candidate-receipt",
            "baseline-receipt",
        ],
    )
    def test_a_malformed_part_is_refused(
        self, changes: dict[str, Any], error: type[Exception]
    ) -> None:
        """The envelope and every witness given are validated."""
        witnesses = replace(promote("dynamics")["family_witnesses"], **changes)
        with pytest.raises(error):
            validate_family_witnesses(witnesses)
