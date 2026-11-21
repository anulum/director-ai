# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Port, Post-Swap and Submission Guard Tests

"""The ports' shapes, the typed spend refusals, post-swap records and the guard."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any

import pytest

from director_ai.model_lifecycle import (
    OBSERVATION_OUTCOMES,
    BarrenPlateauWitnessReceipt,
    CloudComputeDriverPort,
    ComputeAndSpendPort,
    DeploymentRecordError,
    LiveReservationProjection,
    MalformedDigestError,
    ModelLifecycleError,
    PaidRunApprovalRecord,
    PostSwapEvidence,
    PostSwapVerification,
    QuantumAcceptanceError,
    SpendBindingError,
    SpendReservationBinding,
    SpendReservationRefusedError,
    SpendSettlementRefusedError,
    canonical_json_bytes,
    from_canonical_json_bytes,
    guard_job_submission,
    ports,
    record_digest,
    validate_post_swap_evidence,
    validate_post_swap_verification,
)
from tests.model_lifecycle_support import digest_of
from tests.test_model_lifecycle_recipes import BASE, recipe
from tests.test_model_lifecycle_spend import binding, estimate, live


def members(port: type) -> list[str]:
    """Return a port's public method and property names."""
    return sorted(name for name in vars(port) if not name.startswith("_"))


class TestPortShapes:
    """Every port is a runtime-checkable protocol of its methods."""

    @pytest.mark.parametrize("name", ports.__all__)
    def test_a_complete_adapter_is_an_instance(self, name: str) -> None:
        """An object with every member is an instance; one missing a member is not."""
        port = getattr(ports, name)
        names = members(port)
        assert names, name
        complete = type("Adapter", (), {member: lambda self: None for member in names})
        assert isinstance(complete(), port)
        partial = type(
            "Adapter", (), {member: lambda self: None for member in names[1:]}
        )
        assert not isinstance(partial(), port)

    def test_the_facade_lists_sixteen_ports_and_no_readiness_port(self) -> None:
        """Sixteen ports are defined; the operator-readiness port is not the kernel's."""
        assert len(ports.__all__) == 16
        assert "OperatorReadinessPort" not in ports.__all__
        assert members(CloudComputeDriverPort) == [
            "cancel_job",
            "driver_name",
            "estimate_cost",
            "poll_status",
            "submit_job",
        ]


class TestSpendRefusals:
    """B2-5: refusals are typed and keyword-only."""

    def test_reservation_refusal(self) -> None:
        """The pool and the ledger's reason are kept."""
        error = SpendReservationRefusedError(
            pool_id="pool-cloud", ledger_reason="over budget"
        )
        assert (error.pool_id, error.ledger_reason) == ("pool-cloud", "over budget")
        assert isinstance(error, ModelLifecycleError)
        assert "pool-cloud" in str(error) and "over budget" in str(error)

    def test_settlement_refusal(self) -> None:
        """The pool, reservation and reason are kept."""
        error = SpendSettlementRefusedError(
            pool_id="p", reservation_id="r", ledger_reason="closed"
        )
        assert (error.pool_id, error.reservation_id, error.ledger_reason) == (
            "p",
            "r",
            "closed",
        )
        assert "'p'/'r'" in str(error)

    def test_refusals_take_keywords_only(self) -> None:
        """Positional arguments are refused."""
        with pytest.raises(TypeError):
            SpendReservationRefusedError("pool", "reason")  # type: ignore[misc]
        with pytest.raises(TypeError):
            SpendSettlementRefusedError("pool", "res", "reason")  # type: ignore[misc]


def evidence(**changes: object) -> PostSwapEvidence:
    """Return complete evidence of a swap from generation 3 to 4."""
    base = PostSwapEvidence(
        "/srv/chain",
        "/srv/start.json",
        "rev-7",
        3,
        4,
        4,
        digest_of("slot"),
        digest_of("start"),
        "1763700000000000000",
        4242,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def verification(
    observation: str = "published_and_converged", /, **changes: object
) -> PostSwapVerification:
    """Return a verification with the effect and result its observation fixes."""
    effect, result = OBSERVATION_OUTCOMES[observation]
    fields: dict[str, Any] = {
        "schema": "post-swap-verification.v1",
        "operation_id": "op-1",
        "attempt_id": "attempt-1",
        "effect": effect,
        "result": result,
        "observation": observation,
        "evidence": evidence(),
    }
    return replace(PostSwapVerification(**fields), **changes)  # type: ignore[arg-type]


class TestPostSwapRecords:
    """C-2: each observation fixes one effect and one result."""

    @pytest.mark.parametrize("observation", sorted(OBSERVATION_OUTCOMES))
    def test_every_row_validates_and_round_trips(self, observation: str) -> None:
        """Each of the eleven rows is valid with its own effect and result."""
        item = verification(observation)
        assert validate_post_swap_verification(item) == item
        assert (
            from_canonical_json_bytes(PostSwapVerification, canonical_json_bytes(item))
            == item
        )

    def test_the_table_is_the_contracts(self) -> None:
        """Eleven observations; one verified, one unknown."""
        assert len(OBSERVATION_OUTCOMES) == 11
        assert [
            o for o, pair in OBSERVATION_OUTCOMES.items() if pair[1] == "verified"
        ] == ["published_and_converged"]
        assert OBSERVATION_OUTCOMES["chain_unreadable"] == ("unknown", "not_checked")

    @pytest.mark.parametrize(
        "observation", ["published_and_converged", "chain_unreadable", "not_published"]
    )
    def test_every_other_pair_is_refused(self, observation: str) -> None:
        """No other (effect, result) is accepted for an observation."""
        for effect, result in product(
            ("unknown", "absent", "observed"),
            ("verified", "not_verified", "not_checked"),
        ):
            if (effect, result) == OBSERVATION_OUTCOMES[observation]:
                continue
            with pytest.raises(DeploymentRecordError, match="fixes the effect"):
                validate_post_swap_verification(
                    verification(observation, effect=effect, result=result)
                )

    @pytest.mark.parametrize(
        "missing", ["slot_record_sha256", "start_record_sha256", "observed_start_pid"]
    )
    def test_a_verified_swap_names_its_evidence(self, missing: str) -> None:
        """A verified swap needs both digests and the start pid; others need not."""
        with pytest.raises(DeploymentRecordError, match="verified swap"):
            validate_post_swap_verification(
                verification(evidence=evidence(**{missing: None}))
            )
        assert validate_post_swap_verification(
            verification("not_published", evidence=evidence(**{missing: None}))
        )

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"target_generation": 5}, DeploymentRecordError),
            ({"base_generation": -1}, DeploymentRecordError),
            ({"chain_head_generation": -1}, DeploymentRecordError),
            ({"start_record_mtime_ns": "17e9"}, DeploymentRecordError),
            ({"start_record_mtime_ns": "١٢"}, DeploymentRecordError),
            ({"observed_start_pid": 0}, DeploymentRecordError),
            ({"chain_dir": ""}, DeploymentRecordError),
            ({"slot_record_sha256": "x"}, MalformedDigestError),
        ],
    )
    def test_evidence_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """The target is the next generation; every field is checked."""
        with pytest.raises(error):
            validate_post_swap_evidence(evidence(**changes))
        assert validate_post_swap_evidence(
            evidence(chain_head_generation=None, start_record_mtime_ns=None)
        )

    @pytest.mark.parametrize(
        "changes",
        [
            {"schema": "swap.v2"},
            {"operation_id": ""},
            {"attempt_id": ""},
            {"observation": "unknown_row"},
            {"effect": "maybe"},
            {"result": "partly"},
        ],
    )
    def test_verification_refusals(self, changes: dict[str, object]) -> None:
        """Every field and literal is checked."""
        with pytest.raises(DeploymentRecordError):
            validate_post_swap_verification(verification(**changes))
        with pytest.raises(DeploymentRecordError):
            validate_post_swap_verification(evidence())  # type: ignore[arg-type]
        with pytest.raises(DeploymentRecordError):
            validate_post_swap_evidence(verification())  # type: ignore[arg-type]


@dataclass
class FakeSpendPort:
    """A spend port whose live view can change after the caller looked."""

    projection: LiveReservationProjection
    queries: list[SpendReservationBinding] = field(default_factory=list)

    def reserve(self, **_: object) -> str:
        """Not used by the guard."""
        raise AssertionError("the guard never reserves")

    def verify_live_reservation(
        self, *, binding: SpendReservationBinding
    ) -> LiveReservationProjection:
        """Return the ledger's current view and record the query."""
        self.queries.append(binding)
        return self.projection

    def settle(self, **_: object) -> None:
        """Not used by the guard."""
        raise AssertionError("the guard never settles")


@dataclass
class GuardedDriver:
    """A driver that calls the guard last and submits with its projection."""

    spend_port: FakeSpendPort
    sent: list[LiveReservationProjection] = field(default_factory=list)

    def submit(self, **kwargs: Any) -> None:
        """Guard, then send exactly once with the guard's projection."""
        self.sent.append(guard_job_submission(self.spend_port, **kwargs))


def job(**changes: Any) -> dict[str, Any]:
    """Return a valid local job's guard inputs with ``changes`` applied."""
    plan = changes.pop("recipe", recipe())
    cost = changes.pop("cost_estimate", estimate(recipe_digest=record_digest(plan)))
    base: dict[str, Any] = {
        "recipe": plan,
        "funding_kind": plan.funding_kind,
        "cost_estimate": cost,
        "spend_binding": binding(),
        "paid_approval": None,
        "barren_plateau_receipt": None,
    }
    return {**base, **changes}


def preflight(**changes: object) -> BarrenPlateauWitnessReceipt:
    """Return a passing preflight bound to the default family contract."""
    base = BarrenPlateauWitnessReceipt(
        "barren-plateau-witness-receipt.v1",
        digest_of("family"),
        digest_of("pool"),
        64,
        0.02,
        0.01,
        True,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


VARIATIONAL = {
    "family": "quantum_variational",
    "adaptation_mode": "full_safetensors",
    "base_checkpoint_digests": (BASE,),
    "architecture_spec": None,
}
QUANTUM = recipe(**VARIATIONAL, funding_kind="monthly_cloud_credit")
PAID = recipe(funding_kind="per_run_owner_approved_paid")


def approval(plan: Any = PAID, **changes: object) -> PaidRunApprovalRecord:
    """Return the owner's approval of the paid job's estimate and binding."""
    base = PaidRunApprovalRecord(
        "paid-run-approval.v1",
        record_digest(estimate(recipe_digest=record_digest(plan))),
        "owner-ok-1",
        "pool-cloud",
        binding(),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestSubmissionGuard:
    """B2-3, TSP-1 and TQV-7: the driver's last check before any I/O."""

    def test_a_valid_job_returns_the_projection_it_checked(self) -> None:
        """The guard queries the port itself and returns that very projection."""
        port = FakeSpendPort(live())
        driver = GuardedDriver(port)
        driver.submit(**job())
        assert driver.sent == [port.projection] and driver.sent[0] is port.projection
        assert port.queries == [binding()]
        assert isinstance(port, ComputeAndSpendPort)

    def test_tsp1_a_reservation_settled_after_the_caller_looked_is_refused(
        self,
    ) -> None:
        """The caller saw it open; the ledger now says settled; nothing is sent."""
        caller_snapshot = live()
        assert caller_snapshot.state == "open"
        port = FakeSpendPort(live(state="settled"))
        driver = GuardedDriver(port)
        with pytest.raises(SpendBindingError, match="found, open"):
            driver.submit(**job())
        assert driver.sent == [] and port.queries == [binding()]

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"funding_kind": "monthly_cloud_credit"}, "not its recipe's"),
            ({"cost_estimate": estimate()}, "another recipe"),
            ({"funding_kind": "donated"}, "funding_kind"),
            ({"paid_approval": approval(recipe())}, "only a paid run"),
        ],
    )
    def test_binding_refusals(self, changes: dict[str, Any], fragment: str) -> None:
        """The funding and estimate are the recipe's; only a paid run has an approval."""
        port = FakeSpendPort(live())
        with pytest.raises(SpendBindingError, match=fragment):
            GuardedDriver(port).submit(**job(**changes))
        assert port.queries == []

    def test_a_paid_run_needs_its_own_approval(self) -> None:
        """The approval is of this estimate and this binding."""
        port = FakeSpendPort(live())
        assert guard_job_submission(port, **job(recipe=PAID, paid_approval=approval()))
        for wrong, fragment in (
            (None, "needs the owner"),
            (approval(cost_estimate_digest=digest_of("x")), "another cost estimate"),
            (
                approval(spend_binding=binding(idempotency_key="idem-2")),
                "another spend binding",
            ),
        ):
            with pytest.raises(SpendBindingError, match=fragment):
                guard_job_submission(port, **job(recipe=PAID, paid_approval=wrong))

    def test_tqv7_a_credit_funded_variational_job_needs_a_passed_preflight(
        self,
    ) -> None:
        """Missing, failed or foreign preflights are refused before any query."""
        port = FakeSpendPort(live())
        cases = (
            (None, "needs its preflight"),
            (preflight(min_observed_grad_variance=0.005, passed=False), "did not pass"),
            (preflight(companion_digest=digest_of("other")), "another family contract"),
            (preflight(passed=False), "passed must say"),
        )
        for receipt, fragment in cases:
            with pytest.raises(QuantumAcceptanceError, match=fragment):
                guard_job_submission(
                    port, **job(recipe=QUANTUM, barren_plateau_receipt=receipt)
                )
        assert port.queries == []
        assert guard_job_submission(
            port, **job(recipe=QUANTUM, barren_plateau_receipt=preflight())
        )

    def test_tqv7_follows_the_family_and_the_funding(self) -> None:
        """A local variational job and a credit-funded language job need no preflight."""
        port = FakeSpendPort(live())
        local = recipe(**VARIATIONAL)
        assert guard_job_submission(port, **job(recipe=local))
        assert guard_job_submission(
            port, **job(recipe=recipe(funding_kind="monthly_cloud_credit"))
        )
        with pytest.raises(QuantumAcceptanceError, match="another family contract"):
            guard_job_submission(
                port,
                **job(
                    barren_plateau_receipt=preflight(companion_digest=digest_of("x"))
                ),
            )

    def test_malformed_inputs_are_refused_before_the_ledger_is_queried(self) -> None:
        """A bad recipe, binding or estimate never reaches the ledger."""
        port = FakeSpendPort(live())
        bad_recipe = recipe(dedup_report_digest="x")
        with pytest.raises(MalformedDigestError):
            guard_job_submission(port, **job(recipe=bad_recipe))
        with pytest.raises(SpendBindingError, match="task"):
            guard_job_submission(port, **job(spend_binding=binding(task="")))
        plan = recipe()
        greedy = estimate(
            recipe_digest=record_digest(plan), estimated_expected_cost="41.00"
        )
        with pytest.raises(SpendBindingError, match="exceeds the hard upper bound"):
            guard_job_submission(port, **job(cost_estimate=greedy))
        assert port.queries == []

    def test_the_live_cover_is_checked(self) -> None:
        """An uncovered reservation is refused after the query."""
        port = FakeSpendPort(live(remaining_exposure="1.00"))
        with pytest.raises(SpendBindingError, match="remaining exposure"):
            guard_job_submission(port, **job())
        assert port.queries == [binding()]
