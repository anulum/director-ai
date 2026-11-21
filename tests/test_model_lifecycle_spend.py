# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Spend, Job and Promotion Record Tests

"""Spend cover, cost estimates, job records and promotion record shapes."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from director_ai.model_lifecycle import (
    CloudJobCancelReceipt,
    CloudJobStatusReport,
    CloudJobSubmissionReceipt,
    CostEstimateRecord,
    LiveReservationProjection,
    MalformedDigestError,
    PaidRunApprovalRecord,
    PromotionDecisionError,
    PromotionDependencyPins,
    SandboxConfinementSpec,
    ServingMemoryBudget,
    SpendBindingError,
    SpendReservationBinding,
    StudyDeclineRecord,
    amount,
    canonical_json_bytes,
    from_canonical_json_bytes,
    validate_cost_estimate,
    validate_dependency_pins,
    validate_job_cancel,
    validate_job_status,
    validate_job_submission,
    validate_paid_run_approval,
    validate_sandbox_confinement,
    validate_serving_memory_budget,
    validate_spend_binding,
    validate_spend_reservation_live,
    validate_study_decline,
)
from tests.model_lifecycle_support import UTC, digest_of


def binding(**changes: object) -> SpendReservationBinding:
    """Return a valid binding with ``changes`` applied."""
    base = SpendReservationBinding(
        "spend-reservation-binding.v1",
        "pool-cloud",
        "res-1",
        "seat-1",
        "director-ai",
        "train",
        "adapter-sweep",
        "idem-1",
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def estimate(**changes: object) -> CostEstimateRecord:
    """Return a valid cost estimate with a hard bound of 40.00 EUR."""
    base = CostEstimateRecord(
        schema="training-cost-estimate.v1",
        recipe_digest=digest_of("recipe"),
        driver_name="cloud-driver",
        accelerator_type="gpu-a",
        accelerator_count=1,
        replica_count=1,
        scheduling_strategy="SPOT",
        max_wall_seconds=7200,
        unit="EUR",
        estimated_expected_cost="25.00",
        hard_upper_bound_cost="40.00",
        price_source_and_revision="price-sheet r7",
        estimated_at_utc=UTC,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def live(**changes: object) -> LiveReservationProjection:
    """Return an open reservation that covers the estimate exactly at its floors."""
    base = LiveReservationProjection(
        found=True,
        reservation_id="res-1",
        pool_id="pool-cloud",
        state="open",
        exposure="100.00",
        requested_upper="60.00",
        settled="20.00",
        remaining_exposure="45.00",
        minimum_charge="10.00",
        fixed_fee="5.00",
        unit="EUR",
        epoch=3,
        window="2026-11",
        seat="seat-1",
        project="director-ai",
        task="train",
        operation="adapter-sweep",
        expires_at=UTC,
        overrun_open=False,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestSpendCover:
    """Row 7: SV-2 to SV-9."""

    def test_cover_exactly_at_the_floors_passes(self) -> None:
        """45.00 >= max(40, 10) + 5 and 60 - 20 >= 40."""
        item = live()
        assert validate_spend_reservation_live(binding(), item, estimate()) == item

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"remaining_exposure": "44.99"}, "remaining exposure"),
            ({"settled": "20.01"}, "unsettled requested"),
            ({"found": False}, "found, open"),
            ({"state": "settled"}, "found, open"),
            ({"overrun_open": True}, "found, open"),
            ({"overrun_open": None}, "found, open"),
            ({"pool_id": "pool-other"}, "pool_id"),
            ({"reservation_id": "res-2"}, "reservation_id"),
            ({"project": "other"}, "project"),
            ({"task": "eval"}, "task"),
            ({"operation": "other"}, "operation"),
            ({"unit": "USD"}, "another unit"),
            ({"minimum_charge": None}, "minimum_charge"),
            ({"fixed_fee": "-1"}, "plain decimal text"),
            ({"remaining_exposure": "NaN"}, "plain decimal text"),
            ({"requested_upper": "1e400x"}, "plain decimal text"),
        ],
        ids=[
            "M-B2-SPEND-1-remaining",
            "M-B2-SPEND-3",
            "not-found",
            "settled",
            "overrun",
            "overrun-unknown",
            "pool",
            "reservation",
            "project",
            "task",
            "operation",
            "unit",
            "minimum-missing",
            "fee-negative",
            "nan",
            "garbage",
        ],
    )
    def test_cover_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """Each SV rule is refused by name."""
        with pytest.raises(SpendBindingError, match=fragment):
            validate_spend_reservation_live(binding(), live(**changes), estimate())

    def test_m_b2_spend_1_exposure_is_not_remaining_exposure(self) -> None:
        """A large total exposure does not cover a short remaining exposure."""
        short = live(exposure="1000.00", remaining_exposure="44.00")
        with pytest.raises(SpendBindingError, match="remaining exposure"):
            validate_spend_reservation_live(binding(), short, estimate())

    def test_m_b2_spend_2_minimum_charge_and_fee_count(self) -> None:
        """A minimum charge above the bound, plus the fee, must be covered."""
        with pytest.raises(SpendBindingError, match="remaining exposure"):
            validate_spend_reservation_live(
                binding(), live(minimum_charge="41.00"), estimate()
            )
        with pytest.raises(SpendBindingError, match="remaining exposure"):
            validate_spend_reservation_live(
                binding(), live(fixed_fee="5.01"), estimate()
            )
        validate_spend_reservation_live(
            binding(),
            live(minimum_charge="41.00", remaining_exposure="46.00"),
            estimate(),
        )

    def test_m_b2_spend_4_a_zero_hard_bound_is_refused(self) -> None:
        """Class A local runs too must state a positive hard bound."""
        zero = estimate(estimated_expected_cost="0", hard_upper_bound_cost="0")
        with pytest.raises(SpendBindingError, match="above zero"):
            validate_spend_reservation_live(binding(), live(), zero)

    def test_the_binding_and_estimate_are_checked_first(self) -> None:
        """A binding without a task and a malformed estimate are refused."""
        with pytest.raises(SpendBindingError, match="task"):
            validate_spend_reservation_live(binding(task=""), live(task=""), estimate())
        with pytest.raises(SpendBindingError):
            validate_spend_reservation_live(binding(), estimate(), estimate())  # type: ignore[arg-type]

    def test_amounts_are_decimal_text(self) -> None:
        """Amounts are text, finite and not negative."""
        assert str(amount("0.10", "x") + amount("0.20", "y")) == "0.30"
        for bad in (0.1, "", "Infinity", "-0.01"):
            with pytest.raises(SpendBindingError):
                amount(bad, "x")

    @pytest.mark.parametrize(
        "text",
        [
            "1E+100",
            "4.5e1",
            "1_000",
            "+45",
            "-1",
            "00.5",
            ".5",
            "5.",
            "NaN",
            "\u0664\u0665",
        ],
    )
    def test_only_plain_decimal_text_is_read(self, text: str) -> None:
        """A sign, exponent, separator, leading zero or non-ASCII digit is refused."""
        with pytest.raises(SpendBindingError, match="plain decimal text"):
            amount(text, "x")
        for good in ("0", "0.00", "45", "45.00", "1" + "0" * 100):
            assert amount(good, "x") == Decimal(good)


TEN_100 = "1" + "0" * 100
TEN_50 = "1" + "0" * 50


class TestCoverNeverRounds:
    """The architect's three cases: each is one short and must not round to cover."""

    def test_a_fee_lost_in_rounding_is_refused(self) -> None:
        """Hard 10^100 plus fee 1 needs 101 digits; the floor must not round to 10^100."""
        wide = estimate(estimated_expected_cost="0", hard_upper_bound_cost=TEN_100)
        item = live(
            minimum_charge="0",
            fixed_fee="1",
            remaining_exposure=TEN_100,
            requested_upper=TEN_100,
            settled="0",
        )
        with pytest.raises(SpendBindingError, match="more precision"):
            validate_spend_reservation_live(binding(), item, wide)

    def test_a_settled_amount_lost_in_rounding_is_refused(self) -> None:
        """10^100 - 1 is below the hard bound; the difference must not round to 10^100."""
        wide = estimate(estimated_expected_cost="0", hard_upper_bound_cost=TEN_100)
        item = live(
            minimum_charge="0",
            fixed_fee="0",
            remaining_exposure=TEN_100,
            requested_upper=TEN_100,
            settled="1",
        )
        with pytest.raises(SpendBindingError, match="more precision"):
            validate_spend_reservation_live(binding(), item, wide)

    def test_a_51_digit_bound_is_not_rounded_down(self) -> None:
        """A bound of 10^50 + 1 against an exposure of 10^50 is one short."""
        bound = "1" + "0" * 49 + "1"
        wide = estimate(estimated_expected_cost="0", hard_upper_bound_cost=bound)
        item = live(
            minimum_charge="0",
            fixed_fee="0",
            remaining_exposure=TEN_50,
            requested_upper=bound,
            settled="0",
        )
        with pytest.raises(SpendBindingError, match="more precision"):
            validate_spend_reservation_live(binding(), item, wide)

    def test_wide_exact_amounts_still_pass(self) -> None:
        """Exact cover with 101-digit amounts passes; only rounding is refused."""
        wide = estimate(estimated_expected_cost="0", hard_upper_bound_cost=TEN_100)
        item = live(
            minimum_charge="0",
            fixed_fee="0",
            remaining_exposure=TEN_100,
            requested_upper=TEN_100,
            settled="0",
        )
        assert validate_spend_reservation_live(binding(), item, wide) == item


class TestEstimatesAndApprovals:
    """Cost estimates and paid-run approvals."""

    @pytest.mark.parametrize(
        "changes",
        [
            {"estimated_expected_cost": "40.01"},
            {"accelerator_count": 0},
            {"max_wall_seconds": 0},
            {"scheduling_strategy": "PREEMPTIBLE"},
            {"unit": ""},
            {"estimated_at_utc": "today"},
            {"schema": "estimate.v2"},
        ],
    )
    def test_estimate_refusals(self, changes: dict[str, object]) -> None:
        """The expected cost stays within the hard bound; each field is checked."""
        with pytest.raises(SpendBindingError):
            validate_cost_estimate(estimate(**changes))
        with pytest.raises(MalformedDigestError):
            validate_cost_estimate(estimate(recipe_digest="x"))

    def test_approval_binds_its_pool(self) -> None:
        """An approval's binding names the approval's pool."""
        approval = PaidRunApprovalRecord(
            "paid-run-approval.v1",
            digest_of("estimate"),
            "owner-ok-17",
            "pool-cloud",
            binding(),
        )
        assert validate_paid_run_approval(approval) == approval
        with pytest.raises(SpendBindingError, match="another pool"):
            validate_paid_run_approval(replace(approval, pool_id="pool-other"))
        with pytest.raises(SpendBindingError):
            validate_paid_run_approval(replace(approval, owner_approval_ref=""))

    def test_records_of_other_types_are_refused(self) -> None:
        """Each spend validator checks its own record type."""
        for check in (
            validate_spend_binding,
            validate_cost_estimate,
            validate_paid_run_approval,
            validate_sandbox_confinement,
            validate_job_submission,
            validate_job_status,
            validate_job_cancel,
        ):
            with pytest.raises(SpendBindingError):
                check(live())  # type: ignore[arg-type]
        with pytest.raises(SpendBindingError):
            validate_spend_reservation_live(binding(), binding(), estimate())  # type: ignore[arg-type]


CONFINED = SandboxConfinementSpec("sandbox-confinement-spec.v1", True, "landlock_v4")


def submission(**changes: object) -> CloudJobSubmissionReceipt:
    """Return a valid cloud-credit submission with ``changes`` applied."""
    base = CloudJobSubmissionReceipt(
        schema="cloud-job-submission.v1",
        driver_name="cloud-driver",
        funding_kind="monthly_cloud_credit",
        job_id="job-9",
        dry_run=False,
        container_image_digest=digest_of("image"),
        confinement=CONFINED,
        input_shards_digests=(
            ("train-0", digest_of("t0")),
            ("train-1", digest_of("t1")),
        ),
        spend_binding_digest=digest_of("binding"),
        paid_approval_digest=None,
        stop_lease_operation_id="op-stop-9",
        submitted_at_utc=UTC,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestJobRecords:
    """Confinement, submission, status and cancellation."""

    def test_valid_job_records_round_trip(self) -> None:
        """Each job record validates and decodes through the codec."""
        paid = submission(
            funding_kind="per_run_owner_approved_paid",
            paid_approval_digest=digest_of("ok"),
        )
        status = CloudJobStatusReport(
            "cloud-job-status.v1", "job-9", "running", None, 120, UTC
        )
        cancel = CloudJobCancelReceipt(
            "cloud-job-cancel.v1", "job-9", True, "owner stop", UTC
        )
        for item in (
            submission(),
            paid,
            status,
            cancel,
            CONFINED,
            binding(),
            estimate(),
        ):
            assert (
                from_canonical_json_bytes(type(item), canonical_json_bytes(item))
                == item
            )
        validate_job_status(replace(status, latest_checkpoint_digest=digest_of("ckpt")))

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"funding_kind": "per_run_owner_approved_paid"}, "paid_approval_digest"),
            ({"paid_approval_digest": digest_of("ok")}, "only a paid run"),
            (
                {
                    "input_shards_digests": (
                        ("a", digest_of("a")),
                        ("a", digest_of("b")),
                    )
                },
                "repeat",
            ),
            ({"input_shards_digests": (("a",),)}, "pairs"),
            (
                {
                    "confinement": SandboxConfinementSpec(
                        "sandbox-confinement-spec.v1",
                        False,  # type: ignore[arg-type]
                        "landlock_v4",
                    )
                },
                "seccomp",
            ),
            (
                {
                    "confinement": SandboxConfinementSpec(
                        "sandbox-confinement-spec.v1",
                        True,
                        "iptables",  # type: ignore[arg-type]
                    )
                },
                "network_isolation_layer",
            ),
            ({"funding_kind": "donated"}, "funding_kind"),
            ({"dry_run": 0}, "dry_run"),
            ({"job_id": ""}, "job_id"),
        ],
    )
    def test_submission_refusals(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """A paid run names its approval and only a paid run does."""
        with pytest.raises((SpendBindingError, MalformedDigestError), match=fragment):
            validate_job_submission(submission(**changes))

    @pytest.mark.parametrize(
        "item",
        [
            CloudJobStatusReport("cloud-job-status.v1", "job-9", "lost", None, 1, UTC),  # type: ignore[arg-type]
            CloudJobStatusReport(
                "cloud-job-status.v1", "job-9", "running", None, -1, UTC
            ),
            CloudJobCancelReceipt("cloud-job-cancel.v1", "job-9", 1, "x", UTC),  # type: ignore[arg-type]
            CloudJobCancelReceipt("cloud-job-cancel.v1", "job-9", True, "", UTC),
        ],
    )
    def test_status_and_cancel_refusals(self, item: object) -> None:
        """Each status and cancellation field is checked."""
        with pytest.raises(SpendBindingError):
            if isinstance(item, CloudJobStatusReport):
                validate_job_status(item)
            else:
                validate_job_cancel(item)  # type: ignore[arg-type]


def decline(**changes: object) -> StudyDeclineRecord:
    """Return a valid decline record with ``changes`` applied."""
    base = StudyDeclineRecord(
        "model-study-decline.v1",
        "study-1",
        digest_of("prereg"),
        digest_of("result"),
        (("candidate", ("gain interval below margin",)),),
        True,
        True,
        UTC,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestPromotionRecords:
    """Serving memory, dependency pins and study declines."""

    def test_valid_records_round_trip(self) -> None:
        """Each promotion record validates and decodes through the codec."""
        budget = ServingMemoryBudget(
            "serving-memory-budget.v1", "host_ram", 6, 3, 1, 10
        )
        pins = PromotionDependencyPins(
            "promotion-dependency-pins.v1",
            "scope-1",
            digest_of("f"),
            (),
            (digest_of("w"),),
        )
        for item in (budget, pins, decline()):
            assert (
                from_canonical_json_bytes(type(item), canonical_json_bytes(item))
                == item
            )

    def test_serving_memory_must_fit(self) -> None:
        """Weights plus cache plus overhead may equal the ceiling, not exceed it."""
        with pytest.raises(PromotionDecisionError, match="exceed"):
            validate_serving_memory_budget(
                ServingMemoryBudget("serving-memory-budget.v1", "gpu_vram", 6, 3, 2, 10)
            )
        for bad in (
            ("cpu_cache", 6, 0, 0, 10),
            ("host_ram", 0, 0, 0, 10),
            ("host_ram", 1, -1, 0, 10),
        ):
            with pytest.raises(PromotionDecisionError):
                validate_serving_memory_budget(
                    ServingMemoryBudget("serving-memory-budget.v1", *bad)  # type: ignore[arg-type]
                )

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"witness_and_parity_digests": ()}, PromotionDecisionError),
            (
                {"secondary_family_contract_digests": (digest_of("f"),)},
                PromotionDecisionError,
            ),
            ({"pin_scope": ""}, PromotionDecisionError),
            ({"family_contract_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_pin_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """A scope pins at least one witness, and no digest twice."""
        pins = PromotionDependencyPins(
            "promotion-dependency-pins.v1",
            "scope-1",
            digest_of("f"),
            (),
            (digest_of("w"),),
        )
        with pytest.raises(error):
            validate_dependency_pins(replace(pins, **changes))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "changes",
        [
            {"failed_arm_reasons": ()},
            {"failed_arm_reasons": (("candidate", ()),)},
            {"failed_arm_reasons": (("a", ("x",)), ("a", ("y",)))},
            {"failed_arm_reasons": (("a",),)},
            {"failed_arm_reasons": (("a", ("",)),)},
            {"null_control_rejected": 1},
            {"study_id": ""},
            {"declined_at_utc": "now"},
            {"schema": "decline.v2"},
        ],
        ids=[
            "M-B2-DECLINE-1-empty",
            "no-reason",
            "repeat",
            "pairs",
            "empty-reason",
            "bool",
            "study",
            "time",
            "schema",
        ],
    )
    def test_decline_refusals(self, changes: dict[str, object]) -> None:
        """M-B2-DECLINE-1: every failed arm names its reasons."""
        with pytest.raises(PromotionDecisionError):
            validate_study_decline(decline(**changes))

    def test_records_of_other_types_are_refused(self) -> None:
        """Each promotion validator checks its own record type."""
        for check in (
            validate_serving_memory_budget,
            validate_dependency_pins,
            validate_study_decline,
        ):
            with pytest.raises(PromotionDecisionError):
                check(live())  # type: ignore[arg-type]
