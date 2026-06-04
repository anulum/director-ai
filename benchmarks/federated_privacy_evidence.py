# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — federated privacy evidence packet

"""Generate local evidence for privacy-preserving federated sharing.

The packet checks the production-relevant R13 primitives without external
services:

* tenant-safe Director safety signals aggregate into a DP-noised histogram;
* one tenant can contribute at most once per category in a release window;
* release is blocked until the minimum tenant cohort is present;
* additive secret sharing reconstructs the aggregate without serialising
  individual party values.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from director_ai.core.federated_privacy import (
    FederatedSafetySignalAggregator,
    PrivacyAccountant,
    SecureAggregator,
)
from director_ai.core.federated_privacy.secret_sharing import split
from director_ai.core.safety_event import SafetyEvent
from director_ai.core.safety_protocol import director_safety_signal_from_event


def _git_commit() -> str:
    git = shutil.which("git")
    if not git:
        return "unknown"
    try:
        completed = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _signal(*, tenant_id: str, decision: str):
    event = SafetyEvent.from_policy_decision(
        hook_id="federated-evidence",
        hook_scope="streaming",
        policy_decision=decision,
        halt_reason="coherence",
        tenant_safe_explanation="Tenant-safe aggregate signal.",
        tenant_id=tenant_id,
        observed_score=0.2,
        threshold=0.5,
        evidence_refs=("chunk:0",),
    )
    return director_safety_signal_from_event(
        event,
        producer_id="producer-a",
        framework="evidence",
    )


def run_dp_signal_probe() -> dict[str, Any]:
    """Return DP safety-signal aggregation evidence."""
    accountant = PrivacyAccountant(max_epsilon=2.0)
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.9,
        accountant=accountant,
        min_tenants=2,
        seed=0,
        allow_insecure_seed=True,
    )
    first = _signal(tenant_id="tenant-a", decision="halt")
    duplicate_signal = first
    same_tenant_new_signal = _signal(tenant_id="tenant-a", decision="halt")
    second_tenant = _signal(tenant_id="tenant-b", decision="warn")

    accepted_first = aggregator.submit_signal(first)
    accepted_duplicate = aggregator.submit_signal(duplicate_signal)
    accepted_same_tenant = aggregator.submit_signal(same_tenant_new_signal)
    accepted_second = aggregator.submit_signal(second_tenant)
    release = aggregator.release()
    payload = release.to_dict()
    serialised = json.dumps(payload, sort_keys=True)

    return {
        "name": "dp_signal_aggregation",
        "accepted_first": accepted_first,
        "accepted_duplicate": accepted_duplicate,
        "accepted_same_tenant": accepted_same_tenant,
        "accepted_second": accepted_second,
        "signal_count": release.signal_count,
        "distinct_tenants": release.distinct_tenants,
        "epsilon_spent": release.epsilon_spent,
        "accountant_epsilon": round(accountant.cumulative_epsilon(), 6),
        "raw_halt_count": release.raw_counts["decision:halt"],
        "raw_warn_count": release.raw_counts["decision:warn"],
        "payload_raw_counts_included": "raw_counts" in payload,
        "tenant_ids_leaked": "tenant-a" in serialised or "tenant-b" in serialised,
        "passed": bool(
            accepted_first == ("decision:halt", "scope:streaming")
            and accepted_duplicate == ()
            and accepted_same_tenant == ()
            and accepted_second == ("decision:warn", "scope:streaming")
            and release.signal_count == 2
            and release.distinct_tenants == 2
            and release.raw_counts["decision:halt"] == 1
            and release.raw_counts["decision:warn"] == 1
            and "raw_counts" not in payload
            and "tenant-a" not in serialised
            and "tenant-b" not in serialised
            and accountant.cumulative_epsilon() == release.epsilon_spent
        ),
    }


def run_min_tenant_probe() -> dict[str, Any]:
    """Return evidence that undersized cohorts do not charge DP budget."""
    accountant = PrivacyAccountant(max_epsilon=2.0)
    aggregator = FederatedSafetySignalAggregator(
        epsilon=0.9,
        accountant=accountant,
        min_tenants=2,
        seed=0,
        allow_insecure_seed=True,
    )
    aggregator.submit_signal(_signal(tenant_id="tenant-a", decision="halt"))
    error = ""
    try:
        aggregator.release()
    except ValueError as exc:
        error = str(exc)
    return {
        "name": "minimum_tenant_gate",
        "release_blocked": "min_tenants" in error,
        "accountant_epsilon": accountant.cumulative_epsilon(),
        "passed": bool("min_tenants" in error and accountant.cumulative_epsilon() == 0.0),
    }


def run_secret_sharing_probe() -> dict[str, Any]:
    """Return additive secret-sharing aggregate evidence."""
    party_count = 3
    aggregate = SecureAggregator(party_count=party_count)
    for index, value in enumerate((2, 3, 5)):
        share = split(
            value,
            party_count=party_count,
            seed=100 + index,
            allow_insecure_seed=True,
        )
        aggregate.submit(share)
    reconstructed = aggregate.reconstruct()
    payload = {
        "name": "secure_additive_aggregation",
        "party_count": party_count,
        "submissions": aggregate.submissions,
        "aggregate_total": reconstructed,
        "individual_party_values_included": False,
    }
    payload["passed"] = bool(
        reconstructed == 10
        and aggregate.submissions == 3
        and payload["individual_party_values_included"] is False
    )
    return payload


def run_federated_privacy_evidence() -> dict[str, Any]:
    """Return the complete local R13 federated privacy evidence packet."""
    dp_signals = run_dp_signal_probe()
    min_tenant = run_min_tenant_probe()
    secret_sharing = run_secret_sharing_probe()
    passed = bool(
        dp_signals["passed"] and min_tenant["passed"] and secret_sharing["passed"]
    )
    return {
        "schema_version": "director-ai.federated-privacy-evidence.v1",
        "benchmark": "federated_privacy_evidence",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "dp_signal_aggregation": bool(dp_signals["passed"]),
                "minimum_tenant_gate": bool(min_tenant["passed"]),
                "secure_additive_aggregation": bool(secret_sharing["passed"]),
            },
            "limits": {
                "local_only": True,
                "external_federation_included": False,
                "malicious_secure_aggregation_proof_included": False,
            },
        },
        "probes": {
            "dp_signal_aggregation": dp_signals,
            "minimum_tenant_gate": min_tenant,
            "secure_additive_aggregation": secret_sharing,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI federated privacy evidence packet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_federated_privacy_evidence()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"federated_privacy_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
