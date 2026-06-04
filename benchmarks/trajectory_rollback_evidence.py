# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — trajectory rollback evidence packet

"""Generate local evidence for trajectory rollback hardening.

The packet checks the production-relevant R11 primitives without model
downloads:

* Monte-Carlo preflight produces proceed, warn, and halt action bands;
* warn arms a rollback handle without executing it;
* halt executes the registered rollback hook exactly once;
* hook failures report tenant-safe error types without raw backend messages.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess  # nosec B404
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks._common import save_results
from director_ai.core.trajectory import (
    RollbackHandle,
    TrajectoryRollbackManager,
    TrajectorySimulator,
)


@dataclass(frozen=True)
class _Score:
    score: float


class _FixedActor:
    def sample(self, _prompt: str, seed: int) -> list[str]:
        return [f"draw:{seed}"]


class _SequenceScorer:
    def __init__(self, sequence: Sequence[tuple[bool, float]]) -> None:
        self._sequence = tuple(sequence)
        self._index = 0

    def review(
        self,
        prompt: str,
        action: str,
        tenant_id: str = "",
    ) -> tuple[bool, _Score]:
        del prompt, action, tenant_id
        approved, score = self._sequence[self._index % len(self._sequence)]
        self._index += 1
        return approved, _Score(score=score)


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


def _preflight(sequence: Sequence[tuple[bool, float]], *, simulations: int):
    simulator = TrajectorySimulator(
        actor=_FixedActor(),
        scorer=_SequenceScorer(sequence),
        n_simulations=simulations,
        halt_rate_warn=0.25,
        halt_rate_halt=0.50,
        base_seed=101,
    )
    return simulator.preflight("tenant-safe-action")


def run_preflight_rollback_probe(*, simulations: int = 4) -> dict[str, Any]:
    """Return proceed/warn/halt rollback evidence from trajectory preflight."""
    if simulations < 4:
        raise ValueError("simulations must be >= 4")

    manager = TrajectoryRollbackManager()
    calls: list[tuple[str, str]] = []

    def hook(handle: RollbackHandle, reason: str) -> dict[str, str]:
        calls.append((handle.rollback_id, reason))
        return {"rollback_store": "local-audit", "undo_steps": "1"}

    cases = [
        ("proceed", [(True, 0.92)] * simulations, "not_required", False),
        (
            "warn",
            [(False, 0.18)] + [(True, 0.86)] * (simulations - 1),
            "armed",
            False,
        ),
        (
            "halt",
            [(False, 0.12), (False, 0.2)]
            + [(True, 0.88)] * (simulations - 2),
            "executed",
            True,
        ),
    ]
    records = []
    for case_name, sequence, expected_status, expected_executed in cases:
        rollback_id = f"rollback-{case_name}"
        handle = manager.register(
            rollback_id=rollback_id,
            action_id=f"trajectory-action-{case_name}",
            tenant_id="tenant-a",
            hook=hook,
            evidence_refs=(f"change:{case_name}",),
            metadata={"owner": "safety"},
        )
        verdict = _preflight(sequence, simulations=simulations)
        outcome = manager.evaluate_preflight(handle.rollback_id, verdict)
        records.append(
            {
                "case": case_name,
                "recommended": verdict.recommended,
                "halt_rate": round(verdict.halt_rate, 4),
                "failed_trajectories": sum(
                    not trajectory.approved for trajectory in verdict.trajectories
                ),
                "expected_status": expected_status,
                "status": outcome.status,
                "expected_executed": expected_executed,
                "executed": outcome.executed,
                "evidence_refs": list(outcome.evidence_refs),
                "matched": (
                    outcome.status == expected_status
                    and outcome.executed is expected_executed
                ),
            }
        )

    repeat = manager.execute("rollback-halt", reason="trajectory_preflight_halt")
    return {
        "name": "preflight_rollback_paths",
        "simulations": simulations,
        "records": records,
        "hook_calls": calls,
        "repeat_status": repeat.status,
        "repeat_executed": repeat.executed,
        "raw_prompt_payload_included": False,
        "passed": bool(
            all(record["matched"] for record in records)
            and calls == [("rollback-halt", "trajectory_preflight_halt")]
            and repeat.status == "already_executed"
            and repeat.executed is False
        ),
    }


def run_failure_probe() -> dict[str, Any]:
    """Return tenant-safe rollback failure evidence."""
    manager = TrajectoryRollbackManager()

    def hook(_handle: RollbackHandle, _reason: str) -> None:
        raise RuntimeError("raw backend detail must not be emitted")

    handle = manager.register(
        rollback_id="rollback-failure",
        action_id="trajectory-action-failure",
        hook=hook,
    )
    outcome = manager.execute(handle.rollback_id, reason="manual_halt")
    serialised = json.dumps(outcome.to_dict(), sort_keys=True)
    return {
        "name": "rollback_failure_sanitisation",
        "status": outcome.status,
        "executed": outcome.executed,
        "error_type": outcome.error_type,
        "raw_error_payload_included": "raw backend detail" in serialised,
        "passed": bool(
            outcome.status == "failed"
            and outcome.executed is False
            and outcome.error_type == "RuntimeError"
            and "raw backend detail" not in serialised
        ),
    }


def run_trajectory_rollback_evidence(*, simulations: int = 4) -> dict[str, Any]:
    """Return the complete local R11 trajectory rollback evidence packet."""
    paths = run_preflight_rollback_probe(simulations=simulations)
    failure = run_failure_probe()
    passed = bool(paths["passed"] and failure["passed"])
    return {
        "schema_version": "director-ai.trajectory-rollback-evidence.v1",
        "benchmark": "trajectory_rollback_evidence",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "acceptance": {
            "passed": passed,
            "checks": {
                "preflight_rollback_paths": bool(paths["passed"]),
                "rollback_failure_sanitisation": bool(failure["passed"]),
            },
            "limits": {
                "local_only": True,
                "external_operator_signoff_included": False,
                "live_undo_backend_included": False,
            },
        },
        "probes": {
            "preflight_rollback_paths": paths,
            "rollback_failure_sanitisation": failure,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Director-AI trajectory rollback evidence packet.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=4,
        help="Number of trajectory simulations per action-band probe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to benchmarks/results/.",
    )
    args = parser.parse_args(argv)

    payload = run_trajectory_rollback_evidence(simulations=args.simulations)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        save_results(payload, f"trajectory_rollback_evidence_{stamp}.json")
    print(json.dumps(payload["acceptance"], indent=2))
    return 0 if payload["acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
