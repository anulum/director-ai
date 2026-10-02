# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — required aggregate CI result
"""Refuse missing, failed, cancelled or unexpectedly skipped CI categories."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.ci_workflow_inventory import ROOT, load_ci_workflow_policy, object_mapping


def check_ci_results(results: object, event: str, root: Path = ROOT) -> list[str]:
    """Check every exclusively owned category's actual native result.

    Parameters
    ----------
    results : object
        Decoded GitHub ``toJSON(needs)`` result object.
    event : str
        Actual workflow event name, never an inferred branch/event.
    root : Path
        Checkout containing the versioned category policy.

    Returns
    -------
    list[str]
        Authored deterministic violations; empty only when all required
        categories pass or satisfy their declared event-specific condition.
    """
    policy = load_ci_workflow_policy(root)
    actual = object_mapping(results)
    expected = {category.identity for category in policy.categories}
    errors = []
    if set(actual) != expected:
        errors.append("Aggregate category inventory differs from the required policy")
    for category in policy.categories:
        value = actual.get(category.identity)
        if not isinstance(value, dict):
            errors.append(f"Missing native category result: {category.identity}")
            continue
        result = object_mapping(value).get("result")
        if result == "success":
            continue
        if result == "skipped" and event in category.skip_events:
            continue
        errors.append(f"Required category did not succeed: {category.identity}")
    return errors


def main() -> int:
    """Validate the real native result object supplied through the CLI.

    Returns
    -------
    int
        Zero for a complete passing aggregate, one for any refusal.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-env", default="CI_RESULTS")
    parser.add_argument("--event", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        results: object = json.loads(os.environ[args.results_env])
        errors = check_ci_results(results, args.event, args.root)
    except (OSError, ValueError, KeyError):
        print("Required CI result or ownership policy is missing or malformed")
        return 1
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
