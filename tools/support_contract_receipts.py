# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — bind the contract's availability to installed probe receipts
"""Check that every claimed installation was shown to run, in the same job.

``scripts/probe_support_contract.py`` writes one receipt per installed
combination. A row's availability holds only if the probe it names passed in
every receipt whose installation contains the row's installation, and the
required negative probes passed too. No availability is kept by hand.

A model-backed row also needs a positive receipt from a declared host: the
same wheels, matched by digest, with the model snapshot's revision and
verified digest. The hosted job cannot fetch the model, so its receipt
covers the refusals and the dry configuration only.

Every receipt binds the code it tested: ``tree`` is the Git tree the wheels
were built from, ``commit`` is that tree's commit, or its parent when the
tree was only staged (``staged``), and ``machine`` names where it ran, beside
the ``host`` mode. The check needs the expected tree, the landing commit's,
and refuses any receipt of another tree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from director_ai.support_contract import INSTALLATIONS, SupportContract

RECEIPT_SCHEMA = "director-ai.support-receipt.v1"
HOSTS = ("declared", "hosted-ci")
"""A receipt comes from the hosted CI job or from a declared host."""
REQUIRED_NEGATIVES: dict[str, tuple[str, ...]] = {
    "core": (
        "evidence-emission-refused",
        "overlay-authority-refused",
        "paid-import-boundary",
        "tampered-packet-refused",
    ),
    "core+pro": ("overlay-authority-refused",),
    "core+full": ("overlay-authority-refused",),
}
"""The negative probes each hosted receipt must show passing."""
CONDITION_STATES = ("not_applicable", "passed")
"""How a passed probe records a condition it names."""
REQUIRED_CONDITIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "pii-redactor": {"strict-trail-refusal": ("core+pro", "core+full")},
}
"""Conditions a probe must record wherever it passed: ``passed`` on the
listed installations, and ``passed`` or ``not_applicable`` on the others."""

_DIGEST = re.compile(r"[0-9a-f]{64}")
_OBJECT = re.compile(r"[0-9a-f]{40}")
"""A Git commit or tree identifier."""


def _contains(receipt: str, row: str) -> bool:
    """Return whether a receipt's installation includes a row's installation."""
    return INSTALLATIONS.index(receipt) >= INSTALLATIONS.index(row)


def _shape_refusals(receipt: Mapping[str, object], name: str) -> list[str]:
    """Return the refusals of one receipt's own form."""
    refusals = []
    if receipt.get("schema") != RECEIPT_SCHEMA:
        refusals.append(f"{name}: unknown receipt schema")
    if receipt.get("installation") not in INSTALLATIONS:
        refusals.append(f"{name}: unknown installation")
    if receipt.get("host") not in HOSTS:
        refusals.append(f"{name}: unknown host")
    for field in ("commit", "tree"):
        value = receipt.get(field)
        if not isinstance(value, str) or not _OBJECT.fullmatch(value):
            refusals.append(f"{name}: invalid {field}")
    if not isinstance(receipt.get("staged"), bool):
        refusals.append(f"{name}: invalid staged flag")
    machine = receipt.get("machine")
    if not isinstance(machine, str) or not machine.strip():
        refusals.append(f"{name}: invalid machine")
    wheels = receipt.get("wheels")
    if (
        not isinstance(wheels, dict)
        or not wheels
        or any(
            not isinstance(v, str) or not _DIGEST.fullmatch(v) for v in wheels.values()
        )
    ):
        refusals.append(f"{name}: invalid wheel digests")
    for section in ("probes", "negatives"):
        value = receipt.get(section)
        if not isinstance(value, dict) or any(
            not isinstance(result, dict)
            or not isinstance(result.get("passed"), bool)
            or not _valid_conditions(result.get("conditions", {}))
            for result in value.values()
        ):
            refusals.append(f"{name}: invalid {section}")
    model = receipt.get("model")
    if receipt.get("host") == "declared" and (
        not isinstance(model, dict)
        or not isinstance(model.get("revision"), str)
        or not _DIGEST.fullmatch(str(model.get("digest")))
    ):
        refusals.append(f"{name}: a declared-host receipt names no verified model")
    return refusals


def _valid_conditions(conditions: object) -> bool:
    """Return whether a result's conditions map names to known states."""
    return isinstance(conditions, dict) and all(
        isinstance(state, str) and state in CONDITION_STATES
        for state in conditions.values()
    )


def _condition_refusals(name: str, receipt: Mapping[str, object]) -> list[str]:
    """Return the refusals of the conditions one passed probe must record."""
    refusals = []
    installation = str(receipt["installation"])
    probes = cast("dict[str, dict[str, object]]", receipt["probes"])
    for probe, required in sorted(REQUIRED_CONDITIONS.items()):
        result = probes.get(probe)
        if result is None or result["passed"] is not True:
            continue
        conditions = cast("dict[str, str]", result.get("conditions", {}))
        for condition, installations in sorted(required.items()):
            state = conditions.get(condition)
            if state is None:
                refusals.append(
                    f"{name}: probe {probe} does not record its condition {condition}"
                )
            elif installation in installations and state != "passed":
                refusals.append(
                    f"{name}: probe {probe} must pass its condition {condition} "
                    f"on {installation}"
                )
    return refusals


def _passed(receipt: Mapping[str, object], section: str, probe: str) -> bool:
    """Return whether a receipt shows one probe passing."""
    results = receipt.get(section)
    result = results.get(probe) if isinstance(results, dict) else None
    return isinstance(result, dict) and result.get("passed") is True


def receipt_refusals(
    contract: SupportContract,
    receipts: Mapping[str, Mapping[str, object]],
    *,
    tree: str,
) -> tuple[str, ...]:
    """Return every reason the receipts do not support the claimed availability.

    Parameters
    ----------
    contract : SupportContract
        The contract whose rows claim installations.
    receipts : Mapping[str, Mapping[str, object]]
        The decoded receipts by file name.
    tree : str
        The Git tree every receipt must have tested: the landing commit's.

    Returns
    -------
    tuple[str, ...]
        The refusals in a stable order; empty means availability holds.
    """
    if not _OBJECT.fullmatch(tree):
        return (f"Invalid expected tree: {tree!r}",)
    refusals = []
    for name, receipt in sorted(receipts.items()):
        refusals.extend(_shape_refusals(receipt, name))
    if refusals:
        return tuple(refusals)
    refusals.extend(
        f"{name}: the receipt tested tree {receipt['tree']}, not {tree}"
        for name, receipt in sorted(receipts.items())
        if receipt["tree"] != tree
    )
    commits = {str(receipt["commit"]) for receipt in receipts.values()}
    if len(commits) != 1:
        refusals.append("The receipts come from different commits")
    if len({receipt["staged"] for receipt in receipts.values()}) != 1:
        refusals.append("The receipts disagree on whether the tree was staged")
    seen: dict[str, str] = {}
    for name, receipt in sorted(receipts.items()):
        wheels = cast("dict[str, str]", receipt["wheels"])
        for wheel, digest in sorted(wheels.items()):
            if seen.setdefault(wheel, digest) != digest:
                refusals.append(
                    f"{name}: a wheel digest differs between receipts: {wheel}"
                )
    hosted = {
        str(receipt["installation"]): (name, receipt)
        for name, receipt in sorted(receipts.items())
        if receipt["host"] == "hosted-ci"
    }
    for installation in INSTALLATIONS:
        if installation not in hosted:
            refusals.append(f"No hosted receipt for the {installation} installation")
            continue
        name, receipt = hosted[installation]
        refusals.extend(
            f"{name}: the negative probe failed or did not run: {negative}"
            for negative in REQUIRED_NEGATIVES[installation]
            if not _passed(receipt, "negatives", negative)
        )
    for name, receipt in sorted(receipts.items()):
        refusals.extend(_condition_refusals(name, receipt))
    declared = [
        receipt for receipt in receipts.values() if receipt["host"] == "declared"
    ]
    for operation in contract.operations:
        if operation.installation is None or operation.probe is None:
            continue
        for installation, (_name, receipt) in sorted(hosted.items()):
            if _contains(installation, operation.installation) and not _passed(
                receipt, "probes", operation.probe
            ):
                refusals.append(
                    f"{operation.identifier}: probe {operation.probe} failed or did not "
                    f"run on {installation}"
                )
        if operation.model_backed and not any(
            _contains(str(receipt["installation"]), operation.installation)
            and _passed(receipt, "probes", operation.probe)
            for receipt in declared
        ):
            refusals.append(
                f"{operation.identifier}: no positive receipt from a declared host"
            )
    return tuple(refusals)


def read_receipts(directory: Path) -> dict[str, dict[str, object]]:
    """Read every ``*.json`` receipt in a directory.

    Parameters
    ----------
    directory : Path
        The directory the probe runs wrote into.

    Returns
    -------
    dict[str, dict[str, object]]
        The decoded receipts by file name.

    Raises
    ------
    ValueError
        If a receipt is not a JSON object, or the directory holds none.
    """
    receipts: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"A receipt is not a JSON object: {path.name}")
        receipts[path.name] = value
    if not receipts:
        raise ValueError(f"No receipt in {directory}")
    return receipts


def check_receipts(
    contract: SupportContract, directories: Sequence[Path], *, tree: str
) -> tuple[str, ...]:
    """Read the receipts from directories and check them together.

    Parameters
    ----------
    contract : SupportContract
        The contract.
    directories : Sequence[Path]
        The receipt directories, such as the hosted job's and a declared host's.
    tree : str
        The Git tree every receipt must have tested.

    Returns
    -------
    tuple[str, ...]
        The refusals.

    Raises
    ------
    ValueError
        If two directories hold a receipt of the same name, or one is invalid.
    """
    merged: dict[str, dict[str, object]] = {}
    for directory in directories:
        for name, receipt in read_receipts(directory).items():
            if name in merged:
                raise ValueError(f"Two receipts share a name: {name}")
            merged[name] = receipt
    return receipt_refusals(contract, merged, tree=tree)
