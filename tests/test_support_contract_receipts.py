# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the availability rule over installed probe receipts
"""Pin every refusal that binds availability to the receipts (C-4)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from director_ai.support_contract import MEANING, SCHEMA, Operation, SupportContract
from tools.support_contract_receipts import (
    RECEIPT_SCHEMA,
    REQUIRED_NEGATIVES,
    check_receipts,
    read_receipts,
    receipt_refusals,
)

COMMIT = "1" * 40
TREE = "3" * 40
WHEELS = {"director_ai-1-py3-none-any.whl": "a" * 64}
CORE_ROW = Operation(
    identifier="core-row",
    title="Core row",
    support="experimental",
    activation="default",
    installation="core",
    probe="core-probe",
)
PRO_ROW = dataclasses.replace(
    CORE_ROW, identifier="pro-row", installation="core+pro", probe="pro-probe"
)
CONTRACT = SupportContract(
    schema=SCHEMA,
    meaning=MEANING,
    operations=(CORE_ROW, PRO_ROW),
    experimental_exports=(),
    experimental_commands=(),
    experimental_namespaces=(),
    production_defaults=(),
)


def receipt(installation: str, host: str = "hosted-ci") -> dict[str, object]:
    """Return a passing receipt for one installation.

    Parameters
    ----------
    installation : str
        The installed combination.
    host : str
        Where it ran.

    Returns
    -------
    dict[str, object]
        The decoded receipt.
    """
    probes = {"core-probe": {"passed": True, "detail": "ok"}}
    if installation != "core":
        probes["pro-probe"] = {"passed": True, "detail": "ok"}
    return {
        "schema": RECEIPT_SCHEMA,
        "installation": installation,
        "host": host,
        "commit": COMMIT,
        "tree": TREE,
        "staged": False,
        "machine": "runner-1",
        "wheels": dict(WHEELS),
        "model": {"revision": "r1", "digest": "c" * 64} if host == "declared" else None,
        "probes": probes,
        "negatives": {
            name: {"passed": True, "detail": "ok"}
            for name in REQUIRED_NEGATIVES[installation]
        },
    }


def receipts() -> dict[str, dict[str, object]]:
    """Return a passing hosted receipt set."""
    return {f"{name}.json": receipt(name) for name in ("core", "core+pro", "core+full")}


def test_a_complete_receipt_set_supports_the_rows() -> None:
    """Accept passing receipts for every installation."""
    assert receipt_refusals(CONTRACT, receipts(), tree=TREE) == ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other", "unknown receipt schema"),
        ("installation", "core+lite", "unknown installation"),
        ("host", "laptop", "unknown host"),
        ("commit", "abc", "invalid commit"),
        ("tree", "4" * 39, "invalid tree"),
        ("staged", "no", "invalid staged flag"),
        ("machine", " ", "invalid machine"),
        ("machine", None, "invalid machine"),
        ("wheels", {}, "invalid wheel digests"),
        ("wheels", {"w.whl": "xyz"}, "invalid wheel digests"),
        ("probes", {"p": {"passed": "yes"}}, "invalid probes"),
        ("negatives", [], "invalid negatives"),
    ],
)
def test_each_receipt_field_is_validated(
    field: str, value: object, message: str
) -> None:
    """Refuse a malformed receipt by field, before any availability is judged."""
    given = receipts()
    given["core.json"][field] = value
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (f"core.json: {message}",)


def test_receipts_must_share_their_commit_and_wheel_digests() -> None:
    """Refuse receipts from two commits or with two digests of one wheel."""
    given = receipts()
    given["core.json"]["commit"] = "2" * 40
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        "The receipts come from different commits",
    )
    given = receipts()
    given["core+pro.json"]["wheels"] = {"director_ai-1-py3-none-any.whl": "b" * 64}
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        "core+pro.json: a wheel digest differs between receipts: "
        "director_ai-1-py3-none-any.whl",
    )


def test_the_receipts_must_bind_the_expected_tree() -> None:
    """Refuse a receipt of another tree, a split staged flag and a bad expected tree."""
    given = receipts()
    given["core+pro.json"]["tree"] = "5" * 40
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        f"core+pro.json: the receipt tested tree {'5' * 40}, not {TREE}",
    )
    assert receipt_refusals(CONTRACT, receipts(), tree="6" * 40) == tuple(
        f"{name}: the receipt tested tree {TREE}, not {'6' * 40}"
        for name in ("core+full.json", "core+pro.json", "core.json")
    )
    given = receipts()
    given["core.json"]["staged"] = True
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        "The receipts disagree on whether the tree was staged",
    )
    assert receipt_refusals(CONTRACT, receipts(), tree="HEAD") == (
        "Invalid expected tree: 'HEAD'",
    )


def _probes(given: dict[str, dict[str, object]], name: str) -> dict[str, object]:
    """Return one receipt's probe results, typed for editing."""
    probes = given[name]["probes"]
    assert isinstance(probes, dict)
    return probes


def _redactor(given: dict[str, dict[str, object]], name: str) -> dict[str, object]:
    """Return one receipt's redactor result, typed for editing."""
    result = _probes(given, name)["pii-redactor"]
    assert isinstance(result, dict)
    return result


def test_a_probe_records_the_conditions_its_installation_requires() -> None:
    """Require the strict-trail condition: passed on Pro and Full, stated on Core."""
    given = receipts()
    for name, state in (
        ("core.json", "not_applicable"),
        ("core+pro.json", "passed"),
        ("core+full.json", "passed"),
    ):
        _probes(given, name)["pii-redactor"] = {
            "passed": True,
            "detail": "ok",
            "conditions": {"strict-trail-refusal": state},
        }
    assert receipt_refusals(CONTRACT, given, tree=TREE) == ()
    weak = copy.deepcopy(given)
    _redactor(weak, "core+pro.json")["conditions"] = {
        "strict-trail-refusal": "not_applicable"
    }
    del _redactor(weak, "core.json")["conditions"]
    assert receipt_refusals(CONTRACT, weak, tree=TREE) == (
        "core+pro.json: probe pii-redactor must pass its condition "
        "strict-trail-refusal on core+pro",
        "core.json: probe pii-redactor does not record its condition "
        "strict-trail-refusal",
    )
    failed = copy.deepcopy(given)
    _probes(failed, "core.json")["pii-redactor"] = {"passed": False, "detail": "x"}
    assert receipt_refusals(CONTRACT, failed, tree=TREE) == ()
    unknown = copy.deepcopy(given)
    _redactor(unknown, "core.json")["conditions"] = {"x": "maybe"}
    assert receipt_refusals(CONTRACT, unknown, tree=TREE) == (
        "core.json: invalid probes",
    )


def test_every_installation_needs_a_hosted_receipt_and_its_negatives() -> None:
    """Refuse a missing installation and a failed or missing negative probe."""
    given = receipts()
    del given["core+full.json"]
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        "No hosted receipt for the core+full installation",
    )
    given = receipts()
    negatives = given["core.json"]["negatives"]
    assert isinstance(negatives, dict)
    negatives["tampered-packet-refused"] = {"passed": False, "detail": "accepted"}
    del negatives["paid-import-boundary"]
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        "core.json: the negative probe failed or did not run: paid-import-boundary",
        "core.json: the negative probe failed or did not run: tampered-packet-refused",
    )


def test_a_row_fails_where_its_probe_failed_or_did_not_run() -> None:
    """Refuse a row whose probe failed on any containing installation."""
    given = receipts()
    probes = given["core+full.json"]["probes"]
    assert isinstance(probes, dict)
    probes["core-probe"] = {"passed": False, "detail": "broke"}
    del probes["pro-probe"]
    assert receipt_refusals(CONTRACT, given, tree=TREE) == (
        "core-row: probe core-probe failed or did not run on core+full",
        "pro-row: probe pro-probe failed or did not run on core+full",
    )


def test_a_row_without_a_claimed_installation_needs_no_receipt() -> None:
    """Skip a row that claims no availability."""
    unclaimed = dataclasses.replace(
        CORE_ROW, identifier="free", installation=None, probe=None
    )
    changed = dataclasses.replace(
        CONTRACT, operations=CONTRACT.operations + (unclaimed,)
    )
    assert receipt_refusals(changed, receipts(), tree=TREE) == ()


def test_a_model_backed_row_needs_a_declared_positive_receipt() -> None:
    """Refuse a model-backed row until a declared host shows it passing."""
    model = dataclasses.replace(PRO_ROW, model_backed=True)
    changed = dataclasses.replace(CONTRACT, operations=(CORE_ROW, model))
    assert receipt_refusals(changed, receipts(), tree=TREE) == (
        "pro-row: no positive receipt from a declared host",
    )
    given = receipts()
    given["declared.json"] = receipt("core+pro", host="declared")
    assert receipt_refusals(changed, given, tree=TREE) == ()
    missing = copy.deepcopy(given)
    missing["declared.json"]["model"] = None
    assert receipt_refusals(changed, missing, tree=TREE) == (
        "declared.json: a declared-host receipt names no verified model",
    )


def test_receipts_are_read_from_directories(tmp_path: Path) -> None:
    """Read JSON receipts; refuse a non-object, an empty directory and a duplicate."""
    hosted, declared = tmp_path / "hosted", tmp_path / "declared"
    hosted.mkdir()
    declared.mkdir()
    for name, value in receipts().items():
        (hosted / name).write_text(json.dumps(value), encoding="utf-8")
    assert check_receipts(CONTRACT, [hosted], tree=TREE) == ()
    with pytest.raises(ValueError, match="No receipt"):
        read_receipts(declared)
    (declared / "core.json").write_text(json.dumps(receipt("core")), encoding="utf-8")
    with pytest.raises(ValueError, match="Two receipts share a name"):
        check_receipts(CONTRACT, [hosted, declared], tree=TREE)
    (declared / "core.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        read_receipts(declared)
