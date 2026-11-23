# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the support contract's records, validation and authority
"""Pin every refusal of the contract validator and the single-authority check."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import os
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

import director_ai.support_contract as reader
import director_ai.support_contract_catalog as catalog
from director_ai.support_contract import (
    MEANING,
    SCHEMA,
    Custody,
    Operation,
    ProfileDefault,
    SupportContract,
    SupportContractError,
    contract_refusals,
    load_contract,
    verify_single_authority,
)

DIGEST = "a" * 64
EXPORTS = ("Alpha", "Beta", "Gamma")
COMMANDS = ("report", "serve", "status")
SUBCOMMANDS = {"report": ("daily", "weekly")}
BASE = Operation(
    identifier="alpha-row",
    title="Alpha operation",
    support="experimental",
    activation="default",
    installation="core",
    probe="alpha-probe",
    exports=("Alpha",),
)
QUALIFIED = (
    Custody("classification", "REVIEW_20261123T121107Z", DIGEST),
    Custody("evidence", "REVIEW_20261123T120810Z", "b" * 64),
)


def operation(**overrides: Any) -> Operation:
    """Return the base operation with some fields replaced.

    Parameters
    ----------
    **overrides : Any
        Field values to use instead.

    Returns
    -------
    Operation
        The operation.
    """
    return dataclasses.replace(BASE, **overrides)


def contract(*operations: Operation, **overrides: Any) -> SupportContract:
    """Return a contract over the test inventory, every other name experimental.

    Parameters
    ----------
    *operations : Operation
        The rows; by default the base row.
    **overrides : Any
        Field values to use instead.

    Returns
    -------
    SupportContract
        The contract.
    """
    rows = operations or (BASE,)
    named = {name for row in rows for name in row.exports}
    commands = {name for row in rows for name in row.commands}
    base = SupportContract(
        schema=SCHEMA,
        meaning=MEANING,
        operations=rows,
        experimental_exports=tuple(n for n in EXPORTS if n not in named),
        experimental_commands=tuple(n for n in COMMANDS if n not in commands),
        experimental_namespaces=("director_ai.model_lifecycle",),
        production_defaults=(ProfileDefault("use_nli", rows[0].identifier),),
    )
    return dataclasses.replace(base, **overrides)


def refusals(
    value: SupportContract, extras: tuple[str, ...] | None = ("server",)
) -> tuple[str, ...]:
    """Return the refusals of a contract against the test inventory.

    Parameters
    ----------
    value : SupportContract
        The contract.
    extras : tuple[str, ...] | None
        The known extras.

    Returns
    -------
    tuple[str, ...]
        The refusals.
    """
    return contract_refusals(
        value,
        exports=EXPORTS,
        commands=COMMANDS,
        subcommands=SUBCOMMANDS,
        extras=extras,
    )


def test_a_valid_contract_has_no_refusal() -> None:
    """Accept the base contract, a qualified row with custody and a split command."""
    qualified = operation(support="production-qualified", custody=QUALIFIED)
    split = operation(
        identifier="report-row",
        exports=(),
        commands=("report",),
        subcommands=("daily",),
        probe="report-probe",
    )
    assert refusals(contract(qualified, split)) == ()
    assert refusals(contract(), extras=None) == ()


def test_the_schema_and_meaning_must_be_the_reviewed_ones() -> None:
    """Refuse another schema version or a reworded meaning sentence."""
    assert refusals(contract(schema="director-ai.support-contract.v2")) == (
        "Unknown support contract schema: 'director-ai.support-contract.v2'",
    )
    assert refusals(contract(meaning=MEANING + " Mostly.")) == (
        "The meaning sentence differs from the reviewed one",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"identifier": "Alpha_Row"}, "Invalid operation identifier: 'Alpha_Row'"),
        ({"title": "  "}, "alpha-row: the title is empty"),
        ({"support": "beta"}, "alpha-row: unknown support 'beta'"),
        ({"activation": "always"}, "alpha-row: unknown activation 'always'"),
        ({"installation": "core+lite"}, "alpha-row: unknown installation 'core+lite'"),
        ({"probe": None}, "alpha-row: a claimed installation needs a valid probe"),
        (
            {"probe": "Bad Probe"},
            "alpha-row: a claimed installation needs a valid probe",
        ),
        ({"installation": None}, "alpha-row: a probe without a claimed installation"),
        (
            {"prerequisites": ("vector", "nli")},
            "alpha-row: prerequisites are not sorted and unique",
        ),
        ({"prerequisites": ("vector",)}, "alpha-row: unknown prerequisite 'vector'"),
        (
            {"exports": ("Alpha", "Alpha")},
            "alpha-row: exports are not sorted and unique",
        ),
        ({"routes": ("health",)}, "alpha-row: a route must start with '/': 'health'"),
        (
            {"imports": ("other.module",)},
            "alpha-row: an import outside the package: 'other.module'",
        ),
        (
            {"subcommands": ("daily",)},
            "alpha-row: subcommands need exactly one command",
        ),
    ],
)
def test_each_operation_field_is_validated(
    overrides: dict[str, object], message: str
) -> None:
    """Refuse each malformed operation field by name."""
    assert message in refusals(contract(operation(**overrides)))


def test_a_qualified_row_must_claim_an_installation() -> None:
    """Refuse a production-qualified row that claims no installation."""
    row = operation(
        support="production-qualified", installation=None, probe=None, custody=QUALIFIED
    )
    assert refusals(contract(row)) == (
        "alpha-row: a production-qualified row claims no installation",
    )


def test_an_experimental_row_may_claim_no_installation() -> None:
    """Accept an experimental row without an installation or a probe."""
    assert refusals(contract(operation(installation=None, probe=None))) == ()


@pytest.mark.parametrize(
    ("custody", "message"),
    [
        (
            (Custody("friend", "REVIEW_20261123T121107Z", DIGEST),),
            "alpha-row: unknown custody role 'friend'",
        ),
        (
            (Custody("evidence", " ", DIGEST),),
            "alpha-row: a custody record is not a record stem: ' '",
        ),
        (
            (Custody("evidence", "review_20261123T121107Z.md", DIGEST),),
            "alpha-row: a custody record is not a record stem: "
            "'review_20261123T121107Z.md'",
        ),
        (
            (Custody("evidence", "REVIEW_20261123T1211Z", DIGEST),),
            "alpha-row: a custody record is not a record stem: 'REVIEW_20261123T1211Z'",
        ),
        (
            (Custody("evidence", "REVIEW_20261123T121107Z", "abc"),),
            "alpha-row: invalid custody digest for 'REVIEW_20261123T121107Z'",
        ),
        (
            (Custody("evidence", "REVIEW_20261123T121107Z", "A" * 64),),
            "alpha-row: invalid custody digest for 'REVIEW_20261123T121107Z'",
        ),
    ],
)
def test_each_custody_record_is_validated(
    custody: tuple[Custody, ...], message: str
) -> None:
    """Refuse an unknown role, a label that is not a stem, or a bad digest."""
    assert message in refusals(contract(operation(custody=custody)))


def test_a_qualified_row_needs_its_signed_records() -> None:
    """Need classification and evidence, and quality for a model-backed row."""
    bare = operation(support="production-qualified")
    assert refusals(contract(bare)) == (
        "alpha-row: a production-qualified row needs a classification custody record",
        "alpha-row: a production-qualified row needs a evidence custody record",
    )
    model = operation(
        support="production-qualified", model_backed=True, custody=QUALIFIED
    )
    assert refusals(contract(model)) == (
        "alpha-row: a production-qualified row needs a quality custody record",
    )
    signed = QUALIFIED + (Custody("quality", "GATE_Q_20261201T000000Z", "c" * 64),)
    assert refusals(contract(dataclasses.replace(model, custody=signed))) == ()


def test_every_export_is_classified_exactly_once() -> None:
    """Refuse an export that a row and the experimental list both name."""
    twice = contract(experimental_exports=("Alpha", "Beta", "Gamma"))
    assert refusals(twice) == ("An export is classified 2 times: Alpha",)


def test_an_unknown_or_missing_export_is_named() -> None:
    """Name an export the inventory lacks and one the contract misses."""
    assert refusals(contract(experimental_exports=("Delta", "Gamma"))) == (
        "An unknown export is classified: Delta",
        "An export is not classified: Beta",
    )


def test_every_command_is_classified_exactly_once() -> None:
    """Refuse a command classified twice, whole and split, or not at all."""
    whole = operation(
        identifier="serve-row", exports=(), commands=("serve",), probe="serve-probe"
    )
    assert refusals(
        contract(BASE, whole, experimental_commands=("report", "serve", "status"))
    ) == ("A command is classified more than once: serve",)
    split = operation(
        identifier="report-row",
        exports=(),
        commands=("report",),
        subcommands=("daily",),
        probe="report-probe",
    )
    assert refusals(
        contract(BASE, split, experimental_commands=("report", "serve", "status"))
    ) == ("A command is classified more than once: report",)
    assert refusals(contract(experimental_commands=("report", "serve", "stop"))) == (
        "An unknown command is classified: stop",
        "A command is not classified: status",
    )


def test_a_split_command_is_checked_at_subcommand_level() -> None:
    """Refuse a repeated, an unknown or an uninventoried subcommand."""
    first = operation(
        identifier="report-a",
        exports=(),
        commands=("report",),
        subcommands=("daily",),
        probe="report-a",
    )
    second = dataclasses.replace(first, identifier="report-b", probe="report-b")
    assert refusals(contract(BASE, first, second)) == (
        "A subcommand is classified more than once: report daily",
    )
    unknown = dataclasses.replace(first, subcommands=("monthly",))
    assert refusals(contract(BASE, unknown)) == (
        "An unknown subcommand is classified: report monthly",
    )
    other = operation(
        identifier="status-row",
        exports=(),
        commands=("status",),
        subcommands=("now",),
        probe="status-probe",
    )
    assert "The subcommands of status were not inventoried" in refusals(
        contract(BASE, other)
    )


def test_identifiers_lists_and_defaults_are_validated() -> None:
    """Refuse a repeated identifier, unsorted lists and an unowned default."""
    beta = operation(exports=("Beta",))
    assert "An operation identifier repeats: alpha-row" in refusals(
        contract(BASE, beta)
    )
    assert refusals(contract(experimental_exports=("Gamma", "Beta"))) == (
        "The experimental exports are not sorted and unique",
    )
    assert refusals(contract(experimental_commands=("status", "serve", "report"))) == (
        "The experimental commands are not sorted and unique",
    )
    assert refusals(contract(experimental_namespaces=("b", "a"))) == (
        "The experimental namespaces are not sorted and unique",
    )
    defaults = (
        ProfileDefault("use_nli", "alpha-row"),
        ProfileDefault("log_json", "alpha-row"),
    )
    assert refusals(contract(production_defaults=defaults)) == (
        "The production defaults are not sorted and unique",
    )
    assert refusals(
        contract(production_defaults=(ProfileDefault("use_nli", "ghost"),))
    ) == ("A production default has an unknown owner: use_nli ghost",)


def test_an_operation_is_found_by_identifier() -> None:
    """Return the named row and refuse an unknown one."""
    assert contract().operation("alpha-row") is BASE
    with pytest.raises(KeyError):
        contract().operation("ghost")


def _record_digest(data: bytes) -> str:
    """Return a file digest in the wheel RECORD form."""
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


class RecordedDistribution(metadata.Distribution):
    """An installed distribution whose RECORD lists one file at a chosen place."""

    def __init__(self, name: str, record: str, site: Path) -> None:
        """Hold the name, the RECORD text and the site directory.

        Parameters
        ----------
        name : str
            The project name.
        record : str
            The RECORD file's text.
        site : Path
            The directory its paths are relative to.
        """
        self._name, self._record, self._site = name, record, site

    def read_text(self, filename: str) -> str | None:
        """Return the stand-in's metadata files."""
        if filename == "METADATA":
            return f"Metadata-Version: 2.1\nName: {self._name}\nVersion: 0\n"
        if filename == "RECORD":
            return self._record
        return None

    def locate_file(self, path: str | os.PathLike[str]) -> Path:
        """Return a recorded path inside the site directory."""
        return self._site / str(path)


CATALOG_FILE = Path(catalog.__file__ or "").resolve()
SITE = CATALOG_FILE.parents[1]
RELATIVE = "director_ai/support_contract_catalog.py"


def test_no_recorded_copy_means_the_source_module() -> None:
    """Report a module no distribution records as unverified, as in a checkout."""
    assert verify_single_authority(catalog, []) is False
    unrelated = RecordedDistribution("other", "other/module.py,,\n", SITE)
    assert verify_single_authority(catalog, [unrelated]) is False


def test_the_core_copy_must_match_its_record(tmp_path: Path) -> None:
    """Accept the recorded bytes; refuse other bytes, no hash or another file."""
    good = f"{RELATIVE},sha256={_record_digest(CATALOG_FILE.read_bytes())},1\n"
    core = RecordedDistribution("Director_AI", good, SITE)
    assert verify_single_authority(catalog, [core]) is True
    bad = f"{RELATIVE},sha256={_record_digest(b'other')},1\n"
    with pytest.raises(
        SupportContractError, match="differs from the director-ai record"
    ):
        verify_single_authority(
            catalog, [RecordedDistribution("director-ai", bad, SITE)]
        )
    unhashed = f"{RELATIVE},,\n"
    with pytest.raises(
        SupportContractError, match="differs from the director-ai record"
    ):
        verify_single_authority(
            catalog, [RecordedDistribution("director-ai", unhashed, SITE)]
        )
    copy = tmp_path / RELATIVE
    copy.parent.mkdir(parents=True)
    copy.write_bytes(CATALOG_FILE.read_bytes())
    with pytest.raises(SupportContractError, match="imported from outside director-ai"):
        verify_single_authority(
            catalog, [RecordedDistribution("director-ai", good, tmp_path)]
        )


def test_another_distribution_may_not_ship_the_contract() -> None:
    """Refuse a paid overlay that records the catalog path."""
    overlay = RecordedDistribution("director-ai-pro", f"{RELATIVE},,\n", SITE)
    with pytest.raises(SupportContractError, match="Another distribution ships"):
        verify_single_authority(catalog, [overlay])


def test_a_module_without_a_file_is_refused() -> None:
    """Refuse a module that has no source file."""
    import builtins

    with pytest.raises(SupportContractError, match="has no file"):
        verify_single_authority(builtins, [])


def test_load_contract_checks_both_files_and_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the catalog, refuse an overlay of either file, and refuse a schema."""
    assert load_contract([]) is catalog.CONTRACT
    assert load_contract() is catalog.CONTRACT
    reader_site = Path(reader.__file__ or "").resolve().parents[1]
    overlay = RecordedDistribution(
        "director-ai-full", "director_ai/support_contract.py,,\n", reader_site
    )
    with pytest.raises(SupportContractError, match="Another distribution ships"):
        load_contract([overlay])
    catalog_overlay = RecordedDistribution("director-ai-pro", f"{RELATIVE},,\n", SITE)
    with pytest.raises(
        SupportContractError, match=f"Another distribution ships.*{RELATIVE}"
    ):
        load_contract([catalog_overlay])
    altered = RecordedDistribution(
        "director-ai", f"{RELATIVE},sha256={_record_digest(b'other')},1\n", SITE
    )
    with pytest.raises(
        SupportContractError, match="differs from the director-ai record"
    ):
        load_contract([altered])
    monkeypatch.setattr(
        catalog, "CONTRACT", dataclasses.replace(catalog.CONTRACT, schema="v0")
    )
    with pytest.raises(SupportContractError, match="Unknown support contract schema"):
        load_contract([])
