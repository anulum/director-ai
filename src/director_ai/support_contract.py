# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — the production support contract: records, validation, authority
"""Read and validate the production support contract.

The contract says, for each named operation, three separate things:

- **availability**: the smallest installation that runs it (``core``,
  ``core+pro`` or ``core+full``) and its runtime prerequisites;
- **support**: ``production-qualified`` or ``experimental``;
- **activation**: ``default``, ``opt-in`` or ``command``.

Every public export and every CLI command is classified exactly once, either
in an operation or in the explicit experimental lists. Anything outside that
declared boundary is experimental. The records live in
:mod:`director_ai.support_contract_catalog`, which ships in the Core
distribution only. :func:`load_contract` checks, before it returns a record,
that the imported reader and catalog files are the ones the ``director-ai``
distribution recorded and that no other installed distribution records them,
so a paid overlay cannot replace the contract.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType

SCHEMA = "director-ai.support-contract.v1"
"""The schema the reader accepts; any other version is refused."""

MEANING = (
    "production-qualified means qualified and maintained for production use on "
    "the named installation. It is not a service-level, warranty or "
    "support-contract commitment, and it grants no licence entitlement."
)
"""The sentence that heads the catalog and every rendered table."""

INSTALLATIONS = ("core", "core+pro", "core+full")
"""The installed distributions an operation may need, smallest first."""

SUPPORT_LEVELS = ("production-qualified", "experimental")
ACTIVATIONS = ("default", "opt-in", "command")
CUSTODY_ROLES = ("classification", "evidence", "owner", "quality")
"""Who signed a row: the classification and evidence reviewers, the owner, and the quality gates."""

DISTRIBUTION = "director-ai"
"""The Core distribution, the only one that may ship the contract files."""

_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RECORD_STEM = re.compile(r"[A-Z]+(?:_[A-Z]+)*_[0-9]{8}T[0-9]{6}Z")


class SupportContractError(ValueError):
    """The contract or its installed copy does not qualify."""


@dataclass(frozen=True)
class Custody:
    """One signed record behind a row's classification.

    Attributes
    ----------
    role : str
        One of :data:`CUSTODY_ROLES`.
    record : str
        The record's opaque stem, its kind and UTC time, such as
        ``REVIEW_20261123T121107Z``. It is only a label: stems can collide.
    digest : str
        SHA-256 of the record's bytes, which identifies it.
    """

    role: str
    record: str
    digest: str


@dataclass(frozen=True)
class Operation:
    """One named operation and its availability, support and activation.

    Attributes
    ----------
    identifier : str
        A lower-case, hyphenated name, unique in the contract.
    title : str
        What the operation does, in a few words.
    support : str
        One of :data:`SUPPORT_LEVELS`.
    activation : str
        One of :data:`ACTIVATIONS`.
    installation : str | None
        The smallest installation that runs it, one of :data:`INSTALLATIONS`;
        None claims no availability, which only an experimental row may do.
    probe : str | None
        The installed probe whose receipt shows it runs on ``installation``;
        required exactly when an installation is claimed.
    prerequisites : tuple[str, ...]
        The package extras it needs on top of the installation, sorted.
    exports : tuple[str, ...]
        The public exports it covers, sorted.
    commands : tuple[str, ...]
        The CLI commands it covers, sorted.
    subcommands : tuple[str, ...]
        With exactly one command, the subcommands it covers; the command's
        other subcommands stay experimental.
    imports : tuple[str, ...]
        Direct imports it covers, inside the declared boundary.
    routes : tuple[str, ...]
        HTTP routes it covers.
    model_backed : bool
        Whether its result rests on a model-backed verdict.
    custody : tuple[Custody, ...]
        The signed records behind its classification.
    note : str
        A scope statement the rendered table prints beside the row.
    """

    identifier: str
    title: str
    support: str
    activation: str
    installation: str | None = None
    probe: str | None = None
    prerequisites: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    subcommands: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    routes: tuple[str, ...] = ()
    model_backed: bool = False
    custody: tuple[Custody, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ProfileDefault:
    """One setting the production profile turns on, and the row that owns it.

    Attributes
    ----------
    setting : str
        The configuration key the ``production`` profile sets to true.
    owner : str
        The identifier of the operation that owns the setting.
    """

    setting: str
    owner: str


@dataclass(frozen=True)
class SupportContract:
    """The whole contract.

    Attributes
    ----------
    schema : str
        Must equal :data:`SCHEMA`.
    meaning : str
        Must equal :data:`MEANING`.
    operations : tuple[Operation, ...]
        The named operations, in display order.
    experimental_exports : tuple[str, ...]
        Every other public export, sorted.
    experimental_commands : tuple[str, ...]
        Every other CLI command, sorted.
    experimental_namespaces : tuple[str, ...]
        Importable namespaces named experimental although they export nothing
        from the package root.
    production_defaults : tuple[ProfileDefault, ...]
        Every setting the ``production`` profile turns on, sorted by setting.
    """

    schema: str
    meaning: str
    operations: tuple[Operation, ...]
    experimental_exports: tuple[str, ...]
    experimental_commands: tuple[str, ...]
    experimental_namespaces: tuple[str, ...]
    production_defaults: tuple[ProfileDefault, ...]

    def operation(self, identifier: str) -> Operation:
        """Return the operation with this identifier.

        Parameters
        ----------
        identifier : str
            An operation identifier.

        Returns
        -------
        Operation
            The operation.

        Raises
        ------
        KeyError
            If no operation has that identifier.
        """
        for operation in self.operations:
            if operation.identifier == identifier:
                return operation
        raise KeyError(identifier)


def _sorted_unique(values: Sequence[str]) -> bool:
    """Return whether a sequence is strictly increasing."""
    return all(a < b for a, b in zip(values, values[1:], strict=False))


def _operation_refusals(
    operation: Operation, extras: frozenset[str] | None
) -> list[str]:
    """Return the refusals of one operation's own fields."""
    name = operation.identifier
    refusals = []
    if not _IDENTIFIER.fullmatch(name):
        refusals.append(f"Invalid operation identifier: {name!r}")
    if not operation.title.strip():
        refusals.append(f"{name}: the title is empty")
    for value, allowed, field in (
        (operation.support, SUPPORT_LEVELS, "support"),
        (operation.activation, ACTIVATIONS, "activation"),
    ):
        if value not in allowed:
            refusals.append(f"{name}: unknown {field} {value!r}")
    if operation.installation is None:
        if operation.support == "production-qualified":
            refusals.append(
                f"{name}: a production-qualified row claims no installation"
            )
        if operation.probe is not None:
            refusals.append(f"{name}: a probe without a claimed installation")
    else:
        if operation.installation not in INSTALLATIONS:
            refusals.append(f"{name}: unknown installation {operation.installation!r}")
        if operation.probe is None or not _IDENTIFIER.fullmatch(operation.probe):
            refusals.append(f"{name}: a claimed installation needs a valid probe")
    for field, values in (
        ("prerequisites", operation.prerequisites),
        ("exports", operation.exports),
        ("commands", operation.commands),
        ("subcommands", operation.subcommands),
        ("imports", operation.imports),
        ("routes", operation.routes),
    ):
        if not _sorted_unique(values):
            refusals.append(f"{name}: {field} are not sorted and unique")
    if extras is not None:
        refusals.extend(
            f"{name}: unknown prerequisite {extra!r}"
            for extra in operation.prerequisites
            if extra not in extras
        )
    if operation.subcommands and len(operation.commands) != 1:
        refusals.append(f"{name}: subcommands need exactly one command")
    refusals.extend(
        f"{name}: an import outside the package: {module!r}"
        for module in operation.imports
        if module != "director_ai" and not module.startswith("director_ai.")
    )
    refusals.extend(
        f"{name}: a route must start with '/': {route!r}"
        for route in operation.routes
        if not route.startswith("/")
    )
    refusals.extend(_custody_refusals(operation))
    return refusals


def _custody_refusals(operation: Operation) -> list[str]:
    """Return the refusals of one operation's custody records."""
    name = operation.identifier
    refusals = []
    for record in operation.custody:
        if record.role not in CUSTODY_ROLES:
            refusals.append(f"{name}: unknown custody role {record.role!r}")
        if not _RECORD_STEM.fullmatch(record.record):
            refusals.append(
                f"{name}: a custody record is not a record stem: {record.record!r}"
            )
        if not _DIGEST.fullmatch(record.digest):
            refusals.append(f"{name}: invalid custody digest for {record.record!r}")
    if operation.support == "production-qualified":
        roles = {record.role for record in operation.custody}
        needed = {"classification", "evidence"}
        if operation.model_backed:
            needed.add("quality")
        refusals.extend(
            f"{name}: a production-qualified row needs a {role} custody record"
            for role in sorted(needed - roles)
        )
    return refusals


def _coverage_refusals(
    contract: SupportContract,
    exports: Iterable[str],
    commands: Iterable[str],
    subcommands: Mapping[str, Iterable[str]],
) -> list[str]:
    """Return the refusals of the export and command classification."""
    known_exports, known_commands = set(exports), set(commands)
    refusals: list[str] = []
    named = Counter(
        name for operation in contract.operations for name in operation.exports
    )
    named.update(contract.experimental_exports)
    refusals.extend(
        f"An export is classified {count} times: {name}"
        for name, count in sorted(named.items())
        if count > 1
    )
    refusals.extend(
        f"An unknown export is classified: {name}"
        for name in sorted(set(named) - known_exports)
    )
    refusals.extend(
        f"An export is not classified: {name}"
        for name in sorted(known_exports - set(named))
    )
    whole: Counter[str] = Counter(contract.experimental_commands)
    split: dict[str, list[Operation]] = {}
    for operation in contract.operations:
        for command in operation.commands:
            if operation.subcommands:
                split.setdefault(command, []).append(operation)
            else:
                whole[command] += 1
    for command, count in sorted(whole.items()):
        if count > 1 or command in split:
            refusals.append(f"A command is classified more than once: {command}")
    for command, operations in sorted(split.items()):
        seen = Counter(sub for operation in operations for sub in operation.subcommands)
        refusals.extend(
            f"A subcommand is classified more than once: {command} {sub}"
            for sub, count in sorted(seen.items())
            if count > 1
        )
        if command not in subcommands:
            refusals.append(f"The subcommands of {command} were not inventoried")
            continue
        refusals.extend(
            f"An unknown subcommand is classified: {command} {sub}"
            for sub in sorted(set(seen) - set(subcommands[command]))
        )
    classified = set(whole) | set(split)
    refusals.extend(
        f"An unknown command is classified: {command}"
        for command in sorted(classified - known_commands)
    )
    refusals.extend(
        f"A command is not classified: {command}"
        for command in sorted(known_commands - classified)
    )
    return refusals


def _defaults_refusals(contract: SupportContract) -> list[str]:
    """Return the refusals of the production defaults' ownership."""
    settings = [default.setting for default in contract.production_defaults]
    refusals = []
    if not _sorted_unique(settings):
        refusals.append("The production defaults are not sorted and unique")
    owners = {operation.identifier for operation in contract.operations}
    refusals.extend(
        f"A production default has an unknown owner: {default.setting} {default.owner}"
        for default in contract.production_defaults
        if default.owner not in owners
    )
    return refusals


def contract_refusals(
    contract: SupportContract,
    *,
    exports: Iterable[str],
    commands: Iterable[str],
    subcommands: Mapping[str, Iterable[str]] | None = None,
    extras: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return every reason the contract does not qualify against an inventory.

    Parameters
    ----------
    contract : SupportContract
        The contract to check.
    exports : Iterable[str]
        Every name the package root exports.
    commands : Iterable[str]
        Every CLI command.
    subcommands : Mapping[str, Iterable[str]] | None
        For each command that rows split, its subcommands; a split command
        missing here is refused.
    extras : Iterable[str] | None
        The distribution's extras, to check prerequisites against; None skips
        that check, as an installed reader cannot read the project file.

    Returns
    -------
    tuple[str, ...]
        The refusals in a stable order; empty means the contract qualifies.
    """
    refusals = []
    if contract.schema != SCHEMA:
        refusals.append(f"Unknown support contract schema: {contract.schema!r}")
    if contract.meaning != MEANING:
        refusals.append("The meaning sentence differs from the reviewed one")
    identifiers = Counter(operation.identifier for operation in contract.operations)
    refusals.extend(
        f"An operation identifier repeats: {name}"
        for name, count in sorted(identifiers.items())
        if count > 1
    )
    known = frozenset(extras) if extras is not None else None
    for operation in contract.operations:
        refusals.extend(_operation_refusals(operation, known))
    for field, values in (
        ("experimental exports", contract.experimental_exports),
        ("experimental commands", contract.experimental_commands),
        ("experimental namespaces", contract.experimental_namespaces),
    ):
        if not _sorted_unique(values):
            refusals.append(f"The {field} are not sorted and unique")
    refusals.extend(_coverage_refusals(contract, exports, commands, subcommands or {}))
    refusals.extend(_defaults_refusals(contract))
    return tuple(refusals)


def _record_digest(data: bytes) -> str:
    """Return a file's digest in the wheel RECORD form."""
    digest = hashlib.sha256(data).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _distribution_name(distribution: metadata.Distribution) -> str:
    """Return a distribution's normalised project name."""
    name = distribution.metadata["Name"] or ""
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_single_authority(
    module: ModuleType,
    distributions: Iterable[metadata.Distribution] | None = None,
) -> bool:
    """Refuse a contract file that an overlay could have replaced.

    Two limits hold by design. A source or editable run, where no
    distribution records the file, is unverified and must never be cited as
    authority. And the check compares the file on disk with the record at
    check time: it does not cover the executed bytecode or a change between
    import and check, since a writer to site-packages already owns the
    process.

    Parameters
    ----------
    module : ModuleType
        An imported contract module.
    distributions : Iterable[metadata.Distribution] | None
        The installed distributions; None reads them from the environment.

    Returns
    -------
    bool
        True when the Core distribution records the imported file with a
        matching sha256; False when no distribution records it, which is
        a source checkout and verifies nothing.

    Raises
    ------
    SupportContractError
        If another distribution records the module's path, or the Core
        distribution records it and the imported file is not that file or
        its bytes differ from the record.
    """
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise SupportContractError(
            f"The contract module has no file: {module.__name__}"
        )
    imported = Path(origin).resolve()
    relative = module.__name__.replace(".", "/") + ".py"
    found = metadata.distributions() if distributions is None else distributions
    verified = False
    for distribution in found:
        entries = [
            entry for entry in distribution.files or () if str(entry) == relative
        ]
        if not entries:
            continue
        name = _distribution_name(distribution)
        if name != DISTRIBUTION:
            raise SupportContractError(
                f"Another distribution ships the support contract: {name} {relative}"
            )
        entry = entries[0]
        located = Path(str(distribution.locate_file(entry))).resolve()
        if located != imported:
            raise SupportContractError(
                f"The support contract was imported from outside {DISTRIBUTION}: {relative}"
            )
        recorded = entry.hash
        if (
            recorded is None
            or recorded.mode != "sha256"
            or recorded.value != _record_digest(imported.read_bytes())
        ):
            raise SupportContractError(
                f"The support contract differs from the {DISTRIBUTION} record: {relative}"
            )
        verified = True
    return verified


def load_contract(
    distributions: Sequence[metadata.Distribution] | None = None,
) -> SupportContract:
    """Return the installed contract after checking that Core owns it.

    Parameters
    ----------
    distributions : Sequence[metadata.Distribution] | None
        The installed distributions; None reads them from the environment.

    Returns
    -------
    SupportContract
        The contract from :mod:`director_ai.support_contract_catalog`.

    Raises
    ------
    SupportContractError
        If either contract file fails :func:`verify_single_authority`, or the
        catalog's schema is not :data:`SCHEMA`.
    """
    import sys

    from director_ai import support_contract_catalog as catalog

    found = list(metadata.distributions()) if distributions is None else distributions
    verify_single_authority(sys.modules[__name__], found)
    verify_single_authority(catalog, found)
    if catalog.CONTRACT.schema != SCHEMA:
        raise SupportContractError(
            f"Unknown support contract schema: {catalog.CONTRACT.schema!r}"
        )
    return catalog.CONTRACT
