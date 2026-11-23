# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests of the shipped support contract records
"""Pin the shipped catalog to the source tree and to the reviewers' rulings."""

from __future__ import annotations

import re
from pathlib import Path

from director_ai.support_contract import load_contract
from director_ai.support_contract_catalog import CONTRACT
from tools.support_contract_check import tree_refusals

ROOT = Path(__file__).resolve().parents[1]


def test_the_catalog_qualifies_against_the_source_tree() -> None:
    """Classify every export and command and match every source fact."""
    assert tree_refusals(CONTRACT, ROOT) == ()


def test_the_catalog_loads_through_the_authority_check() -> None:
    """Return the shipped records after the single-authority check."""
    assert load_contract() is CONTRACT


def test_model_lifecycle_is_experimental() -> None:
    """Name the model-lifecycle contracts experimental (C-6)."""
    assert CONTRACT.experimental_namespaces == ("director_ai.model_lifecycle",)


def test_rows_that_rest_on_a_model_verdict_are_experimental() -> None:
    """Keep model scoring and every row built on its verdict experimental (K-1, K-2)."""
    for identifier in (
        "model-scoring",
        "production-guard",
        "stream-guard",
        "streaming-contradiction-halt",
        "injection-detection",
    ):
        row = CONTRACT.operation(identifier)
        assert row.model_backed and row.support == "experimental", identifier


def test_streaming_is_opt_in() -> None:
    """Never turn streaming contradiction handling on by default (K-3)."""
    assert CONTRACT.operation("streaming-contradiction-halt").activation == "opt-in"


def test_the_transport_rows_name_their_verdict_routes_experimental() -> None:
    """State beside the transport rows that their verdict routes are experimental (T-2)."""
    assert (
        "/v1/review is experimental" in CONTRACT.operation("rest-server-transport").note
    )
    assert "experimental" in CONTRACT.operation("proxy-transport").note
    for identifier in (
        "rest-server-transport",
        "proxy-transport",
        "compliance-reports",
    ):
        assert CONTRACT.operation(identifier).installation == "core+pro", identifier


def test_compliance_is_split_at_subcommand_level() -> None:
    """Split compliance by subcommand: the reports and the evidence kit (b0f9, O-2)."""
    row = CONTRACT.operation("compliance-reports")
    assert (row.commands, row.subcommands) == (
        ("compliance",),
        ("governance", "report"),
    )
    kit = CONTRACT.operation("compliance-evidence-kit")
    assert (kit.commands, kit.subcommands) == (("compliance",), ("evidence-kit",))
    assert "verdicts it packages are not qualified" in kit.note


def test_the_owner_added_rows_claim_their_installations() -> None:
    """Add the redactor, the evidence kit and evidence emission and verification (O-2)."""
    claimed = {
        identifier: CONTRACT.operation(identifier).installation
        for identifier in (
            "pii-redactor",
            "compliance-evidence-kit",
            "evidence-emission",
            "evidence-verification",
        )
    }
    assert claimed == {
        "pii-redactor": "core",
        "compliance-evidence-kit": "core+pro",
        "evidence-emission": "core+pro",
        "evidence-verification": "core",
    }
    assert not CONTRACT.operation("evidence-emission").model_backed


def test_the_injection_firewall_is_the_rule_based_sanitizer() -> None:
    """Cover InputSanitizer only; the model-backed detector is another row."""
    assert CONTRACT.operation("input-sanitizer").exports == (
        "InputSanitizer",
        "SanitizeResult",
    )
    assert "InjectionDetector" in CONTRACT.operation("injection-detection").exports


def test_the_redactor_stays_on_and_names_its_classes() -> None:
    """Own redact_pii by the redactor row, which names every class it covers (R-1)."""
    owners = {
        default.setting: default.owner for default in CONTRACT.production_defaults
    }
    assert owners["redact_pii"] == "pii-redactor"
    note = CONTRACT.operation("pii-redactor").note
    for name in (
        "e-mail",
        "card",
        "social security",
        "phone",
        "MRN",
        "DOB",
        "NHS",
        "IBAN",
        "passport",
        "IPv4",
    ):
        assert name in note, name


def test_each_scope_note_bounds_what_its_probe_shows() -> None:
    """Say what a functional probe does not measure, and name what a row lacks."""
    for identifier, bound in (
        (
            "input-sanitizer",
            "its detection rate on injection attacks is not measured or qualified",
        ),
        ("pii-redactor", "its recall on real data is not measured or qualified"),
        ("fact-retrieval", "retrieval relevance is not qualified"),
        ("vector-store", "embedding and ranking quality are not qualified"),
        ("evidence-emission", "demonstration verdicts use the fast heuristic"),
        ("proxy-transport", "It exposes no metrics route and applies no rate limit."),
    ):
        assert bound in CONTRACT.operation(identifier).note, identifier
    assert CONTRACT.operation("evidence-emission").note.endswith(
        "and are not qualified."
    )
    assert "Exact" not in CONTRACT.operation("fact-retrieval").title
    assert "(core+pro, core+full)" in CONTRACT.operation("pii-redactor").note


def test_every_row_with_an_installation_names_its_probe() -> None:
    """Bind every claimed installation to an installed probe (C-4)."""
    claimed = {
        row.identifier: row.probe for row in CONTRACT.operations if row.installation
    }
    assert claimed == {
        "input-sanitizer": "input-sanitizer",
        "fact-retrieval": "fact-retrieval",
        "vector-store": "vector-store",
        "guard-fail-closed": "guard-fail-closed",
        "compliance-reports": "compliance-reports",
        "rest-server-transport": "rest-server-transport",
        "proxy-transport": "proxy-transport",
        "pii-redactor": "pii-redactor",
        "compliance-evidence-kit": "compliance-evidence-kit",
        "evidence-emission": "evidence-emission",
        "evidence-verification": "evidence-verification",
    }


QUALIFIED = (
    "compliance-evidence-kit",
    "compliance-reports",
    "evidence-emission",
    "evidence-verification",
    "fact-retrieval",
    "guard-fail-closed",
    "input-sanitizer",
    "pii-redactor",
    "proxy-transport",
    "rest-server-transport",
    "vector-store",
)
"""The rows the stage-1 classification and evidence records approved."""


def test_the_qualified_set_is_the_reviewed_one() -> None:
    """Qualify exactly the approved rows, each with its three custody records."""
    qualified = {
        row.identifier: row
        for row in CONTRACT.operations
        if row.support == "production-qualified"
    }
    assert tuple(sorted(qualified)) == QUALIFIED
    for identifier, row in qualified.items():
        assert not row.model_backed, identifier
        assert row.installation is not None and row.probe == identifier
        assert tuple(record.role for record in row.custody) == (
            "classification",
            "evidence",
            "owner",
        ), identifier
        for record in row.custody:
            assert re.fullmatch(
                r"[A-Z]+(?:_[A-Z]+)*_[0-9]{8}T[0-9]{6}Z", record.record
            ), record
            assert re.fullmatch(r"[0-9a-f]{64}", record.digest), record
