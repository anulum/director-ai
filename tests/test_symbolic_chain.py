# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — neural-symbolic reasoning tests

"""Multi-angle coverage: Claim / ClaimRelation validation,
GraphProver polarity + direct + transitive contradiction detection,
custom prover backends via Protocol, NeuralSymbolicVerifier
adapter, error paths."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import pytest

from director_ai.core.symbolic_chain import (
    Claim,
    ClaimRelation,
    ConsistencyReport,
    GraphProver,
    NeuralSymbolicVerifier,
)

# --- Claim ----------------------------------------------------------


class TestClaim:
    def test_valid_claim_constructs(self):
        c = Claim(id="x", text="the sky is blue")
        assert c.negated is False

    def test_negated_flag(self):
        c = Claim(id="x", text="not blue", negated=True)
        assert c.negated

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="id"):
            Claim(id="", text="t")

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError, match="text"):
            Claim(id="x", text="")


# --- ClaimRelation -------------------------------------------------


class TestClaimRelation:
    def test_valid_relation(self):
        r = ClaimRelation(source="a", target="b", kind="implies")
        assert r.kind == "implies"

    def test_invalid_kind_rejected(self):
        bad_kind = cast(Any, "proves")
        with pytest.raises(ValueError, match="kind"):
            ClaimRelation(source="a", target="b", kind=bad_kind)

    def test_self_relation_rejected(self):
        with pytest.raises(ValueError, match="distinct claims"):
            ClaimRelation(source="a", target="a", kind="implies")

    def test_empty_endpoints_rejected(self):
        with pytest.raises(ValueError):
            ClaimRelation(source="", target="b", kind="implies")


# --- GraphProver ---------------------------------------------------


class TestGraphProver:
    def test_consistent_set(self):
        prover = GraphProver()
        claims = [Claim(id="a", text="A"), Claim(id="b", text="B")]
        report = prover.check(claims, [ClaimRelation("a", "b", "implies")])
        assert report.is_consistent
        assert report.status == "consistent"

    def test_polarity_conflict(self):
        prover = GraphProver()
        claims = [
            Claim(id="x", text="X"),
            Claim(id="x", text="not X", negated=True),
        ]
        report = prover.check(claims, [])
        assert report.status == "inconsistent"
        assert any("polarity" in c[2] for c in report.conflicts)

    def test_direct_contradicts(self):
        prover = GraphProver()
        claims = [Claim(id="a", text="A"), Claim(id="b", text="B")]
        relations = [ClaimRelation("a", "b", "contradicts")]
        report = prover.check(claims, relations)
        assert not report.is_consistent

    def test_transitive_contradiction(self):
        """A implies B, B contradicts C  →  A contradicts C."""
        prover = GraphProver()
        claims = [
            Claim(id="a", text="A"),
            Claim(id="b", text="B"),
            Claim(id="c", text="C"),
        ]
        relations = [
            ClaimRelation("a", "b", "implies"),
            ClaimRelation("b", "c", "contradicts"),
        ]
        report = prover.check(claims, relations)
        assert not report.is_consistent
        reasons = {c[2] for c in report.conflicts}
        assert any("transitive" in r for r in reasons)

    def test_equivalent_is_non_conflicting(self):
        prover = GraphProver()
        report = prover.check(
            [Claim(id="a", text="A"), Claim(id="b", text="B")],
            [ClaimRelation("a", "b", "equivalent")],
        )
        assert report.is_consistent

    def test_empty_inputs(self):
        prover = GraphProver()
        report = prover.check([], [])
        assert report.is_consistent
        assert report.conflicts == ()


# --- ProverBackend Protocol ----------------------------------------


class TestProverProtocol:
    def test_custom_backend_plugs_in(self):
        class _AlwaysInconsistent:
            def check(
                self,
                claims: Iterable[Claim],
                relations: Iterable[ClaimRelation],
            ) -> ConsistencyReport:
                return ConsistencyReport(
                    status="inconsistent",
                    conflicts=(("x", "y", "stub"),),
                )

        v = NeuralSymbolicVerifier(prover=_AlwaysInconsistent())
        report = v.verify([Claim(id="x", text="X")], [])
        assert report.status == "inconsistent"
        assert report.conflicts[0][2] == "stub"

    def test_default_prover_is_graph(self):
        v = NeuralSymbolicVerifier()
        report = v.verify([Claim(id="a", text="A")], [])
        assert report.is_consistent


# --- NeuralSymbolicVerifier end-to-end ------------------------------


class TestEndToEnd:
    def test_well_formed_chain_passes(self):
        v = NeuralSymbolicVerifier()
        claims = [
            Claim(id="paris_capital", text="Paris is the capital of France"),
            Claim(id="paris_city", text="Paris is a city"),
        ]
        relations = [ClaimRelation("paris_capital", "paris_city", "implies")]
        report = v.verify(claims, relations)
        assert report.is_consistent

    def test_contradictory_chain_fails(self):
        v = NeuralSymbolicVerifier()
        claims = [
            Claim(id="paris_capital", text="Paris is the capital"),
            Claim(id="lyon_capital", text="Lyon is the capital"),
        ]
        relations = [ClaimRelation("paris_capital", "lyon_capital", "contradicts")]
        report = v.verify(claims, relations)
        assert not report.is_consistent
