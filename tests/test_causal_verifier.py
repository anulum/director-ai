# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — causal counterfactual verifier tests

"""Multi-angle coverage: CausalGraph construction + cycle rejection
+ topological order, Intervention do-operator semantics, and
CounterfactualVerifier branch aggregation with safety invariants."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import cast

import pytest

import director_ai.core.causal_verifier.counterfactual as counterfactual_module
from director_ai.core.causal_verifier import (
    CausalGraph,
    CounterfactualVerifier,
    GraphCycleError,
    Intervention,
)
from director_ai.core.types import EvidenceChunk


def _as_int(value: object) -> int:
    """Cast helper — the graph stores values as ``object`` by design
    (booleans, floats, strings all flow through); tests know the
    concrete type at each assert site."""
    return cast(int, value)


# --- CausalGraph ----------------------------------------------------


def _const(value: object) -> Callable[[Mapping[str, object]], object]:
    def _eq(_: Mapping[str, object]) -> object:
        return value

    return _eq


def _sum_eq(parents: Mapping[str, object]) -> object:
    return sum(_as_int(v) for v in parents.values())


class TestCausalGraph:
    def test_linear_chain_evaluates(self) -> None:
        g = CausalGraph()
        g.add("x", _const(3))
        g.add("y", lambda p: _as_int(p["x"]) * 2, parents=("x",))
        g.add("z", lambda p: _as_int(p["y"]) + 1, parents=("y",))
        out = g.evaluate({})
        assert out["z"] == 7

    def test_exogenous_from_inputs(self) -> None:
        g = CausalGraph()
        g.add("a", _const(0))  # default structural eq
        out = g.evaluate({"a": 42})
        # Inputs override exogenous equations — the structural
        # equation only fires when the input is missing.
        assert out["a"] == 42

    def test_unknown_parent_rejected(self) -> None:
        g = CausalGraph()
        with pytest.raises(ValueError, match="unknown parent"):
            g.add("y", _sum_eq, parents=("x",))

    def test_self_loop_rejected(self) -> None:
        g = CausalGraph()
        with pytest.raises(ValueError, match="self-loop"):
            g.add("x", _const(1), parents=("x",))

    def test_duplicate_rejected(self) -> None:
        g = CausalGraph()
        g.add("x", _const(1))
        with pytest.raises(ValueError, match="duplicate"):
            g.add("x", _const(2))

    def test_empty_name_rejected(self) -> None:
        g = CausalGraph()
        with pytest.raises(ValueError, match="non-empty"):
            g.add("", _const(1))

    def test_topological_order_is_cached(self) -> None:
        g = CausalGraph()
        g.add("x", _const(1))
        g.add("y", _sum_eq, parents=("x",))
        first = g.topological_order()
        assert first == g.topological_order()

    def test_cycle_raises_on_hand_mutation(self) -> None:
        """Defence-in-depth: if a caller tampers with _nodes and
        introduces a cycle, topological_order must detect it."""
        g = CausalGraph()
        g.add("x", _const(1))
        g.add("y", _sum_eq, parents=("x",))
        # Manufacture a cycle via direct tampering. dataclasses.replace
        # keeps the _Node type opaque to the test.
        g._nodes["x"] = dataclasses.replace(
            g._nodes["x"], parents=("y",), equation=_sum_eq
        )
        g._topo_cache = None
        with pytest.raises(GraphCycleError):
            g.topological_order()


# --- Intervention ---------------------------------------------------


class TestIntervention:
    def _graph(self) -> CausalGraph:
        g = CausalGraph()
        g.add("x", _const(10))
        g.add("y", lambda p: _as_int(p["x"]) * 2, parents=("x",))
        g.add("z", lambda p: _as_int(p["y"]) + 1, parents=("y",))
        return g

    def test_do_operator_overrides_variable(self) -> None:
        g = self._graph()
        iv = Intervention({"x": 5})
        values = iv.apply(g, {})
        assert values["x"] == 5
        assert values["y"] == 10
        assert values["z"] == 11

    def test_intervening_mid_chain_isolates_upstream(self) -> None:
        g = self._graph()
        iv = Intervention({"y": 100})
        values = iv.apply(g, {})
        # x still takes the structural equation (10).
        assert values["x"] == 10
        # y is pinned to 100, so z = 101 irrespective of x.
        assert values["y"] == 100
        assert values["z"] == 101

    def test_input_root_value_survives_unrelated_intervention(self) -> None:
        """Input-supplied root variables should bypass root equations."""
        g = self._graph()
        iv = Intervention({"y": 100})

        values = iv.apply(g, {"x": 7})

        assert values["x"] == 7
        assert values["y"] == 100
        assert values["z"] == 101

    def test_empty_fixes_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Intervention({})

    def test_unknown_target_rejected(self) -> None:
        g = self._graph()
        iv = Intervention({"not_in_graph": 0})
        with pytest.raises(ValueError, match="not in graph"):
            iv.apply(g, {})

    def test_multi_target_intervention(self) -> None:
        g = self._graph()
        iv = Intervention({"x": 1, "y": 2})
        values = iv.apply(g, {})
        assert values["x"] == 1
        assert values["y"] == 2
        assert values["z"] == 3


# --- CounterfactualVerifier -----------------------------------------


class TestCounterfactualVerifier:
    def _graph(self) -> CausalGraph:
        g = CausalGraph()
        g.add("action", _const("deploy"))
        g.add(
            "blast_radius",
            lambda p: {"deploy": 5, "rollback": 0, "abort": 0}.get(
                cast(str, p["action"]), 10
            ),
            parents=("action",),
        )
        return g

    def _invariant(self, values: Mapping[str, object]) -> bool:
        return _as_int(values["blast_radius"]) <= 3

    def test_safe_branch_passes(self) -> None:
        g = self._graph()
        verifier = CounterfactualVerifier(g, safety_invariant=self._invariant)
        verdict = verifier.verify(
            inputs={},
            branches=[
                ("rollback", Intervention({"action": "rollback"})),
                ("abort", Intervention({"action": "abort"})),
            ],
        )
        assert verdict.total == 2
        assert verdict.safe == 2
        assert verdict.safety_rate == 1.0
        assert verdict.unsafe == 0

    def test_unsafe_branches_surfaced(self) -> None:
        g = self._graph()
        verifier = CounterfactualVerifier(g, safety_invariant=self._invariant)
        verdict = verifier.verify(
            inputs={},
            branches=[
                ("deploy", Intervention({"action": "deploy"})),
                ("rollback", Intervention({"action": "rollback"})),
            ],
        )
        assert verdict.safe == 1
        labels = {b.label for b in verdict.unsafe_branches}
        assert "deploy" in labels

    def test_empty_branches_rejected(self) -> None:
        g = self._graph()
        verifier = CounterfactualVerifier(g, safety_invariant=self._invariant)
        with pytest.raises(ValueError, match="at least one branch"):
            verifier.verify(inputs={}, branches=[])

    def test_branch_captures_full_values(self) -> None:
        g = self._graph()
        verifier = CounterfactualVerifier(g, safety_invariant=self._invariant)
        verdict = verifier.verify(
            inputs={},
            branches=[("deploy", Intervention({"action": "deploy"}))],
        )
        unsafe_branch = verdict.unsafe_branches[0]
        assert unsafe_branch.values["blast_radius"] == 5
        assert unsafe_branch.values["action"] == "deploy"
        assert unsafe_branch.outcome == "unsafe"

    def test_halt_fact_change_diagnostic_identifies_single_change(self) -> None:
        diagnostic = CounterfactualVerifier.explain_halt_fact_change(
            observed_score=0.3,
            threshold=0.5,
            evidence_chunks=[
                EvidenceChunk(
                    text="The source fact says treatment A failed.",
                    distance=0.0,
                    source="paper-1",
                ),
            ],
            proposed_fact="The source fact says treatment A succeeded.",
        )

        assert diagnostic.question
        assert diagnostic.observed_score == 0.3
        assert diagnostic.threshold == 0.5
        assert len(diagnostic.candidates) == 1
        assert diagnostic.best_change is not None
        assert diagnostic.best_change.fact_source == "paper-1"
        assert diagnostic.best_change.required_score_delta == pytest.approx(0.2)
        assert diagnostic.best_change.prevented_halt is True

    def test_halt_fact_change_diagnostic_handles_missing_facts(self) -> None:
        diagnostic = CounterfactualVerifier.explain_halt_fact_change(
            observed_score=0.3,
            threshold=0.5,
            evidence_chunks=[],
            proposed_fact="unavailable",
        )

        assert diagnostic.best_change is None
        assert diagnostic.candidates == []

    def test_halt_fact_change_diagnostic_clips_long_fact_text(self) -> None:
        diagnostic = CounterfactualVerifier.explain_halt_fact_change(
            observed_score=0.3,
            threshold=0.5,
            evidence_chunks=[
                EvidenceChunk(text="original " * 100, distance=0.0, source="paper-1")
            ],
            proposed_fact="replacement " * 100,
        )

        candidate = diagnostic.candidates[0]
        assert len(candidate.original_fact) == 500
        assert candidate.original_fact.endswith("...")
        assert len(candidate.proposed_fact) == 500
        assert candidate.proposed_fact.endswith("...")

    def test_counterfactual_float_rejects_non_numeric_values(self) -> None:
        with pytest.raises(TypeError, match="not numeric"):
            counterfactual_module._counterfactual_float({"score": object()}, "score")

    def test_integer_sum_falls_back_to_python_when_accelerator_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(counterfactual_module, "_RUST_COUNTERFACTUAL", False)

        assert counterfactual_module._sum_int([1, 0, 1]) == 2
