# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — emergence oracle tests

"""Multi-angle coverage: SwarmEvent validation, InteractionGraph
density / clustering / cycle detection, RandomWalkSpectrum
convergence and invariants, CommunityDetector deterministic
label propagation, EmergenceOracle composite risk across canonical
scenarios (hub attractor, fragmented swarm, clean pipeline)."""

from __future__ import annotations

import pytest

import director_ai.core.emergence_oracle.graph as graph_mod
import director_ai.core.emergence_oracle.oracle as oracle_mod
import director_ai.core.emergence_oracle.spectrum as spectrum_mod
from director_ai.core.emergence_oracle import (
    CommunityDetector,
    EmergenceOracle,
    EmergenceVerdict,
    InteractionGraph,
    RandomWalkSpectrum,
    SwarmEvent,
)

# --- SwarmEvent ----------------------------------------------------


class TestSwarmEvent:
    def test_valid(self):
        e = SwarmEvent(source="a", target="b", timestamp=1.0)
        assert e.action == ""

    def test_empty_source_rejected(self):
        with pytest.raises(ValueError, match="source"):
            SwarmEvent(source="", target="b", timestamp=1.0)

    def test_empty_target_rejected(self):
        with pytest.raises(ValueError, match="target"):
            SwarmEvent(source="a", target="", timestamp=1.0)

    def test_self_interaction_rejected(self):
        with pytest.raises(ValueError, match="self-interaction"):
            SwarmEvent(source="a", target="a", timestamp=1.0)

    def test_negative_timestamp_rejected(self):
        with pytest.raises(ValueError, match="timestamp"):
            SwarmEvent(source="a", target="b", timestamp=-1.0)


# --- InteractionGraph ----------------------------------------------


def _pipeline_events() -> list[SwarmEvent]:
    """Linear pipeline a -> b -> c -> d. No cycles, uniform traffic."""
    return [
        SwarmEvent("a", "b", 1.0),
        SwarmEvent("b", "c", 2.0),
        SwarmEvent("c", "d", 3.0),
    ]


def _hub_events() -> list[SwarmEvent]:
    """Star-of-David pattern. Every peripheral agent funnels
    requests into a central hub."""
    peripheral = [f"p{i}" for i in range(6)]
    return [
        SwarmEvent(source=p, target="hub", timestamp=float(i))
        for i, p in enumerate(peripheral)
    ] + [
        SwarmEvent(source="hub", target=p, timestamp=float(100 + i))
        for i, p in enumerate(peripheral)
    ]


def _cycle_events() -> list[SwarmEvent]:
    """Strongly connected 3-cycle a -> b -> c -> a."""
    return [
        SwarmEvent("a", "b", 1.0),
        SwarmEvent("b", "c", 2.0),
        SwarmEvent("c", "a", 3.0),
    ]


def _fragmented_events() -> list[SwarmEvent]:
    """Two disconnected cliques."""
    return [
        SwarmEvent("a", "b", 1.0),
        SwarmEvent("b", "a", 2.0),
        SwarmEvent("x", "y", 3.0),
        SwarmEvent("y", "x", 4.0),
    ]


class TestInteractionGraph:
    def test_from_pipeline(self):
        g = InteractionGraph.from_events(_pipeline_events())
        assert g.node_count == 4
        assert g.edge_count == 3

    def test_weights_accumulate(self):
        g = InteractionGraph.from_events(
            [SwarmEvent("a", "b", 1.0), SwarmEvent("a", "b", 2.0)]
        )
        assert g.edge_weight("a", "b") == 2

    def test_density_pipeline(self):
        g = InteractionGraph.from_events(_pipeline_events())
        # 3 edges out of 4*3 = 12 possible directed edges.
        assert g.density() == pytest.approx(3 / 12)

    def test_density_empty(self):
        assert InteractionGraph().density() == 0.0

    def test_out_and_in_weight(self):
        g = InteractionGraph.from_events(
            [SwarmEvent("a", "b", 0.0), SwarmEvent("a", "c", 1.0)]
        )
        assert g.out_weight("a") == 2
        assert g.in_weight("a") == 0
        assert g.in_weight("b") == 1

    def test_has_cycle_false_on_pipeline(self):
        g = InteractionGraph.from_events(_pipeline_events())
        assert not g.has_cycle()

    def test_has_cycle_true_on_three_cycle(self):
        g = InteractionGraph.from_events(_cycle_events())
        assert g.has_cycle()

    def test_has_cycle_skips_nodes_finished_by_an_earlier_dfs(self):
        """The sorted start loop must hit the already-visited `continue` arc.

        A single chain component guarantees the DFS from the sorted-first
        node colours every other node, so each later start takes the
        ``continue`` branch — deterministically, independent of hash seed.
        """
        g = InteractionGraph.from_events(
            [SwarmEvent("a", "b", 0.0), SwarmEvent("b", "c", 1.0)]
        )
        assert not g.has_cycle()

    def test_local_clustering_isolated(self):
        g = InteractionGraph.from_events(_pipeline_events())
        # b has two neighbours (a, c) but no edge between them.
        assert g.local_clustering("b") == 0.0

    def test_local_clustering_triangle(self):
        g = InteractionGraph.from_events(
            [
                SwarmEvent("a", "b", 0.0),
                SwarmEvent("b", "c", 1.0),
                SwarmEvent("a", "c", 2.0),
            ]
        )
        # a's neighbours are b and c; b-c edge exists → C(a) = 1.
        assert g.local_clustering("a") == pytest.approx(1.0)

    def test_mean_clustering(self):
        g = InteractionGraph.from_events(_pipeline_events())
        # Every node either has <2 neighbours or no triangle, so
        # mean clustering is 0.
        assert g.mean_clustering() == 0.0

    def test_mean_clustering_empty_graph(self):
        assert InteractionGraph().mean_clustering() == 0.0

    def test_mean_clustering_python_sum_when_rust_disabled(self, monkeypatch):
        monkeypatch.setattr(graph_mod, "_RUST_GRAPH", False)
        g = InteractionGraph.from_events(_pipeline_events())
        assert g.mean_clustering() == 0.0

    def test_has_cycle_false_on_diamond_revisiting_finished_node(self):
        # Diamond DAG: a→b, a→c, b→d, c→d. The second path into d finds it
        # already finished, exercising the "already-visited, not on stack"
        # branch without a false cycle report.
        g = InteractionGraph.from_events(
            [
                SwarmEvent("a", "b", 0.0),
                SwarmEvent("a", "c", 1.0),
                SwarmEvent("b", "d", 2.0),
                SwarmEvent("c", "d", 3.0),
            ]
        )
        assert not g.has_cycle()

    def test_unknown_node_in_clustering(self):
        g = InteractionGraph.from_events(_pipeline_events())
        with pytest.raises(KeyError):
            g.local_clustering("ghost")

    def test_out_neighbours(self):
        g = InteractionGraph.from_events(_pipeline_events())
        assert g.out_neighbours("b") == ("c",)

    def test_nodes_sorted(self):
        g = InteractionGraph.from_events(_pipeline_events())
        assert g.nodes() == ("a", "b", "c", "d")


# --- RandomWalkSpectrum -------------------------------------------


class TestRandomWalkSpectrum:
    def test_uniform_graph_returns_uniform_stationary(self):
        """Fully-connected directed graph over 4 nodes — the
        uniform stationary distribution is the unique fixed point."""
        events: list[SwarmEvent] = []
        t = 0.0
        for a in ("n0", "n1", "n2", "n3"):
            for b in ("n0", "n1", "n2", "n3"):
                if a != b:
                    events.append(SwarmEvent(a, b, t))
                    t += 1.0
        g = InteractionGraph.from_events(events)
        spectrum = RandomWalkSpectrum(tolerance=1e-9)
        result = spectrum.stationary(g)
        assert result.converged
        for prob in result.probabilities.values():
            assert prob == pytest.approx(0.25, abs=1e-3)

    def test_hub_concentrates_mass(self):
        g = InteractionGraph.from_events(_hub_events())
        spectrum = RandomWalkSpectrum()
        result = spectrum.stationary(g)
        assert result.probabilities["hub"] > max(
            prob for node, prob in result.probabilities.items() if node != "hub"
        )

    def test_probabilities_sum_to_one(self):
        g = InteractionGraph.from_events(_hub_events())
        spectrum = RandomWalkSpectrum()
        result = spectrum.stationary(g)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)

    def test_empty_graph(self):
        spectrum = RandomWalkSpectrum()
        result = spectrum.stationary(InteractionGraph())
        assert result.probabilities == {}
        assert result.converged

    def test_stationary_python_sum_when_rust_disabled(self, monkeypatch):
        # With the accelerator off, the final-mass normalisation (and the
        # _sum_float reducer it calls) takes the pure-python branch.
        monkeypatch.setattr(spectrum_mod, "_RUST_SPECTRUM", False)
        g = InteractionGraph.from_events(_hub_events())
        result = RandomWalkSpectrum().stationary(g)
        assert sum(result.probabilities.values()) == pytest.approx(1.0)

    def test_sum_float_empty_is_zero(self):
        assert spectrum_mod._sum_float([]) == 0.0

    def test_non_convergence_exhausts_iteration_cap(self):
        # A single permitted iteration cannot reach the tolerance, so the power
        # iteration runs the loop to exhaustion without flagging convergence.
        g = InteractionGraph.from_events(_hub_events())
        result = RandomWalkSpectrum(max_iterations=1, tolerance=1e-12).stationary(g)
        assert not result.converged

    def test_top_nodes(self):
        g = InteractionGraph.from_events(_hub_events())
        spectrum = RandomWalkSpectrum()
        result = spectrum.stationary(g)
        top = result.top_nodes(k=1)
        assert top[0][0] == "hub"

    def test_top_nodes_bad_k(self):
        g = InteractionGraph.from_events(_hub_events())
        result = RandomWalkSpectrum().stationary(g)
        with pytest.raises(ValueError, match="k"):
            result.top_nodes(k=0)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"max_iterations": 0}, "max_iterations"),
            ({"tolerance": 0}, "tolerance"),
            ({"laziness": 0.0}, "laziness"),
            ({"laziness": 1.0}, "laziness"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            RandomWalkSpectrum(**kwargs)


# --- CommunityDetector --------------------------------------------


class TestCommunityDetector:
    def test_two_fragments(self):
        """Two disconnected cliques must land in separate communities."""
        g = InteractionGraph.from_events(_fragmented_events())
        assignment = CommunityDetector().detect(g)
        communities = assignment.communities()
        assert assignment.community_count == 2
        # Every {a, b} pair lives together; every {x, y} pair lives together.
        sets = [set(members) for members in communities.values()]
        assert {"a", "b"} in sets
        assert {"x", "y"} in sets

    def test_single_community(self):
        g = InteractionGraph.from_events(_pipeline_events())
        assignment = CommunityDetector().detect(g)
        # Linear chains label-propagate into a single community.
        assert assignment.community_count == 1

    def test_deterministic_across_runs(self):
        g = InteractionGraph.from_events(_fragmented_events())
        detector = CommunityDetector()
        a = detector.detect(g).labels
        b = detector.detect(g).labels
        assert a == b

    def test_bad_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            CommunityDetector(max_iterations=0)

    def test_iteration_cap_stops_before_convergence(self):
        # One pass changes labels on the fragmented graph, so a single-iteration
        # cap exits via exhaustion rather than the stabilised-labels break.
        g = InteractionGraph.from_events(_fragmented_events())
        assignment = CommunityDetector(max_iterations=1).detect(g)
        assert assignment.iterations == 1
        assert not assignment.converged

    def test_empty_graph(self):
        assignment = CommunityDetector().detect(InteractionGraph())
        assert assignment.community_count == 0


# --- EmergenceOracle ----------------------------------------------


class TestEmergenceOracle:
    def test_pipeline_has_low_risk(self):
        oracle = EmergenceOracle(
            attractor_top_k=2,
            weight_attractor=0.5,
            weight_cycle=0.3,
            weight_imbalance=0.2,
        )
        verdict = oracle.analyse(_pipeline_events())
        assert isinstance(verdict, EmergenceVerdict)
        assert not verdict.cycle_detected
        assert verdict.risk < 0.5
        assert verdict.safe

    def test_hub_raises_attractor_signal(self):
        oracle = EmergenceOracle(
            attractor_top_k=1,
            weight_attractor=1.0,
            weight_cycle=0.0,
            weight_imbalance=0.0,
        )
        verdict = oracle.analyse(_hub_events())
        # The hub holds more than the uniform expectation.
        assert verdict.attractor_mass > 0.0
        assert verdict.top_hubs[0][0] == "hub"

    def test_cycle_adds_to_risk(self):
        oracle = EmergenceOracle(
            weight_attractor=0.0,
            weight_cycle=1.0,
            weight_imbalance=0.0,
        )
        verdict = oracle.analyse(_cycle_events())
        assert verdict.cycle_detected
        assert verdict.risk == pytest.approx(1.0)

    def test_fragmentation_adds_imbalance(self):
        oracle = EmergenceOracle(
            weight_attractor=0.0,
            weight_cycle=0.0,
            weight_imbalance=1.0,
        )
        verdict = oracle.analyse(_fragmented_events())
        # Two equal-sized communities -> imbalance = 0 (uniform).
        assert verdict.community_imbalance == pytest.approx(0.0)
        # Now tilt: add more events into one clique.
        tilted = _fragmented_events() + [
            SwarmEvent("a", "b", 10.0),
            SwarmEvent("b", "a", 11.0),
            SwarmEvent("a", "b", 12.0),
        ]
        verdict_tilted = oracle.analyse(tilted)
        assert verdict_tilted.community_imbalance >= verdict.community_imbalance

    def test_empty_trace(self):
        oracle = EmergenceOracle()
        verdict = oracle.analyse([])
        assert verdict.risk == 0.0
        assert verdict.safe

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"attractor_top_k": 0}, "attractor_top_k"),
            (
                {
                    "weight_attractor": -0.1,
                    "weight_cycle": 0.5,
                    "weight_imbalance": 0.6,
                },
                "non-negative",
            ),
            (
                {"weight_attractor": 0.9, "weight_cycle": 0.5, "weight_imbalance": 0.6},
                "sum to 1",
            ),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            EmergenceOracle(**kwargs)


class TestEmergenceRustSums:
    def test_sum_float_empty_and_python_paths(self, monkeypatch):
        monkeypatch.setattr(oracle_mod, "_RUST_EMERGENCE", False)

        assert oracle_mod._sum_float([]) == 0.0
        assert oracle_mod._sum_float([0.25, 0.75]) == pytest.approx(1.0)

    def test_rust_sum_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(oracle_mod, "_RUST_EMERGENCE", True)
        called = {"count": 0}

        def _sum(values: list[float]) -> float:
            called["count"] += 1
            return sum(values)

        monkeypatch.setattr(oracle_mod, "rust_sum_f64", _sum, raising=True)
        total = oracle_mod._sum_float([0.25, 0.75])
        assert total == pytest.approx(1.0)
        assert called["count"] == 1

    def test_rust_sum_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(oracle_mod, "_RUST_EMERGENCE", True)
        monkeypatch.setattr(
            oracle_mod,
            "rust_sum_f64",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        assert oracle_mod._sum_float([1.0, 2.0, 3.0]) == pytest.approx(6.0)


class TestSpectrumRustSums:
    def test_spectrum_rust_sum_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(spectrum_mod, "_RUST_SPECTRUM", True)
        called = {"count": 0}

        def _sum(values: list[float]) -> float:
            called["count"] += 1
            return sum(values)

        monkeypatch.setattr(spectrum_mod, "rust_sum_f64", _sum, raising=True)
        graph = InteractionGraph.from_events(_hub_events())
        result = RandomWalkSpectrum(max_iterations=3).stationary(graph)
        assert called["count"] >= 1
        assert sum(result.probabilities.values()) == pytest.approx(1.0)

    def test_spectrum_rust_sum_type_error_falls_back(self, monkeypatch):
        monkeypatch.setattr(spectrum_mod, "_RUST_SPECTRUM", True)
        monkeypatch.setattr(
            spectrum_mod,
            "rust_sum_f64",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        graph = InteractionGraph.from_events(_cycle_events())
        result = RandomWalkSpectrum().stationary(graph)
        assert result.probabilities


class TestGraphRustSums:
    def test_graph_rust_sum_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(graph_mod, "_RUST_GRAPH", True)
        called = {"count": 0}

        def _sum(values: list[float]) -> float:
            called["count"] += 1
            return sum(values)

        monkeypatch.setattr(graph_mod, "rust_sum_f64", _sum, raising=True)
        graph = InteractionGraph.from_events(_pipeline_events())
        assert graph.out_weight("a") == 1
        assert called["count"] >= 1

    def test_graph_rust_sum_type_error_falls_back(self, monkeypatch):
        monkeypatch.setattr(graph_mod, "_RUST_GRAPH", True)
        monkeypatch.setattr(
            graph_mod,
            "rust_sum_f64",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        graph = InteractionGraph.from_events(_pipeline_events())
        assert graph.in_weight("b") == 1


def test_mean_clustering_and_sum_float_paths(monkeypatch):
    import director_ai.core.emergence_oracle.graph as g

    graph = g.InteractionGraph()
    graph._add_edge("a", "b")
    graph._add_edge("b", "c")
    graph._add_edge("a", "c")
    assert graph.mean_clustering() > 0.0

    monkeypatch.setattr(g, "_RUST_GRAPH", True)
    monkeypatch.setattr(g, "rust_sum_f64", lambda v: sum(v), raising=True)
    assert g._sum_float([1.0, 2.0]) == pytest.approx(3.0)
    monkeypatch.setattr(g, "_RUST_GRAPH", False)
    assert g._sum_float([]) == 0.0
