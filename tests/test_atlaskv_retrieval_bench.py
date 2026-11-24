# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AtlasKV retrieval comparison tests

"""The comparison harness on a small corpus with the hashing encoder.

The real arms, the real hybrid backend and the real boundary checks run;
only the encoder is the deterministic hashing embedder, so no model is
loaded and nothing is downloaded.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import benchmarks.atlaskv_retrieval_bench as bench
from benchmarks.atlaskv_corpus import Corpus, Query, build_corpus


@pytest.fixture(scope="module")
def small() -> Corpus:
    """Build a small corpus with the full adversarial structure."""
    return build_corpus(entities=30, similar_pairs=3)


def test_the_hashing_embedder_is_unit_and_deterministic() -> None:
    """Vectors are normalised on request and repeatable."""
    embedder = bench.HashingEmbedder(dimensions=64)
    vector = embedder.encode("Velorix Labs")
    assert vector.shape == (64,) and abs(float(np.linalg.norm(vector)) - 1.0) < 1e-6
    assert np.array_equal(vector, embedder.encode("Velorix Labs"))
    assert (
        float(np.linalg.norm(embedder.encode("ab", normalize_embeddings=False))) > 1.0
    )


def test_an_unknown_arm_is_refused(small: Corpus) -> None:
    """Only the three preregistered arms exist."""
    with pytest.raises(ValueError, match="unknown arm"):
        bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), "attention")


@pytest.mark.parametrize("arm", bench.ARMS)
def test_each_arm_answers_inside_the_tenant_and_without_retractions(
    small: Corpus, arm: str
) -> None:
    """No arm returns another tenant's, a retracted or a stale document."""
    index = bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), arm)
    for query in small.queries:
        result = index.query(query, g=2, tau=0.3)
        assert len(result.docs) <= bench.TOP_K
        assert sum(result.leaks.values()) == 0
        assert all(
            doc.tenant_id == query.tenant_id and not doc.retracted
            for doc in result.docs
        )


def test_read_back_counts_every_leak(small: Corpus) -> None:
    """A result from another snapshot, a retracted or a foreign document is counted."""
    index = bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), "hybrid")
    retracted = next(item for item in small.snapshot.evidence if item.retracted)
    foreign = next(
        item for item in small.snapshot.evidence if item.tenant_id == "tenant-b"
    )
    good = next(
        item
        for item in small.snapshot.evidence
        if item.tenant_id == "tenant-a" and not item.retracted
    )
    sid = small.snapshot.snapshot_id
    kept, leaks = index._checked(
        [
            (good.doc_id, sid),
            (good.doc_id, "other"),
            ("missing", sid),
            (retracted.doc_id, sid),
            (foreign.doc_id, sid),
        ],
        "tenant-a",
    )
    assert kept == [good]
    assert leaks == {"tenant": 1, "retracted": 1, "snapshot": 2}


def test_keys_return_one_paraphrase_per_fact_first(small: Corpus) -> None:
    """A contradicted key returns both sources before repeating one."""
    index = bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), "keys")
    query = next(q for q in small.queries if q.kind == "answer" and q.gold_contradict)
    key = f"{query.tenant_id}|{query.entity_id}|{query.relation}"
    first_two = index.key_docs[key][:2]
    assert {doc.source_id for doc in first_two} == {"registry-a", "registry-b"}


def test_the_pruned_arm_falls_back_below_its_threshold(small: Corpus) -> None:
    """A threshold above any similarity forces the flat-key fallback."""
    index = bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), "pruned")
    query = next(q for q in small.queries if q.kind == "answer")
    assert index.query(query, g=2, tau=2.0).fallback
    assert not index.query(query, g=8, tau=-1.0).fallback


def test_the_value_check_needs_agreement(small: Corpus) -> None:
    """A claim is approved only when the relevant documents all state it."""
    query = next(q for q in small.queries if q.kind == "answer" and q.gold_contradict)
    docs = [
        item for item in small.snapshot.evidence if item.doc_id in query.gold_support
    ]
    rival = [
        item for item in small.snapshot.evidence if item.doc_id in query.gold_contradict
    ]
    assert bench.value_check(query, docs, query.true_value)
    assert not bench.value_check(query, docs + rival, query.true_value)
    assert not bench.value_check(query, [], query.true_value)


def test_the_value_check_sees_a_similar_name_partner(small: Corpus) -> None:
    """A partner's documents name the base entity, so they count for it."""
    query = next(
        q
        for q in small.queries
        if q.kind == "answer" and q.false_value_class == "partner"
    )
    partner_docs = [
        item
        for item in small.snapshot.evidence
        if item.relation == query.relation
        and item.entity_id != query.entity_id
        and item.value == query.false_value
        and query.entity_name in item.text
    ]
    assert partner_docs
    assert bench.relevant_docs(query, partner_docs) == partner_docs
    assert bench.value_check(query, partner_docs, query.false_value)


def test_partner_classes_are_reported_by_direction(small: Corpus) -> None:
    """The partner class splits by which member of the pair was asked about."""
    partners = [q for q in small.queries if q.false_value_class == "partner"]
    names = {bench._report_class(q) for q in partners}
    assert names <= {"partner_on_longer_name", "partner_on_shorter_name"}
    longer = next(q for q in partners if q.entity_name.endswith(" North"))
    assert bench._report_class(longer) == "partner_on_longer_name"
    other = next(q for q in small.queries if q.false_value_class == "other_entity")
    assert bench._class_counts([longer, other]) == {
        "other_entity": 1,
        "partner_on_longer_name": 1,
    }


def test_summaries_and_empty_rows() -> None:
    """Rates are percentages and an empty arm reports zeros."""
    assert bench.summarise([], [])["p95_ms"] == 0.0
    assert bench._rate([], "support_hit") == 0.0


def test_the_bootstrap_is_seeded_and_brackets_the_difference(small: Corpus) -> None:
    """The interval is repeatable and contains a constant difference."""
    answers = [q for q in small.queries if q.kind == "answer"]
    left: list[bench.Row] = [{"support_hit": True} for _ in answers]
    right: list[bench.Row] = [
        {"support_hit": index % 2 == 0} for index, _ in enumerate(answers)
    ]
    groups = small.entity_group
    low, high = bench.paired_bootstrap(
        answers, left, right, "support_hit", groups=groups, reps=200
    )
    assert (low, high) == bench.paired_bootstrap(
        answers, left, right, "support_hit", groups=groups, reps=200
    )
    assert 0.0 < low <= high < 100.0


def _summary(**pruned: float) -> dict[str, dict[str, float]]:
    """Return three arms' metrics with ``pruned`` overrides."""
    base = {
        "r_sup_at_k": 70.0,
        "r_con_at_k": 80.0,
        "false_approval": 1.0,
        "leaks": 0.0,
        "p95_ms": 10.0,
    }
    return {
        "hybrid": dict(base),
        "keys": {**base, "p95_ms": 12.0},
        "pruned": {**base, **pruned},
    }


def test_the_acceptance_rules() -> None:
    """Each preregistered rule can fail on its own."""
    cost = {arm: {"ram_bytes": 100.0, "index_bytes": 100.0} for arm in bench.ARMS}
    good = _summary(r_sup_at_k=80.0, p95_ms=11.0)
    gain, safe = (2.0, 9.0), (-0.4, 0.4)
    assert bench.decide(good, gain, cost, safe)["accepted"]
    assert not bench.decide(good, (-1.0, 9.0), cost, safe)["gain"]
    assert not bench.decide(
        _summary(r_sup_at_k=80.0, r_con_at_k=70.0), gain, cost, safe
    )["contradictions"]
    assert not bench.decide(good, gain, cost, (-0.1, 0.6))["false_approvals"]
    assert not bench.decide(_summary(r_sup_at_k=80.0, leaks=1.0), gain, cost, safe)[
        "boundaries"
    ]
    heavy = {**cost, "pruned": {"ram_bytes": 300.0, "index_bytes": 100.0}}
    assert not bench.decide(good, gain, heavy, safe)["cost"]
    assert not bench.decide(_summary(r_sup_at_k=80.0, p95_ms=20.0), gain, cost, safe)[
        "h2_faster"
    ]


def test_overlap_digests_and_percentiles(small: Corpus) -> None:
    """The overlap share, source digests and empty percentiles are reported."""
    share = bench.key_overlap_share(list(small.queries), small)
    assert 0.0 <= share <= 100.0
    assert bench.key_overlap_share([], small) == 0.0
    digests = bench.source_digests()
    assert set(digests) == {"generator_sha256", "harness_sha256", "templates_sha256"}
    assert all(len(value) == 64 for value in digests.values())
    assert bench._percentile_ms([], 0.95) == 0.0
    assert bench._percentile_ms([0.001, 0.002], 0.5) == 1.0


def test_evaluate_and_dev_selection(small: Corpus) -> None:
    """A full run reports every arm, the swap and a decision."""
    result = bench.evaluate(small, bench.HashingEmbedder(), "test", g=4, tau=0.3)
    summary, swap, decision = result["summary"], result["swap"], result["decision"]
    assert isinstance(summary, dict) and set(summary) == set(bench.ARMS)
    assert isinstance(swap, dict) and set(swap) == set(bench.ARMS)
    assert isinstance(decision, dict) and "accepted" in decision
    selection = bench.select_on_dev(small, bench.HashingEmbedder())
    grid = selection["grid"]
    assert isinstance(grid, list) and len(grid) == 9
    assert selection["g"] in bench.GRID_G and selection["tau"] in bench.GRID_TAU


def test_the_encoder_loader_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hashing needs no model; a named model loads through the factory with its cached revision."""
    assert bench.load_encoder("hashing")[1] == "builtin"
    monkeypatch.setenv("HOME", str(tmp_path))
    ref = tmp_path / ".cache/huggingface/hub/models--org--model/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("abc123\n", encoding="utf-8")
    loaded: list[str] = []

    def factory(name: str) -> object:
        """Record the requested model instead of loading it."""
        loaded.append(name)
        return bench.HashingEmbedder()

    encoder, revision = bench.load_encoder("org/model", factory)
    assert loaded == ["org/model"] and revision == "abc123"
    assert bench._cached_revision("org/absent") == ""


def test_the_environment_probe(tmp_path: Path) -> None:
    """The CPU model is read when ``cpuinfo`` exists and empty otherwise."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor : 0\nmodel name : Test CPU 9000\n", encoding="utf-8")
    assert bench._environment(cpuinfo)["cpu"] == "Test CPU 9000"
    assert bench._environment(tmp_path / "absent")["cpu"] == ""


def _loader(name: str) -> tuple[object, str]:
    """Return the hashing encoder for any requested model."""
    return bench.HashingEmbedder(), "test"


def test_main_runs_dev_and_test_and_writes_the_artefact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command line runs both splits and refuses a test run without parameters."""
    small = ["--entities", "20", "--similar-pairs", "2", "--encoder", "hashing"]
    dev = bench.main(
        ["--split", "dev", *small, "--output", str(tmp_path / "dev.json")], load=_loader
    )
    assert dev["selection"] and (tmp_path / "dev.json").is_file()
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path / "results")
    test = bench.main(
        [
            "--split",
            "test",
            "--g",
            "2",
            "--tau",
            "0.3",
            *small,
            "--preregistration-sha256",
            "ab",
        ],
        load=_loader,
    )
    written = json.loads(
        (tmp_path / "results" / "atlaskv_retrieval_test.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        written["preregistration_sha256"] == "ab"
        and written["split"] == "test" == test["split"]
    )
    with pytest.raises(SystemExit):
        bench.main(["--split", "test", *small], load=_loader)


def test_a_query_with_no_candidates_reports_nothing(small: Corpus) -> None:
    """A query for an empty tenant returns no documents in every arm."""
    query: Query = replace(small.queries[0], tenant_id="tenant-none")
    for arm in bench.ARMS:
        result = bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), arm).query(
            query, g=2, tau=0.3
        )
        assert result.docs == []
