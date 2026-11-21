# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AtlasKV fallback diagnostic tests

"""The fallback diagnostic splits the pruned arm's fallbacks by condition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmarks.atlaskv_fallback_diagnostic as diagnostic
import benchmarks.atlaskv_retrieval_bench as bench
from benchmarks.atlaskv_corpus import Corpus, build_corpus


@pytest.fixture(scope="module")
def small() -> Corpus:
    """Build a small corpus with the full adversarial structure."""
    return build_corpus(entities=30, similar_pairs=3)


@pytest.fixture(scope="module")
def pruned(small: Corpus) -> bench.ArmIndex:
    """Build the pruned arm on the small corpus with the hashing encoder."""
    return bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), "pruned")


def test_conditions_match_the_arm_on_every_query(
    small: Corpus, pruned: bench.ArmIndex
) -> None:
    """The recomputed conditions agree with the arm and sum to the query count."""
    queries = [q for q in small.queries if q.split == "test"]
    for tau in (0.2, 0.9, 2.0):
        result = diagnostic.count_conditions(pruned, queries, g=2, tau=tau)
        counts = result["by_condition"]
        assert isinstance(counts, dict)
        assert sum(counts.values()) == result["queries"] == len(queries)
        assert result["fallbacks"] == len(queries) - counts["neither"]
        fell_back = sum(pruned.query(q, g=2, tau=tau).fallback for q in queries)
        assert result["fallbacks"] == fell_back


def test_a_threshold_above_every_similarity_always_falls_back(
    small: Corpus, pruned: bench.ArmIndex
) -> None:
    """With tau above any cosine, every query falls back for its similarity."""
    queries = [q for q in small.queries if q.split == "test"][:20]
    result = diagnostic.count_conditions(pruned, queries, g=2, tau=2.0)
    counts = result["by_condition"]
    assert isinstance(counts, dict)
    assert counts["neither"] == counts["too_few_items_only"] == 0
    assert counts["similarity_below_tau_only"] + counts["both"] == len(queries)


def test_too_few_items_is_counted_on_its_own(
    small: Corpus, pruned: bench.ArmIndex
) -> None:
    """One group cannot fill top_k, so a low tau leaves only the count condition."""
    queries = [q for q in small.queries if q.split == "test"][:20]
    result = diagnostic.count_conditions(pruned, queries, g=1, tau=-2.0)
    counts = result["by_condition"]
    assert isinstance(counts, dict)
    assert counts["similarity_below_tau_only"] == counts["both"] == 0
    assert counts["too_few_items_only"] + counts["neither"] == len(queries)


def test_other_arms_are_refused(small: Corpus) -> None:
    """Only the pruned arm has entity groups."""
    keys = bench.ArmIndex(small.snapshot, bench.HashingEmbedder(), "keys")
    with pytest.raises(ValueError, match="pruned arm"):
        diagnostic.conditions(keys, small.queries[0], g=2, tau=0.4)


def test_a_disagreement_with_the_arm_is_refused(
    small: Corpus, pruned: bench.ArmIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run stops if its conditions differ from the arm's own fallback."""
    monkeypatch.setattr(diagnostic, "conditions", lambda *_a, **_k: (True, True))
    query = next(
        q for q in small.queries if not pruned.query(q, g=2, tau=-2.0).fallback
    )
    with pytest.raises(RuntimeError, match="disagree"):
        diagnostic.count_conditions(pruned, [query], g=2, tau=-2.0)


def test_main_writes_the_artefact(tmp_path: Path) -> None:
    """The command builds the arm, counts the split and writes its artefact."""
    output = tmp_path / "fallback.json"
    artefact = diagnostic.main(
        [
            "--split",
            "test",
            "--g",
            "2",
            "--tau",
            "0.4",
            "--entities",
            "20",
            "--similar-pairs",
            "2",
            "--encoder",
            "hashing",
            "--output",
            str(output),
        ],
        load=lambda name: (bench.HashingEmbedder(), name),
    )
    assert json.loads(output.read_text()) == artefact
    assert artefact["schema"] == "director-ai.atlaskv-fallback-diagnostic.v1"
    assert artefact["encoder_revision"] == "hashing"
    assert (
        diagnostic.main(
            [
                "--split",
                "dev",
                "--g",
                "4",
                "--tau",
                "0.2",
                "--entities",
                "20",
                "--similar-pairs",
                "2",
            ],
            load=lambda name: (bench.HashingEmbedder(), name),
        )["split"]
        == "dev"
    )
