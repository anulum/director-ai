# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — training provenance tests

"""Row identities, provenance records and cluster-disjoint splits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from director_ai.core.training.training_provenance import (
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA,
    SOURCE_CLUSTER_FIELD,
    TrainingProvenance,
    cluster_digest,
    content_key,
    row_cluster,
    split_rows_by_cluster,
)

Row = dict[str, str | int]


def _row(premise: str, hypothesis: str, cluster: str | None = None) -> Row:
    """Build one loaded NLI row, with a source cluster when given."""
    row: Row = {"premise": premise, "hypothesis": hypothesis, "label": 1}
    if cluster is not None:
        row[SOURCE_CLUSTER_FIELD] = cluster
    return row


class TestRowIdentity:
    """Content keys and source clusters of single rows."""

    def test_content_key_ignores_case_and_whitespace(self) -> None:
        """Case and spacing differences give the same content key."""
        assert content_key("The Sky  is blue.", "It is BLUE") == content_key(
            "the sky is blue.", " it  is blue "
        )

    def test_content_key_separates_premise_and_hypothesis(self) -> None:
        """Swapping or merging the two fields changes the key."""
        assert content_key("a", "b") != content_key("b", "a")
        assert content_key("a b", "c") != content_key("a", "b c")

    def test_row_cluster_reads_a_valid_cluster(self) -> None:
        """A recorded cluster is returned and an absent one is None."""
        assert row_cluster(_row("p", "h", "doc-7")) == "doc-7"
        assert row_cluster(_row("p", "h")) is None

    @pytest.mark.parametrize("value", ["", " padded", "padded ", 7, ["x"]])
    def test_row_cluster_refuses_malformed_values(self, value: object) -> None:
        """Empty, padded or non-string clusters are refused."""
        with pytest.raises(ValueError, match=SOURCE_CLUSTER_FIELD):
            row_cluster(
                {"premise": "p", "hypothesis": "h", SOURCE_CLUSTER_FIELD: value}
            )

    def test_rows_without_text_are_refused(self) -> None:
        """Provenance needs the premise and hypothesis text."""
        with pytest.raises(ValueError, match="premise"):
            TrainingProvenance.from_rows([{"hypothesis": "h", "label": 1}])


class TestTrainingProvenance:
    """Recording, overlap counting and the provenance file."""

    def test_overlap_counts_shared_content_and_shared_clusters(self) -> None:
        """Rows sharing content or a cluster with training are counted."""
        provenance = TrainingProvenance.from_rows(
            [_row("alpha premise", "alpha claim", "doc-1"), _row("beta", "claim")]
        )
        evaluation = [
            _row("ALPHA  premise", "alpha claim"),
            _row("unrelated", "text", "doc-1"),
            _row("fresh premise", "fresh claim", "doc-9"),
        ]
        assert provenance.overlapping_rows(evaluation) == 2
        assert provenance.overlapping_rows(evaluation[2:]) == 0

    def test_file_round_trip(self, tmp_path: Path) -> None:
        """A written provenance file reads back unchanged."""
        provenance = TrainingProvenance.from_rows(
            [_row("p1", "h1", "doc-1"), _row("p2", "h2")]
        )
        path = provenance.write(tmp_path)
        assert path == tmp_path / PROVENANCE_FILENAME
        assert TrainingProvenance.read(path) == provenance
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert payload["schema"] == PROVENANCE_SCHEMA
        assert payload["content_keys"] == sorted(provenance.content_keys)
        assert payload["cluster_digests"] == [cluster_digest("doc-1")]
        assert "doc-1" not in text

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            ([], "schema"),
            ({"schema": "other", "content_keys": [], "cluster_digests": []}, "schema"),
            (
                {
                    "schema": PROVENANCE_SCHEMA,
                    "content_keys": ["xyz"],
                    "cluster_digests": [],
                },
                "content keys",
            ),
            (
                {
                    "schema": PROVENANCE_SCHEMA,
                    "content_keys": [],
                    "cluster_digests": [],
                },
                "no training rows",
            ),
            (
                {
                    "schema": PROVENANCE_SCHEMA,
                    "content_keys": ["a" * 64],
                    "cluster_digests": ["doc-1"],
                },
                "cluster digests",
            ),
        ],
    )
    def test_malformed_files_are_refused(
        self, tmp_path: Path, payload: object, message: str
    ) -> None:
        """Unknown schemas and malformed members are refused."""
        path = tmp_path / PROVENANCE_FILENAME
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            TrainingProvenance.read(path)


class TestSplitRowsByCluster:
    """Train and evaluation splits keep groups whole."""

    def test_clusters_and_repeated_content_stay_on_one_side(self) -> None:
        """No cluster or content key appears on both sides."""
        rows = [_row(f"premise {i}", f"claim {i}", f"doc-{i % 5}") for i in range(30)]
        rows += [
            _row("repeated text", "same claim"),
            _row("REPEATED text", "same claim"),
        ]
        rows.append(_row("premise 0", "claim 0", "doc-0"))
        train, evaluation = split_rows_by_cluster(rows, eval_fraction=0.2, seed=3)
        assert train and evaluation
        assert len(train) + len(evaluation) == len(rows)
        held_out = TrainingProvenance.from_rows(evaluation)
        assert held_out.overlapping_rows(train) == 0

    def test_clusters_chained_through_one_content_stay_together(self) -> None:
        """Three clusters sharing one content key always land together."""
        chain = [
            _row("shared text", "shared claim", "chain-b"),
            _row("Shared Text", "shared claim", "chain-a"),
            _row("shared  text", "shared claim", "chain-c"),
        ]
        rows = chain + [_row(f"p{i}", f"h{i}", f"doc-{i}") for i in range(9)]
        for seed in range(10):
            train, evaluation = split_rows_by_cluster(
                rows, eval_fraction=0.3, seed=seed
            )
            sides = {
                row[SOURCE_CLUSTER_FIELD]
                in [r.get(SOURCE_CLUSTER_FIELD) for r in evaluation]
                for row in chain
            }
            assert len(sides) == 1
            assert TrainingProvenance.from_rows(evaluation).overlapping_rows(train) == 0

    def test_split_reaches_the_fraction_and_keeps_input_order(self) -> None:
        """Evaluation holds at least the fraction and both sides keep order."""
        rows = [_row(f"premise {i}", f"claim {i}") for i in range(20)]
        train, evaluation = split_rows_by_cluster(rows, eval_fraction=0.25, seed=1)
        assert len(evaluation) >= 5
        assert train == [row for row in rows if row in train]
        assert evaluation == [row for row in rows if row in evaluation]

    def test_split_is_reproducible_and_seed_dependent(self) -> None:
        """The same seed repeats the split and other seeds vary it."""
        rows = [_row(f"premise {i}", f"claim {i}", f"doc-{i}") for i in range(12)]
        first = split_rows_by_cluster(rows, eval_fraction=0.25, seed=5)
        assert first == split_rows_by_cluster(rows, eval_fraction=0.25, seed=5)
        variants = {
            tuple(
                str(row["premise"])
                for row in split_rows_by_cluster(rows, eval_fraction=0.25, seed=seed)[1]
            )
            for seed in range(6)
        }
        assert len(variants) > 1

    def test_training_keeps_at_least_one_group(self) -> None:
        """A large fraction still leaves one group for training."""
        rows = [_row("a", "b", "doc-1"), _row("c", "d", "doc-2")]
        train, evaluation = split_rows_by_cluster(rows, eval_fraction=0.99, seed=0)
        assert len(train) == 1 and len(evaluation) == 1

    def test_single_group_is_refused(self) -> None:
        """Rows from one cluster cannot form independent sides."""
        rows = [_row(f"p{i}", f"h{i}", "only") for i in range(5)]
        with pytest.raises(ValueError, match="independent groups"):
            split_rows_by_cluster(rows, eval_fraction=0.2, seed=0)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, float("nan")])
    def test_fraction_must_be_inside_the_open_interval(self, fraction: float) -> None:
        """The evaluation fraction must lie strictly between 0 and 1."""
        with pytest.raises(ValueError, match="eval_fraction"):
            split_rows_by_cluster(
                [_row("a", "b"), _row("c", "d")], eval_fraction=fraction, seed=0
            )
