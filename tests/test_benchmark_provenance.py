# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — benchmark artefact reproducibility provenance tests

from __future__ import annotations

import argparse
import re
from typing import Any

import pytest

from benchmarks import _provenance as prov
from benchmarks import grounded_operating_point_campaign as campaign
from benchmarks import ragtruth_eval
from benchmarks.e2e_eval import E2EMetrics, E2ESample

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


# --------------------------------------------------------------------------- #
# git-state capture
# --------------------------------------------------------------------------- #
def test_resolve_git_sha_returns_head_in_repo() -> None:
    sha = prov.resolve_git_sha()
    assert _HEX40.match(sha), sha
    assert sha != prov.UNKNOWN_SHA


def test_resolve_git_sha_unknown_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov.shutil, "which", lambda _name: None)
    assert prov.resolve_git_sha() == prov.UNKNOWN_SHA


def test_resolve_git_sha_unknown_on_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prov.shutil, "which", lambda _name: "/usr/bin/git")

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("git exploded")

    monkeypatch.setattr(prov.subprocess, "run", _boom)
    assert prov.resolve_git_sha() == prov.UNKNOWN_SHA


def test_working_tree_dirty_returns_bool_in_repo() -> None:
    assert isinstance(prov.working_tree_dirty(), bool)


def test_working_tree_dirty_none_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov.shutil, "which", lambda _name: None)
    assert prov.working_tree_dirty() is None


# --------------------------------------------------------------------------- #
# provenance block + stamp
# --------------------------------------------------------------------------- #
def test_provenance_block_shape_in_repo() -> None:
    block = prov.provenance_block()
    assert _HEX40.match(block["git_sha"])
    assert block["git_short"] == block["git_sha"][:12]
    assert isinstance(block["git_dirty"], bool)
    assert block["python"] and block["platform"] and block["generated_at"]


def test_provenance_block_explicit_sha_overrides_head() -> None:
    block = prov.provenance_block(git_sha="cafe1234")
    assert block["git_sha"] == "cafe1234"
    assert block["git_short"] == "cafe1234"


def test_provenance_block_unknown_git_short(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov.shutil, "which", lambda _name: None)
    block = prov.provenance_block()
    assert block["git_sha"] == prov.UNKNOWN_SHA
    assert block["git_short"] == prov.UNKNOWN_SHA
    assert block["git_dirty"] is None


def test_stamp_attaches_block_and_returns_same_object() -> None:
    payload: dict[str, Any] = {"rows": [1]}
    returned = prov.stamp(payload, git_sha="abc123")
    assert returned is payload
    assert returned["provenance"]["git_sha"] == "abc123"


# --------------------------------------------------------------------------- #
# assert_reproducible — the fail-closed gate
# --------------------------------------------------------------------------- #
def _good_payload() -> dict[str, object]:
    return prov.stamp({"rows": [{"score": 0.9}]}, git_sha="a" * 40)


def test_assert_reproducible_passes_for_complete_artefact() -> None:
    prov.assert_reproducible(_good_payload())


def test_assert_reproducible_rejects_missing_provenance() -> None:
    with pytest.raises(prov.ProvenanceError, match="provenance"):
        prov.assert_reproducible({"rows": [{"score": 0.9}]})


@pytest.mark.parametrize(
    "provenance", [{}, {"git_sha": ""}, {"git_sha": prov.UNKNOWN_SHA}]
)
def test_assert_reproducible_rejects_unresolved_sha(
    provenance: dict[str, object],
) -> None:
    # Hand-build the block: ``stamp`` would resolve an empty sha to HEAD, so the
    # guard's own git_sha check is exercised directly here.
    payload = {"rows": [{"score": 0.9}], "provenance": provenance}
    with pytest.raises(prov.ProvenanceError, match="git_sha"):
        prov.assert_reproducible(payload)


@pytest.mark.parametrize("empty", [[], {}, None])
def test_assert_reproducible_rejects_absent_rows(empty: object) -> None:
    payload = prov.stamp({"rows": empty}, git_sha="a" * 40)
    with pytest.raises(prov.ProvenanceError, match="per-sample"):
        prov.assert_reproducible(payload)


def test_assert_reproducible_rejects_aggregate_only_keys() -> None:
    # per_task / matrix are aggregate breakdowns, not per-sample rows.
    payload = prov.stamp(
        {"per_task": {"qa": {"f1": 0.5}}, "matrix": {"a": 1}}, git_sha="a" * 40
    )
    with pytest.raises(prov.ProvenanceError, match="per-sample"):
        prov.assert_reproducible(payload)


@pytest.mark.parametrize("key", ["rows", "per_sample", "samples"])
def test_assert_reproducible_accepts_each_sample_key(key: str) -> None:
    payload = prov.stamp({key: [{"score": 0.1}]}, git_sha="a" * 40)
    prov.assert_reproducible(payload)


def test_assert_reproducible_require_clean_rejects_dirty_tree() -> None:
    payload = prov.stamp({"rows": [{"score": 0.9}]}, git_sha="a" * 40)
    payload["provenance"]["git_dirty"] = True
    with pytest.raises(prov.ProvenanceError, match="dirty"):
        prov.assert_reproducible(payload, require_clean=True)


def test_assert_reproducible_require_clean_allows_clean_tree() -> None:
    payload = prov.stamp({"rows": [{"score": 0.9}]}, git_sha="a" * 40)
    payload["provenance"]["git_dirty"] = False
    prov.assert_reproducible(payload, require_clean=True)


# --------------------------------------------------------------------------- #
# RAGTruth wiring — per-sample extraction + reproducible artefact
# --------------------------------------------------------------------------- #
def _metrics_with_samples() -> E2EMetrics:
    metrics = E2EMetrics()
    metrics.samples.append(
        E2ESample(
            task="ragtruth",
            context="ctx",
            response="grounded",
            is_hallucinated=False,
            coherence_score=0.91,
            approved=True,
            latency_ms=12.3456,
        )
    )
    metrics.samples.append(
        E2ESample(
            task="ragtruth",
            context="ctx",
            response="made up",
            is_hallucinated=True,
            coherence_score=0.12,
            approved=False,
            latency_ms=8.0,
        )
    )
    return metrics


def test_per_sample_rows_extracts_scores() -> None:
    rows = ragtruth_eval.per_sample_rows(_metrics_with_samples())
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[0]["is_hallucinated"] is False
    assert rows[0]["approved"] is True
    assert rows[0]["coherence_score"] == 0.91
    assert rows[1]["is_hallucinated"] is True
    assert rows[1]["approved"] is False
    assert rows[1]["latency_ms"] == 8.0


def test_build_artefact_is_reproducible_and_keeps_aggregates() -> None:
    payload = ragtruth_eval.build_artefact(_metrics_with_samples())
    # aggregates survive
    assert payload["total"] == 2
    assert "f1" in payload and "catch_rate" in payload
    # per-sample rows + resolved provenance
    assert len(payload["rows"]) == 2
    assert _HEX40.match(payload["provenance"]["git_sha"])
    prov.assert_reproducible(payload)  # does not raise


def test_build_artefact_fails_closed_on_empty_metrics() -> None:
    with pytest.raises(prov.ProvenanceError, match="per-sample"):
        ragtruth_eval.build_artefact(E2EMetrics())


def test_build_artefact_records_explicit_git_sha() -> None:
    payload = ragtruth_eval.build_artefact(_metrics_with_samples(), git_sha="feed" * 10)
    assert payload["provenance"]["git_sha"] == "feed" * 10


# --------------------------------------------------------------------------- #
# operating-point campaign wiring — git_sha default + fail-closed gate
# --------------------------------------------------------------------------- #
def _patch_campaign_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavy download/model/scoring path with tiny deterministic fakes."""
    monkeypatch.setattr(
        campaign, "load_halueval_qa", lambda _url, _max: [{"question": "q"}]
    )
    monkeypatch.setattr(campaign, "build_scorer", lambda _backend: (object(), object()))
    monkeypatch.setattr(
        campaign,
        "score_pairs",
        lambda _scorer, _store, _samples: [
            {"sample": 0, "label": "right", "coherence": 0.9},
            {"sample": 0, "label": "hallucinated", "coherence": 0.1},
        ],
    )
    monkeypatch.setattr(campaign, "sweep", lambda _rows: {"n_good": 1, "n_bad": 1})


def _campaign_args(git_sha: str = "") -> argparse.Namespace:
    return argparse.Namespace(
        git_sha=git_sha, data_url="http://example/qa", max_samples=1, backend="deberta"
    )


def test_campaign_run_stamps_reproducible_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_campaign_scoring(monkeypatch)
    payload = campaign.run_campaign(_campaign_args())
    assert _HEX40.match(payload["provenance"]["git_sha"])
    assert len(payload["rows"]) == 2
    prov.assert_reproducible(payload)  # does not raise


def test_campaign_git_sha_defaults_to_head(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_campaign_scoring(monkeypatch)
    payload = campaign.run_campaign(_campaign_args())
    assert payload["git_sha"] == prov.resolve_git_sha()


def test_campaign_explicit_git_sha_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_campaign_scoring(monkeypatch)
    payload = campaign.run_campaign(_campaign_args(git_sha="dead" * 10))
    assert payload["git_sha"] == "dead" * 10
    assert payload["provenance"]["git_sha"] == "dead" * 10
