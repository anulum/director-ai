# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FactCG Public Claim Tests
"""Regression tests for public FactCG benchmark claim wording."""

from __future__ import annotations

import json
import tomllib
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest
from backfire_kernel import rust_wilson_score_interval

from director_ai import FeedbackStore, OnlineCalibrator

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "public_accuracy_manifest.toml"
FACTCG_RESULT = (
    ROOT / "benchmarks" / "results" / "aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json"
)
README = ROOT / "README.md"
ARCHITECTURE = ROOT / "ARCHITECTURE.md"
PRODUCTION_CHECKLIST = ROOT / "docs" / "PRODUCTION_CHECKLIST.md"


def _text(path: Path) -> str:
    """Return UTF-8 text from a repository file."""
    return path.read_text(encoding="utf-8")


def _mode_card_metric(card_id: str) -> str:
    """Return the public metric for one benchmark mode card."""
    payload = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    cards = cast(list[dict[str, Any]], payload["benchmark_mode_cards"])
    metrics = {card["id"]: str(card["public_metric"]) for card in cards}
    return metrics[card_id]


def _factcg_result() -> dict[str, Any]:
    """Return the committed global-threshold FactCG result packet."""
    return cast(dict[str, Any], json.loads(FACTCG_RESULT.read_text(encoding="utf-8")))


def test_factcg_manifest_keeps_global_and_tuned_metrics_separate() -> None:
    """The mode cards should keep default and tuned FactCG claims separate.

    The default card carries the committed local-packet operating point
    (per-dataset mean balanced accuracy at threshold 0.46 = 75.8%); the tuned
    card carries the per-dataset threshold replay (77.76%). Neither is conflated
    with the leaderboard's threshold-0.50 operating point (75.6%).
    """
    result = _factcg_result()

    assert result["avg_balanced_accuracy_pct"] == 75.8
    assert result["threshold"] == 0.46
    assert (
        _mode_card_metric("pure_nli_aggrefact_global")
        == "75.8% per-dataset mean balanced accuracy at threshold 0.46"
    )
    assert (
        _mode_card_metric("tuned_threshold_aggrefact")
        == "77.76% per-dataset mean balanced accuracy"
    )


def test_public_factcg_docs_pair_leaderboard_and_packet_operating_points() -> None:
    """Public docs should pair both AggreFact operating points, not one headline.

    Every headline surface reports the leaderboard number (75.6% at threshold
    0.50, #6) alongside the local-packet number (75.8% at threshold 0.46), and
    never presents the tuned replay (77.76%) as the default NLI score.
    """
    readme = _text(README)
    architecture = _text(ARCHITECTURE)
    production = _text(PRODUCTION_CHECKLIST)

    assert (
        'pip install "director-ai[nli]"                     # NLI scoring '
        "(75.6% leaderboard / 75.8% packet BA; 77.76% tuned replay)"
    ) in readme
    assert (
        "| **5** | NLI (FactCG) | **75.6% leaderboard / 75.8% packet BA** "
        "(77.76% tuned replay) | see latency table | `[nli]` |"
    ) in readme
    assert (
        "| **#6** | **Director-AI (FactCG)** | "
        "**75.6%** | 0.4B | see latency table | **Yes** |"
    ) in readme
    assert (
        "the committed local packet reports **75.8%** on the same "
        "per-dataset-mean metric at threshold 0.46"
    ) in readme
    assert (
        "With per-dataset threshold replay (no retraining), FactCG reaches "
        "**77.76%** in the committed threshold packet."
    ) in readme

    assert (
        "| DeBERTa | `[nli]` | 75.6% leaderboard / 75.8% packet "
        "| see benchmark packets |"
    ) in architecture
    assert (
        "| FactCG (ONNX) | `[nli,onnx]` | 75.6% leaderboard / 75.8% packet "
        "(77.76% tuned replay) | see benchmark packets |"
    ) in architecture

    assert (
        '| **NLI** | `pip install "director-ai[nli]"` | '
        "75.6% leaderboard / 75.8% packet BA |"
    ) in production
    assert (
        '| **NLI+RAG** | `pip install "director-ai[nli,vector]"` | '
        "75.6% leaderboard / 75.8% packet BA |"
    ) in production
    assert (
        '| **Full** | `pip install "director-ai[nli,vector,server]"` | '
        "75.6% leaderboard / 75.8% packet BA |"
    ) in production

    # The retracted 75.86% earlier-run figure must not resurface as a claim.
    assert "75.86%" not in readme
    assert "75.86% global" not in readme
    assert "77.8% BA" not in readme
    assert "| DeBERTa | `[nli]` | 77.8%" not in architecture
    assert "77.8% BA" not in production


def test_calibration_guide_keeps_measured_operating_points_separate() -> None:
    """Bind each documented false-rejection rate to its evaluation packet."""
    guide = _text(ROOT / "docs-site/guide/online-calibration.md")
    results = ROOT / "benchmarks/results"
    calibration = json.loads(_text(results / "wcs2a_operating_points_calibration.json"))
    eval_packets = {
        "dialogue": ("Dialogue", "dialogue_raw"),
        "summarization": ("Summarization, weakest link", "weakest_link"),
    }
    for point in calibration["points"]:
        task = point["task"]
        label, suffix = eval_packets[task]
        evaluation = json.loads(
            _text(results / f"e2e_nli_only_200_wcs2a_{suffix}.json")
        )
        outcome = evaluation["per_task"][task]
        n_cal = point["n_good"]
        cal_rate = point["actual_fpr"]
        n_eval = outcome["fp"] + outcome["tn"]
        assert evaluation["calibration_source"] == (
            "wcs2a_operating_points_calibration.json"
        )
        assert (
            f"| {label} | {point['target_fpr']:.1%} | "
            f"{round(n_cal * cal_rate)}/{n_cal} ({cal_rate:.1%}) | "
            f"{outcome['fp']}/{n_eval} ({outcome['fp'] / n_eval:.1%}) |"
        ) in guide

    factcg = _factcg_result()
    assert "leaderboard reports 75.6% at 0.50" in guide
    assert (
        f"packet reports {factcg['avg_balanced_accuracy_pct']:.1f}% "
        f"at {factcg['threshold']:.2f}" in guide
    )
    assert "replay reports 77.76%" in guide


def test_calibration_guide_reports_wilson_endpoints_and_insufficient_data(
    tmp_path: Path,
) -> None:
    """Verify the guide against native intervals and real labelled feedback."""
    guide = _text(ROOT / "docs-site/guide/online-calibration.md")
    production = _text(ROOT / "docs-site/deployment/production.md")
    with closing(FeedbackStore(str(tmp_path / "feedback.db"))) as store:
        for index in range(50):
            store.report(
                "Confirm the quoted price",
                f"Reviewed response {index}",
                guardrail_approved=index < 3,
                human_approved=False,
                guardrail_score=0.7,
                domain="pricing",
            )
        report = OnlineCalibrator(store).calibrate(domain="pricing")
    low, high = rust_wilson_score_interval(report.fpr, report.correction_count, 0.95)
    assert report.fpr == pytest.approx(0.06)
    assert report.fpr_ci == pytest.approx((high - low) / 2)
    assert report.fnr_ci == 1
    assert report.optimal_threshold is None
    assert f"`3/50 = {report.fpr:.4f}`" in guide
    assert f"`[{low:.4f}, {high:.4f}]`" in guide
    assert "no examples of that class is an unavailable estimate" in guide
    assert "exchangeable calibration and evaluation examples" in guide
    assert "online-calibration.md#task-routed-operating-points" in production
