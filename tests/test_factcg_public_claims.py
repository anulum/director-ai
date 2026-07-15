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
from pathlib import Path
from typing import Any, cast

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
    for card in cards:
        if card["id"] == card_id:
            return str(card["public_metric"])
    raise AssertionError(f"missing benchmark mode card {card_id!r}")


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
