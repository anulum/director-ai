# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — jailbreak benchmark harness tests (offline)

from __future__ import annotations

import base64
from dataclasses import dataclass

from benchmarks import jailbreak_detection as jb


def test_rate_handles_zero_total() -> None:
    assert jb._rate(0, 0) == 0.0
    assert jb._rate(3, 4) == 0.75
    assert jb._rate(1, 3) == 0.3333


def test_every_template_embeds_the_goal() -> None:
    goal = "do the forbidden thing"
    for name, template in jb._JAILBREAK_TEMPLATES:
        wrapped = template(goal)
        assert wrapped != goal, name
        if name == "base64_obfuscation":
            # the goal is encoded, so the plaintext is absent but recoverable
            assert goal not in wrapped
            assert base64.b64encode(goal.encode()).decode() in wrapped
        else:
            assert goal in wrapped, name


def test_column_selects_first_present_candidate() -> None:
    csv_text = "Index,Goal,Other\n0,attack one,x\n1,attack two,y\n"
    assert jb._column(csv_text, "Behavior", "Goal") == ["attack one", "attack two"]


def test_column_raises_when_no_candidate_matches() -> None:
    try:
        jb._column("A,B\n1,2\n", "Goal", "Behavior")
    except ValueError as exc:
        assert "Goal" in str(exc)
    else:  # pragma: no cover - failure path
        raise AssertionError("expected ValueError")


@dataclass
class _FakeResult:
    blocked: bool


class _MarkerSanitizer:
    """Blocks any text containing the marker — deterministic stand-in."""

    block_threshold = 0.5

    def __init__(self, marker: str) -> None:
        self._marker = marker

    def score(self, text: str) -> _FakeResult:
        return _FakeResult(blocked=self._marker in text)


def test_injection_flag_rate_counts_blocked() -> None:
    san = _MarkerSanitizer("BAD")
    flagged, total = jb._injection_flag_rate(san, ["BAD one", "ok", "BAD two"])
    assert (flagged, total) == (2, 3)


def test_injection_jailbroken_aggregates_per_template() -> None:
    # The marker only appears via the prefix_injection template ("developer mode"),
    # so exactly one template family fires across all goals.
    san = _MarkerSanitizer("developer mode")
    goals = ["alpha", "beta"]
    out = jb._injection_jailbroken(san, goals)
    assert out["total"] == len(goals) * len(jb._JAILBREAK_TEMPLATES)
    assert out["flagged"] == len(goals)  # only prefix_injection matches
    assert out["per_template"]["prefix_injection"]["rate"] == 1.0
    assert out["per_template"]["dan_roleplay"]["rate"] == 0.0


def test_run_structure_and_rates_without_toxicity(monkeypatch) -> None:
    monkeypatch.setattr(
        jb,
        "_load_datasets",
        lambda **kw: jb._Datasets(
            harmbench=["harm a", "harm b"],
            jbb_harmful=["harm c"],
            jbb_benign=["please summarise this", "what is the capital of France"],
            sources={"harmbench_all": {"url": "x", "rows": 2}},
        ),
    )
    monkeypatch.setattr(
        jb, "InputSanitizer", lambda: _MarkerSanitizer("developer mode")
    )

    result = jb.run(with_toxicity=False, with_artifacts=False)

    assert result["benchmark"] == "jailbreak_detection"
    # raw harmful carries no injection marker -> zero
    assert result["raw_baseline"]["detection_rate"] == 0.0
    assert result["raw_baseline"]["total"] == 3
    # only prefix_injection injects the marker -> 3 goals / (3*5) wrapped
    templates = result["template_families"]
    assert templates["flagged"] == 3
    assert templates["total"] == 15
    # held-out families exist and (with this marker) do not fire
    assert result["heldout_constructed"]["detection_rate"] == 0.0
    # benign carries no marker -> zero false positives
    fp = result["false_positives"]["injection_guard"]
    assert fp["false_positive_rate"] == 0.0
    assert fp["total"] == 2
    # optional layers omitted when disabled
    assert "moderation_toxicity" not in result
    assert "real_artifacts" not in result


def test_run_respects_max_samples(monkeypatch) -> None:
    monkeypatch.setattr(
        jb,
        "_load_datasets",
        lambda **kw: jb._Datasets(
            harmbench=["a", "b", "c", "d"],
            jbb_harmful=["e", "f", "g"],
            jbb_benign=["h", "i", "j"],
            sources={},
            benign_extra=["k", "l"],
        ),
    )
    monkeypatch.setattr(jb, "InputSanitizer", lambda: _MarkerSanitizer("zzz"))

    result = jb.run(max_samples=2, with_toxicity=False, with_artifacts=False)
    # harmbench[:2] + jbb_harmful[:2] = 4 raw harmful
    assert result["raw_baseline"]["total"] == 4
    # jbb_benign[:2] + benign_extra[:2] = 4 benign
    assert result["false_positives"]["injection_guard"]["total"] == 4
