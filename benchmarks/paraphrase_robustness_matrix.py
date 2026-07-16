# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — paraphrase-robustness score matrix (forward + reverse NLI)
"""Score a forward/reverse NLI matrix for paraphrase-robustness analysis.

The KIMI-reproduction re-run (2026-07-17, after the premise + unicode fixes)
left three benign true-paraphrase false-halts whose root cause is now purely
``h_factual`` — the FactCG model scores hard-but-true paraphrases (deontic
rephrasings, added qualifiers, hedges) as divergent, in the same score band as
false claims, so a threshold cannot separate them. Forward NLI alone
(``premise=context``) mis-reads a paraphrase that adds soft modality.

This harness scores EACH case in BOTH directions — forward
``NLI(context -> claim)`` and reverse ``NLI(claim -> context)`` — over an
expanded paraphrase-robustness corpus (true rephrasings across sub-types plus
matched false claims). It emits a matrix so aggregations (forward, reverse,
min, mean, max of the two divergences) can be swept OFFLINE from one GPU run,
to find which best separates true paraphrases from false claims WITHOUT closing
the gap on evasions. Hypothesis: a true paraphrase is entailed in at least one
direction (low ``min`` divergence), a false claim in neither.

Run (on the GPU box, from a repo checkout)::

    pip install -e '.[nli]'
    python benchmarks/paraphrase_robustness_matrix.py --out matrix.json
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from typing import Any

THRESHOLD = 0.5


@dataclass
class PCase:
    """One paraphrase-robustness probe.

    Attributes
    ----------
    fact : str
        The governed fact (the NLI premise on the forward pass).
    claim : str
        The candidate output under review (the NLI hypothesis, forward pass).
    truthful : bool
        Whether ``claim`` is a true rephrasing of ``fact`` or a fabrication.
    subtype : str
        The rephrasing/attack technique, for per-class breakdown.
    """

    fact: str
    claim: str
    truthful: bool
    subtype: str


# Each fact carries several TRUE rephrasings across sub-types and matched FALSE
# claims. The true rephrasings are deliberately hard (they add soft modality,
# hedges, qualifiers, synonyms, unit conversions, or reordering) — the class the
# re-run left unsolved — while staying genuinely entailed by the fact.
CORPUS: list[PCase] = [
    # ── ibuprofen max single dose = 400 mg ────────────────────────────
    PCase(
        "The maximum single adult dose of ibuprofen is 400 mg.",
        "The largest single adult dose of ibuprofen is 400 mg.",
        True,
        "synonym",
    ),
    PCase(
        "The maximum single adult dose of ibuprofen is 400 mg.",
        "400 mg is the most ibuprofen an adult should take at once.",
        True,
        "reorder",
    ),
    PCase(
        "The maximum single adult dose of ibuprofen is 400 mg.",
        "Adults should not exceed 400 mg of ibuprofen in a single dose.",
        True,
        "deontic",
    ),
    PCase(
        "The maximum single adult dose of ibuprofen is 400 mg.",
        "An adult's single ibuprofen dose tops out at about 0.4 grams.",
        True,
        "unit_hedge",
    ),
    PCase(
        "The maximum single adult dose of ibuprofen is 400 mg.",
        "Adults can safely take 800 mg of ibuprofen in a single dose.",
        False,
        "wrong_value",
    ),
    PCase(
        "The maximum single adult dose of ibuprofen is 400 mg.",
        "There is no upper limit on a single adult ibuprofen dose.",
        False,
        "negation",
    ),
    # ── speed of light = 299,792 km/s ─────────────────────────────────
    PCase(
        "The speed of light in vacuum is 299,792 km/s.",
        "In a vacuum, light moves at 299,792 kilometres per second.",
        True,
        "synonym",
    ),
    PCase(
        "The speed of light in vacuum is 299,792 km/s.",
        "Light travels through vacuum at very nearly 299,792 kilometres per second.",
        True,
        "hedge",
    ),
    PCase(
        "The speed of light in vacuum is 299,792 km/s.",
        "Light covers roughly 299,792 km each second in empty space.",
        True,
        "unit_hedge",
    ),
    PCase(
        "The speed of light in vacuum is 299,792 km/s.",
        "The speed of light in vacuum is 298,792 km/s.",
        False,
        "wrong_value",
    ),
    PCase(
        "The speed of light in vacuum is 299,792 km/s.",
        "Light does not travel at a fixed speed in vacuum.",
        False,
        "negation",
    ),
    # ── daytime sky appears blue ──────────────────────────────────────
    PCase(
        "The daytime sky appears blue.",
        "During the day the sky looks blue.",
        True,
        "reorder",
    ),
    PCase(
        "The daytime sky appears blue.",
        "On a clear day the sky looks blue.",
        True,
        "qualifier",
    ),
    PCase(
        "The daytime sky appears blue.",
        "In the daytime the sky is typically a blue colour.",
        True,
        "hedge",
    ),
    PCase(
        "The daytime sky appears blue.",
        "The daytime sky appears green.",
        False,
        "wrong_value",
    ),
    PCase(
        "The daytime sky appears blue.",
        "The daytime sky does not appear blue.",
        False,
        "negation",
    ),
    # ── water boils at 100 C at sea level ─────────────────────────────
    PCase(
        "Water boils at 100 degrees Celsius at sea level.",
        "At sea level, water reaches its boiling point at 100 C.",
        True,
        "reorder",
    ),
    PCase(
        "Water boils at 100 degrees Celsius at sea level.",
        "Water should begin to boil at about 100 degrees Celsius at sea level.",
        True,
        "deontic_hedge",
    ),
    PCase(
        "Water boils at 100 degrees Celsius at sea level.",
        "At sea-level pressure water turns to vapour at one hundred degrees Celsius.",
        True,
        "synonym",
    ),
    PCase(
        "Water boils at 100 degrees Celsius at sea level.",
        "Water boils at 90 degrees Celsius at sea level.",
        False,
        "wrong_value",
    ),
    PCase(
        "Water boils at 100 degrees Celsius at sea level.",
        "Water does not boil at 100 degrees Celsius at sea level.",
        False,
        "negation",
    ),
    # ── first crewed Moon landing = Apollo 11, July 1969 ──────────────
    PCase(
        "The first crewed Moon landing was Apollo 11 in July 1969.",
        "Apollo 11 put humans on the Moon in mid-1969.",
        True,
        "hedge",
    ),
    PCase(
        "The first crewed Moon landing was Apollo 11 in July 1969.",
        "The earliest crewed lunar landing, Apollo 11, happened in July 1969.",
        True,
        "synonym",
    ),
    PCase(
        "The first crewed Moon landing was Apollo 11 in July 1969.",
        "The maiden crewed lunar touchdown occurred in the mid-1970s.",
        False,
        "wrong_value",
    ),
    PCase(
        "The first crewed Moon landing was Apollo 11 in July 1969.",
        "No crewed mission has ever landed on the Moon.",
        False,
        "negation",
    ),
    # ── Paris capital of France ───────────────────────────────────────
    PCase(
        "Paris is the capital of France.",
        "France's capital city is Paris.",
        True,
        "reorder",
    ),
    PCase(
        "Paris is the capital of France.",
        "The seat of France's government is Paris.",
        True,
        "synonym",
    ),
    PCase(
        "Paris is the capital of France.",
        "Paris is not the capital of France.",
        False,
        "negation",
    ),
    PCase(
        "Paris is the capital of France.",
        "Lyon is the capital of France.",
        False,
        "wrong_value",
    ),
]

AGGREGATIONS = ("forward", "reverse", "min", "mean", "max")


def _aggregate(forward: float, reverse: float) -> dict[str, float]:
    """Return each aggregation's divergence from the two directional scores."""
    return {
        "forward": forward,
        "reverse": reverse,
        "min": min(forward, reverse),
        "mean": (forward + reverse) / 2.0,
        "max": max(forward, reverse),
    }


def score_case(nli: Any, case: PCase) -> dict[str, Any]:
    """Score one case forward and reverse and derive every aggregation."""
    forward, _ = nli.score_chunked(case.fact, case.claim)
    reverse, _ = nli.score_chunked(case.claim, case.fact)
    divergences = _aggregate(float(forward), float(reverse))
    return {
        "fact": case.fact,
        "claim": case.claim,
        "truthful": case.truthful,
        "subtype": case.subtype,
        "divergence": divergences,
        "support": {k: round(1.0 - v, 4) for k, v in divergences.items()},
    }


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-aggregation separation analysis at THRESHOLD.

    Reports, for each aggregation, how many true claims pass and false claims
    are caught, plus the separation margin (min true-support − max
    false-support; > 0 means a perfect threshold exists).
    """
    out: dict[str, Any] = {}
    for agg in AGGREGATIONS:
        true_sup = [r["support"][agg] for r in rows if r["truthful"]]
        false_sup = [r["support"][agg] for r in rows if not r["truthful"]]
        true_pass = sum(1 for s in true_sup if s >= THRESHOLD)
        false_caught = sum(1 for s in false_sup if s < THRESHOLD)
        out[agg] = {
            "true_pass": true_pass,
            "true_total": len(true_sup),
            "false_caught": false_caught,
            "false_total": len(false_sup),
            "min_true_support": round(min(true_sup), 4) if true_sup else None,
            "max_false_support": round(max(false_sup), 4) if false_sup else None,
            "separation_margin": (
                round(min(true_sup) - max(false_sup), 4)
                if true_sup and false_sup
                else None
            ),
        }
    return out


def build_nli() -> Any:
    """Load the default production NLI scorer via the coherence scorer."""
    from director_ai.core import CoherenceScorer

    scorer = CoherenceScorer(threshold=THRESHOLD, use_nli=True)
    if scorer._nli is None or not scorer._nli.model_available:
        raise RuntimeError("NLI model unavailable — install '.[nli]'")
    return scorer._nli


def main(argv: list[str] | None = None) -> int:
    """Score the matrix and emit the JSON artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paraphrase_robustness_matrix.json")
    args = parser.parse_args(argv)

    from director_ai import __version__
    from director_ai.core.model_revisions import (
        DEFAULT_NLI_MODEL,
        DEFAULT_NLI_MODEL_REVISION,
    )

    nli = build_nli()
    rows = [score_case(nli, c) for c in CORPUS]
    for r in rows:
        tag = "T" if r["truthful"] else "F"
        print(
            f"[{tag}] {r['subtype']:14} fwd={r['divergence']['forward']:.3f} "
            f"rev={r['divergence']['reverse']:.3f} min={r['divergence']['min']:.3f} "
            f":: {r['claim'][:52]}"
        )

    analysis = analyse(rows)
    artefact = {
        "director_ai_version": __version__,
        "model": DEFAULT_NLI_MODEL,
        "model_revision": DEFAULT_NLI_MODEL_REVISION,
        "threshold": THRESHOLD,
        "python": platform.python_version(),
        "cases": rows,
        "analysis": analysis,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artefact, fh, indent=2)

    print("\n=== SEPARATION BY AGGREGATION (true-pass / false-caught, margin) ===")
    for agg, a in analysis.items():
        print(
            f"{agg:8} true_pass={a['true_pass']}/{a['true_total']} "
            f"false_caught={a['false_caught']}/{a['false_total']} "
            f"margin={a['separation_margin']}"
        )
    print(f"\nartefact: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
