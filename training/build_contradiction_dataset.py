# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — contradiction-focused 3-class training set builder

"""Build a 3-class NLI training set that teaches contradiction ≠ unsupported.

The streaming halt fails because the base NLI model does not separate a claim
that *contradicts* the grounding from one that is merely *unsupported* — it fires
on both, so it false-halts correct-but-unsupported text. This builder produces a
dataset that targets exactly that boundary, from LLM-AggreFact:

* **entailment** — a supported claim paired with its most relevant document
  passage (the passage entails the claim);
* **contradiction** — the same passage paired with a meaning-flipped variant of
  the supported claim (see :class:`ContradictionInjector`); the passage now
  contradicts the claim;
* **neutral** — an unsupported (AggreFact label 0) claim paired with its most
  relevant passage; unsupported but not contradicting, so the model must learn
  to stay neutral rather than halt.

Premise is the retrieved passage (not the truncated whole document), matching the
production halt. Output is a HuggingFace ``DatasetDict`` saved to disk in the
``(premise, hypothesis, label, source)`` schema consumed by
``train_full_finetune.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks._common import select_relevant_passages
from benchmarks.contradiction_injection import ContradictionInjector

LABEL_ENTAILMENT = 0
LABEL_NEUTRAL = 1
LABEL_CONTRADICTION = 2

_AGGREFACT = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "aggrefact_test.jsonl"
)


def _best_passage(doc: str, claim: str) -> str:
    passages = select_relevant_passages(doc, claim, top_k=1)
    return passages[0] if passages else doc[:2000]


def build_examples(
    rows: list[dict],
    injector: ContradictionInjector,
    *,
    neutral_cap: int | None = None,
    cross_doc_neutral: int = 0,
) -> list[dict]:
    """Turn AggreFact rows into ``(premise, hypothesis, label, source)`` examples.

    Supported rows yield an entailment example plus one contradiction example per
    applicable injection strategy. Unsupported rows yield a neutral example, up
    to *neutral_cap* (to keep the neutral class from swamping the others).

    *cross_doc_neutral* adds that many extra neutral examples by pairing a
    supported claim with the most relevant passage of a *different* document.
    The claim is neither entailed nor contradicted by an unrelated passage, so
    it teaches the model the retrieved-but-irrelevant case — the dominant source
    of false halts — which AggreFact's small unsupported class underrepresents.
    """
    examples: list[dict] = []
    supported: list[dict] = []
    neutral_seen = 0
    for r in rows:
        doc, claim = r["doc"], r["claim"]
        if int(r["label"]) == 1:
            supported.append(r)
            premise = _best_passage(doc, claim)
            examples.append(
                {
                    "premise": premise,
                    "hypothesis": claim,
                    "label": LABEL_ENTAILMENT,
                    "source": "aggrefact_supported",
                }
            )
            for variant in injector.inject_all(claim):
                examples.append(
                    {
                        "premise": premise,
                        "hypothesis": variant.perturbed,
                        "label": LABEL_CONTRADICTION,
                        "source": f"aggrefact_injected_{variant.strategy}",
                    }
                )
        else:
            if neutral_cap is not None and neutral_seen >= neutral_cap:
                continue
            examples.append(
                {
                    "premise": _best_passage(doc, claim),
                    "hypothesis": claim,
                    "label": LABEL_NEUTRAL,
                    "source": "aggrefact_unsupported",
                }
            )
            neutral_seen += 1

    examples.extend(
        _cross_doc_neutrals(supported, cross_doc_neutral)
    )
    return examples


def _cross_doc_neutrals(supported: list[dict], count: int) -> list[dict]:
    """Pair supported claims with an unrelated document's most relevant passage.

    Deterministic: claim *i* is grounded against document *i + stride* (wrapping
    around), where the stride spreads pairings across the corpus rather than
    using adjacent — and possibly same-topic — documents.
    """
    if count <= 0 or len(supported) < 2:
        return []
    count = min(count, len(supported))
    stride = max(1, len(supported) // 2 + 1)
    out: list[dict] = []
    for i in range(count):
        claim = supported[i]["claim"]
        other_doc = supported[(i + stride) % len(supported)]["doc"]
        out.append(
            {
                "premise": _best_passage(other_doc, claim),
                "hypothesis": claim,
                "label": LABEL_NEUTRAL,
                "source": "cross_doc_neutral",
            }
        )
    return out


def _load(max_samples: int | None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in _AGGREFACT.read_text(encoding="utf-8").splitlines()
    ]
    return rows[:max_samples] if max_samples is not None else rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_samples", nargs="?", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).parent / "data_contradiction"),
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument(
        "--neutral-cap",
        type=int,
        default=None,
        help="Max neutral (unsupported) examples; default keeps all.",
    )
    parser.add_argument(
        "--cross-doc-neutral",
        type=int,
        default=8000,
        help="Extra neutral examples pairing claims with unrelated documents.",
    )
    args = parser.parse_args()

    from datasets import ClassLabel, Dataset, DatasetDict

    rows = _load(args.max_samples)
    examples = build_examples(
        rows,
        ContradictionInjector(),
        neutral_cap=args.neutral_cap,
        cross_doc_neutral=args.cross_doc_neutral,
    )
    counts = Counter(e["label"] for e in examples)
    print(
        f"examples={len(examples)} "
        f"entail={counts[LABEL_ENTAILMENT]} "
        f"neutral={counts[LABEL_NEUTRAL]} "
        f"contradiction={counts[LABEL_CONTRADICTION]}"
    )
    print("by source:", dict(Counter(e["source"] for e in examples)))

    ds = Dataset.from_list(examples)
    ds = ds.cast_column(
        "label",
        ClassLabel(num_classes=3, names=["entailment", "neutral", "contradiction"]),
    )
    split = ds.train_test_split(
        test_size=args.test_size, seed=42, stratify_by_column="label"
    )
    dataset = DatasetDict({"train": split["train"], "eval": split["test"]})
    dataset.save_to_disk(args.out_dir)
    print(f"saved train={len(dataset['train'])} eval={len(dataset['eval'])} -> {args.out_dir}")


if __name__ == "__main__":
    main()
