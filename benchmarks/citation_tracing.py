# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation tracing accuracy benchmark

"""Measure citation tracing.

The tracer is deterministic character-offset interval mapping over the citation
parser — there is no scoring or polyglot kernel, so the measurements are
attachment accuracy and throughput:

* **Attachment accuracy** — labelled snippets, each tagged with how many of their
  claim sentences should be cited, are run through ``trace_citations``; we report
  how often the measured cited-claim count matches the label.
* **Throughput** — traces per second over the corpus.

Output: ``benchmarks/results/citation_tracing.json``. Reproduce with
``python -m benchmarks.citation_tracing``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.verification.citation_tracer import trace_citations

# (text, expected_cited_claim_count)
_LABELLED: list[tuple[str, int]] = [
    (
        "Transformers came in 2017 [1]. They scale well.\n\nReferences:\n"
        "[1] https://arxiv.org/abs/1706.03762\n",
        1,
    ),
    (
        "BERT is bidirectional (Devlin et al., 2019). It pretrains on text. "
        "Fine-tuning adapts it (Howard and Ruder, 2018).",
        2,
    ),
    ("The result holds for all inputs. No citation is given here.", 0),
    ("See the DOI 10.1038/nature14539 for details. Deep learning is broad.", 1),
    (
        "All three are cited here [1][2]. A second uncited sentence follows.\n\n"
        "References:\n[1] https://a.example/x\n[2] https://b.example/y\n",
        1,
    ),
]


def attachment_accuracy() -> dict:
    hits = sum(
        1 for text, expected in _LABELLED if len(trace_citations(text).cited) == expected
    )
    return {"n": len(_LABELLED), "accuracy": round(hits / len(_LABELLED), 4)}


def throughput(repeats: int) -> dict:
    corpus = [t for t, _ in _LABELLED]
    t0 = time.perf_counter()
    for _ in range(repeats):
        for text in corpus:
            trace_citations(text)
    elapsed = time.perf_counter() - t0
    rate = (len(corpus) * repeats) / elapsed if elapsed else 0.0
    return {"traces_per_sec": round(rate, 1)}


def run(*, repeats: int = 3000) -> dict:
    return {
        "benchmark": "citation_tracing",
        "attachment": attachment_accuracy(),
        "throughput": throughput(repeats),
        "backend": "python-deterministic (offset interval mapping; no kernel)",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "citation_tracing.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    a = result["attachment"]
    print(f"\nCitation tracing (n={a['n']}):")
    print(f"  attachment accuracy={a['accuracy']:.2f}")
    print(f"  throughput {result['throughput']['traces_per_sec']:.0f}/s")


if __name__ == "__main__":
    main()
