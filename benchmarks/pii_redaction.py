# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PII redaction precision/recall + Rust-vs-Python benchmark

"""Measure the PII detector on a labelled corpus and compare the Rust and Python
backends.

Two things are reported:

* **Accuracy** — span-level precision / recall / F1 of ``RegexPIIDetector``
  against a hand-labelled set covering all eight categories (email, credit_card,
  ssn, phone, phi, iban, passport, ipv4) plus negative texts that must not
  produce a finding. A detected ``(category, start, end)`` is a true positive
  only when it exactly matches a gold span.
* **Backend comparison** — the same corpus is scanned with ``prefer_rust=True``
  (``backfire_kernel.PiiScanner``) and ``prefer_rust=False`` (pure-Python
  regex). The two must agree exactly (parity) and both throughputs are recorded
  so the production dispatch can be chosen on measured speed, not assumption.

Output: ``benchmarks/results/pii_redaction.json``. Reproduce with
``python -m benchmarks.pii_redaction``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.safety.moderation.pii import RegexPIIDetector

# (text, [(category, pii_substring), ...]) — each substring must occur once.
_LABELLED: list[tuple[str, list[tuple[str, str]]]] = [
    ("Email me at alice.smith@example.com today.", [("email", "alice.smith@example.com")]),
    ("Card on file 4111 1111 1111 1111 expires soon.", [("credit_card", "4111 1111 1111 1111")]),
    ("Her SSN is 123-45-6789 per the form.", [("ssn", "123-45-6789")]),
    ("Call the desk at (555) 123-4567 before noon.", [("phone", "(555) 123-4567")]),
    ("Patient record MRN: AB12345 was updated.", [("phi", "MRN: AB12345")]),
    ("Wire to GB82WEST12345698765432 by Friday.", [("iban", "GB82WEST12345698765432")]),
    ("Passport B2345678 is on file.", [("passport", "B2345678")]),
    ("The server at 192.168.1.100 is unreachable.", [("ipv4", "192.168.1.100")]),
    (
        "Reach bob@corp.io or call +1-555-987-6543 for support.",
        [("email", "bob@corp.io"), ("phone", "+1-555-987-6543")],
    ),
    (
        "Please email carol.jones@mail.org to confirm the pending order.",
        [("email", "carol.jones@mail.org")],
    ),
    (
        "The saved card 5500-0000-0000-0004 should be removed.",
        [("credit_card", "5500-0000-0000-0004")],
    ),
    ("Tax id on the W-2 reads 987-65-4321 exactly.", [("ssn", "987-65-4321")]),
    ("Logs show a hit from 10.0.42.7 overnight.", [("ipv4", "10.0.42.7")]),
    ("Send invoices to billing@vendor.co.uk please.", [("email", "billing@vendor.co.uk")]),
    ("DOB: 1984-02-29 noted in the chart.", [("phi", "DOB: 1984-02-29")]),
    ("Reception line 212.555.0198 rings through.", [("phone", "212.555.0198")]),
    # Negatives — must produce no PII finding.
    ("The quarterly report is due next Tuesday.", []),
    ("Order 12 units of part XJ and ship standard.", []),
    ("Meeting moved to room 4 at 3pm sharp.", []),
    ("Temperature held at 451 degrees overnight.", []),
    ("Version 3 of the policy supersedes the draft.", []),
]


def _gold_spans(text: str, items: list[tuple[str, str]]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    for category, sub in items:
        start = text.index(sub)
        spans.add((category, start, start + len(sub)))
    return spans


def _detected_spans(detector, text: str) -> set[tuple[str, int, int]]:
    result = detector.analyse(text)
    return {(m.category, m.start, m.end) for m in result.matches}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate(detector, samples: list[tuple[str, list[tuple[str, str]]]]) -> dict:
    """Span-level precision/recall/F1, overall and per category."""
    from collections import defaultdict

    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    tp = fp = fn = 0
    for text, items in samples:
        gold = _gold_spans(text, items)
        got = _detected_spans(detector, text)
        for span in got & gold:
            per_cat[span[0]][0] += 1
        for span in got - gold:
            per_cat[span[0]][1] += 1
        for span in gold - got:
            per_cat[span[0]][2] += 1
        tp += len(got & gold)
        fp += len(got - gold)
        fn += len(gold - got)
    return {
        "overall": _prf(tp, fp, fn),
        "per_category": {
            cat: _prf(c[0], c[1], c[2]) for cat, c in sorted(per_cat.items())
        },
    }


def _throughput(detector, texts: list[str], repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        for text in texts:
            detector.analyse(text)
    elapsed = time.perf_counter() - t0
    return (len(texts) * repeats) / elapsed if elapsed else 0.0


def run(*, repeats: int = 200) -> dict:
    rust_detector = RegexPIIDetector(prefer_rust=True)
    py_detector = RegexPIIDetector(prefer_rust=False)
    texts = [t for t, _ in _LABELLED]

    accuracy = evaluate(py_detector, _LABELLED)

    parity = all(
        _detected_spans(rust_detector, t) == _detected_spans(py_detector, t)
        for t in texts
    )
    rust_tps = _throughput(rust_detector, texts, repeats)
    py_tps = _throughput(py_detector, texts, repeats)
    return {
        "benchmark": "pii_redaction",
        "n_samples": len(_LABELLED),
        "accuracy": accuracy,
        "backends": {
            "rust_available": rust_detector.backend == "rust",
            "parity_rust_equals_python": parity,
            "rust_texts_per_sec": round(rust_tps, 1),
            "python_texts_per_sec": round(py_tps, 1),
            "rust_speedup": round(rust_tps / py_tps, 3) if py_tps else None,
            "fastest": "rust" if rust_tps >= py_tps else "python",
        },
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "pii_redaction.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    o = result["accuracy"]["overall"]
    b = result["backends"]
    print(f"\nPII redaction (n={result['n_samples']}):")
    print(f"  overall P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}")
    print(
        f"  backend rust_available={b['rust_available']} "
        f"parity={b['parity_rust_equals_python']}"
    )
    print(
        f"  throughput rust={b['rust_texts_per_sec']:.0f}/s "
        f"python={b['python_texts_per_sec']:.0f}/s "
        f"speedup={b['rust_speedup']} fastest={b['fastest']}"
    )


if __name__ == "__main__":
    main()
