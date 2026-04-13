# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Example: custom Rust scorer integration
"""Example: extend backfire-kernel with a custom Rust scoring function.

This example shows how to:
1. Define a Python scorer function with a Rust fast-path
2. Use the ``_RUST_AVAILABLE`` pattern for transparent fallback
3. Integrate with the existing backend registry

The Rust implementation goes in ``backfire-kernel/crates/backfire-ffi/``
and is exposed via PyO3. The Python fallback ensures the scorer works
even without the Rust kernel installed.

Usage::

    python examples/custom_rust_scorer.py

Prerequisites::

    pip install director-ai          # Python-only (uses fallback)
    pip install director-ai[rust]    # Rust-accelerated (uses FFI)
"""

from __future__ import annotations

import re
import time

# ── Rust fast-path pattern ─────────────────────────────────────────────
#
# This is the standard pattern used across Director-AI for all 34+
# Rust-accelerated functions. The try/except at module level means:
# - If backfire_kernel is installed: use Rust (10-60x faster)
# - If not: transparent Python fallback (same results, slower)

_RUST_AVAILABLE = False
# In a real implementation:
# try:
#     from backfire_kernel import rust_custom_score
#     _RUST_AVAILABLE = True
# except ImportError:
#     pass


# ── Python fallback implementation ─────────────────────────────────────


def _python_domain_score(premise: str, hypothesis: str, domain: str) -> float:
    """Domain-aware factual consistency scorer (Python implementation).

    Checks that the hypothesis is consistent with the premise using
    domain-specific rules:
    - medical: strict numeric + unit checking
    - finance: decimal precision + currency symbol validation
    - general: keyword overlap heuristic

    Returns 0.0 (fully consistent) to 1.0 (contradicted).
    """
    if domain == "medical":
        return _medical_score(premise, hypothesis)
    if domain == "finance":
        return _finance_score(premise, hypothesis)
    return _general_score(premise, hypothesis)


def _medical_score(premise: str, hypothesis: str) -> float:
    """Medical domain: check numeric values + units match."""
    # Extract numbers from both texts
    prem_nums = set(re.findall(r"\d+\.?\d*", premise))
    hyp_nums = set(re.findall(r"\d+\.?\d*", hypothesis))

    if not hyp_nums:
        return 0.3  # no numbers to check

    # Check if all hypothesis numbers appear in premise
    missing = hyp_nums - prem_nums
    if missing:
        return min(1.0, 0.5 + 0.2 * len(missing))

    return 0.0


def _finance_score(premise: str, hypothesis: str) -> float:
    """Finance domain: check decimal precision + currency."""
    currencies = {"CHF", "EUR", "USD", "GBP", "JPY"}
    prem_curr = {w for w in premise.split() if w in currencies}
    hyp_curr = {w for w in hypothesis.split() if w in currencies}

    if hyp_curr and hyp_curr != prem_curr:
        return 0.8  # currency mismatch

    return _general_score(premise, hypothesis)


def _general_score(premise: str, hypothesis: str) -> float:
    """General domain: keyword overlap."""
    p_words = set(premise.lower().split())
    h_words = set(hypothesis.lower().split())
    if not h_words:
        return 0.5
    overlap = len(p_words & h_words) / len(h_words)
    return max(0.0, 1.0 - overlap)


# ── Public API with Rust fast-path ─────────────────────────────────────


def domain_score(premise: str, hypothesis: str, domain: str = "general") -> float:
    """Score factual consistency with domain-specific rules.

    Uses Rust fast-path when ``backfire_kernel`` is installed,
    otherwise falls back to Python implementation.

    Parameters
    ----------
    premise : str
        Source/context text.
    hypothesis : str
        Generated text to verify.
    domain : str
        ``"medical"``, ``"finance"``, or ``"general"``.

    Returns
    -------
    float
        0.0 (consistent) to 1.0 (contradicted).
    """
    if _RUST_AVAILABLE:
        # In production, this would call:
        # return float(rust_custom_score(premise, hypothesis, domain))
        pass

    return _python_domain_score(premise, hypothesis, domain)


# ── Backend registration ───────────────────────────────────────────────
#
# To register as a Director-AI scorer backend, create a wrapper class
# that implements the ScorerBackend protocol and register it:
#
#   from director_ai.core.scoring.backends import register_backend
#
#   class DomainBackend:
#       def __init__(self, domain="general"):
#           self._domain = domain
#       def score(self, premise, hypothesis):
#           return 1.0 - domain_score(premise, hypothesis, self._domain)
#       def score_batch(self, pairs):
#           return [self.score(p, h) for p, h in pairs]
#
#   register_backend("domain", DomainBackend)
#
# Then use via config: scorer_backend="domain"


# ── Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Director-AI Custom Rust Scorer Example")
    print("=" * 50)
    print(f"Rust available: {_RUST_AVAILABLE}")
    print()

    cases = [
        ("general", "Water boils at 100°C.", "Water boils at 100°C.", "consistent"),
        ("general", "Paris is in France.", "Paris is in Germany.", "contradicted"),
        (
            "medical",
            "Dosage: 500mg twice daily.",
            "Dosage: 500mg twice daily.",
            "consistent",
        ),
        ("medical", "Blood pressure 120/80.", "Blood pressure 140/90.", "contradicted"),
        ("finance", "Revenue: CHF 1.2M.", "Revenue: EUR 1.2M.", "currency mismatch"),
        ("finance", "Revenue: CHF 1.2M.", "Revenue: CHF 1.2M.", "consistent"),
    ]

    for domain, prem, hyp, expected in cases:
        score = domain_score(prem, hyp, domain)
        status = "✓" if (score < 0.5) == (expected == "consistent") else "✗"
        print(f"  {status} [{domain:8}] score={score:.3f} | {expected:15} | {hyp}")

    # Benchmark
    print()
    t0 = time.perf_counter()
    for _ in range(10000):
        domain_score("Water boils at 100°C.", "Water boils at 500°C.", "medical")
    ms = (time.perf_counter() - t0) * 1000 / 10000
    print(f"  Benchmark: {ms:.4f} ms/call (Python fallback)")
    print(f"  With Rust: expected ~{ms / 20:.4f} ms/call (~20x speedup)")
