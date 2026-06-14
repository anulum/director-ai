# Live temporal-claim refresh

> **Status: advanced tier (BUSL-1.1), opt-in, makes live web requests.** The
> reliable payload is the *retrieved current evidence*. The verdict is decided by
> an injected NLI scorer when present, and falls back to a coarse lexical triage
> otherwise — treat a lexical `drift_suspected` as "verify", not "false".

[Temporal freshness scoring](../guide/scoring.md) flags claims that *may* rely on
outdated knowledge — a named office-holder, a statistic, a record — but it cannot
tell whether they are still true; it only knows they are the kind of thing that
goes stale. The refresher closes that gap: it takes the flagged claims, queries a
live web-search provider, and reports whether current sources still support them.

Where [temporal consistency](../guide/scoring.md) checks a claim against *this
system's own history*, the refresher checks it against *the live web*.

## How it works

1. Score the response for staleness (`score_temporal_freshness`).
2. For each sufficiently-stale claim, split it into an **unbiased subject** (the
   office or metric) and its **asserted value** (the incumbent or number). The
   query uses the subject only — querying with the value would just return pages
   echoing it, and every claim would look confirmed.
3. Fetch the top results and decide a verdict:
   * with an injected NLI `ContradictionScorer`, the fresh evidence is the
     premise and the claim the hypothesis — high `P(contradiction)` ⇒
     `contradicted`, else `supported`. This is the dependable path and reuses the
     same scorer as the [streaming contradiction halt](streaming.md);
   * without it, a lexical heuristic checks whether the asserted value appears in
     the top result — `supported` if so, else `drift_suspected`.
4. Always return the retrieved evidence (title, snippet, URL) so a human or a
   downstream verifier can compare.

## Usage

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard()
report = guard.refresh_temporal("The CEO of Twitter is Jack Dorsey.")
for r in report.refreshes:
    print(r.verdict, r.verdict_source, [h.title for h in r.evidence])
```

For adjudicated verdicts, inject the NLI scorer:

```python
from director_ai.core.scoring.contradiction import ContradictionScorer
from director_ai.core.scoring.temporal_refresh import TemporalRefresher

refresher = TemporalRefresher(nli=ContradictionScorer.from_pretrained())
report = refresher.refresh_response("The president of the United States is Joe Biden.")
```

## Polyglot backend

The `topical_overlap` diagnostic runs through the Rust `rust_word_overlap` kernel
with a bit-exact pure-Python fallback (case-folded, whitespace-split, punctuation
retained), so the two backends are identical and the dispatch is a speed choice.
The HTTP layer is injected through the `HttpGetter` protocol, so the parser is
tested without a network. See [Rust Acceleration](rust-acceleration.md).

## Measured

`python -m benchmarks.temporal_refresh` (offline; the live path is not run in the
committed benchmark):

| Metric | Value |
|---|---|
| DuckDuckGo result parser | 2 hits parsed, ~42k parses/s |
| Rust ↔ Python `topical_overlap` parity | exact |
| Lexical drift accuracy (clean position cases, n=10) | 0.90 |
| Lexical drift precision / recall | 1.00 / 0.80 |

The lexical path is precise but not complete: a former office-holder's name
persisting in current coverage can mask drift, and a sparse top result can
over-flag it. That ceiling is exactly why the NLI engine exists — these numbers
are the dependency-free floor, published as-is.

## Limits

This is an advisory enrichment, not a fact oracle. It performs no natural-language
inference on the lexical path, makes live network requests (rate limits and
provider HTML changes apply), and a `drift_suspected` verdict means *verify*, not
*false*. The retrieved evidence is the primary, reliable output.
