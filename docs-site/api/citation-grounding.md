# Citation Grounding

The citation-grounding subsystem checks whether a generated answer's factual
assertions are actually supported by the sources it cites — the operationalisation
of groundedness used by HalluHard-style evaluation. The first step is locating
and resolving the citations in the answer.

## Citation extraction

`extract_inline_citations` finds every citing marker in an answer, across five
styles:

| Style | Example | Resolved identifier |
| --- | --- | --- |
| numeric | `[1]`, `[2, 3]`, `[4-6]` | the label (resolved via the reference list) |
| DOI | `10.3847/2041-8213/ab50c5`, `doi:…`, `https://doi.org/…` | the DOI |
| arXiv | `arXiv:2411.04368`, `arxiv.org/abs/…`, `cond-mat/0211034` | the arXiv id |
| URL | `https://example.org/x` | the URL (trailing punctuation trimmed) |
| author-year | `(Riess et al., 2022)`, `(Doe 2023)` | `Author YEAR` |

A DOI or arXiv id appearing inside a URL is reported once, as the more specific
citation. Numeric markers are expanded (`[2, 3]` → two citations, `[4-6]` → three).

```python
from director_ai.core.citation_grounding import resolve_citations

answer = """Radii constrain the EOS [1]. NICER measured this [2].

References:
[1] Bogdanov 10.3847/2041-8213/ab50c5
[2] NICER arXiv:2411.04368
"""

for cite in resolve_citations(answer):
    print(cite.kind.value, cite.identifier)
# doi   10.3847/2041-8213/ab50c5
# arxiv 2411.04368
```

## Reference resolution

`parse_reference_section` reads a trailing *References* / *Bibliography* block and
maps each numeric label to the concrete DOI / arXiv id / URL it points at,
preferring a DOI or arXiv id over a bare landing-page URL. `resolve_citations`
combines extraction and resolution: numeric markers are rewritten to the concrete
identifier their label references (an unresolvable marker is dropped), while
inline DOI/arXiv/URL/author-year citations are already concrete and pass through.
Citations that fall inside the reference list itself are excluded, so a work is
never counted both as a citation and as its own bibliography entry.

## Grounding judge

`CitationGroundingJudge` decides whether each assertion in an answer is grounded
in what it cites — the core of the HalluHard groundedness metric. The answer is
split into sentence-level assertions; each is matched to the citations occurring
within it; and the cited sources' text is scored against the assertion with an
NLI scorer. An assertion is **grounded** only when it carries a citation *and*
the cited material entails it. An uncited factual sentence, or one whose cited
source fails to support it, is a hallucination.

```python
from director_ai.core import NLIScorer
from director_ai.core.citation_grounding import CitationGroundingJudge

judge = CitationGroundingJudge(scorer=NLIScorer(use_nli=True), support_threshold=0.6)
report = judge.assess(answer, sources)  # sources: {identifier: fetched_text}

print(report.grounded_fraction, report.citation_coverage)
for claim in report.hallucinated:
    print("ungrounded:", claim.claim)
```

The judge is backend-agnostic — it accepts anything exposing
`score(premise, hypothesis) -> float` (the `Scorer` protocol, satisfied by
`NLIScorer`), so its logic is fully exercised in tests with a stub and no model.
A citation whose identifier is missing from `sources` (the fetch failed)
contributes no evidence, so the assertion is judged ungrounded rather than
silently passed.

## Full API

::: director_ai.core.citation_grounding.citations.resolve_citations

::: director_ai.core.citation_grounding.citations.extract_inline_citations

::: director_ai.core.citation_grounding.citations.parse_reference_section

::: director_ai.core.citation_grounding.citations.Citation

::: director_ai.core.citation_grounding.citations.CitationKind

::: director_ai.core.citation_grounding.judge.CitationGroundingJudge

::: director_ai.core.citation_grounding.judge.GroundingReport

::: director_ai.core.citation_grounding.judge.ClaimGrounding
