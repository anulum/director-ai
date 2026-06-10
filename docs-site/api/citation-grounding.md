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

## Full API

::: director_ai.core.citation_grounding.citations.resolve_citations

::: director_ai.core.citation_grounding.citations.extract_inline_citations

::: director_ai.core.citation_grounding.citations.parse_reference_section

::: director_ai.core.citation_grounding.citations.Citation

::: director_ai.core.citation_grounding.citations.CitationKind
