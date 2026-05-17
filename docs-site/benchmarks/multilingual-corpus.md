# Multilingual Corpus

Director-AI ships a deterministic multilingual evaluation corpus for EU-market
regression testing. It is not a leaderboard claim; it is a fixture for
catching locale regressions in factual consistency, numeric consistency, policy
compliance, temporal freshness, and retrieval grounding.

## Coverage

| Property | Value |
|----------|-------|
| File | `benchmarks/multilingual_corpus.jsonl` |
| Rows | 200 |
| Languages | English, German, French, Spanish, Italian, Polish, Czech, Dutch |
| Rows per language | 25 |
| Labels | `supported`, `contradicted` |
| Expected decisions | `allow`, `halt` |
| Builder | `tools/build_multilingual_corpus.py` |
| Validator | `tools/validate_multilingual_corpus.py` |

Every row contains:

- `id`
- `language`
- `language_name`
- `category`
- `prompt`
- `source`
- `response`
- `label`
- `expected_decision`
- `risk_tags`

## Validation

```bash
uv run --frozen python tools/validate_multilingual_corpus.py .
```

The validator fails when:

- the corpus does not contain exactly 200 rows;
- any of the 8 expected languages has anything other than 25 rows;
- a required field is missing or empty;
- row IDs do not match their language prefix;
- labels and expected decisions disagree;
- category or label coverage is incomplete;
- risk tags are absent or malformed.

## Regeneration

```bash
uv run --frozen python tools/build_multilingual_corpus.py
uv run --frozen python tools/validate_multilingual_corpus.py .
```

The builder is deterministic. Review generated diffs before publishing because
the JSONL file is the benchmark fixture used by downstream regression checks.
