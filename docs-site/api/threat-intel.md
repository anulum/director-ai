# Threat-Intelligence Integration (STIX)

Check prompts and responses against known indicators of compromise and report not
just a block but the **attribution** — "matches the APT29 phishing kit". Indicators
are STIX-aligned, so they can be imported from any threat-intel feed; the matcher
is the local detection half, independent of the transport that delivered them.

> **TAXII transport.** TAXII is the network protocol that delivers a STIX bundle
> from a feed server. That client is out of scope here — supply the bundle (from a
> TAXII pull, a file, or a vendor export) to `from_stix_bundle()`.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.threat_intel import from_stix_bundle

ti = ProductionGuard(DirectorConfig()).threat_intel
ti.add_many(from_stix_bundle(my_stix_bundle))   # import a feed

matches = ti.match(prompt_or_response)
if matches:
    top = matches[0]                            # highest severity first
    print(top.attribution, top.severity)        # "APT29", "high"
print(ti.attributions(text))                    # ('APT28', 'APT29')
```

## Indicators

A `ThreatIndicator` carries the IOC plus what makes a hit actionable: its
`attribution` (intrusion set / actor) and `severity` (`low` … `critical`).
`indicator_type` selects how it matches:

| `IndicatorType` | Match |
|-----------------|-------|
| `SUBSTRING` | Case-insensitive containment. |
| `REGEX` | Regular-expression search (case-sensitive as authored). |
| `SHA256` | SHA-256 of the whole text equals the pattern. |

`ThreatIntelligenceMatcher.match(text)` returns every fired indicator as a
`ThreatMatch` (highest severity first); `is_threat(text)` is the boolean form and
`attributions(text)` the distinct actors. A duplicate indicator id is rejected.
`ThreatMatch.to_dict()` is tenant-safe — indicator id, name, type, attribution,
and severity only, never the inspected text.

## STIX import

`from_stix_bundle(bundle)` reads the bundle's `indicator` objects and resolves
attribution from `intrusion-set` / `threat-actor` / `malware` / `campaign` objects
via `relationship` objects (`indicates` / `attributed-to`).

**Supported pattern subset (documented, not full STIX patterning):** simple
comparison terms `[path OP 'value']`. `=` becomes a `SUBSTRING` (or `SHA256` when
the path is a SHA-256 hash); `MATCHES` becomes a `REGEX`. Compound patterns
(several comparisons joined by `AND`/`OR`) are decomposed into one indicator per
comparison. A term this subset cannot represent is **skipped, never guessed** — so
the importer never fabricates an indicator it could not actually parse.
