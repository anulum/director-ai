# Continuous Guard Fuzzing

A static adversarial suite checks a fixed list of attacks; a fuzzer mutates a seed
corpus round after round and surfaces the variant a guard fails to flag. The
`ContinuousFuzzer` takes a guard predicate and a corpus of strings the guard
*should* flag, applies obfuscating mutations, and reports every one that slipped
through — plus any seed the guard missed outright. The RNG is seeded, so each
finding is replayable as a regression case.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig

fuzzer = ProductionGuard(DirectorConfig()).continuous_fuzzer(seed=0)

# predicate: True means "the guard flags this as an attack".
def guard_flags(text: str) -> bool:
    return my_injection_detector.is_attack(text)

report = fuzzer.run(guard_flags, rounds_per_seed=50)   # default attack corpus
print(report.ok)                 # False if any mutation bypassed the guard
for bypass in report.bypasses:
    print(bypass.operator, "→", bypass.mutation)        # replayable obfuscation
```

## Mutation operators

Each operator perturbs an attack string while a human still reads the same intent:

| Operator | Obfuscation |
|----------|-------------|
| `case_flip` | Randomised upper/lower casing. |
| `whitespace_inject` | Stray spaces, tabs, newlines between characters. |
| `zero_width_inject` | Zero-width spaces/joiners/BOM sprinkled in. |
| `homoglyph_substitute` | Latin letters → confusable Cyrillic homoglyphs. |
| `leetspeak` | `a→4`, `e→3`, `i→1`, `o→0`, `s→5`, `t→7`. |
| `char_duplicate` | Doubles a sample of characters. |
| `delimiter_inject` | Splices chat/template delimiters (`</s>`, `[INST]`, …). |

Pass a custom `mutators` mapping to `ContinuousFuzzer` to add domain-specific
operators.

## The report

`run(predicate, corpus=None, rounds_per_seed=25)` returns a `FuzzReport`:

| Field | Meaning |
|-------|---------|
| `seeds_tested` | Number of corpus seeds. |
| `mutations_run` | Total mutations evaluated. |
| `bypasses` | `Bypass(operator, seed, mutation)` for each mutation the guard missed. |
| `seed_misses` | Seeds the guard failed to flag even unmutated (a baseline gap). |
| `operators_used` | Which operators were exercised. |
| `ok` | True only when there were no bypasses and no seed misses. |

A seed the guard cannot even flag unmutated is reported as a `seed_miss` and is
not mutated further — fix the baseline first. Wire `run()` into CI to fail the
build when `report.ok` is False, turning bypass discovery into a standing
regression gate. The bypass payloads are attack-corpus mutations, not tenant
data, so they are safe to surface to the security team for triage.
