# NeMo Guardrails Rails-as-Config

Teams migrating from NeMo Guardrails keep their rails as configuration —
a `config.yml` plus Colang `.co` flow files. `load_rails_config()` maps
the honest subset of those files onto Director's native declarative
`Policy`, and reports everything it could not map instead of silently
dropping it.

```python
from director_ai.integrations.rails_config import (
    RailsLoadResult,
    load_rails_config,
)

result: RailsLoadResult = load_rails_config("./nemo_config_dir")
violations = result.policy.check(candidate_answer)
print(result.to_dict())   # audit: what mapped, what did not
```

Accepted inputs: a NeMo config **directory** (`config.yml` + `*.co`), a
single YAML config, or a single Colang file.

## What maps

| NeMo construct | Director equivalent |
|---|---|
| Colang v1 topical rail — `define flow` pairing `user <intent>` with `bot refuse …` | Every example utterance of the intent becomes a `Policy` forbidden phrase (word-boundary, case-insensitive) |
| `rails.input/output.flows` entries `self check input`, `self check output`, `content safety check …` | Enables Director's dependency-free moderation detectors (keyword toxicity + regex PII); recorded in `notes` as a semantic substitution, not a re-implementation of the NeMo prompt-based checks |

## What is reported instead of translated

Bot message definitions, subflows, `execute` actions, variables and
conditionals, model/prompt configuration, unrecognised flow names, and
refusal intents with no example utterances all land in
`RailsLoadResult.unsupported` — one entry per construct, so the
migration gap is auditable. Non-refusal flows are listed there too:
their intents are observed, not enforced.

Guardrails AI RAIL XML is intentionally not translated: that ecosystem
is integrated natively as a validator (see
[Guardrails AI](guardrails-ai.md)), so `.rail`/`.xml` inputs raise a
`ValueError` pointing there.
