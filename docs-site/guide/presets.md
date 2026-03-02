# Domain Presets

`DirectorConfig.from_profile(name)` loads a tuned parameter set for common use cases.

## Profile Reference

| Profile | Threshold | Hard Limit | Soft Limit | NLI | Reranker | W_Logic | W_Fact |
|---------|-----------|------------|------------|-----|----------|---------|--------|
| `fast` | 0.50 | default | default | no | no | default | default |
| `thorough` | 0.60 | default | default | yes | no | default | default |
| `research` | 0.70 | default | default | yes | no | default | default |
| `medical` | 0.75 | 0.55 | 0.75 | yes | yes | 0.5 | 0.5 |
| `finance` | 0.70 | 0.50 | 0.70 | yes | yes | 0.4 | 0.6 |
| `legal` | 0.68 | 0.45 | 0.68 | yes | no | 0.6 | 0.4 |
| `creative` | 0.40 | 0.30 | 0.45 | no | no | 0.7 | 0.3 |
| `customer_support` | 0.55 | 0.40 | 0.60 | no | no | 0.5 | 0.5 |

"default" means the field inherits the `DirectorConfig` dataclass default (hard_limit=0.5, soft_limit=0.6, w_logic/w_fact=0.0 which defers to `CoherenceScorer` class defaults).

## Profile Rationale

**fast** — Heuristic scoring only, no model loading. Sub-millisecond latency for dev loops and high-throughput pipelines where approximate filtering is acceptable.

**thorough** — Adds NLI inference (FactCG-DeBERTa) to catch logical contradictions that heuristics miss. Standard production baseline.

**research** — Higher threshold (0.70) for academic and analytical workloads where factual precision matters more than recall.

**medical** — Tight hard limit (0.55) prevents any medical misinformation from reaching users. Reranker enabled for precise KB retrieval. Equal logic/fact weighting reflects the need for both clinical reasoning and factual accuracy.

**finance** — Fact-weighted (0.6) because numerical claims and regulatory data dominate. Reranker sharpens retrieval against financial KB documents.

**legal** — Logic-weighted (0.6) because legal reasoning chains (statute + precedent + application) matter more than isolated facts. No reranker; legal KBs tend to be smaller and well-structured.

**creative** — Permissive thresholds (0.40/0.30/0.45) allow divergent generation. NLI disabled to avoid penalising metaphor and fiction. Logic-weighted (0.7) because internal narrative consistency matters more than factual grounding.

**customer_support** — Moderate thresholds balance helpfulness with accuracy. NLI disabled for latency (support bots need fast responses). Equal weights suit mixed queries (policy facts + troubleshooting logic).

## Usage

```python
from director_ai import DirectorConfig

config = DirectorConfig.from_profile("medical")
```

Load via CLI:

```bash
director-ai quickstart --profile medical
```

Load via environment variable:

```bash
export DIRECTOR_PROFILE=finance
```

## Customising a Profile

`from_profile` returns a regular `DirectorConfig` dataclass. Override fields after loading:

```python
from dataclasses import replace
from director_ai import DirectorConfig

base = DirectorConfig.from_profile("medical")
config = replace(base, hard_limit=0.60, nli_model="lytang/MiniCheck-DeBERTa-L")
```

Or override via environment variables (env vars take precedence when using `from_env` after profile):

```python
config = DirectorConfig.from_profile("finance")
config.coherence_threshold = 0.72
config.reranker_top_k_multiplier = 5
```

## Profile + YAML

Combine a profile base with a YAML overlay:

```yaml
# config.yaml
coherence_threshold: 0.72
chroma_persist_dir: /data/chroma
audit_log_path: /var/log/director/audit.jsonl
```

```python
from director_ai import DirectorConfig

config = DirectorConfig.from_profile("finance")
yaml_overrides = DirectorConfig.from_yaml("config.yaml")

# Merge: YAML values override profile values
for field_name in DirectorConfig.__dataclass_fields__:
    yaml_val = getattr(yaml_overrides, field_name)
    default_val = DirectorConfig.__dataclass_fields__[field_name].default
    if yaml_val != default_val:
        setattr(config, field_name, yaml_val)
```

## Adding Custom Profiles

For organisation-specific profiles, subclass or wrap:

```python
from director_ai import DirectorConfig

INTERNAL_PROFILES = {
    "compliance": {
        "coherence_threshold": 0.80,
        "hard_limit": 0.60,
        "soft_limit": 0.80,
        "use_nli": True,
        "reranker_enabled": True,
        "w_logic": 0.5,
        "w_fact": 0.5,
    },
}

def load_profile(name: str) -> DirectorConfig:
    if name in INTERNAL_PROFILES:
        return DirectorConfig(**INTERNAL_PROFILES[name], profile=name)
    return DirectorConfig.from_profile(name)
```
