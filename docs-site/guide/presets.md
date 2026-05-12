# Domain Presets

`DirectorConfig.from_profile(name)` loads a preset parameter set for common use cases.
These are starting points, not a substitute for calibration on your own
evaluation set. Domain profiles should be treated as a starting config until
`director-ai tune` has measured the expected false-halt and miss rates.

## Profile Reference

Each built-in profile has runtime metadata available through
`DirectorConfig.profile_metadata(name)`: intended workload, validation status,
expected false-halt risk, required dependency extras, and whether labelled
calibration is required before strict deployment.

| Profile | Threshold | Hard Limit | Soft Limit | NLI | Reranker | W_Logic | W_Fact |
|---------|-----------|------------|------------|-----|----------|---------|--------|
| `fast` | 0.50 | default | default | no | no | default | default |
| `lite` | 0.50 | default | default | no | no | default | default |
| `rules` | 0.50 | default | default | no | no | default | default |
| `embed` | 0.60 | default | default | no | no | default | default |
| `thorough` | 0.60 | default | default | yes | no | default | default |
| `research` | 0.70 | default | default | yes | no | default | default |
| `medical` | 0.30 | 0.20 | 0.35 | yes | yes | 0.5 | 0.5 |
| `finance` | 0.30 | 0.20 | 0.35 | yes | yes | 0.4 | 0.6 |
| `legal` | 0.30 | 0.20 | 0.35 | yes | no | 0.6 | 0.4 |
| `creative` | 0.40 | 0.30 | 0.45 | no | no | 0.7 | 0.3 |
| `customer_support` | 0.55 | 0.40 | 0.60 | no | no | 0.5 | 0.5 |
| `summarization` | 0.15 | 0.08 | 0.25 | yes | no | 0.0 | 1.0 |

"default" means the field inherits the `DirectorConfig` dataclass default (hard_limit=0.5, soft_limit=0.6, w_logic/w_fact=0.0 which defers to `CoherenceScorer` class defaults).

## Profile Metadata

| Profile | Intended Workload | Validation Status | False-Halt Risk | Required Extras |
|---------|-------------------|-------------------|-----------------|-----------------|
| `fast` | Development loops and heuristic screening | smoke-tested heuristic baseline | low for obvious checks, unknown for factual QA | none |
| `lite` | Offline approximate local scoring | smoke-tested lite scorer baseline | medium without calibration | none |
| `rules` | Deterministic local checks | deterministic baseline | low for exact rules, high for semantic hallucinations | none |
| `embed` | Semantic similarity screening | benchmarked approximate scorer | medium; tune per corpus | `embed` |
| `thorough` | General production baseline | standard validated baseline | medium until tuned | `nli` |
| `research` | Academic precision-heavy review | experimental high-threshold baseline | high by design | `nli` |
| `medical` | Biomedical fact-heavy review with curated KB | limited PubMedQA validation; requires KB grounding | very high without KB and calibration | `nli`, `vector` |
| `finance` | Financial and regulatory KB review | limited FinanceBench validation; requires recalibration | very high without KB and calibration | `nli`, `vector` |
| `legal` | Legal reasoning over curated KBs | not independently validated | unknown; treat as high until tuned | `nli` |
| `creative` | Drafting, fiction, and non-factual generation | heuristic permissive preset | low for creative drift, high for factual safety | none |
| `customer_support` | Policy bots and troubleshooting assistants | latency-first starter preset | medium; depends on policy KB coverage | none |
| `summarization` | Source-grounded summaries | validated with summarization FPR diagnostics | low after claim coverage; tune per corpus | `nli` |

High-stakes profiles (`medical`, `finance`, `legal`) and source-grounded
`summarization` expose `calibration_required=True`,
`min_calibration_samples`, and a `director-ai tune --profile ...` command in
metadata. Treat the built-in threshold as an onboarding default until labelled
local traces have measured false-halt and miss rates for the deployment corpus.

## Starter YAML Presets

The built-in profiles above are compact runtime defaults. The repository also
ships fuller starter YAML files in `configs/starter-presets/` for teams that
want a ready-to-edit deployment config:

| File | Workload | Starting stance |
|------|----------|-----------------|
| `customer_support.yaml` | Policy and troubleshooting assistants | latency-first, injection checks on, retrieval disabled by default |
| `summarization.yaml` | Source-grounded summaries | fact-only NLI, prompt-as-premise, claim coverage |
| `rag_qa.yaml` | Retrieval-grounded QA | grounded mode, reranker, HyDE, decomposition, compression |
| `finance.yaml` | Numeric and regulatory claims | high-stakes grounded mode, audit path, PII redaction |
| `legal.yaml` | Legal drafting and review | logic-weighted grounded mode, audit path, PII redaction |
| `medical.yaml` | Biomedical or clinical fact review | high-stakes grounded mode, stricter claim support |
| `creative_drafting.yaml` | Fiction and exploratory drafting | permissive lite scoring with basic injection checks |
| `edge_offline.yaml` | Offline or constrained edge runtime | rules backend, no vector or heavyweight model path |
| `stem_fact_heavy.yaml` | Scientific and technical fact workflows | grounded mode, stronger claim support, parent-child retrieval |
| `code_generation.yaml` | Code and tool-output review | logic-weighted hybrid scoring, retrieval disabled by default |
| `multi_agent_swarm.yaml` | Multi-agent supervision | review queue batching, retrieval routing, trace-friendly logging |
| `voice_agents.yaml` | Real-time dialogue and voice agents | lite scoring, dialogue thresholds, low-latency defaults |
| `high_stakes_medical_review.yaml` | Clinical review workflows | strict grounded review, higher retrieval and claim-support gates |

Load a starter preset directly when you want the full YAML surface:

```python
from director_ai import DirectorConfig

config = DirectorConfig.from_yaml("configs/starter-presets/rag_qa.yaml")
```

Grounded presets assume a populated vector store. They intentionally omit
production enforcement, auth key lists, cloud endpoints, and sensitive values;
add those in an ignored deployment overlay after local validation.

## Profile Rationale

**fast** — Heuristic scoring only, no model loading. Sub-millisecond latency for dev loops and high-throughput pipelines where approximate filtering is acceptable.

**lite** — Lite scorer backend with no heavyweight NLI dependency. Use for
offline trials and latency-sensitive routing where approximate scores are
acceptable.

**rules** — Rules-only scorer. Use when deployments need deterministic local
checks and no model downloads.

**embed** — Embedding scorer backend. Use when semantic similarity is the
primary signal and a full NLI model is not available.

**thorough** — Adds NLI inference (FactCG-DeBERTa) to catch logical contradictions that heuristics miss. Standard production baseline.

**research** — Higher threshold (0.70) for academic and analytical workloads where factual precision matters more than recall.

**medical** — Equal logic/fact weighting reflects the need for both clinical reasoning and factual accuracy. Reranker enabled for precise KB retrieval. NLI-only eval on PubMedQA (1000 samples, 2026-03-20): F1=61.9% at t=0.30, but FPR=100% (all responses flagged). **KB grounding or customer-specific calibration required for usable precision.** Scores without KB cluster 0.25-0.35.

**finance** — Fact-weighted (0.6) because numerical claims and regulatory data dominate. Reranker sharpens retrieval against financial KB documents. NLI-only eval on FinanceBench (150 clean samples, 2026-03-20): FPR=100%, precision=0% — all clean responses were flagged. **These thresholds need KB grounding or recalibration before production use.**

**legal** — Logic-weighted (0.6) because legal reasoning chains (statute + precedent + application) matter more than isolated facts. No reranker; legal KBs tend to be smaller and well-structured. **Not validated** — CUAD benchmark OOM on 6GB VRAM. No domain-specific artefact exists.

**creative** — Permissive thresholds (0.40/0.30/0.45) allow divergent generation. NLI disabled to avoid penalising metaphor and fiction. Logic-weighted (0.7) because internal narrative consistency matters more than factual grounding.

**customer_support** — Moderate thresholds balance helpfulness with accuracy. NLI disabled for latency (support bots need fast responses). Equal weights suit mixed queries (policy facts + troubleshooting logic).

**summarization** — Fact-only weighting with prompt-as-premise scoring,
trimmed-mean aggregation, and claim coverage enabled. Use for source-grounded
summaries, then tune on your own clean and adversarial samples.

## Usage

```python
from director_ai import DirectorConfig

config = DirectorConfig.from_profile("medical")
```

Load via CLI:

```bash
director-ai quickstart --profile medical
```

Generate the Docker Compose quickstart and start it immediately:

```bash
director-ai quickstart --profile customer_support --run
```

Tune against a labelled evaluation set before production:

```bash
director-ai tune --dataset my_eval.jsonl --profile medical --output medical_tuned.yaml
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
