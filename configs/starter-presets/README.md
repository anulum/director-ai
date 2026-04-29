# Starter Presets

These YAML files are opinionated starting configurations for common
Director-AI deployments. Load one directly with:

```python
from director_ai import DirectorConfig

config = DirectorConfig.from_yaml("configs/starter-presets/rag_qa.yaml")
```

Then tune against local labelled data before strict enforcement:

```bash
director-ai tune --dataset my_eval.jsonl --profile rag_qa --output rag_qa_tuned.yaml
```

For YAML-only preset names such as `rag_qa` or `edge_offline`, omit
`--profile` and merge the output into the selected preset after review.

| File | Workload | Starting stance |
|------|----------|-----------------|
| `customer_support.yaml` | Policy and troubleshooting assistants | latency-first, optional retrieval, injection checks on |
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

Presets with grounded mode assume a populated vector store. They intentionally
do not set production enforcement, auth key lists, cloud endpoints, or sensitive
values. Enable those in an ignored deployment override after local validation.
