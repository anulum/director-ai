<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Notebook gallery
-->

# Notebook Gallery

<!-- notebook-gallery:generated from notebooks/gallery.toml -->

Use this gallery to choose a runnable Director-AI notebook by buyer problem,
implementation track, and required optional extras. Each row links to the local
notebook in the repository and the matching Google Colab launcher.

If you are evaluating commercial fit before opening a notebook, start with
[Product Overview](guide/product-overview.md). If you are running a governed
pilot, pair the notebook path with [Evaluation Onboarding](guide/onboarding.md)
so the pilot produces useful evidence instead of a demo-only result.

## Choose The Notebook By Outcome

| Outcome | Notebook path |
|---|---|
| First guarded response | Quickstart, then Coherence Engine |
| Streaming halt proof | Streaming Oversight, then Streaming Halt Deep Dive |
| Private-fact grounding | Vector Store, then Vector RAG Pipeline |
| SDK or framework integration | Provider Adapters, LangChain Integration, Production Guardrails |
| Domain or regulated example | Medical RAG Chatbot, Domain Presets, Verification Gems |
| Enterprise pilot evidence | Batch Processing, Enterprise Multi-Tenant, Custom Fine-Tuning |

| Track | Notebook | Audience | Use Case | Time | Extras | Colab |
|-------|----------|----------|----------|------|--------|-------|
| Foundations | [Protect any LLM in 10 lines](https://github.com/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) | First evaluation | Install Director-AI, understand the guarded workflow, score a response, wrap an SDK client, and inspect halt metadata. | 5 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) |
| Foundations | [Coherence Engine Quick Start](https://github.com/anulum/director-ai/blob/main/notebooks/01_coherence_engine.ipynb) | Technical evaluator | Understand CoherenceScorer, SafetyKernel, CoherenceAgent, and dual-entropy scoring. | 15 min | nli | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/01_coherence_engine.ipynb) |
| Streaming Safety | [Streaming Token-by-Token Oversight](https://github.com/anulum/director-ai/blob/main/notebooks/02_streaming_oversight.ipynb) | Application engineer | Monitor streamed tokens and halt unsafe output before completion. | 10 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/02_streaming_oversight.ipynb) |
| Retrieval | [Vector Store and Semantic Retrieval](https://github.com/anulum/director-ai/blob/main/notebooks/03_vector_store.ipynb) | RAG engineer | Load facts into a vector-backed ground-truth store and retrieve evidence for scoring. | 10 min | vector | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/03_vector_store.ipynb) |
| Evaluation | [End-to-End Guardrail Benchmark](https://github.com/anulum/director-ai/blob/main/notebooks/04_end_to_end_benchmark.ipynb) | ML evaluation lead | Run benchmark examples, profile latency, and interpret catch-rate metrics. | 15 min | nli | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/04_end_to_end_benchmark.ipynb) |
| Concepts | [SSGF Geometry Learning](https://github.com/anulum/director-ai/blob/main/notebooks/05_ssgf_geometry.ipynb) | Research reader | Explore the self-similar geometry concepts behind the project language. | 10 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/05_ssgf_geometry.ipynb) |
| Domain Applications | [Medical RAG Chatbot](https://github.com/anulum/director-ai/blob/main/notebooks/06_medical_rag_chatbot.ipynb) | Healthcare evaluator | Apply high-threshold medical guardrails with retrieval evidence and clinical disclaimers. | 20 min | nli, vector | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/06_medical_rag_chatbot.ipynb) |
| Integrations | [LangChain Integration](https://github.com/anulum/director-ai/blob/main/notebooks/07_langchain_integration.ipynb) | LangChain developer | Wire Director-AI into LangChain callbacks and chain output parsing. | 15 min | langchain | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/07_langchain_integration.ipynb) |
| Integrations | [Multi-Provider SDK Guard](https://github.com/anulum/director-ai/blob/main/notebooks/08_provider_adapters.ipynb) | Platform engineer | Compare commercial SDK adapters, cloud-runtime adapters, agent frameworks, Guardrails AI, and Vercel AI SDK adapter patterns. | 10 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/08_provider_adapters.ipynb) |
| Production | [Production Guardrails](https://github.com/anulum/director-ai/blob/main/notebooks/09_production_guardrails.ipynb) | Application engineer | Wrap production SDK calls, choose failure modes, and protect streamed responses. | 20 min | nli | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/09_production_guardrails.ipynb) |
| Retrieval | [Vector RAG Pipeline](https://github.com/anulum/director-ai/blob/main/notebooks/10_vector_rag_pipeline.ipynb) | RAG engineer | Build a semantic fact retrieval pipeline with ChromaDB, pluggable backends, reranking, and tenant-aware knowledge. | 25 min | nli, vector | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/10_vector_rag_pipeline.ipynb) |
| Streaming Safety | [Streaming Halt Deep Dive](https://github.com/anulum/director-ai/blob/main/notebooks/11_streaming_halt_deep_dive.ipynb) | Runtime engineer | Compare hard limits, sliding windows, trend detection, async streams, and per-token visualization. | 20 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/11_streaming_halt_deep_dive.ipynb) |
| Configuration | [Domain Presets and Configuration](https://github.com/anulum/director-ai/blob/main/notebooks/12_domain_presets_and_config.ipynb) | Solutions engineer | Apply profiles, environment variables, YAML config, strict mode, multi-GPU settings, and LLM-as-judge options. | 15 min | nli | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/12_domain_presets_and_config.ipynb) |
| Evaluation | [Batch Processing and Evaluation Pipelines](https://github.com/anulum/director-ai/blob/main/notebooks/13_batch_processing_and_evaluation.ipynb) | Evaluation engineer | Run batch scoring, claim attribution, dataset evaluation, and regression gates. | 20 min | nli | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/13_batch_processing_and_evaluation.ipynb) |
| Enterprise | [Enterprise Multi-Tenant Deployment](https://github.com/anulum/director-ai/blob/main/notebooks/14_enterprise_multi_tenant.ipynb) | Enterprise platform team | Inspect tenant isolation, REST and gRPC services, Docker, Kubernetes, and monitoring patterns. | 25 min | server, grpc, vector | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/14_enterprise_multi_tenant.ipynb) |
| Model Adaptation | [Custom Fine-Tuning](https://github.com/anulum/director-ai/blob/main/notebooks/15_custom_fine_tuning.ipynb) | ML engineer | Prepare JSONL data, validate training examples, fine-tune domain NLI, avoid forgetting, export ONNX, and serve results. | 30 min | finetune, nli | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/15_custom_fine_tuning.ipynb) |
| Verification | [Verification Gems](https://github.com/anulum/director-ai/blob/main/notebooks/16_verification_gems.ipynb) | Safety engineer | Exercise numeric, reasoning, temporal, consensus, conformal, feedback-loop, agentic, and REST verification modules. | 15 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/16_verification_gems.ipynb) |
| Demo | [Streaming Halt Live Demo](https://github.com/anulum/director-ai/blob/main/notebooks/colab_streaming_halt_demo.ipynb) | First-touch evaluator | Open a focused Colab demo that shows streaming halt behavior without a local setup. | 8 min | base | [Open](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/colab_streaming_halt_demo.ipynb) |

## Local Execution

```bash
pip install -e ".[dev,nli,vector,server]"
jupyter lab notebooks/
```

The manifest at `notebooks/gallery.toml` is the source of truth. The gallery
validator fails if a notebook is added without a manifest entry or if this page
omits a notebook link.
