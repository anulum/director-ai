# Tutorials

Interactive Jupyter notebooks covering Director-AI from first principles to production deployment.

## Getting Started

| # | Notebook | Topics | Colab |
|---|----------|--------|-------|
| 00 | [Quickstart](https://github.com/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) | Install, score, guard, stream, presets | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/quickstart.ipynb) |
| 01 | [Coherence Engine](https://github.com/anulum/director-ai/blob/main/notebooks/01_coherence_engine.ipynb) | CoherenceScorer, SafetyKernel, CoherenceAgent | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/01_coherence_engine.ipynb) |

## Core Features

| # | Notebook | Topics | Colab |
|---|----------|--------|-------|
| 09 | [Production Guardrails](https://github.com/anulum/director-ai/blob/main/notebooks/09_production_guardrails.ipynb) | `guard()` for OpenAI/Anthropic/Bedrock/Gemini/Cohere, failure modes, streaming | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/09_production_guardrails.ipynb) |
| 10 | [Vector RAG Pipeline](https://github.com/anulum/director-ai/blob/main/notebooks/10_vector_rag_pipeline.ipynb) | Semantic retrieval, ChromaDB, pluggable backends, reranking, multi-tenant | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/10_vector_rag_pipeline.ipynb) |
| 11 | [Streaming Halt Deep Dive](https://github.com/anulum/director-ai/blob/main/notebooks/11_streaming_halt_deep_dive.ipynb) | Hard limit, sliding window, trend detection, async, visualisation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/11_streaming_halt_deep_dive.ipynb) |
| 12 | [Domain Presets & Config](https://github.com/anulum/director-ai/blob/main/notebooks/12_domain_presets_and_config.ipynb) | 8 profiles, env vars, YAML, backends, strict mode, multi-GPU, LLM-as-judge | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/12_domain_presets_and_config.ipynb) |

## Advanced

| # | Notebook | Topics | Colab |
|---|----------|--------|-------|
| 02 | [Streaming Oversight](https://github.com/anulum/director-ai/blob/main/notebooks/02_streaming_oversight.ipynb) | StreamingKernel basics | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/02_streaming_oversight.ipynb) |
| 06 | [Medical RAG Chatbot](https://github.com/anulum/director-ai/blob/main/notebooks/06_medical_rag_chatbot.ipynb) | Healthcare-specific guardrails | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/06_medical_rag_chatbot.ipynb) |
| 07 | [LangChain Integration](https://github.com/anulum/director-ai/blob/main/notebooks/07_langchain_integration.ipynb) | CoherenceCallbackHandler for LangChain | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/07_langchain_integration.ipynb) |
| 08 | [Provider Adapters](https://github.com/anulum/director-ai/blob/main/notebooks/08_provider_adapters.ipynb) | OpenAI, Anthropic, HuggingFace adapters | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/08_provider_adapters.ipynb) |
| 13 | [Batch Processing & Evaluation](https://github.com/anulum/director-ai/blob/main/notebooks/13_batch_processing_and_evaluation.ipynb) | BatchProcessor, evaluation pipelines, claim attribution, regression gates | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/13_batch_processing_and_evaluation.ipynb) |

## Enterprise & Production

| # | Notebook | Topics | Colab |
|---|----------|--------|-------|
| 04 | [End-to-End Benchmark](https://github.com/anulum/director-ai/blob/main/notebooks/04_end_to_end_benchmark.ipynb) | Full benchmark suite | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/04_end_to_end_benchmark.ipynb) |
| 14 | [Enterprise Multi-Tenant](https://github.com/anulum/director-ai/blob/main/notebooks/14_enterprise_multi_tenant.ipynb) | Tenant isolation, REST/gRPC servers, Docker, Kubernetes, monitoring | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/14_enterprise_multi_tenant.ipynb) |
| 15 | [Custom Fine-Tuning](https://github.com/anulum/director-ai/blob/main/notebooks/15_custom_fine_tuning.ipynb) | JSONL data prep, validation, training, anti-forgetting, ONNX export, REST API | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anulum/director-ai/blob/main/notebooks/15_custom_fine_tuning.ipynb) |

## Prerequisites

All notebooks run on **Python 3.11+** with `pip install director-ai`.

Notebooks requiring optional extras note this in their first cell:

- NLI scoring: `pip install director-ai[nli]`
- Vector store: `pip install director-ai[vector]`
- Fine-tuning: `pip install director-ai[finetune]`
- Server: `pip install director-ai[server]`
- gRPC: `pip install director-ai[grpc]`
