# Glossary

Key terms used throughout Director-AI documentation.

---

**Balanced Accuracy (BA)**
:   Macro-averaged recall across supported/not-supported classes. The standard metric for the [LLM-AggreFact](https://llm-aggrefact.github.io/) leaderboard. See [Benchmarks](benchmarks.md).

**Bidirectional NLI**
:   Scoring the premise→hypothesis pair *and* the reverse (hypothesis→premise), then combining scores. Reduces false positives on paraphrases. See [Threshold Tuning](guide/threshold-tuning.md).

**Coherence Score**
:   The final 0–1 output of `CoherenceScorer.review()`. Computed as `1 - (w_logic * H_logical + w_factual * H_factual)`. Higher means more coherent. See [Scoring](guide/scoring.md).

**Contradiction**
:   An NLI label indicating the hypothesis *negates* the premise. Director-AI treats high contradiction probability as evidence of hallucination. See [NLI Backends](api/nli.md).

**DeBERTa**
:   The transformer architecture (He et al., 2021) used by Director-AI's default NLI model. DeBERTa-v3-Large has 0.4B parameters. See [Scoring](guide/scoring.md).

**DirectorConfig**
:   Centralized configuration object for threshold, backend, cache, and device settings. Serializable to YAML/JSON. See [Configuration](guide/config.md).

**Dual-Entropy Scoring**
:   Director-AI's approach of combining two independent entropy signals — H_logical (NLI) and H_factual (RAG retrieval) — into a single coherence score.

**Entailment**
:   An NLI label indicating the hypothesis *follows from* the premise. High entailment probability means the LLM response is consistent with ground truth.

**Evidence Chunk**
:   A specific passage from the knowledge base returned with every rejection. Tells the user *why* the response was flagged. See [Evidence & Fallback](guide/evidence.md).

**FactCG**
:   The fine-tuned DeBERTa-v3-Large model from Li et al. (NAACL 2025) that Director-AI uses as its default NLI backend. 77.2% BA (paper) / 75.86% BA (our eval). See [Benchmarks](benchmarks.md).

**False-Halt Rate**
:   Percentage of *correct* responses incorrectly halted by the streaming kernel. 0.0% on Wikipedia passages in heuristic mode. See [Benchmarks — Streaming False-Halt](benchmarks.md#streaming-false-halt).

**False-Positive Rate (FPR)**
:   Percentage of *correct* premise-hypothesis pairs incorrectly flagged as incoherent. See [Benchmarks — False-Positive Rate](benchmarks.md#false-positive-rate).

**Ground Truth Store**
:   The knowledge base that `CoherenceScorer` checks LLM responses against. Implementations: `GroundTruthStore` (dict-based), `VectorGroundTruthStore` (vector DB). See [KB Ingestion](guide/kb-ingestion.md).

**Guard**
:   The top-level `guard()` function that wraps an LLM SDK client (OpenAI, Anthropic, etc.) with automatic coherence checking. See [Quickstart](quickstart.md).

**H_factual**
:   The retrieval-based entropy component. Measures how well the LLM response is supported by retrieved KB chunks. Range 0–1 (0 = fully supported).

**H_logical**
:   The NLI-based entropy component. Measures logical contradiction between the LLM response and ground truth. Range 0–1 (0 = no contradiction).

**Hallucination**
:   An LLM output that contradicts the provided ground truth or makes unsupported factual claims. Director-AI detects this via NLI + RAG, not via content moderation.

**Hard Limit**
:   The `threshold` value. Responses scoring below this are rejected outright. See [Threshold Tuning](guide/threshold-tuning.md).

**Heuristic Scoring**
:   The fallback scoring mode (no NLI model loaded). Uses text overlap, entity matching, and n-gram similarity. Fast (<0.2 ms) but less accurate than NLI. See [Scoring](guide/scoring.md).

**Hybrid Judge**
:   A mode where NLI scoring is combined with an LLM-as-judge (GPT-4o-mini, Claude Sonnet, or local DeBERTa classifier) for higher catch rates. See [Benchmarks](benchmarks.md).

**Knowledge Base (KB)**
:   Synonym for Ground Truth Store (see above). See [KB Ingestion](guide/kb-ingestion.md).

**LLM-AggreFact**
:   A 29,320-sample benchmark for factual consistency evaluation (Tang et al., 2024). Director-AI's primary accuracy benchmark. See [Benchmarks](benchmarks.md).

**MiniCheck**
:   An alternative NLI model family (Tang et al., EMNLP 2024). Supported as a backend via `scorer_backend="minicheck"`. See [NLI Backends](api/nli.md).

**NLI (Natural Language Inference)**
:   The task of determining whether a hypothesis is *entailed*, *contradicted*, or *neutral* given a premise. Director-AI uses NLI to detect factual inconsistency. See [NLI Backends](api/nli.md).

**ONNX**
:   Open Neural Network Exchange format. Director-AI exports DeBERTa to ONNX for faster inference via `OnnxBackend`. See [ONNX Export](guide/onnx-export.md).

**Premise**
:   In NLI, the known-true statement (from your KB). The LLM response is the *hypothesis* tested against it.

**Hypothesis**
:   In NLI, the statement being evaluated (the LLM's response). Tested against the *premise* (your KB facts).

**RAG (Retrieval-Augmented Generation)**
:   A pattern where an LLM's response is grounded in retrieved documents. Director-AI scores the *output* of RAG pipelines, not the retrieval itself.

**Reranker**
:   A cross-encoder model that re-scores retrieved chunks for relevance before they reach the scorer. See [Vector Store API](api/vector-store.md).

**Scorer Backend**
:   The engine that computes NLI scores. Options: `deberta` (default), `onnx`, `minicheck`, `lite` (heuristic), `rust` (PyO3 FFI). See [NLI Backends](api/nli.md).

**Sliding Window**
:   In streaming mode, the scorer evaluates the most recent N tokens rather than the full response. Keeps latency constant as responses grow. See [Streaming Halt](guide/streaming.md).

**Soft Limit**
:   The `soft_limit` threshold. Scores between `threshold` and `soft_limit` trigger a warning but don't reject. See [Threshold Tuning](guide/threshold-tuning.md).

**Streaming Halt**
:   Director-AI's signature feature: stopping LLM token generation mid-stream when coherence drops below threshold. See [Streaming Halt](guide/streaming.md).

**Streaming Kernel**
:   The `StreamingKernel` class that wraps a token stream and injects coherence checks every N tokens. See [API — StreamingKernel](api/streaming.md).

**Threshold**
:   The coherence score cutoff (0–1). Responses below this are rejected. Domain-dependent: 0.55 for support, 0.6 default, 0.7+ for medical/legal. See [Threshold Tuning](guide/threshold-tuning.md).

**Token-Level Scoring**
:   Evaluating coherence incrementally as each token (or batch of tokens) arrives, rather than waiting for the complete response.

**Trend Detection**
:   In streaming mode, detecting a *downward trend* in coherence scores across consecutive windows — an early warning before the score crosses the threshold.

**Vector Backend**
:   The storage engine for `VectorGroundTruthStore`. Options: `memory` (in-process), `chroma`, `faiss`, `qdrant`, `pinecone`, `weaviate`, `elasticsearch`. See [Vector Store API](api/vector-store.md).
