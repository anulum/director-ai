# Operational Runbooks

Decision-tree troubleshooting for production Director-AI deployments.

---

## 1. NLI Model Fails to Load

**Symptom:** `CoherenceScorer` raises `RuntimeError` or `OSError` on init. Scoring falls back to heuristic mode.

```
Step 1 — Check VRAM
├─ nvidia-smi shows < 2 GB free
│  └─ Fix: kill competing GPU processes, or switch to ONNX CPU
│     pip install director-ai[onnx]
│     scorer = CoherenceScorer(scorer_backend="onnx", nli_device="cpu")
│
├─ nvidia-smi shows sufficient VRAM
│  └─ Step 2 — Check disk space
│     ├─ < 3 GB free in HF cache (~/.cache/huggingface/)
│     │  └─ Fix: clear old models
│     │     huggingface-cli delete-cache
│     │
│     └─ Disk space OK
│        └─ Step 3 — Check model download
│           ├─ Behind corporate proxy / air-gapped
│           │  └─ Fix: pre-download model
│           │     huggingface-cli download yaxili96/FactCG-DeBERTa-v3-Large
│           │     # Or copy model dir to HF_HOME
│           │
│           └─ Network OK → file a bug report
```

**Fallback:** Director-AI automatically degrades to heuristic scoring if NLI load fails. Check logs for `"Falling back to heuristic scorer"`.

---

## 2. Scores Consistently Too Low or Too High

**Symptom:** All responses score below 0.3 (mass rejection) or above 0.9 (nothing flagged).

```
Scores too LOW (mass rejection)
├─ Is the KB populated?
│  ├─ No → scorer has no facts to match against
│  │  └─ Fix: ingest your knowledge base
│  │     store.ingest(documents)
│  │
│  └─ Yes → Step 2 — Check threshold
│     ├─ threshold too high for content type
│     │  └─ Fix: use measured profiles
│     │     Medical/legal/finance: 0.30
│     │     General: 0.50
│     │     Creative: 0.40
│     │
│     └─ Threshold reasonable → Step 3 — Check KB freshness
│        ├─ KB docs are outdated (facts changed)
│        │  └─ Fix: re-ingest updated documents
│        │
│        └─ KB current → Step 4 — Check NLI calibration
│           └─ Run threshold sweep on your data
│              python -m benchmarks.aggrefact_eval --sweep

Scores too HIGH (nothing flagged)
├─ Is NLI loaded? Check scorer.backend_name
│  ├─ Returns "lite" or "heuristic"
│  │  └─ Fix: install NLI extras
│  │     pip install director-ai[nli]
│  │
│  └─ NLI loaded → Step 2 — Check w_logic / w_factual weights
│     ├─ w_logic=0 → only RAG scoring (misses logical contradictions)
│     │  └─ Fix: restore default weights
│     │     config = DirectorConfig(w_logic=0.6, w_factual=0.4)
│     │
│     └─ Weights OK → threshold too low
│        └─ Fix: raise threshold from 0.3 → 0.5+
```

---

## 3. Latency Degradation

**Symptom:** P95 latency increases 2x+ from baseline.

```
Step 1 — Check cache hit rate
├─ Metrics endpoint: GET /v1/metrics/prometheus → director_cache_hit_ratio
│  ├─ Hit rate < 50%
│  │  └─ Fix: increase cache_size or cache_ttl
│  │     scorer = CoherenceScorer(cache_size=8192, cache_ttl=7200)
│  │
│  └─ Hit rate OK → Step 2 — Check GPU thermal
│     ├─ nvidia-smi shows GPU temp > 85°C
│     │  └─ Fix: check cooling, reduce batch concurrency
│     │
│     └─ Temp OK → Step 3 — Check batch queue
│        ├─ Using review() in a loop?
│        │  └─ Fix: switch to review_batch()
│        │     results = scorer.review_batch(pairs)  # 2.5x faster
│        │
│        └─ Already batched → Step 4 — Check model backend
│           ├─ Using PyTorch FP32?
│           │  └─ Fix: switch to ONNX GPU or FP16
│           │     scorer = CoherenceScorer(scorer_backend="onnx")
│           │
│           └─ Already ONNX → check for driver/CUDA version mismatch
│              nvidia-smi  # driver version
│              python -c "import onnxruntime; print(onnxruntime.get_device())"
```

**Baseline reference** (see [Benchmarks — Latency](../benchmarks.md#latency)):

| Backend | Expected per-pair |
|---------|------------------|
| Heuristic | < 0.2 ms |
| ONNX GPU batch | 0.9–15 ms |
| PyTorch GPU batch | 1.2–20 ms |
| ONNX CPU batch | 380+ ms |

---

## 4. High False-Positive Spike

**Symptom:** Correct responses suddenly getting rejected at a much higher rate than usual.

```
Step 1 — Did you update the KB recently?
├─ Yes → new KB entries may conflict with valid responses
│  └─ Fix: review recently added entries for overly broad claims
│
└─ No → Step 2 — Check bidirectional NLI
   ├─ bidirectional=False
   │  └─ Fix: enable bidirectional NLI
   │     scorer = CoherenceScorer(bidirectional=True)
   │     # Reduces FPR by up to 89% (v3.5 benchmarks)
   │
   └─ Already bidirectional → Step 3 — Check premise_ratio
      ├─ premise_ratio=1.0 (default) — all chunks used as premises
      │  └─ Fix: lower to 0.85 to discard noisy retrieval hits
      │     scorer = CoherenceScorer(premise_ratio=0.85)
      │
      └─ premise_ratio already tuned → Step 4 — Domain mismatch
         ├─ KB domain ≠ query domain (e.g., legal KB, product queries)
         │  └─ Fix: use domain-specific KBs or TenantRouter
         │
         └─ Domain matches → run falsepositive_eval on your data
            python -m benchmarks.falsepositive_eval --data your_data.jsonl
```

---

## 5. Streaming False-Halts

**Symptom:** `StreamingKernel` halts generation on correct responses.

```
Step 1 — Check score_every_n
├─ score_every_n < 3
│  └─ Fix: increase to 5–10
│     Scoring every 1–2 tokens catches partial words mid-sentence.
│     kernel = StreamingKernel(scorer=scorer, score_every_n=5)
│
└─ score_every_n >= 5 → Step 2 — Check window_size
   ├─ window_size < 20 tokens
   │  └─ Fix: increase to 50–100
   │     Small windows lack context for accurate NLI.
   │     kernel = StreamingKernel(scorer=scorer, window_size=50)
   │
   └─ Window OK → Step 3 — Check trend sensitivity
      ├─ trend_threshold too aggressive (< 0.05)
      │  └─ Fix: relax to 0.1
      │     kernel = StreamingKernel(scorer=scorer, trend_threshold=0.1)
      │
      └─ Trend OK → Step 4 — Is heuristic mode adequate?
         ├─ Using NLI in streaming (expensive + noisy on fragments)
         │  └─ Fix: use heuristic for streaming, NLI for final check
         │     kernel = StreamingKernel(scorer=heuristic_scorer)
         │     # Then full NLI review on complete response
         │
         └─ Already heuristic → file a bug with the halted text
```

---

## 6. Out-of-Memory on GPU

**Symptom:** `torch.cuda.OutOfMemoryError` or `CUDA error: out of memory`.

```
Step 1 — Check model + data fit in VRAM
├─ DeBERTa-v3-Large FP32: ~1.6 GB
│  DeBERTa-v3-Large FP16: ~0.8 GB
│  ONNX quantized INT8: ~0.4 GB
│
├─ GPU has < 2 GB free
│  └─ Step 2 — Reduce memory usage
│     ├─ Option A: Switch to FP16
│     │  scorer = CoherenceScorer(nli_device="cuda", dtype="float16")
│     │
│     ├─ Option B: Switch to ONNX
│     │  pip install director-ai[onnx]
│     │  scorer = CoherenceScorer(scorer_backend="onnx")
│     │
│     ├─ Option C: 8-bit quantization
│     │  pip install director-ai[quantize]
│     │  scorer = CoherenceScorer(quantize="8bit")
│     │
│     └─ Option D: CPU fallback
│        scorer = CoherenceScorer(nli_device="cpu", scorer_backend="onnx")
│
└─ GPU has sufficient free VRAM
   └─ Step 3 — Check batch size
      ├─ Sending large batches (> 32 items)
      │  └─ Fix: split into smaller batches
      │     results = scorer.review_batch(pairs[:16])
      │
      └─ Batch size OK → check for memory leaks
         ├─ Long-running server accumulating tensors
         │  └─ Fix: ensure torch.no_grad() on inference path
         │     (Director-AI does this internally — check custom code)
         │
         └─ No leak → file a bug with nvidia-smi output
```

**Memory planning guide:**

| GPU VRAM | Recommended Backend | Max Batch |
|----------|-------------------|-----------|
| 4 GB | ONNX INT8 or CPU | 8 |
| 6 GB | ONNX FP16 | 16 |
| 8 GB | PyTorch FP16 | 16 |
| 16 GB+ | PyTorch FP32 or ONNX FP16 | 32+ |
| 24 GB+ | Any | 64+ |

---

## General Diagnostics

**Check Director-AI version and backend:**

```python
import director_ai
print(director_ai.__version__)

scorer = CoherenceScorer()
print(f"Backend: {scorer.backend_name}")
print(f"Device: {scorer.device}")
print(f"NLI loaded: {scorer.nli_loaded}")
```

**Check GPU status:**

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv
```

**Run built-in health check:**

```bash
director-ai health
```

---

*For issues not covered here, check [Troubleshooting](../guide/troubleshooting.md) or open an [issue](https://github.com/anulum/director-ai/issues/new).*
