# Troubleshooting

## Import Errors

**`ImportError: No module named 'transformers'`**

NLI models require PyTorch + Transformers:

```bash
pip install director-ai[nli]
```

**`ImportError: No module named 'onnxruntime'`**

ONNX backend requires ONNX Runtime:

```bash
pip install director-ai[onnx]
```

**`ImportError: No module named 'chromadb'`**

Vector store with ChromaDB:

```bash
pip install director-ai[vector]
```

## Validation Errors (v2.2.0+)

**`ValueError: threshold must be in [0, 1]`**

The `threshold` parameter is out of range. Pass a float between 0.0 and 1.0:

```python
scorer = CoherenceScorer(threshold=0.6)  # valid
```

**`ValueError: w_logic + w_fact must equal 1.0`**

Divergence weights must sum to exactly 1.0:

```python
scorer = CoherenceScorer(w_logic=0.6, w_fact=0.4)  # valid
```

**`ValueError: score_every_n must be >= 1`**

Scoring cadence must be a positive integer:

```python
kernel = StreamingKernel(hard_limit=0.4, score_every_n=4)  # valid
```

## Scoring Issues

**Score always 0.5**

No NLI model loaded — heuristic fallback is active. Install the NLI extras and enable it:

```python
pip install director-ai[nli]
scorer = CoherenceScorer(threshold=0.6, use_nli=True)
```

Or use `strict_mode=True` to make 0.5 explicit (neutral, no heuristic guessing).

**Score always 1.0**

Empty or trivial prompt/response pair. The scorer returns perfect coherence when there is nothing to contradict.

**Streaming halts too aggressively**

Lower `hard_limit` or increase `window_size` to smooth out transient dips:

```python
kernel = StreamingKernel(hard_limit=0.35, window_size=15)
```

**Streaming never halts**

Raise `hard_limit`. Verify your `coherence_callback` returns a float in [0, 1]:

```python
def cb(token):
    score = my_scorer.review(prompt, accumulated_text)[1].score
    return score  # must be float in [0, 1]
```

## Performance

**Slow first review**

NLI model loading takes 2-5 seconds on first call. Mitigations:

- Use `cache_size > 0` to cache repeated prompt/response pairs
- Use `scorer_backend="onnx"` for faster cold-start (~1s vs ~3s)
- Pre-warm the scorer at startup: `scorer.review("warmup", "warmup")`

**High memory usage**

DeBERTa-v3-Large uses ~1.5 GB GPU / ~2 GB CPU. Options:

- Switch to ONNX: `scorer_backend="onnx"` (~400 MB)
- Use 8-bit quantization: `pip install director-ai[quantize]`

**Streaming throughput**

Scoring every token is expensive. Use cadence control:

```python
kernel = StreamingKernel(hard_limit=0.4, score_every_n=4)
# Or adaptive cadence:
kernel = StreamingKernel(hard_limit=0.4, adaptive=True, max_cadence=8)
```

See [Streaming Overhead](streaming-overhead.md) for tokens/sec benchmarks by cadence.

## Server Issues

**429 Too Many Requests**

`rate_limit_rpm` exceeded. Increase the limit or disable rate limiting:

```yaml
rate_limit_rpm: 0  # disabled
```

**403 Forbidden**

Missing or invalid `X-API-Key` header. Check `api_keys` in your config:

```yaml
api_keys:
  - "your-api-key-here"
```

Pass the key in requests:

```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8080/review
```

**Metrics endpoint empty**

Metrics collection is disabled. Enable it in config:

```yaml
metrics_enabled: true
```

Then access `/metrics` for Prometheus-format output.
