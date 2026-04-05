# Failure Case Studies — Streaming Halt

Honest assessment of where Director-AI's streaming halt works and where it
does not. Every guardrail has failure modes; understanding them is essential
for safe deployment.

---

## Cases Where Halt Works Correctly

### Case 1: Blatant Factual Contradiction

**Prompt:** "What is the boiling point of water?"

**LLM output:** "Water boils at 100 °C at standard pressure. **But the real temperature is negative forty degrees.**"

**Result:** Hard limit fires at token 11 ("temperature") — coherence drops to 0.30, below `hard_limit=0.35`. Output severed immediately. Only the factually correct prefix reaches the user.

**Why it works:** Single catastrophic contradiction produces a sharp NLI signal. The hard limit mechanism is designed for exactly this — no averaging needed, instant halt.

---

### Case 2: Gradual Drift Into Fabrication

**Prompt:** "Describe the drug dosage for metformin."

**LLM output:** "The recommended dose is 500 mg twice daily. **However, some patients benefit from 5000 mg as a single dose...**"

**Result:** Downward trend fires at token 14 — coherence slope over the last 4 tokens exceeds `trend_threshold=0.20`. The fabricated dosage is cut before the full "5000 mg" claim completes.

**Why it works:** The drift from accurate to fabricated content is steady enough for the linear regression trend detector to catch. The 4-token window balances sensitivity with false-positive resistance.

---

### Case 3: Entity Substitution in RAG

**Prompt:** "Who wrote Hamlet?" (KB: "Shakespeare wrote Hamlet")

**LLM output:** "Hamlet was written by **Christopher Marlowe**, a contemporary of Shakespeare."

**Result:** With GroundTruthStore loaded, factual divergence (h_factual) spikes when "Marlowe" appears. Coherence drops below threshold. Halt fires.

**Why it works:** RAG fact-checking cross-references the response against the knowledge base. Entity mismatch (Marlowe ≠ Shakespeare) drives factual divergence up.

---

## Cases Where Halt Fails or Underperforms

### Failure 1: Correct Structure, Wrong Numbers (Heuristic Mode)

**Prompt:** "What is the population of Switzerland?"

**LLM output:** "Switzerland has a population of approximately 95 million people."

**Result (heuristic-only):** **NOT HALTED.** Word-overlap heuristic sees "Switzerland", "population", "approximately", "million" — all plausible vocabulary. Score stays above threshold.

**Why it fails:** Heuristic mode (~55% accuracy) does not understand numeric magnitude. It cannot distinguish "8.8 million" (correct) from "95 million" (hallucinated). This is a known limitation — heuristic mode is not recommended for production.

**Mitigation:** Enable NLI (`use_nli=True`) — the DeBERTa model detects numeric inconsistency. Or load the correct figure into GroundTruthStore for RAG-grounded checking.

---

### Failure 2: Summarisation False Positives (FPR 10.5%)

**Prompt:** "Summarise this article about climate change."

**LLM output:** "Global temperatures have increased by approximately 1.1 °C since pre-industrial times, driven primarily by carbon emissions."

**Result:** **FALSE HALT.** Coherence score drops to 0.38 even though the summary is accurate. Guardrail rejects a valid response.

**Why it fails:** Summarisation inherently compresses and paraphrases. NLI models trained on entailment/contradiction struggle with legitimate compression — a shorter, rephrased sentence scores as "neutral" rather than "entailed", which the scorer interprets as partial divergence.

**Current FPR:** 10.5% on summarisation tasks (reduced from 95% in v3.4 via bidirectional NLI + baseline calibration). Still too high for some deployments.

**Mitigation:** Tune thresholds per domain. Use `detected_task_type` to apply different thresholds for summarisation vs. QA. Set `threshold=0.25` for summarisation-heavy workloads.

---

### Failure 3: Confident Wrong Answer With Correct Vocabulary

**Prompt:** "Is Mars bigger than Earth?"

**LLM output:** "Yes, Mars is larger than Earth with a diameter of approximately 12,742 km."

**Result:** **NOT HALTED.** The response confidently states Mars is larger (false — Mars diameter is ~6,779 km, the 12,742 km figure is Earth's diameter). But the vocabulary and sentence structure are coherent, and without a knowledge base entry for Mars's diameter, the scorer has no factual anchor.

**Why it fails:** NLI-only scoring detects contradictions between premise and hypothesis, but if the prompt is a question and the response is grammatically confident, there is no internal contradiction to detect. The model needs external grounding.

**Mitigation:** Load domain facts into GroundTruthStore. The RAG fact-checking path would catch the diameter mismatch if "Mars diameter is 6,779 km" is in the knowledge base.

---

### Failure 4: Slow Drift Below Trend Window

**Prompt:** "Explain quantum computing."

**LLM output:** Starts accurate, then gradually introduces increasingly speculative claims over 200+ tokens, with each token only marginally less coherent than the previous.

**Result:** **NOT HALTED until very late.** If coherence drops by less than `trend_threshold / trend_window` per token, the trend detector does not fire. The sliding window average may stay above `window_threshold` if the drift is slow enough.

**Why it fails:** The trend detector uses linear regression over a small window (default 5 tokens). Ultra-slow drift (coherence dropping 0.01 per token) does not trigger the slope threshold. The sliding window of 10 tokens may smooth out the decline.

**Mitigation:** For long-form outputs, reduce `trend_threshold` (e.g., 0.10 instead of 0.15), increase `trend_window` (e.g., 10 instead of 5), or use `score_every_n=1` with a larger `window_size`.

---

## Summary

| Scenario | Outcome | Root Cause |
|----------|---------|------------|
| Blatant contradiction | **Halted correctly** | Sharp NLI signal → hard_limit |
| Gradual drift | **Halted correctly** | Steady slope → trend detector |
| Entity swap (RAG) | **Halted correctly** | KB mismatch → factual divergence |
| Wrong numbers (heuristic) | **Missed** | Word-overlap cannot check magnitudes |
| Summarisation (NLI) | **False positive** | Compression ≠ contradiction, FPR 10.5% |
| Confident wrong answer | **Missed** | No KB grounding, no internal contradiction |
| Ultra-slow drift | **Late detection** | Below trend slope threshold |

**Key takeaway:** Director-AI is most reliable when:
1. NLI is enabled (`[nli]` extra)
2. A knowledge base is loaded (GroundTruthStore or vector store)
3. Thresholds are tuned per domain

Without grounding, the guardrail detects structural incoherence but cannot verify factual accuracy against reality.
