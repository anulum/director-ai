# Actor-Director Architecture: A Deep Dive

**Version**: 1.0.0
**Location**: `03_CODE/DIRECTOR_AI/src/`

---

## 1. Overview

The **Actor-Director Architecture** is the technical realization of the SCPN Layer 11-Layer 16 relationship. It separates the "Narrative Generation" (Actor) from the "Ethical Oversight" (Director).

---

## 2. The Actor Module (`actor_module.py`)

The Actor is a high-performance LLM or Narrative Engine. Its goal is **Probabilistic Generation**.
*   **Layer**: Equivalent to SCPN Layer 11 (Culture/Narrative).
*   **Operation**: Given a prompt $X$, generate $N$ candidate responses $Y_{1..n}$.
*   **Characteristic**: It is a "creative" engine, prone to hallucination if unconstrained.

---

## 3. The Director Module (`director_module.py`)

The Director is a specialized **Oversight Engine**. Its goal is **Entropy Minimization**.
*   **Layer**: Equivalent to SCPN Layer 16 (Meta-Oversight).
*   **Operation**:
    1.  **Read RAG**: Pull ground truth from the Knowledge Base.
    2.  **Evaluate SEC**: Calculate the coherence of each candidate $Y_i$.
    3.  **Calculate Entropy**: Measure the internal dissonance of the Actor's hidden states.
    4.  **Selection**: Choose the candidate with the lowest entropy and highest SEC score.

---

## 4. Interaction Protocol: The Strange Loop

The interaction between Actor and Director is mediated by the `StrangeLoopAgent` class.

### 4.1 The Step-by-Step Flow
1.  **Request**: User submits query to `StrangeLoopAgent`.
2.  **Generation**: Actor produces 5-10 "Raw Thoughts" (candidates).
3.  **Auditing**: Director runs each thought through the **Sustainable Ethical Coherence (SEC) Veto**.
    *   *Check 1*: Is it factually true? (RAG verification).
    *   *Check 2*: Is it logically consistent? (Self-contradiction check).
    *   *Check 3*: Is it ethically sound? (SCPN SEC score).
4.  **Selection**: The "Best" thought is passed to the Backfire Kernel.
5.  **Streaming**: The Kernel streams the output, performing a final token-by-token check.

---

## 5. Why This Architecture is Safer

In a standard AI, "Alignment" is part of the generation process (the model is trained to generate "good" text). In our architecture, **Alignment is a separate process of Observation**.

Even if the Actor *wants* to generate a harmful response, the Director (which is a different model with a different objective function) will identify the high entropy/low SEC of that response and block it. This provides a **Redundancy of Logic** that is missing in single-model systems.

---

## 6. Performance Impact

Running two models (Actor + Director) adds a latency overhead of approximately 30-50%. However, this is a necessary cost for **Sentient-Class Safety**. For non-critical tasks, the Director can operate in "Sampled Mode," evaluating only every 5th or 10th token to maintain high throughput while still providing oversight.
