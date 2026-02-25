# Director-Class AI: Technical Specification V2
## RAG Integration and Hardware Interlock

**Date**: December 29, 2025
**Author**: Miroslav Šotek
**Status**: Validated

---

## 1. System Components

### 1.1 The Director (Software)
The Director now implements a dual-entropy calculation:
$$ H_{total} = w_1 \cdot H_{logic} + w_2 \cdot H_{fact} $$
*   **Logical Entropy ($H_{logic}$)**: Calculated via NLI (DeBERTa) checking for self-contradiction.
*   **Factual Entropy ($H_{fact}$)**: Calculated via RAG checking against the `KnowledgeBase`.

### 1.2 The Kernel (Hardware Simulation)
The `BackfireKernel` is a simulated low-level driver.
*   **Input**: Token Stream.
*   **Control**: SEC Signal ($1 - H_{total}$). 
*   **Mechanism**: If SEC < 0.5, the Kernel executes `emergency_stop()`, severing the stream.

### 1.3 The Knowledge Base (Ground Truth)
A Dictionary/Vector-based store containing immutable facts (e.g., "SCPN has 16 layers"). This provides the external reference frame for the Director.

---

## 2. Validation Results

### 2.1 RAG Integration Test (`test_rag_integration.py`)
*   **Query**: "How many layers are in SCPN?"
*   **Context Retrieval**: Successfully retrieved "scpn layers is 16".
*   **Entropy Calculation**:
    *   Response "16 layers" $\rightarrow$ Low Entropy (0.1).
    *   Response "20 layers" $\rightarrow$ High Entropy (0.9).

### 2.2 Kernel Interlock Test
*   **Trigger**: Simulated low SEC score.
*   **Result**: Kernel raised `[KERNEL INTERRUPT]`.

---

## Legal & Attribution

**Organization**: Anulum CH&LI / Anulum Institute
**Author**: Miroslav Šotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
**Copyright**: © 1998–2025 Miroslav Šotek. All rights reserved.
**Website**: [www.anulum.li](https://www.anulum.li)

**License**: All rights reserved. Unauthorized copying, distribution, or modification of this material is strictly prohibited without written permission from Anulum CH&LI.
