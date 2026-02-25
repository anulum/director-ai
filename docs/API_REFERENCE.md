# Director-Class AI API Reference

## StrangeLoopAgent

The main entry point for the Director-Class AI system.

```python
from director_ai.src import StrangeLoopAgent

agent = StrangeLoopAgent()
response = agent.process_query("What is the color of the sky?")
```

### Methods

*   `process_query(prompt)`: Processes a user prompt through the Actor-Director-Kernel pipeline.

## DirectorModule

The recursive oversight engine (Layer 16).

### Methods

*   `review_action(prompt, action)`: Evaluates the proposed action against the prompt using Logical and Factual entropy. Returns `(approved: bool, sec_score: float)`.

## KnowledgeBase

The ground truth retrieval system.

### Methods

*   `retrieve_context(query)`: Fetches relevant facts from the internal store.

## BackfireKernel

The simulated hardware interlock.

### Methods

*   `stream_output(token_generator, sec_callback)`: Streams tokens while monitoring SEC. Halts if SEC drops below threshold.

---

## Legal & Attribution

**Organization**: Anulum CH&LI / Anulum Institute
**Author**: Miroslav Šotek — ORCID [0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851)
**Copyright**: © 1998–2025 Miroslav Šotek. All rights reserved.
**Website**: [www.anulum.li](https://www.anulum.li)

**License**: All rights reserved. Unauthorized copying, distribution, or modification of this material is strictly prohibited without written permission from Anulum CH&LI.
