# Mechanistic Interpretability (ReDeEP)

`MechanisticAttributor` traces a hallucination signal to the specific
transformer components that produced it, following the ReDeEP decoupling
(Sun et al., 2025): a model's output is driven by **Knowledge FFNs** (feed-forward
layers injecting *parametric* knowledge learned in training) and **Copying
Heads** (attention heads that copy from the *external* retrieved context). A
response is hallucination-prone when it leans on parametric knowledge while
under-using the supplied context.

The attributor consumes per-layer FFN-knowledge and external-attention signals
plus per-head copying scores and reports an overall risk together with *which*
Knowledge-FFN layers and *which* Copying-Heads drove it — the per-component,
regulator-facing explanation (EU AI Act, FDA) that a single score cannot give.

## Decoupled risk

Per layer: `risk = ffn_weight · ffn_knowledge + attention_weight · (1 − external_attention)`
(weights normalised to sum to 1). High parametric injection with low external
attention → high risk. The overall risk is the mean across layers; at or above
`risk_threshold` the response is flagged.

## Injected signals

Signals are injected, so the attribution logic runs without an ML stack and is
fully deterministic under test. A real deployment extracts them from a
transformer's MLP activations and attention maps (HuggingFace
`output_attentions` / TransformerLens hooks) and feeds them in.

```python
from director_ai.core.interpretability import MechanisticAttributor

attributor = MechanisticAttributor(risk_threshold=0.5, top_k=5)

# From per-layer arrays a real integration has already reduced:
layers = MechanisticAttributor.layer_signals_from_arrays(
    ffn_knowledge=per_layer_mlp_magnitude,        # [0, 1] per layer
    external_attention=per_layer_context_attention,  # [0, 1] per layer
)
heads = MechanisticAttributor.head_signals_from_matrix(per_layer_per_head_copying)

report = attributor.attribute(layers, heads)
if report.is_hallucination:
    worst = report.knowledge_ffn_layers[0]
    print(report.reason)
    # "hallucination: risk=0.90 ...; top Knowledge-FFN layer 17 (ffn=0.93, external=0.07)"
    for head in report.copying_heads:
        print("copying head", head.layer_index, head.head_index, head.copying_score)
```

Implement the `ActivationProvider` protocol to pull signals straight from a model
and call `attribute_from(provider)`.

## Full API

::: director_ai.core.interpretability.redeep.MechanisticAttributor

::: director_ai.core.interpretability.redeep.MechanisticAttributionReport

::: director_ai.core.interpretability.redeep.LayerSignal

::: director_ai.core.interpretability.redeep.HeadSignal

::: director_ai.core.interpretability.redeep.LayerContribution

::: director_ai.core.interpretability.redeep.HeadContribution
