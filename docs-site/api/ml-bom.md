# Supply-Chain ML-BOM

A supply-chain attack swaps a model, dataset, or dependency for a poisoned one
(a backdoored checkpoint on a model hub, a tampered training set). The defence is
provenance: record every component's SHA-256 digest at a known-good point, then
re-verify the deployed artefacts to detect substitution (OWASP ASVS / ML
supply-chain controls).

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.ml_bom import ComponentType

bom = ProductionGuard(DirectorConfig()).ml_bom

# Pin each component to the digest of its known-good bytes.
bom.add_artifact("factcg-onnx", "1.0", ComponentType.MODEL, model_bytes,
                 supplier="anulum", source="hf://anulum/factcg")
bom.add_artifact("aggrefact", "2024", ComponentType.DATASET, dataset_bytes)

print(bom.bom_digest)        # 64-hex fingerprint of the whole inventory

# Later, re-verify what is actually deployed.
report = bom.verify({"factcg-onnx": deployed_model_bytes})
print(report.ok)             # False if any supplied artefact was substituted
print(report.to_dict())      # {"ok", "intact", "tampered", "unverified"}
```

## Components

A `MLBOMComponent` records `name`, `version`, `component_type`, `sha256`, and
optional `supplier` / `source` / `license`. The type is one of:

| `ComponentType` | Tracks |
|-----------------|--------|
| `MODEL` | A weight artefact (checkpoint, ONNX, safetensors). |
| `DATASET` | A training/evaluation dataset. |
| `DEPENDENCY` | A software package. |
| `CODE` | A first-party source artefact or build output. |

`add_artifact(name, version, type, data, **metadata)` digests `data` to pin the
SHA-256 for you; `add(component)` records a pre-built component. A duplicate name
is rejected. `MLBOMComponent.matches(data)` re-derives the digest of `data` and
compares — a mismatch is substitution/poisoning.

## Tamper-evidence and verification

`bom_digest` is a SHA-256 over the canonical component list: any change to a
component (or the set of components) changes it, so a trusted copy of the digest
makes the whole inventory tamper-evident.

`verify(actuals)` maps component name → deployed bytes and classifies each:

| Bucket | Meaning |
|--------|---------|
| `intact` | Supplied bytes match the recorded digest. |
| `tampered` | Supplied bytes differ (poisoned), **or** an unknown name not in the inventory. |
| `unverified` | A recorded component for which no bytes were supplied. |

`report.ok` is `True` only when nothing is `tampered` (an `unverified` component
is not a failure — it simply was not checked this pass). Both `to_dict()`
serialisations are tenant-safe: names, versions, digests, and suppliers only,
never the artefact bytes.
