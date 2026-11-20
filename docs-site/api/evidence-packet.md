# Local production evidence walkthrough

Run the production scaffold checks and score two completed answers against a
bundled knowledge base. The demo uses a local FactCG model and needs no
running server, private upstream API or model-generation service.

## Install and provision

Use a clean virtual environment on a supported Python version. Set
`CORE_WHEEL` and `LICENSED_WHEEL` to the paths of matching Core and licensed
Pro or Full wheels from the same release as this guide. The licensed wheel can be
installed from a local file. The [pricing page](../pricing.md) describes
delivery. Core supplies `quickstart` and `verify-evidence`. In Core alone,
`evidence` exits with the advanced-tier diagnostic before creating a packet
or its output directory. Pro and Full
supply `production-check` and the parts that emit the packet.

```bash
python -m pip install "${CORE_WHEEL}[nli]" "$LICENSED_WHEEL" sentencepiece backfire-kernel==0.1.3
```

The NLI extra installs PyTorch and Transformers. The production profile
uses the native Backfire kernel for adaptive threshold computation. Use
PyTorch and Backfire wheels for your platform. This walkthrough was checked
on CPU with Python 3.12.3, PyTorch 2.13.0, Transformers 5.18.0,
Hugging Face Hub 1.33.0 and
SentencePiece 0.2.2.

Provision `yaxili96/FactCG-DeBERTa-v3-Large` at the pinned revision
[`0430e3509dbd28d2dff7a117c0eae25359ff3e80`](https://huggingface.co/yaxili96/FactCG-DeBERTa-v3-Large/tree/0430e3509dbd28d2dff7a117c0eae25359ff3e80)
in the Hugging Face cache.

Use the same `HF_HOME` for provisioning and scoring. On a connected machine,
populate the cache with the model and tokenizer files:

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in (
    "config.json", "model.safetensors", "spm.model", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
):
    hf_hub_download(
        "yaxili96/FactCG-DeBERTa-v3-Large",
        filename,
        revision="0430e3509dbd28d2dff7a117c0eae25359ff3e80",
    )
PY
```

With the model cached, set `HF_HUB_OFFLINE=1` to run without fetching it.
Leave enough RAM for the model and its tokenizer.

## Validate the scaffold

In an empty working directory, run:

```bash
director-ai quickstart --profile production
director-ai production-check --path director_guard
```

Fill `director_guard/.env`. The generated file gives the formats for the
API-key-to-tenant map, proxy keys, LLM and proxy upstream URLs, knowledge-write
and audit HMAC keys, and exact HTTPS CORS origin. For this local walkthrough,
use dedicated demo keys, loopback upstream URLs and a demo HTTPS origin.

```bash
director-ai production-check --path director_guard --require-secrets
```

The validator checks the generated files and whether the required values are
present and have the expected basic form. It reports redacted results and
exits nonzero if a check fails.

## Emit and verify a local packet

Set `DIRECTOR_API_KEY_TENANT_MAP` in the shell running the demo to the scaffold
map. Set `DIRECTOR_KNOWLEDGE_WRITE_HMAC_KEYS` to the scaffold's
`DIRECTOR_KB_HMAC_KEYS` value. Compose performs this mapping for deployments.
The local CLI takes those values from its process environment.

```bash
director-ai evidence --emit evidence/ --profile production
director-ai verify-evidence evidence/
```

The demo asks which day the library opens. The bundled fact supports opening
on Monday and contradicts the claim that the library stays closed that day. Expect `grounded answer approved: True`,
`hallucinated answer blocked: True` and `Evidence packet VERIFIED`. The
`production` profile requires model-backed NLI. A missing model or required
profile setting makes the command fail. The separate `fast` profile runs a
heuristic demonstration.

`evidence/evidence_packet.json` records the decisions, scores,
[Answer Bills of Materials](answer-bom.md) and [eval records](eval-trace.md).
A Core installation can verify the file offline. Verification checks the
content digest and recorded demo outcomes. An edit that leaves the original
digest unchanged returns `digest_mismatch`. The digest provides content
integrity against the recorded value. Producer authentication requires a
separate trust mechanism.

For an optional streaming test, follow the
[contradiction halt guide](../guide/streaming.md). The packet above contains
the decisions for two completed responses.
