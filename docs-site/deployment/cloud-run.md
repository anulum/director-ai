<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Cloud Run CPU deployment
-->

# Cloud Run CPU deployment

`deploy/cloud-run/Dockerfile.saas` builds a Linux AMD64 image with the free
core wheel, the commercial `director-ai-pro` overlay, the compiled Rust
kernel used by adaptive scoring, and a local FP32 FactCG ONNX model.
Production use of the overlay requires the applicable commercial
licence. The model retains its upstream MIT licence and model card.

```bash
docker build --platform linux/amd64 \
  -f deploy/cloud-run/Dockerfile.saas -t director-ai-saas .
docker run --rm -p 8080:8080 \
  -e DIRECTOR_API_KEYS="$DIRECTOR_API_KEYS" director-ai-saas
```

The image verifies every dependency against
`requirements/docker-saas.txt`. This CPython 3.12/Linux AMD64 profile uses
the official PyTorch CPU wheel at the same upstream version selected by
`uv.lock`; it installs no CUDA packages. Refresh commands are documented in
[Optional Extra Locks](https://github.com/anulum/director-ai/blob/main/requirements/OPTIONAL_EXTRA_LOCKS.md).

FactCG source weights are pinned to revision
`0430e3509dbd28d2dff7a117c0eae25359ff3e80` and checked against their
upstream SHA-256 before export. The runtime contains the exported model,
tokenizer, upstream notices and `provenance.json` under `/app/models/onnx`.
It runs as a non-root user and has Hugging Face offline mode enabled.

## Configuration and authentication

The entrypoint listens on `0.0.0.0`. Its port is `DIRECTOR_PORT`, then the
Cloud Run-injected `PORT`, then `8080`. Leave `DIRECTOR_PORT` unset on Cloud
Run so the injected port controls ingress. Docker's health check uses the
same port precedence and the existing `/v1/health` route. Configure the
Cloud Run startup probe separately; Docker health checks do not configure
the Cloud Run service.

Supply `DIRECTOR_API_KEYS` as a JSON array or comma-separated list, or
`DIRECTOR_API_KEY_TENANT_MAP` as the supported JSON mapping. Empty effective
credentials refuse startup. Store deployment credentials in your secret
manager and inject them at runtime. Review clients use `X-API-Key`:

```bash
curl http://localhost:8080/v1/health
curl http://localhost:8080/v1/review \
  -H "X-API-Key: $DIRECTOR_CLIENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Where is Paris?","response":"Paris is in France."}'
```

This image reads individual `DIRECTOR_*` environment fields through the
default configuration path. Named `DIRECTOR_PROFILE` values are refused
because that path would bypass environment authentication and ONNX settings.
Set `DIRECTOR_PROFILE=default` and configure fields such as
`DIRECTOR_COHERENCE_THRESHOLD` directly. NLI uses the local ONNX artefact
and refuses a missing model-backed scorer. The image does not require a
local LLM judge service.

`DIRECTOR_PRODUCTION_MODE=true` requests the existing hardened production
configuration and its validation requirements. The entrypoint confirms
that mode with the CLI's `--production` flag; API credentials alone do not
satisfy all production requirements. See the
[production checklist](checklist.md) for audit, knowledge-store and provider
configuration.

## Capacity and storage

The default is one worker. Each additional `DIRECTOR_WORKERS` process loads
its own scorer, so measure memory, startup time and concurrency with your
request lengths before increasing it. The FP32 source weights are about
1.74 GB; this is not the total runtime memory requirement. No 2 GiB capacity
or latency guarantee is implied.

Cloud Run's writable filesystem is ephemeral and consumes instance memory.
The default writable model-upload directory is `/tmp/director-models`;
configure persistent external storage for any uploads, audit or knowledge
data that must survive an instance restart. Do not use local files as a
durability contract.

The build and local container checks do not deploy a service. Validate
the platform-specific probes, secrets, storage and capacity against the
[Cloud Run container contract](https://docs.cloud.google.com/run/docs/container-contract)
before deployment.
