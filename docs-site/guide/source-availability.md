# Source Availability

Director-AI is open core: the **core** is Apache-2.0 and the **Advanced & Labs**
tier is BUSL-1.1 (source-available). For transparency, the server exposes a
`/v1/source` endpoint so anyone interacting with a deployment can find the
corresponding source.

## Default Behaviour

Director-AI ships with a `/v1/source` endpoint enabled by default:

```bash
curl https://your-server:8080/v1/source
```

```json
{
  "license": "Apache-2.0 AND BUSL-1.1",
  "version": "3.21.0",
  "repository_url": "https://github.com/anulum/director-ai",
  "instructions": "git clone https://github.com/anulum/director-ai"
}
```

This endpoint is **auth-exempt** — no API key is required.

## Custom Repository URL

If you maintain a private fork:

```bash
export DIRECTOR_SOURCE_REPOSITORY_URL="https://git.internal.corp/ai/director-ai"
```

## Disabling the Endpoint

The endpoint is a transparency convenience, not a licence obligation. Disable it
if you do not want to advertise the source location:

```bash
export DIRECTOR_SOURCE_ENDPOINT_ENABLED=false
```

## The Tier Boundary in Practice

Since 3.18.1 the public PyPI wheel is **core-only**: the BUSL-1.1 advanced
packages and the paid single modules (server, verified scorer, calibration,
training, …) are not in the artefact. Every public name stays importable —
using one in a core-only install raises a clear
`requires the advanced tier` error instead of failing obscurely. The source
of all tiers remains visible in this repository (source-available).

Subscribers install a paid tier from the private index; it layers into the
same `director_ai` namespace:

```bash
pip install director-ai-pro \
  --index-url https://pypi.org/simple/ \
  --extra-index-url https://CUSTOMER:TOKEN@pypi.remanentia.com/simple/
```

An annual subscription (Pro CHF/USD 490, Full CHF/USD 980 per year) covers
index access and all releases of the tier while active. Checkout and plans:
[anulum.li/director-ai/pricing](https://www.anulum.li/director-ai/pricing.html).

## Operator Notes

1. **Unmodified deployments**: the default endpoint points at the upstream GitHub
   repository.
2. **Modified deployments**: point `DIRECTOR_SOURCE_REPOSITORY_URL` at your fork
   so users can find your changes.
3. **Advanced tier in production**: the Apache-2.0 core is free for production;
   running the BUSL-1.1 advanced tier in production or as a hosted service needs
   an active commercial subscription or licence — see the pricing page or
   contact `director.class.ai@anulum.li`.
