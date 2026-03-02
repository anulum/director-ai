# AGPL §13 Compliance

Director-AI is licensed under **AGPL-3.0-or-later**. Section 13 requires that
users interacting with the software over a network can obtain the corresponding
source code.

## Default Behaviour

Director-AI ships with a `/v1/source` endpoint enabled by default:

```bash
curl https://your-server:8080/v1/source
```

```json
{
  "license": "AGPL-3.0-or-later",
  "version": "2.5.0",
  "repository_url": "https://github.com/anulum/director-ai",
  "instructions": "git clone https://github.com/anulum/director-ai",
  "agpl_section": "13"
}
```

This endpoint is **auth-exempt** — no API key is required.

## Custom Repository URL

If you maintain a private fork:

```bash
export DIRECTOR_SOURCE_REPOSITORY_URL="https://git.internal.corp/ai/director-ai"
```

## Disabling the Endpoint

Commercial licensees who have obtained a non-AGPL license may disable the
endpoint:

```bash
export DIRECTOR_SOURCE_ENDPOINT_ENABLED=false
```

!!! warning
    Disabling this endpoint without a commercial license violates AGPL §13.

## Operator Responsibilities

1. **Unmodified deployments**: the default endpoint pointing to the upstream
   GitHub repository satisfies §13.
2. **Modified deployments**: you must publish your modified source and update
   `DIRECTOR_SOURCE_REPOSITORY_URL` to point to it.
3. **Commercial license**: contact `protoscience@anulum.li` for a proprietary
   license that removes the §13 obligation.
