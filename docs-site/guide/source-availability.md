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
  "version": "3.15.3",
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

## Operator Notes

1. **Unmodified deployments**: the default endpoint points at the upstream GitHub
   repository.
2. **Modified deployments**: point `DIRECTOR_SOURCE_REPOSITORY_URL` at your fork
   so users can find your changes.
3. **Advanced tier in production**: the Apache-2.0 core is free for production;
   running the BUSL-1.1 advanced tier in production or as a hosted service needs
   a commercial licence — contact `director.class.ai@anulum.li`.
