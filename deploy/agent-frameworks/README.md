# Agent Framework Guardrail Deploy Pack

This deploy pack gives LangGraph, CrewAI, LlamaIndex, and Vercel AI SDK
applications one shared Director-AI review service. The service runs on Cloud
Run; framework clients call either the Python in-process adapters or the REST
`/v1/review` endpoint.

## Cloud Run

Build and push the service image:

```bash
gcloud builds submit \
  --tag REGION-docker.pkg.dev/PROJECT_ID/director-ai/director-ai:agent-frameworks \
  --file deploy/cloud-run/Dockerfile.saas
```

Create secrets before applying the service template:

```bash
printf '%s' "$DIRECTOR_API_KEYS" | gcloud secrets create director-api-keys --data-file=-
printf '%s' "$DIRECTOR_KB_SIGNING_KEY" | gcloud secrets create director-kb-signing-key --data-file=-
```

Apply the template after replacing `PROJECT_ID`, `REGION`, and service account
placeholders:

```bash
gcloud run services replace deploy/agent-frameworks/cloud-run-service.yaml --region REGION
```

## Vercel AI SDK Clients

Use `deploy/agent-frameworks/vercel.json` in the Vercel application that imports
`@director-ai/vercel-ai`. Configure these environment variables in Vercel:

| Variable | Purpose |
|---|---|
| `DIRECTOR_AI_ENDPOINT` | Cloud Run service root URL |
| `DIRECTOR_API_KEY` | API key accepted by the Cloud Run Director-AI service |

The Vercel app stays lightweight: it does not run the NLI model; it wraps its AI
SDK model with `createDirectorAiMiddleware()` and delegates review decisions to
the Cloud Run service.

## Local Smoke

```bash
PYTHONPATH=src ./.venv/bin/python examples/agent_framework_guardrails.py
```

That smoke exercises the LangGraph node, CrewAI tool, and LlamaIndex
postprocessor contracts without installing the optional framework SDKs.
