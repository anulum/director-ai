# Agent Framework Deploy Pack

LangGraph, CrewAI, LlamaIndex, and Vercel AI SDK applications can share one
Director-AI review service instead of embedding model loading into every agent
process. The tracked deploy pack lives in `deploy/agent-frameworks/`.

## Recommended Topology

| Surface | Runtime | Director-AI attachment |
|---|---|---|
| LangGraph | Python agent worker | In-process `director_ai_node()` plus conditional edge |
| CrewAI | Python crew worker | In-process `DirectorAITool` |
| LlamaIndex | Python RAG worker | In-process `DirectorAIPostprocessor` |
| Vercel AI SDK | Vercel app / edge-adjacent server | `@director-ai/vercel-ai` middleware calling Cloud Run |

Use the in-process Python adapters when the agent worker already runs
Director-AI. Use the Cloud Run service when JavaScript clients, serverless apps,
or multiple teams need a central review gate.

## Executable Smoke

The source-of-truth smoke is `examples/agent_framework_guardrails.py`.

```bash
PYTHONPATH=src ./.venv/bin/python examples/agent_framework_guardrails.py
```

It exercises the LangGraph node, CrewAI tool, and LlamaIndex postprocessor
contracts without installing the optional framework SDKs. This keeps the base
package testable while the integration docs stay copyable for real framework
installs.

## Cloud Run Template

`deploy/agent-frameworks/cloud-run-service.yaml` deploys the existing
`deploy/cloud-run/Dockerfile.saas` image as a shared review service.

Required operator edits before deployment:

- replace `PROJECT_ID` and `REGION`;
- create the `director-api-keys` and `director-kb-signing-key` secrets;
- bind the service account to only the secrets and telemetry sinks it needs.

```bash
gcloud run services replace deploy/agent-frameworks/cloud-run-service.yaml --region REGION
```

## Vercel AI SDK Template

`deploy/agent-frameworks/vercel.json` belongs in the Vercel application that
uses `@director-ai/vercel-ai`.

Set these Vercel environment variables:

| Variable | Purpose |
|---|---|
| `DIRECTOR_AI_ENDPOINT` | Cloud Run service root URL |
| `DIRECTOR_API_KEY` | API key accepted by the Director-AI service |

The Vercel application does not run the NLI model. It wraps the AI SDK model
with `createDirectorAiMiddleware()` and sends review requests to Cloud Run.
