# Helm Chart

*Added in v3.11.0*

Deploy Director-AI to Kubernetes with the bundled Helm chart at `deploy/helm/director-ai/`.

## Install

```bash
helm install director-ai deploy/helm/director-ai/ \
  --set server.profile=thorough \
  --set server.apiKeySecret=director-api-key
```

## Configuration

Key values in `values.yaml`:

| Value | Default | Description |
|-------|---------|-------------|
| `replicaCount` | `1` | Number of replicas |
| `image.repository` | `ghcr.io/anulum/director-ai` | Container image |
| `server.port` | `8080` | API server port |
| `server.workers` | `2` | Uvicorn workers |
| `server.profile` | `fast` | Scoring profile (fast, thorough, medical, etc.) |
| `server.apiKeySecret` | `""` | K8s secret name containing `api-key` |
| `gpu.enabled` | `false` | Enable GPU resources |
| `autoscaling.enabled` | `false` | Enable HPA |
| `autoscaling.targetCPUUtilization` | `70` | HPA CPU target |

## GPU Mode

```bash
helm install director-ai deploy/helm/director-ai/ \
  --set gpu.enabled=true \
  --set server.profile=thorough
```

## API Key

Create a Kubernetes secret first:

```bash
kubectl create secret generic director-api-key \
  --from-literal=api-key=your-secret-key
```

Then reference it:

```bash
helm install director-ai deploy/helm/director-ai/ \
  --set server.apiKeySecret=director-api-key
```

## Security

The chart enforces:

- `runAsNonRoot: true` (UID 1000)
- `readOnlyRootFilesystem: true`
- All capabilities dropped
- `allowPrivilegeEscalation: false`

## Health Probes

- Liveness: `GET /health` (initial delay 10s, period 30s)
- Readiness: `GET /health` (initial delay 5s, period 10s)
