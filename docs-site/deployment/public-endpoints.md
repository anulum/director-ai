# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - public endpoint exposure rules

# Public Endpoint Exposure

Director-AI has a small set of endpoints that may be reached before normal credential checks. Treat them as operational interfaces, not user traffic.

## Default Exposure

| Endpoint | Default | What it returns | Exposure rule |
|----------|---------|-----------------|---------------|
| `GET /v1/live` | Exempt | A single static liveness flag; no version, tier, or licensee details | Private load-balancer probe, private network, or localhost tunnel only |
| `GET /v1/health` | Exempt | Version, mode, profile, NLI loaded flag, uptime, and licence class | Private load-balancer probe, private network, or localhost tunnel only |
| `GET /v1/ready` | Exempt | Ready flag and scorer load failure reason | Private readiness checks only |
| `GET /v1/source` | Exempt | Licence class, version, repository URL, and clone instruction | Publish where the service wishes to advertise its source location |
| `GET /v1/metrics` | Protected when server credentials are configured | JSON counters, histograms, and gauges | Private observability plane only |
| `GET /v1/metrics/prometheus` | Protected when server credentials are configured | Prometheus metric families and labelled samples | Unauthenticated only on a private scrape network |

The default unauthenticated set is `/v1/live`, `/v1/health`, `/v1/ready`, and `/v1/source`. The Prometheus endpoint joins that set only when `metrics_require_auth=false`.

## Operator Controls

| Setting | Default | Use |
|---------|---------|-----|
| `metrics_require_auth` | `true` | Keep Prometheus protected when server credentials exist. Set false only for a private scrape path. |
| `source_endpoint_enabled` | `true` | Keeps the source-availability pointer live. Disable if you do not want to advertise the source location. |
| `cors_origins` | Empty list | Set explicit browser origins; do not use broad wildcard CORS on internet-facing deployments. |
| `production_mode` | `false` | Enables the stricter deployment posture expected for paid or public services. |

When server credentials are configured, all other REST endpoints require the caller credential unless explicitly listed above.
For browser-facing reverse-proxy examples, see [CORS Reverse Proxy](cors-reverse-proxy.md).

## Reverse Proxy Rules

Public internet deployments should keep the operational endpoints behind a private route. For example, let the load balancer call `/v1/health` and `/v1/ready`, but deny direct WAN access to those paths.

Metrics belong on the observability network. If Prometheus scrapes through a sidecar, loopback, service mesh, or private subnet, `metrics_require_auth=false` is acceptable. For any shared or internet-routable path, keep `metrics_require_auth=true`.

The source endpoint is a transparency convenience that advertises where the source lives. If enabled on a public service, it should return only the licence class and repository URL. Do not point it at private forks, internal mirrors, or local filesystem paths.

## Kubernetes Sketch

```yaml
readinessProbe:
  httpGet:
    path: /v1/ready
    port: 8080
livenessProbe:
  httpGet:
    path: /v1/health
    port: 8080
```

Expose these probes to the orchestrator, not to public ingress. Route `/v1/review`, `/v1/stream`, and customer APIs through the authenticated ingress path.

## Audit Checklist

- `/v1/health` is reachable only from the load balancer, orchestrator, or private tunnel.
- `/v1/ready` is reachable only from readiness infrastructure.
- `/v1/source` returns a public repository URL, not an internal path.
- `/v1/metrics/prometheus` is protected unless Prometheus scrapes over a private route.
- `cors_origins` lists exact origins for browser clients.
