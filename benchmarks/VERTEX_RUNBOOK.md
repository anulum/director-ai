# Vertex AI benchmark runbook

The orchestrator (`python -m benchmarks.orchestrator`) runs
locally for smoke tests and on **Vertex AI custom jobs** for full
accuracy / latency / E2E sweeps. This document covers the Vertex
path end to end.

## Prerequisites (once per workstation)

```bash
# Authenticate with the GCP account that owns the project
gcloud auth login fortisstudio.ch@gmail.com

# Default project
gcloud config set project gotm-director-ai

# Default region (GPU quotas T4, L4, V100, A100 are provisioned here)
gcloud config set compute/region europe-west4

# Enable the APIs used by the runner (idempotent)
gcloud services enable \
  cloudbuild.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com
```

## Resources

| Resource | Value |
|---|---|
| Project | `gotm-director-ai` |
| Region | `europe-west4` |
| Bucket | `gs://gotm-director-ai-training` |
| Artifact Registry | `europe-west4-docker.pkg.dev/gotm-director-ai/director-ai-training` |
| Image | `director-ai-benchmarks:<git-sha>` (also `:cache`) |
| GPU quotas (europe-west4) | T4=1, L4=1, V100=1, P4=1, P100=1, A100=1 |

## One-shot full run

```bash
# T4 worker, default suite (smoke + pytest + adversarial + latency)
benchmarks/run_vertex_benchmarks.sh
```

This:

1. `gcloud builds submit` — builds the container remotely from
   `training/Dockerfile.benchmarks`. ~18 min cold, ~4 min warm.
2. `gcloud ai custom-jobs create` — launches a T4 worker with the
   built image.
3. Container runs `benchmarks/run_in_container.sh`, which invokes
   the orchestrator and uploads `/workspace/output/*` to
   `gs://gotm-director-ai-training/benchmarks/<timestamp>-<sha>/`.

## Common invocations

```bash
# Quick single-case run (skips image build if image for this SHA already exists)
benchmarks/run_vertex_benchmarks.sh --only rust_parity_safety --skip-build

# L4 (faster than T4, same cost tier)
benchmarks/run_vertex_benchmarks.sh --accelerator NVIDIA_L4

# CPU only (skip latency GPU + any GPU-gated accuracy cases)
benchmarks/run_vertex_benchmarks.sh --accelerator ""

# Strict mode — non-zero exit on any failure or high-severity regression.
# Useful from CI when you want the job to fail loudly.
benchmarks/run_vertex_benchmarks.sh \
  --baseline gs://gotm-director-ai-training/benchmarks/baseline/run_report.json \
  --strict
```

## Monitoring

```bash
# Running + recent jobs
gcloud ai custom-jobs list --region=europe-west4 --limit=10

# Stream logs (replace <job-id> from the list above)
gcloud ai custom-jobs stream-logs <job-id> --region=europe-west4

# Download results after completion
RUN_PREFIX=20260418T0805-877feea
gcloud storage cp -r \
  gs://gotm-director-ai-training/benchmarks/${RUN_PREFIX}/ \
  ./benchmarks/results/${RUN_PREFIX}/
```

## Result schema

Every JSON under `gs://.../benchmarks/<run>/` follows
`benchmarks/results/SCHEMA.md`. The two canonical files are:

* `run_report.json` — the full `RunReport`: environment
  fingerprint, per-case results, metrics, wall-clock.
* `regression.json` — only present when the job was launched with
  `--baseline`. Lists `findings` (failed rules) and
  `skipped_rules` (rules that could not evaluate).

## Publishing a new baseline

```bash
# 1. Run a full suite on the reference hardware (typically
#    NVIDIA_TESLA_T4 for cost, NVIDIA_L4 when a faster GPU is
#    needed — document the choice in notes).
benchmarks/run_vertex_benchmarks.sh --suffix baseline-candidate

# 2. Fetch the result locally to review before promoting.
gcloud storage cp \
  gs://gotm-director-ai-training/benchmarks/<run>/run_report.json \
  /tmp/candidate.json

# 3. Inspect.
cat /tmp/candidate.json | jq '.environment, .entries[].name'

# 4. Promote by copying to the canonical baseline path.
gcloud storage cp \
  /tmp/candidate.json \
  gs://gotm-director-ai-training/benchmarks/baseline/run_report.json
```

Baselines are never overwritten in place — promote with the new
file alongside and rotate pointers in CI config if you want
historical versions.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Cloud Build fails at `maturin build` | Missing rustup toolchain | Rerun; the toolchain install is cached after first success. Reproduce locally only via `gcloud builds submit` — never `docker build` locally. |
| Vertex job stays `PENDING` > 10 min | GPU quota exhausted in region | `gcloud compute regions describe europe-west4 --format='value(quotas[].metric, quotas[].usage)' \| grep GPU` |
| `DIRECTOR_BENCH_BASELINE` download fails | Incorrect GCS path or no read-access | Path is `gs://…` not a bare name; container SA needs `storage.objectViewer`. |
| Orchestrator reports dataset size 0 | Accuracy cases not yet wired | Expected on the default suite; add `--only` with cases that accept datasets. |
| Rust parity case says 0 tests collected | Wheel built against wrong Python | Cloud Build uses Python from the base image; confirm `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` digest matches in `Dockerfile.benchmarks`. |

## Cost notes

* Cloud Build: ~4–18 min on `E2_HIGHCPU_8` → roughly USD 0.02–0.08 per build.
* Vertex AI custom job on NVIDIA_TESLA_T4 + n1-standard-8: ~USD 0.55 / hour.
  A full smoke-only suite completes in under 2 min, so single runs are
  under USD 0.05. Accuracy cases (AggreFact 29k) on T4 take ~20 min.

All charges hit the `gotm-director-ai` billing account; GCP free-tier
and promo credits cover initial volume (see
`reference/gcloud_credits_state.md` for remaining balances).
