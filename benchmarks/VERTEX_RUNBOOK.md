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
| GPU quotas (europe-west4) | T4 usable; L4 submits but can queue; V100/P100 rejected by Vertex CustomJob quota as of 2026-04-28; A100 unavailable |

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

## Credibility Campaign

The external-credibility benchmark queue is tracked in
`benchmarks/vertex_credibility_campaign.toml` and validated by
`python -m benchmarks.credibility_campaign`.

```bash
# Show structural findings and the next dependency-ready stage.
PYTHONPATH=src python -m benchmarks.credibility_campaign --root .

# Machine-readable view for autonomous execution.
PYTHONPATH=src python -m benchmarks.credibility_campaign --root . --json

# Include gated/prediction-dependent stages only after access and
# prediction provenance are ready.
PYTHONPATH=src python -m benchmarks.credibility_campaign \
  --root . \
  --completed aggrefact_anchor_vertex,ragtruth_vertex,halueval_vertex,financebench_vertex,legal_contractnli_vertex,medical_mednli_pubmedqa_vertex,patronus_halubench_wire \
  --include-gated
```

Campaign stages are intentionally one-at-a-time. The planner does not
launch cloud work by itself; it provides the next stage, command,
claim boundary, expected outputs, and Vertex target so the operator can
submit a controlled CustomJob after checking live quota and any
currently running training job.

## Per-Model Package Campaign

The per-model package runner executes every Vertex-eligible benchmark
stage for every stable public scorer model, in manifest order, inside a
single Vertex custom job:

```bash
python -m benchmarks.model_package_vertex_campaign \
  --bucket gs://gotm-director-ai-training \
  --prefix benchmarks/model-packages/campaigns/<timestamp>-<git-sha> \
  --min-free-gb 25
```

The current package manifest expands to 21 stage executions:

- 3 stable scorer aliases: `balanced-default`, `deberta-small`,
  `deberta-large-nli`
- 7 Vertex-eligible evidence stages per alias: AggreFact, RAGTruth,
  HaluEval, FinanceBench, legal ContractNLI/CUAD, medical
  MedNLI/PubMedQA, and Patronus HaluBench text

Use a larger Vertex boot disk for the full package campaign. The job
should be submitted with at least `500 GiB` SSD boot disk so managed
model artefacts, Hugging Face datasets, and intermediate JSON outputs
can coexist while the stages run one by one. The runner checks free
space before and after every stage and fails loudly if available space
falls below the configured threshold.

Results are isolated by model alias and stage:

```text
gs://gotm-director-ai-training/benchmarks/model-packages/campaigns/<run>/
  balanced-default/aggrefact_anchor_vertex/
  balanced-default/ragtruth_vertex/
  ...
  deberta-large-nli/patronus_halubench_wire/
  campaign_summary.json
```

Download after completion with:

```bash
gcloud storage cp -r \
  gs://gotm-director-ai-training/benchmarks/model-packages/campaigns/<run>/ \
  ./benchmarks/results/model-packages/<run>/
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

## Managed fine-tuning jobs

The benchmark image is also used by managed fine-tuning jobs. These jobs are
submitted through the product CLI and run `director_ai.core.training.vertex_runner`
inside the container.

```bash
director-ai train submit --execute \
  --project gotm-director-ai \
  --region europe-west4 \
  --image europe-west4-docker.pkg.dev/gotm-director-ai/director-ai-training/director-ai-benchmarks:<tag> \
  --dataset-uri gs://gotm-director-ai-training/labels/smoke/director_smoke_train_500_20260428.jsonl \
  --eval-uri gs://gotm-director-ai-training/labels/smoke/director_smoke_eval_500_20260428.jsonl \
  --output-uri gs://gotm-director-ai-training/managed-training/<run-name> \
  --model factcg-deberta-v3-large \
  --epochs 1 \
  --batch-size 1 \
  --lr 1e-5 \
  --machine n1-standard-8 \
  --gpu NVIDIA_T4 \
  --gpu-count 1
```

Use `director-ai train models --include-experimental` to list model choices.
Experimental or explicit model ids require `--allow-experimental-model`.

Use `director-ai train sweep` for reproducible scenario matrices across
models, dataset sizes, label strategies, epoch budgets, and hardware profiles.
The current managed sweep record is documented in
`benchmarks/MANAGED_TRAINING_SWEEPS.md`.

Use `benchmarks/run_vertex_model_benchmark.sh` for the managed-training
model-choice anti-regression benchmark. It builds the same benchmark image,
downloads the selected GCS model artefacts inside Vertex AI, runs
`director-ai train benchmark-models` equivalent logic in the container, and
uploads `model_benchmark_report.json` plus a Markdown summary:

```bash
MODEL_SPECS='{"winner":"gs://gotm-director-ai-training/managed-training/sweeps/managed-full-e1-20260428/naturalfull-deberta-v3-large-nli-e1-b1"}' \
GENERAL_URI='gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl' \
benchmarks/run_vertex_model_benchmark.sh --suffix full-e1
```

For the first model-choice benchmark record, see
`docs/internal/managed_training_model_choice_2026-04-28.md`.
For the completed 2026-04-29 managed sweep and final Vertex anti-regression
report path, see `benchmarks/MANAGED_TRAINING_SWEEPS.md`.

Initial same-data smoke results on 2026-04-28:

| Model | Job | Balanced accuracy | F1 |
|---|---|---:|---:|
| `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | `7743632658080464896` | `0.764` | `0.7079` |
| `roberta-large-mnli` | `6306984376949276672` | `0.722` | `0.6150` |
| `yaxili96/FactCG-DeBERTa-v3-Large` | `3912758225048436736` | `0.680` | `0.7080` |

These are smoke-run metrics from 500 train / 500 eval samples. They are useful
for verifying managed training and model-choice plumbing, but not sufficient for
production model promotion.

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
| Vertex job stays `PENDING` > 10 min | Capacity wait or Vertex CustomJob accelerator quota | Check `gcloud ai custom-jobs describe <job-id> --region=europe-west4`; Compute Engine quota alone is not enough. |
| `unknown enum label "NVIDIA_T4"` | Vertex SDK expects Tesla names for older GPUs | Use current CLI; it normalises `NVIDIA_T4` to `NVIDIA_TESLA_T4`. |
| `custom_model_training_nvidia_*_gpus` quota exceeded | Vertex CustomJob quota missing for that accelerator | Use T4/L4 or request Vertex CustomJob quota increase. |
| CUDA OOM early in fine-tuning | Batch size too high for model/GPU | Use `--batch-size 1` for large DeBERTa/RoBERTa smoke runs on T4. |
| Repeated `huggingface/tokenizers` fork warnings | Tokenizer parallelism was initialised before dataloader workers forked | Current runner sets `TOKENIZERS_PARALLELISM=false`; rebuild the image before expecting this in Vertex logs. |
| Text model import fails with `operator torchvision::nms does not exist` | The PyTorch base image carried an incompatible inherited `torchvision` package, and Transformers imported optional vision helpers before loading text model classes | Rebuild from an image whose Dockerfile uninstalls inherited `torchvision` after the pinned dependency install; confirm the replacement benchmark is run on Vertex AI before promoting any model. |
| `DIRECTOR_BENCH_BASELINE` download fails | Incorrect GCS path or no read-access | Path is `gs://…` not a bare name; container SA needs `storage.objectViewer`. |
| Orchestrator reports dataset size 0 | Accuracy cases not yet wired | Expected on the default suite; add `--only` with cases that accept datasets. |
| Rust parity case says 0 tests collected | Wheel built against wrong Python | Cloud Build uses Python from the base image; confirm `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` digest matches in `Dockerfile.benchmarks`. |

## Cost notes

* Cloud Build: ~4–18 min on `E2_HIGHCPU_8` → roughly USD 0.02–0.08 per build.
* Vertex AI custom job on NVIDIA_TESLA_T4 + n1-standard-8: ~USD 0.55 / hour.
  A full smoke-only suite completes in under 2 min, so single runs are
  under USD 0.05. Accuracy cases (AggreFact 29k) on T4 take ~20 min.
* Managed fine-tune smoke jobs with 500 train / 500 eval samples take roughly
  10 minutes wall time on T4 including provisioning and artefact upload.

All charges hit the `gotm-director-ai` billing account; GCP free-tier
and promo credits cover initial volume (see
`reference/gcloud_credits_state.md` for remaining balances).
