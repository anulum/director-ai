# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Distillation v5: 3-strategy cloud training
"""Distillation v5 — three strategies to fix the 49.8% BA failure.

Strategy A: Larger student (DeBERTa-v3-base, 184M params)
  - Same pipeline, 3x capacity, should reach ~65-70% BA
  - Uses existing FactCG soft labels with template

Strategy B: Hyperparameter sweep on xsmall
  - epochs: [10, 20, 30, 50]
  - alpha: [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
  - lr: [1e-5, 2e-5, 5e-5]
  - Tests if xsmall CAN learn or if capacity is the hard limit

Strategy C: Raw NLI pairs (no FactCG template)
  - Re-generate labels: teacher scores raw (premise, hypothesis) pairs
  - Student trains on raw pairs — simpler task, no template overhead
  - Eval also without template — apples-to-apples

All strategies share:
  - AggreFact soft labels from FactCG teacher
  - KL divergence + hard label blend
  - Automatic eval on AggreFact after training
  - Results saved to training/output/v5_results.json

Usage::

    # Strategy A: large student
    python training/distil_v5_cloud.py --strategy A

    # Strategy B: sweep (many runs)
    python training/distil_v5_cloud.py --strategy B

    # Strategy C: raw NLI (re-generates labels + trains)
    python training/distil_v5_cloud.py --strategy C

    # All strategies
    python training/distil_v5_cloud.py --strategy all
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEACHER_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
TEACHER_REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
AGGREFACT_DATASET = "lytang/LLM-AggreFact"

# FactCG instruction template
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

OUTPUT_DIR = Path("training/output/v5")


# ── Dataset ──────────────────────────────────────────────────────────


class SoftLabelDataset(Dataset):
    """Dataset from pre-computed soft labels."""

    def __init__(
        self,
        records: list[dict],
        tokeniser,
        max_length: int = 256,
        use_template: bool = False,
    ) -> None:
        self.records = records
        self.tokeniser = tokeniser
        self.max_length = max_length
        self.use_template = use_template

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        premise = r["premise"]
        hypothesis = r["hypothesis"]

        if self.use_template:
            # Pre-truncate premise to fit template within max_length
            suffix = _FACTCG_TEMPLATE.format(text_a="", text_b=hypothesis)
            suffix_len = len(self.tokeniser.encode(suffix, add_special_tokens=False))
            max_doc = self.max_length - suffix_len - 4
            if max_doc < 32:
                max_doc = 32
            doc_ids = self.tokeniser.encode(premise, add_special_tokens=False)
            if len(doc_ids) > max_doc:
                premise = self.tokeniser.decode(
                    doc_ids[:max_doc], skip_special_tokens=True
                )
            text = _FACTCG_TEMPLATE.format(text_a=premise, text_b=hypothesis)
            enc = self.tokeniser(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            enc = self.tokeniser(
                premise,
                hypothesis,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "soft_labels": torch.tensor(r["soft_probs"], dtype=torch.float32),
        }


# ── Training ─────────────────────────────────────────────────────────


def train_student(
    student_model: str,
    records: list[dict],
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 2e-5,
    temperature: float = 1.5,
    alpha: float = 0.3,
    max_length: int = 256,
    use_template: bool = False,
) -> dict:
    """Train a student model and return metrics."""
    t0 = time.time()

    tokeniser = AutoTokenizer.from_pretrained(student_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        student_model, num_labels=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Student: %s (%dM params) on %s", student_model, n_params // 1_000_000, device
    )

    dataset = SoftLabelDataset(
        records, tokeniser, max_length=max_length, use_template=use_template
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=len(loader) // 10,
        num_training_steps=len(loader) * epochs,
    )
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels = batch["soft_labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            student_log_probs = torch.log_softmax(logits / temperature, dim=-1)
            teacher_probs = torch.softmax(soft_labels / temperature, dim=-1)
            soft_loss = kl_loss_fn(student_log_probs, teacher_probs) * (temperature**2)

            hard_targets = soft_labels.argmax(dim=-1)
            hard_loss = ce_loss_fn(logits, hard_targets)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)
        logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, epochs, avg)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokeniser.save_pretrained(output_dir)

    # Add factcg flag to config for eval compatibility
    config_path = output_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    cfg["factcg"] = use_template
    config_path.write_text(json.dumps(cfg, indent=2))

    duration = time.time() - t0
    logger.info("Trained in %.1f min, saved to %s", duration / 60, output_dir)

    return {
        "student": student_model,
        "epochs": epochs,
        "lr": lr,
        "alpha": alpha,
        "temperature": temperature,
        "max_length": max_length,
        "use_template": use_template,
        "n_params": n_params,
        "final_loss": losses[-1],
        "losses": losses,
        "duration_s": round(duration, 1),
        "output_dir": str(output_dir),
    }


# ── Evaluation ───────────────────────────────────────────────────────


def evaluate_model(model_dir: Path) -> dict:
    """Run AggreFact eval on a trained model. Returns BA metrics."""
    import subprocess
    import sys

    result_path = model_dir / "aggrefact_eval.json"
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.aggrefact_eval",
        "--model",
        str(model_dir),
        "--save-scores",
        str(result_path),
        "--max-length",
        "256",
    ]
    logger.info("Evaluating: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if proc.returncode != 0:
        logger.error("Eval failed: %s", proc.stderr[-500:] if proc.stderr else "")
        return {"error": proc.stderr[-500:] if proc.stderr else "unknown"}

    if result_path.exists():
        data = json.loads(result_path.read_text())
        ba = data.get("global_balanced_accuracy", 0)
        per_ds_ba = data.get("per_dataset_avg_balanced_accuracy", 0)
        logger.info(
            "BA: %.2f%% (global), %.2f%% (per-dataset avg)", ba * 100, per_ds_ba * 100
        )
        return {
            "global_ba": round(ba, 4),
            "per_dataset_ba": round(per_ds_ba, 4),
            "threshold": data.get("global_threshold", 0),
        }

    return {"error": "no result file"}


# ── Strategy A: Larger student ───────────────────────────────────────


def strategy_a(records: list[dict]) -> dict:
    """DeBERTa-v3-base (184M) with FactCG template."""
    logger.info("=" * 60)
    logger.info("STRATEGY A: DeBERTa-v3-base (184M) + FactCG template")
    logger.info("=" * 60)

    out = OUTPUT_DIR / "strategy_a_base"
    metrics = train_student(
        "microsoft/deberta-v3-base",
        records,
        out,
        epochs=10,
        batch_size=16,  # larger model, smaller batch
        lr=2e-5,
        temperature=1.5,
        alpha=0.3,
        max_length=256,
        use_template=True,
    )
    eval_result = evaluate_model(out)
    metrics["eval"] = eval_result
    return metrics


# ── Strategy B: Hyperparameter sweep ─────────────────────────────────


def strategy_b(records: list[dict]) -> list[dict]:
    """Sweep epochs/alpha/lr on DeBERTa-v3-xsmall."""
    logger.info("=" * 60)
    logger.info("STRATEGY B: Hyperparameter sweep on xsmall")
    logger.info("=" * 60)

    results = []
    configs = [
        # (epochs, alpha, lr) — most promising first
        (20, 0.0, 2e-5),  # pure hard labels
        (20, 0.5, 2e-5),  # balanced blend
        (30, 0.3, 1e-5),  # slower LR, more epochs
        (50, 0.3, 1e-5),  # long training
        (20, 1.0, 2e-5),  # pure soft labels
        (20, 0.3, 5e-5),  # higher LR
    ]

    for i, (epochs, alpha, lr) in enumerate(configs):
        tag = f"b_{i}_ep{epochs}_a{alpha}_lr{lr}"
        out = OUTPUT_DIR / tag
        logger.info("--- Run %d/%d: %s ---", i + 1, len(configs), tag)

        metrics = train_student(
            "microsoft/deberta-v3-xsmall",
            records,
            out,
            epochs=epochs,
            batch_size=32,
            lr=lr,
            temperature=1.5,
            alpha=alpha,
            max_length=256,
            use_template=True,
        )
        eval_result = evaluate_model(out)
        metrics["eval"] = eval_result
        metrics["tag"] = tag
        results.append(metrics)

        ba = eval_result.get("global_ba", 0)
        logger.info(">>> %s: BA=%.2f%%", tag, ba * 100)

        # Early stop if we find something > 60%
        if ba > 0.60:
            logger.info("Found viable config at %.2f%% BA — stopping sweep", ba * 100)
            break

    return results


# ── Strategy C: Raw NLI (no template) ────────────────────────────────


def generate_raw_labels(max_samples: int | None = None) -> list[dict]:
    """Generate teacher labels WITHOUT FactCG template."""
    from datasets import load_dataset

    logger.info("Generating raw NLI labels (no FactCG template)...")
    tokeniser = AutoTokenizer.from_pretrained(TEACHER_MODEL, revision=TEACHER_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, revision=TEACHER_REVISION
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    ds = load_dataset(AGGREFACT_DATASET, split="dev")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    records = []
    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])

        # Raw NLI: premise + hypothesis as pair, NO template
        inputs = tokeniser(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            raw_probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        # Same flip as before: FactCG P[0] inversely maps to AggreFact
        soft_probs = [1.0 - raw_probs[0], raw_probs[0]]

        records.append(
            {
                "premise": premise,
                "hypothesis": hypothesis,
                "hard_label": label,
                "soft_probs": soft_probs,
            }
        )

        if (i + 1) % 1000 == 0:
            logger.info("[%d/%d] raw labels", i + 1, len(ds))

    return records


def strategy_c(
    records_with_template: list[dict], max_label_samples: int | None = None
) -> dict:
    """Raw NLI pairs without FactCG template."""
    logger.info("=" * 60)
    logger.info("STRATEGY C: Raw NLI pairs (no FactCG template)")
    logger.info("=" * 60)

    # Phase 1: generate raw labels (no template)
    raw_labels_path = OUTPUT_DIR / "raw_nli_labels.json"
    if raw_labels_path.exists():
        logger.info("Loading cached raw labels from %s", raw_labels_path)
        raw_records = json.loads(raw_labels_path.read_text())
    else:
        raw_records = generate_raw_labels(max_samples=max_label_samples)
        raw_labels_path.parent.mkdir(parents=True, exist_ok=True)
        raw_labels_path.write_text(json.dumps(raw_records))
        logger.info("Saved %d raw labels to %s", len(raw_records), raw_labels_path)

    # Phase 2: train xsmall on raw pairs
    out_xsmall = OUTPUT_DIR / "strategy_c_xsmall_raw"
    metrics_xsmall = train_student(
        "microsoft/deberta-v3-xsmall",
        raw_records,
        out_xsmall,
        epochs=20,
        batch_size=32,
        lr=2e-5,
        temperature=1.5,
        alpha=0.3,
        max_length=256,
        use_template=False,
    )
    eval_xsmall = evaluate_model(out_xsmall)
    metrics_xsmall["eval"] = eval_xsmall

    # Phase 3: also try base on raw pairs
    out_base = OUTPUT_DIR / "strategy_c_base_raw"
    metrics_base = train_student(
        "microsoft/deberta-v3-base",
        raw_records,
        out_base,
        epochs=10,
        batch_size=16,
        lr=2e-5,
        temperature=1.5,
        alpha=0.3,
        max_length=256,
        use_template=False,
    )
    eval_base = evaluate_model(out_base)
    metrics_base["eval"] = eval_base

    return {
        "xsmall_raw": metrics_xsmall,
        "base_raw": metrics_base,
    }


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description="Distillation v5: 3-strategy training")
    p.add_argument(
        "--strategy",
        choices=["A", "B", "C", "all"],
        required=True,
        help="A=large student, B=hparam sweep, C=raw NLI, all=run all",
    )
    p.add_argument(
        "--labels",
        default="training/output/distil_labels_v3_flipped.json",
        help="Existing soft-label JSON (for strategies A and B)",
    )
    p.add_argument(
        "--max-label-samples",
        type=int,
        default=None,
        help="Max samples for raw label generation (strategy C)",
    )
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Load existing soft labels for A and B
    if args.strategy in ("A", "B", "all"):
        labels_path = Path(args.labels)
        if not labels_path.exists():
            p.error(f"Labels file not found: {labels_path}")
        records = json.loads(labels_path.read_text())
        logger.info("Loaded %d soft-label records from %s", len(records), labels_path)

    if args.strategy in ("A", "all"):
        all_results["strategy_a"] = strategy_a(records)

    if args.strategy in ("B", "all"):
        all_results["strategy_b"] = strategy_b(records)

    if args.strategy in ("C", "all"):
        all_results["strategy_c"] = strategy_c(
            records if args.strategy == "all" else [],
            max_label_samples=args.max_label_samples,
        )

    # Save combined results
    results_path = OUTPUT_DIR / "v5_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("All results saved to %s", results_path)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, res in all_results.items():
        if isinstance(res, list):
            for r in res:
                ba = r.get("eval", {}).get("global_ba", 0)
                logger.info("  %s/%s: BA=%.2f%%", name, r.get("tag", "?"), ba * 100)
        elif isinstance(res, dict) and "eval" in res:
            ba = res["eval"].get("global_ba", 0)
            logger.info("  %s: BA=%.2f%%", name, ba * 100)
        elif isinstance(res, dict):
            for sub, sub_res in res.items():
                if isinstance(sub_res, dict) and "eval" in sub_res:
                    ba = sub_res["eval"].get("global_ba", 0)
                    logger.info("  %s/%s: BA=%.2f%%", name, sub, ba * 100)


if __name__ == "__main__":
    main()
