#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Knowledge Distillation
"""Distil a 3-class NLI teacher into a compact Lite Scorer v2 student.

The student learns from teacher soft labels with KL divergence plus hard labels
with cross-entropy. Public accuracy and latency claims are forbidden until the
held-out evaluator and evidence recorder write a reviewed evidence packet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DIRECTOR_DATA_DIR", Path(__file__).parent / "data"))

SUMM_SOURCES = {
    "halueval_summarization",
    "aggrefact_AggreFact-CNN",
    "aggrefact_AggreFact-XSum",
    "aggrefact_TofuEval-MediaS",
    "aggrefact_TofuEval-MeetB",
}

GENERAL_SOURCES = {
    "halueval_qa",
    "halueval_dialogue",
    "fever",
    "anli_r3",
    "vitaminc",
}


@dataclass(frozen=True)
class TrainingRunConfig:
    teacher: str
    student: str
    output_dir: Path
    lr: float
    epochs: int
    batch_size: int
    max_length: int
    temperature: float
    alpha: float
    summ_target: int
    general_target: int
    seed: int
    eval_limit: int
    num_workers: int
    device: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Knowledge distillation")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher model path")
    parser.add_argument(
        "--student",
        type=str,
        default="microsoft/MiniLM-L6-H384-uncased",
        help="Student base model",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--summ-target", type=int, default=15000)
    parser.add_argument("--general-target", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--eval-limit", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training/output/distilled-lite-scorer"),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingRunConfig:
    return TrainingRunConfig(
        teacher=args.teacher,
        student=args.student,
        output_dir=args.output_dir,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        alpha=args.alpha,
        summ_target=args.summ_target,
        general_target=args.general_target,
        seed=args.seed,
        eval_limit=args.eval_limit,
        num_workers=args.num_workers,
        device=args.device,
    )


def validate_training_run_config(config: TrainingRunConfig) -> list[str]:
    errors: list[str] = []
    if config.epochs <= 0:
        errors.append("epochs must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    if config.max_length <= 0:
        errors.append("max_length must be positive")
    if config.temperature <= 0.0:
        errors.append("temperature must be positive")
    if config.alpha <= 0.0 or config.alpha > 1.0:
        errors.append("alpha must be in (0, 1]")
    if config.lr <= 0.0:
        errors.append("lr must be positive")
    if config.summ_target <= 0:
        errors.append("summ_target must be positive")
    if config.general_target <= 0:
        errors.append("general_target must be positive")
    if config.seed < 0:
        errors.append("seed must be non-negative")
    if config.eval_limit <= 0:
        errors.append("eval_limit must be positive")
    if config.num_workers < 0:
        errors.append("num_workers must be non-negative")
    if config.device not in {"auto", "cpu", "cuda"}:
        errors.append("device must be one of auto, cpu, or cuda")
    return errors


def _cuda_runtime_probe() -> bool:
    try:
        tensor = torch.ones(1, device="cuda")
        _ = float(tensor.cpu()[0])
    except Exception:
        return False
    return True


def resolve_training_device(
    preference: str,
    *,
    cuda_available: Callable[[], bool] = torch.cuda.is_available,
    cuda_probe: Callable[[], bool] = _cuda_runtime_probe,
) -> tuple[torch.device, str | None]:
    if preference == "cpu":
        return torch.device("cpu"), None
    if preference == "cuda":
        if not cuda_available():
            raise RuntimeError("CUDA requested but is not available")
        if not cuda_probe():
            raise RuntimeError("CUDA requested but failed the runtime probe")
        return torch.device("cuda"), None
    if preference == "auto":
        if not cuda_available():
            return torch.device("cpu"), None
        if cuda_probe():
            return torch.device("cuda"), None
        return torch.device(
            "cpu"
        ), "CUDA was visible but failed the runtime probe; using CPU"
    raise RuntimeError("device must be one of auto, cpu, or cuda")


def set_reproducible_seeds(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_subset(
    dataset: Any, summ_target: int, general_target: int, seed: int = 42
) -> Any:
    """Build deterministic summarisation-heavy plus general training subset."""
    rng = np.random.default_rng(seed)
    sources = np.array(dataset["source"])

    summ_mask = np.isin(sources, list(SUMM_SOURCES))
    summ_idx = np.where(summ_mask)[0]
    if len(summ_idx) < summ_target:
        repeats = summ_target // len(summ_idx) + 1
        summ_idx = np.tile(summ_idx, repeats)[:summ_target]
        rng.shuffle(summ_idx)
    else:
        summ_idx = rng.choice(summ_idx, size=summ_target, replace=False)

    gen_mask = np.isin(sources, list(GENERAL_SOURCES))
    gen_idx = np.where(gen_mask)[0]
    if len(gen_idx) > general_target:
        gen_idx = rng.choice(gen_idx, size=general_target, replace=False)

    all_idx = np.concatenate([summ_idx, gen_idx])
    rng.shuffle(all_idx)
    return dataset.select(all_idx.tolist())


class DistillationDataset(torch.utils.data.Dataset):
    """Dataset that returns tokenised pairs for both teacher and student."""

    def __init__(
        self,
        hf_dataset: Any,
        teacher_tok: Any,
        student_tok: Any,
        max_length: int = 384,
    ) -> None:
        self.data = hf_dataset
        self.teacher_tok = teacher_tok
        self.student_tok = student_tok
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data[idx]
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        label = row["label"]

        t_enc = self.teacher_tok(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        s_enc = self.student_tok(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "t_input_ids": t_enc["input_ids"].squeeze(0),
            "t_attention_mask": t_enc["attention_mask"].squeeze(0),
            "s_input_ids": s_enc["input_ids"].squeeze(0),
            "s_attention_mask": s_enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Combined KL divergence soft loss and hard-label cross-entropy."""
    soft_teacher = functional.log_softmax(teacher_logits / temperature, dim=-1)
    soft_student = functional.log_softmax(student_logits / temperature, dim=-1)
    kl_loss = functional.kl_div(
        soft_student, soft_teacher.exp(), reduction="batchmean"
    ) * (temperature**2)
    ce_loss = functional.cross_entropy(student_logits, labels)
    return alpha * kl_loss + (1.0 - alpha) * ce_loss


def _classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    accuracy = float((preds == labels).mean()) if len(labels) else 0.0
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in (0, 1, 2):
        true_positive = int(((preds == label) & (labels == label)).sum())
        false_positive = int(((preds == label) & (labels != label)).sum())
        false_negative = int(((preds != label) & (labels == label)).sum())
        support = true_positive + false_negative
        recall = true_positive / support if support else 0.0
        precision_denominator = true_positive + false_positive
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        denominator = precision + recall
        recalls.append(recall)
        f1_scores.append(
            (2.0 * precision * recall / denominator) if denominator else 0.0
        )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "f1": float(np.mean(f1_scores)),
        "f1_entailment": f1_scores[0],
        "f1_neutral": f1_scores[1],
        "f1_contradiction": f1_scores[2],
    }


def evaluate(
    student: Any, dataloader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Evaluate student model on a 3-class NLI dataloader."""
    student.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            s_ids = batch["s_input_ids"].to(device)
            s_mask = batch["s_attention_mask"].to(device)
            labels = batch["label"]
            out = student(input_ids=s_ids, attention_mask=s_mask)
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return _classification_metrics(np.array(all_labels), np.array(all_preds))


def write_training_run_manifest(
    config: TrainingRunConfig,
    *,
    train_rows: int,
    eval_rows: int,
    device: str,
    teacher_params: int,
    student_params: int,
) -> None:
    payload = {
        "schema_version": "1.0.0",
        "run_id": "lite-scorer-v2-training",
        "public_score_claim": False,
        "claim_boundary": (
            "Training run metadata only; public score claims require held-out evaluation "
            "and recorded evidence."
        ),
        **{
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "device_preference": config.device,
        "resolved_device": device,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / student_params,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "training_run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    errors = validate_training_run_config(config)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = set_reproducible_seeds(config.seed)

    try:
        device, device_warning = resolve_training_device(config.device)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if device_warning is not None:
        logger.warning("%s", device_warning)
    logger.info("Device: %s", device)

    full_dataset = load_from_disk(str(DATA_DIR))
    train_sub = build_subset(
        full_dataset["train"],
        summ_target=config.summ_target,
        general_target=config.general_target,
        seed=config.seed,
    )
    eval_sub = full_dataset["eval"].select(
        range(min(config.eval_limit, len(full_dataset["eval"])))
    )
    logger.info("Train: %d, Eval: %d", len(train_sub), len(eval_sub))

    logger.info("Loading teacher: %s", config.teacher)
    teacher_tok = AutoTokenizer.from_pretrained(config.teacher, use_fast=False)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        config.teacher,
        num_labels=3,
    )
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    logger.info("Loading student: %s", config.student)
    student_tok = AutoTokenizer.from_pretrained(config.student)
    student = AutoModelForSequenceClassification.from_pretrained(
        config.student,
        num_labels=3,
    )
    student.to(device)

    teacher_params = sum(param.numel() for param in teacher.parameters())
    student_params = sum(param.numel() for param in student.parameters())
    logger.info(
        "Teacher: %s params, Student: %s params (%.1fx compression)",
        f"{teacher_params:,}",
        f"{student_params:,}",
        teacher_params / student_params,
    )
    write_training_run_manifest(
        config,
        train_rows=len(train_sub),
        eval_rows=len(eval_sub),
        device=str(device),
        teacher_params=teacher_params,
        student_params=student_params,
    )

    train_ds = DistillationDataset(
        train_sub, teacher_tok, student_tok, config.max_length
    )
    eval_ds = DistillationDataset(eval_sub, teacher_tok, student_tok, config.max_length)
    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=config.batch_size * 2,
        num_workers=config.num_workers,
    )

    optimiser = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max=config.epochs * len(train_dl),
    )

    best_ba = 0.0
    for epoch in range(config.epochs):
        student.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl):
            t_ids = batch["t_input_ids"].to(device)
            t_mask = batch["t_attention_mask"].to(device)
            s_ids = batch["s_input_ids"].to(device)
            s_mask = batch["s_attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                teacher_out = teacher(input_ids=t_ids, attention_mask=t_mask)

            student_out = student(input_ids=s_ids, attention_mask=s_mask)
            loss = distillation_loss(
                student_out.logits,
                teacher_out.logits,
                labels,
                config.temperature,
                config.alpha,
            )

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            total_loss += loss.item()
            if (step + 1) % 50 == 0:
                logger.info(
                    "Epoch %d step %d/%d loss=%.4f lr=%.2e",
                    epoch + 1,
                    step + 1,
                    len(train_dl),
                    total_loss / (step + 1),
                    scheduler.get_last_lr()[0],
                )

        metrics = evaluate(student, eval_dl, device)
        logger.info(
            "Epoch %d: BA=%.4f F1=%.4f (ent=%.4f neu=%.4f con=%.4f)",
            epoch + 1,
            metrics["balanced_accuracy"],
            metrics["f1"],
            metrics["f1_entailment"],
            metrics["f1_neutral"],
            metrics["f1_contradiction"],
        )

        if metrics["balanced_accuracy"] > best_ba:
            best_ba = metrics["balanced_accuracy"]
            student.save_pretrained(str(output_dir))
            student_tok.save_pretrained(str(output_dir))
            logger.info("New best BA=%.4f - saved", best_ba)

    final_metrics: dict[str, Any] = evaluate(student, eval_dl, device)
    final_metrics["config"] = {
        "teacher": config.teacher,
        "student": config.student,
        "lr": config.lr,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "temperature": config.temperature,
        "alpha": config.alpha,
        "seed": config.seed,
        "eval_limit": config.eval_limit,
        "num_workers": config.num_workers,
        "device_preference": config.device,
        "teacher_params": teacher_params,
        "student_params": student_params,
        "compression_ratio": teacher_params / student_params,
        "device": str(device),
    }
    (output_dir / "final_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Final: BA=%.4f, saved to %s",
        final_metrics["balanced_accuracy"],
        output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
