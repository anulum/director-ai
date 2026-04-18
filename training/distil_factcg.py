# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Knowledge distillation: FactCG → MiniLM
"""Distil FactCG-DeBERTa-v3-Large (0.4B) into MiniLM-L6 (22M).

Three-phase pipeline:

1. **Soft-label generation**: Run FactCG on AggreFact dev set (30K
   samples) to produce soft probability labels.
2. **Student training**: Fine-tune ``microsoft/MiniLM-L6-H384-uncased``
   on soft labels via KL divergence loss.
3. **ONNX export**: Export trained student to ONNX + INT8 quantisation.

Usage::

    # Phase 1: generate soft labels (requires GPU for teacher)
    HIP_VISIBLE_DEVICES=1 python training/distil_factcg.py --phase labels \\
        --output training/output/distil_labels.json

    # Phase 2: train student (GPU recommended, CPU possible)
    python training/distil_factcg.py --phase train \\
        --labels training/output/distil_labels.json \\
        --output training/output/distilled-nli

    # Phase 3: export to ONNX
    python training/distil_factcg.py --phase export \\
        --model training/output/distilled-nli \\
        --output training/output/distilled-nli-onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEACHER_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
TEACHER_REVISION = "0430e3509dbd28d2dff7a117c0eae25359ff3e80"
STUDENT_MODEL = "nreimers/MiniLM-L6-H384-uncased"
AGGREFACT_DATASET = "lytang/LLM-AggreFact"

# FactCG requires this instruction template — without it, the model
# returns near-constant probabilities (~0.10/0.90) for all inputs.
_FACTCG_TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)


def phase_labels(args):
    """Phase 1: generate soft labels from FactCG teacher."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading teacher: %s", TEACHER_MODEL)
    tokeniser = AutoTokenizer.from_pretrained(TEACHER_MODEL, revision=TEACHER_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL, revision=TEACHER_REVISION
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    logger.info("Teacher on %s", device)

    ds = load_dataset(AGGREFACT_DATASET, split="dev")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    logger.info("Samples: %d", len(ds))

    records = []
    for i, sample in enumerate(ds):
        premise = sample["doc"]
        hypothesis = sample["claim"]
        label = int(sample["label"])

        # FactCG requires instruction template for correct scoring.
        # The template ENDING ("OPTIONS:\n- Yes\n- No\nI think the
        # answer is ") must NOT be truncated — pre-truncate the doc
        # so that the full templated text fits within max_length.
        suffix = _FACTCG_TEMPLATE.format(text_a="", text_b=hypothesis)
        suffix_len = len(tokeniser.encode(suffix, add_special_tokens=False))
        max_doc_tokens = 512 - suffix_len - 4  # margin for special tokens
        if max_doc_tokens < 32:
            max_doc_tokens = 32
        doc_ids = tokeniser.encode(premise, add_special_tokens=False)
        if len(doc_ids) > max_doc_tokens:
            premise_trunc = tokeniser.decode(
                doc_ids[:max_doc_tokens], skip_special_tokens=True
            )
        else:
            premise_trunc = premise

        text = _FACTCG_TEMPLATE.format(text_a=premise_trunc, text_b=hypothesis)
        inputs = tokeniser(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            raw_probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

        # FactCG convention: raw_probs[0] is P(Yes/entailment) but maps
        # INVERSELY to AggreFact labels (FactCG gives HIGH P[0] for
        # inconsistent samples). Flip: student P[0] = 1-raw_probs[0]
        # = P(supported in AggreFact convention).
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
            logger.info("[%d/%d]", i + 1, len(ds))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2))
    logger.info("Saved %d soft labels to %s", len(records), out)


def phase_train(args):
    """Phase 2: train student on soft labels."""

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    records = json.loads(Path(args.labels).read_text())
    logger.info("Loaded %d soft-label records", len(records))

    # Use --student-model if provided, else local safetensors, else default
    if hasattr(args, "student_model") and args.student_model:
        student_path = args.student_model
    else:
        local_st = Path("training/output/deberta-v3-xsmall-safetensors")
        student_path = str(local_st) if local_st.is_dir() else STUDENT_MODEL
    logger.info("Student model: %s", student_path)

    tokeniser = AutoTokenizer.from_pretrained(student_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        student_path, num_labels=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    class SoftLabelDataset(Dataset):
        def __init__(self, records, tokeniser, max_length=256):
            self.records = records
            self.tokeniser = tokeniser
            self.max_length = max_length

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            enc = self.tokeniser(
                r["premise"],
                r["hypothesis"],
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

    dataset = SoftLabelDataset(records, tokeniser)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=len(loader) // 10,
        num_training_steps=len(loader) * args.epochs,
    )

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    ce_loss = torch.nn.CrossEntropyLoss()
    alpha = args.alpha  # soft/hard label blend ratio

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            soft_labels = batch["soft_labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Soft label loss (KL divergence from teacher)
            student_log_probs = torch.log_softmax(
                outputs.logits / args.temperature, dim=-1
            )
            teacher_probs = torch.softmax(soft_labels / args.temperature, dim=-1)
            soft_loss = kl_loss(student_log_probs, teacher_probs) * (
                args.temperature**2
            )

            # Hard label loss (cross-entropy from AggreFact ground truth)
            hard_targets = soft_labels.argmax(dim=-1)  # 0=supported, 1=contradicted
            hard_loss = ce_loss(outputs.logits, hard_targets)

            # Blended loss
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, args.epochs, avg_loss)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokeniser.save_pretrained(out)
    logger.info("Student saved to %s", out)


def phase_export(args):
    """Phase 3: export to ONNX + optional INT8 quantisation."""
    from transformers import AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(args.model)

    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        ort_model = ORTModelForSequenceClassification.from_pretrained(
            args.model, export=True
        )
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        ort_model.save_pretrained(out)
        tokeniser.save_pretrained(out)
        logger.info("ONNX exported to %s", out)

        if args.quantise:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            onnx_path = out / "model.onnx"
            quant_path = out / "model_quantised.onnx"
            quantize_dynamic(
                str(onnx_path),
                str(quant_path),
                weight_type=QuantType.QInt8,
            )
            logger.info("INT8 quantised: %s", quant_path)

    except ImportError:
        logger.error("ONNX export requires optimum: pip install optimum[onnxruntime]")
        raise


def main():
    p = argparse.ArgumentParser(description="Distil FactCG → MiniLM")
    p.add_argument(
        "--phase",
        choices=["labels", "train", "export"],
        required=True,
    )
    p.add_argument("--output", required=True)
    p.add_argument("--labels", help="Soft-label JSON (phase=train)")
    p.add_argument("--model", help="Model dir (phase=export)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Soft/hard label blend (0=hard only, 1=soft only)",
    )
    p.add_argument(
        "--student-model",
        type=str,
        default="",
        help="Student model path (local safetensors dir or HF ID)",
    )
    p.add_argument("--quantise", action="store_true")
    args = p.parse_args()

    if args.phase == "labels":
        phase_labels(args)
    elif args.phase == "train":
        if not args.labels:
            p.error("--labels required for phase=train")
        phase_train(args)
    elif args.phase == "export":
        if not args.model:
            p.error("--model required for phase=export")
        phase_export(args)


if __name__ == "__main__":
    main()
