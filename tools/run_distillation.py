# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Knowledge Distillation Pipeline

"""Knowledge distillation from a 7B teacher model to DeBERTa student.

Trains the student to match the teacher's output distribution via
KL divergence loss instead of hard labels. This transfers the
teacher's calibration to the student without catastrophic forgetting.

Phase 4A options:
  (a) Bespoke-MiniCheck-7B teacher — non-commercial (CC BY-NC 4.0)
  (b) Claude Haiku API teacher — commercial-safe

Usage::

    # Step 1: Generate soft labels from teacher
    python tools/run_distillation.py --generate-labels \\
        --teacher bespoke-minicheck \\
        --dataset halueval \\
        --output labels/teacher_soft_labels.json

    # Step 2: Train student on soft labels
    python tools/run_distillation.py --train \\
        --labels labels/teacher_soft_labels.json \\
        --output-dir models/distilled-student

    # Alternative: API teacher
    python tools/run_distillation.py --generate-labels \\
        --teacher claude-haiku \\
        --dataset halueval \\
        --output labels/claude_soft_labels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("DirectorAI.Distillation")


def _load_training_data(
    dataset_name: str,
    max_samples: int | None = None,
) -> list[dict]:
    """Load training pairs for distillation."""
    from tools.run_lora_training import load_dataset_pairs

    return load_dataset_pairs(dataset_name, max_samples=max_samples)


def generate_teacher_labels_local(
    pairs: list[dict],
    teacher_model: str = "bespokelabs/Bespoke-MiniCheck-7B",
    batch_size: int = 8,
    output_path: str = "labels/teacher_soft_labels.json",
) -> list[dict]:
    """Generate soft probability labels from a local 7B teacher."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading teacher: %s", teacher_model)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        teacher_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    labeled: list[dict] = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        texts = [f"{p['premise']}\n\n{p['hypothesis']}" for p in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        for j, p in enumerate(batch):
            labeled.append(
                {
                    "premise": p["premise"],
                    "hypothesis": p["hypothesis"],
                    "hard_label": p["label"],
                    "soft_probs": probs[j].tolist(),
                }
            )

        if (i + batch_size) % 1000 < batch_size:
            logger.info(
                "Labeled %d/%d pairs", min(i + batch_size, len(pairs)), len(pairs)
            )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labeled, f)
    logger.info("Saved %d soft labels to %s", len(labeled), output_path)
    return labeled


def generate_teacher_labels_api(
    pairs: list[dict],
    provider: str = "anthropic",
    model_name: str = "claude-haiku-4-5-20251001",
    output_path: str = "labels/api_soft_labels.json",
    max_concurrent: int = 5,
) -> list[dict]:
    """Generate labels from API-based teacher."""
    labeled: list[dict] = []

    for i, p in enumerate(pairs):
        prompt = (
            f"Premise: {p['premise'][:1000]}\n"
            f"Hypothesis: {p['hypothesis'][:500]}\n\n"
            "Is the hypothesis supported by the premise? "
            'Reply JSON: {{"supported_probability": 0.0-1.0}}'
        )

        try:
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic()
                result = client.messages.create(
                    model=model_name,
                    max_tokens=50,
                    messages=[{"role": "user", "content": prompt}],
                )
                reply = result.content[0].text if result.content else ""
            elif provider == "openai":
                import openai

                client = openai.OpenAI()
                result = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    response_format={"type": "json_object"},
                )
                reply = result.choices[0].message.content or ""
            else:
                raise ValueError(f"Unknown provider: {provider}")

            data = json.loads(reply)
            prob = float(data.get("supported_probability", 0.5))

        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            prob = 0.5

        labeled.append(
            {
                "premise": p["premise"],
                "hypothesis": p["hypothesis"],
                "hard_label": p["label"],
                "soft_probs": [1.0 - prob, prob],
            }
        )

        if (i + 1) % 100 == 0:
            logger.info("API-labeled %d/%d pairs", i + 1, len(pairs))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labeled, f)
    logger.info("Saved %d API labels to %s", len(labeled), output_path)
    return labeled


def train_distilled_student(
    labels_path: str,
    student_model: str = "yaxili96/FactCG-DeBERTa-v3-Large",
    output_dir: str = "models/distilled-student",
    temperature: float = 2.0,
    alpha: float = 0.5,
    lr: float = 5e-5,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lora_rank: int = 8,
    lora_alpha_val: int = 16,
    freeze_encoder: bool = False,
    seed: int = 42,
) -> dict:
    """Train student with KL divergence loss from teacher soft labels.

    Loss = alpha * KL(teacher || student) + (1-alpha) * CE(hard_label, student).

    When freeze_encoder=True, only the classification head is trained.
    This avoids encoder representation damage (29/29 LoRA experiments
    degraded FactCG on AggreFact).
    """
    import torch
    import torch.nn.functional as f_nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(labels_path) as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        student_model,
        low_cpu_mem_usage=False,
    )

    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Freeze-encoder mode: %d trainable params (head only)", trainable)
    else:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_rank,
            lora_alpha=lora_alpha_val,
            lora_dropout=0.1,
            target_modules=["query_proj", "value_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    factcg_template = (
        "{text_a}\n\nChoose your answer: based on the paragraph above "
        'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
        "I think the answer is "
    )

    class DistillDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            text = factcg_template.format(
                text_a=r["premise"],
                text_b=r["hypothesis"],
            )
            enc = tokenizer(text, truncation=True, max_length=512, padding="max_length")
            return {
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "hard_label": torch.tensor(r["hard_label"], dtype=torch.long),
                "soft_probs": torch.tensor(r["soft_probs"], dtype=torch.float),
            }

    dataset = DistillDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * epochs // grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            hard_labels = batch["hard_label"].to(device)
            soft_probs = batch["soft_probs"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = outputs.logits

            # KL divergence loss (soft labels)
            student_log_probs = f_nn.log_softmax(student_logits / temperature, dim=1)
            teacher_probs = soft_probs
            # Pad/trim teacher probs to match student output dims
            if teacher_probs.size(1) != student_log_probs.size(1):
                n = student_log_probs.size(1)
                if teacher_probs.size(1) < n:
                    pad = torch.zeros(
                        teacher_probs.size(0),
                        n - teacher_probs.size(1),
                        device=device,
                    )
                    teacher_probs = torch.cat([teacher_probs, pad], dim=1)
                else:
                    teacher_probs = teacher_probs[:, :n]
            teacher_probs = teacher_probs / teacher_probs.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-8)
            kl_loss = f_nn.kl_div(
                student_log_probs, teacher_probs, reduction="batchmean"
            )
            kl_loss = kl_loss * (temperature**2)

            # Hard label CE loss
            ce_loss = f_nn.cross_entropy(student_logits, hard_labels)

            loss = alpha * kl_loss + (1 - alpha) * ce_loss
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum

        avg_loss = total_loss / len(loader)
        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {
        "final_loss": avg_loss,
        "epochs": epochs,
        "temperature": temperature,
        "alpha": alpha,
        "mode": "head_only" if freeze_encoder else f"lora_r{lora_rank}",
        "lora_rank": 0 if freeze_encoder else lora_rank,
        "train_samples": len(data),
    }
    with open(os.path.join(output_dir, "distillation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Distilled student saved to %s", output_dir)
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Knowledge distillation pipeline")
    parser.add_argument("--generate-labels", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--teacher",
        default="bespoke-minicheck",
        help="Teacher: bespoke-minicheck, claude-haiku, gpt-4o-mini",
    )
    parser.add_argument("--dataset", default="halueval")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--labels", default="labels/teacher_soft_labels.json")
    parser.add_argument("--output", default="labels/teacher_soft_labels.json")
    parser.add_argument("--output-dir", default="models/distilled-student")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="KL weight (1-alpha = CE weight)"
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder, train only classification head (safest mode)",
    )
    args = parser.parse_args()

    if args.generate_labels:
        pairs = _load_training_data(args.dataset, max_samples=args.max_samples)

        if args.teacher == "bespoke-minicheck":
            generate_teacher_labels_local(
                pairs,
                teacher_model="bespokelabs/Bespoke-MiniCheck-7B",
                output_path=args.output,
            )
        elif args.teacher in ("claude-haiku", "anthropic"):
            generate_teacher_labels_api(
                pairs,
                provider="anthropic",
                model_name="claude-haiku-4-5-20251001",
                output_path=args.output,
            )
        elif args.teacher in ("gpt-4o-mini", "openai"):
            generate_teacher_labels_api(
                pairs,
                provider="openai",
                model_name="gpt-4o-mini",
                output_path=args.output,
            )

    if args.train:
        train_distilled_student(
            labels_path=args.labels,
            output_dir=args.output_dir,
            temperature=args.temperature,
            alpha=args.alpha,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lora_rank=args.lora_rank,
            freeze_encoder=args.freeze_encoder,
        )
