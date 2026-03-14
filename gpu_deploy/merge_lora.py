# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LoRA Adapter Merge Utility
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Merge a PEFT LoRA adapter into the base model and save as standalone.

Usage::

    python gpu_deploy/merge_lora.py \\
        --adapter models/lora-halueval-r8-lr1e4 \\
        --output models/merged-halueval-r8-lr1e4
"""

from __future__ import annotations

import argparse
from pathlib import Path


def merge(
    adapter_path: str,
    output_path: str,
    base_model: str = "yaxili96/FactCG-DeBERTa-v3-Large",
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        low_cpu_mem_usage=False,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # merge_and_unload() strips PEFT adapter metadata including factcg=True.
    # Restore it so downstream eval applies the FactCG instruction template.
    import json

    config_path = Path(output_path) / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        if not cfg.get("factcg"):
            cfg["factcg"] = True
            config_path.write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument(
        "--base-model",
        default="yaxili96/FactCG-DeBERTa-v3-Large",
        help="Base model to merge adapter into",
    )
    args = parser.parse_args()
    merge(args.adapter, args.output, args.base_model)
