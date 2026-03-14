"""Evaluate a model on full LLM-AggreFact (29K) from local JSONL."""
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tag = sys.argv[1]
model_path = sys.argv[2] if len(sys.argv) > 2 else ""

print(f"Evaluating: {tag} (model_path={model_path!r})")

if tag == "baseline" or not model_path:
    mn = "yaxili96/FactCG-DeBERTa-v3-Large"
    tokenizer = AutoTokenizer.from_pretrained(mn)
    model = AutoModelForSequenceClassification.from_pretrained(mn).cuda().eval()
else:
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_lora:
        from peft import PeftModel

        base = AutoModelForSequenceClassification.from_pretrained(
            "yaxili96/FactCG-DeBERTa-v3-Large"
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained("yaxili96/FactCG-DeBERTa-v3-Large")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.cuda().eval()

print("Loading LLM-AggreFact from local JSONL...")
rows = []
with open("data/aggrefact_test.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
print(f"Loaded {len(rows)} samples")

TEMPLATE = (
    '{text_a}\n\nChoose your answer: based on the paragraph above '
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    'I think the answer is '
)

t0 = time.perf_counter()
by_dataset = {}
for idx, row in enumerate(rows):
    doc = row.get("doc", "")
    claim = row.get("claim", "")
    label = row.get("label")
    ds_name = row.get("dataset", "unknown")
    if label is None or not doc or not claim:
        continue
    text = TEMPLATE.format(text_a=doc[:2000], text_b=claim[:500])
    enc = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)
    score = probs[0, 1].item()
    by_dataset.setdefault(ds_name, []).append((int(label), score))
    if (idx + 1) % 5000 == 0:
        elapsed = time.perf_counter() - t0
        rate = (idx + 1) / elapsed
        eta = (len(rows) - idx - 1) / rate / 60
        print(f"  {idx+1}/{len(rows)} ({rate:.0f}/s, ETA {eta:.0f}m)", flush=True)

elapsed = time.perf_counter() - t0
best_thresh, best_avg = 0.5, 0.0
for t_int in range(10, 91):
    t = t_int / 100.0
    accs = [
        balanced_accuracy_score(
            [p[0] for p in v], [1 if p[1] >= t else 0 for p in v]
        )
        for v in by_dataset.values()
    ]
    avg = float(np.mean(accs))
    if avg > best_avg:
        best_avg, best_thresh = avg, t

per_ds = {}
for ds_name in sorted(by_dataset):
    v = by_dataset[ds_name]
    ba = balanced_accuracy_score(
        [p[0] for p in v], [1 if p[1] >= best_thresh else 0 for p in v]
    )
    per_ds[ds_name] = {"ba": round(float(ba), 4), "n": len(v)}

result = {
    "model": tag,
    "macro_ba": round(float(best_avg), 4),
    "threshold": best_thresh,
    "elapsed": round(elapsed, 1),
    "per_dataset": per_ds,
}
out_path = f"results/distill_eval_{tag}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"{tag}: {best_avg*100:.2f}% BA (t={best_thresh}) in {elapsed/60:.1f}m -> {out_path}")
