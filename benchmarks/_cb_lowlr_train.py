"""CB at LR=5e-6 (catastrophic-forgetting experiment vs original 1e-5)."""

import json
import os
import time

import numpy as np
from datasets import load_dataset
from sklearn.metrics import balanced_accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "yaxili96/FactCG-DeBERTa-v3-Large"
OUTPUT_DIR = "/home/director-ai/models/factcg-cb-lowlr"
TEMPLATE = (
    "{text_a}\n\nChoose your answer: based on the paragraph above "
    'can we conclude that "{text_b}"?\n\nOPTIONS:\n- Yes\n- No\n'
    "I think the answer is "
)

ds_raw = load_dataset("super_glue", "cb")


def convert(ex):
    label = 1 if ex["label"] == 0 else 0
    return {
        "text": TEMPLATE.format(text_a=ex["premise"], text_b=ex["hypothesis"]),
        "label": label,
    }


train = ds_raw["train"].map(convert, remove_columns=ds_raw["train"].column_names)
val_split = ds_raw["validation"].train_test_split(test_size=0.5, seed=42)
val = val_split["train"].map(convert, remove_columns=val_split["train"].column_names)
test = val_split["test"].map(convert, remove_columns=val_split["test"].column_names)
print(f"CB: train={len(train)}, val={len(val)}, test={len(test)}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)


def tok_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)


train = train.map(tok_fn, batched=True, remove_columns=["text"])
val = val.map(tok_fn, batched=True, remove_columns=["text"])
test = test.map(tok_fn, batched=True, remove_columns=["text"])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    learning_rate=5e-6,  # key: half the original 1e-5
    weight_decay=0.01,
    warmup_ratio=0.2,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="balanced_accuracy",
    greater_is_better=True,
    fp16=True,
    logging_steps=10,
    report_to="none",
)

t0 = time.time()
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
test_result = trainer.evaluate(test)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

result = {
    "dataset": "cb_superglue",
    "base_model": BASE_MODEL,
    "learning_rate": 5e-6,
    "test_balanced_accuracy": test_result["eval_balanced_accuracy"],
    "training_time_minutes": round((time.time() - t0) / 60, 1),
}
with open(os.path.join(OUTPUT_DIR, "training_result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(f"CB-lowLR COMPLETE — bal_acc={result['test_balanced_accuracy']:.4f}")
