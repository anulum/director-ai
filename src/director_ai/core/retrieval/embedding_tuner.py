# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Domain Embedding Fine-Tuner

"""Fine-tune embedding models on customer documents for domain adaptation.

Creates contrastive training pairs from document chunks: adjacent
chunks within the same document are positives, chunks from different
documents are negatives. Trains for a few epochs to adapt the embedding
space to the customer's terminology.

Requires ``pip install sentence-transformers>=4``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("DirectorAI.EmbeddingTuner")


@dataclass
class TuneResult:
    """Outcome of an embedding fine-tune: model path and training counts."""

    model_path: str
    train_samples: int
    epochs: int
    loss_start: float
    loss_end: float


def _mean_loss(
    model: Any, train_loss: Any, examples: list[Any], batch_size: int
) -> float:
    """Return the mean loss over *examples* without taking a gradient step.

    Uses the model's batching collate and the configured loss module under
    ``torch.no_grad`` — the same computation the training loop performs, so the
    reported start/end loss reflects the real objective on the training pairs.
    """
    import torch
    from torch.utils.data import DataLoader

    loader: DataLoader[Any] = DataLoader(
        cast(Any, examples),
        shuffle=False,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate,
    )
    total = 0.0
    count = 0
    with torch.no_grad():
        for features, labels in loader:
            total += float(train_loss(features, labels).item()) * len(labels)
            count += len(labels)
    return total / count if count else 0.0


def tune_embeddings(
    documents: list[list[str]],
    base_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "models/tuned_embeddings",
    epochs: int = 3,
    batch_size: int = 16,
    seed: int = 42,
) -> TuneResult:
    """Fine-tune embedding model on document chunks.

    Parameters
    ----------
    documents : list of list of str
        Each inner list is chunks from one document. Adjacent chunks
        form positive pairs; chunks from different docs form negatives.
    base_model : str
        HuggingFace sentence-transformers model to fine-tune.
    output_dir : str
        Where to save the fine-tuned model.
    epochs : int
        Training epochs (2-5 recommended).
    batch_size : int
        Training batch size.

    Returns
    -------
    TuneResult with model path and training metrics.
    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "sentence-transformers required. Install: pip install director-ai[embeddings]"
        ) from e
    losses = import_module("sentence_transformers.losses")

    random.seed(seed)
    model = SentenceTransformer(base_model)

    # Build contrastive pairs from adjacent chunks
    train_examples = []
    for doc_chunks in documents:
        for i in range(len(doc_chunks) - 1):
            # Positive: adjacent chunks from same document
            train_examples.append(
                InputExample(texts=[doc_chunks[i], doc_chunks[i + 1]], label=1.0)
            )
        # Negative: random chunk from a different document
        for chunk in doc_chunks[:3]:
            other_docs = [d for d in documents if d is not doc_chunks]
            if other_docs:
                other_doc = random.choice(other_docs)
                other_chunk = random.choice(other_doc)
                train_examples.append(
                    InputExample(texts=[chunk, other_chunk], label=0.0)
                )

    if not train_examples:
        raise ValueError("Need at least 2 documents with 2+ chunks each for tuning")

    logger.info(
        "Training on %d pairs from %d documents (%d epochs)",
        len(train_examples),
        len(documents),
        epochs,
    )

    # PyTorch DataLoader expects a Dataset; sentence-transformers'
    # ``InputExample`` list is accepted at runtime via its __getitem__
    # protocol. cast pins that contract without a suppression.
    loader: DataLoader[Any] = DataLoader(
        cast(Any, train_examples),
        shuffle=True,
        batch_size=batch_size,
    )
    train_loss = losses.CosineSimilarityLoss(model)

    # Measure the mean training loss before and after fitting so TuneResult
    # reports real diagnostics rather than placeholder zeros.
    loss_start = _mean_loss(model, train_loss, train_examples, batch_size)

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=max(1, len(loader) // 10),
        show_progress_bar=True,
    )

    loss_end = _mean_loss(model, train_loss, train_examples, batch_size)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    logger.info("Saved fine-tuned model to %s", output_dir)

    return TuneResult(
        model_path=output_dir,
        train_samples=len(train_examples),
        epochs=epochs,
        loss_start=loss_start,
        loss_end=loss_end,
    )
